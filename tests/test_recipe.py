"""Stage-plumbing tests for the L2 recipe orchestrator.

The heavy calls (`finetune`, `run_grpo`, `evaluate_dev`) are stubbed: what is
under test is the glue -- which pool each arm builds, that the mixture honors
`--mix-ratio` deterministically, that the GRPO stage starts from the SFT
checkpoint, and that `run.json` records the budget.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atcgen.rl.finetune_lite import finetune
from scripts import bench_devices
from training import recipe as recipe_mod
from training.grpo import Utterance, UtterancePool
from training.recipe import LazyFeatures, RecipeConfig, mixed_pool, run_recipe

SR = 16000


@pytest.fixture(autouse=True)
def tracking_off(monkeypatch):
    monkeypatch.setenv("ATCGAN_TRACKING", "off")


class _StubPool:
    def __init__(self, texts):
        self._texts = list(texts)

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, index):
        return Utterance(np.zeros(SR, dtype=np.float32), self._texts[int(index)])


def _fake_dataset(texts):
    from datasets import Dataset

    return Dataset.from_list([
        {"audio": {"array": np.zeros(64, dtype=np.float32).tolist(), "sampling_rate": SR},
         "text": text}
        for text in texts
    ])


# -- mixture ---------------------------------------------------------------

def test_mixture_hits_the_requested_ratio():
    real = UtterancePool([_fake_dataset([f"r{i}" for i in range(40)])])
    synthetic = UtterancePool([_fake_dataset([f"s{i}" for i in range(40)])])
    mixed = mixed_pool(real, synthetic, ratio=0.75, seed=0)

    texts = [mixed[i].text for i in range(len(mixed))]
    n_real = sum(text.startswith("r") for text in texts)
    assert len(mixed) == 53  # 40 real binds the total at 40/0.75
    assert n_real / len(mixed) == pytest.approx(0.75, abs=0.02)


def test_mixture_is_seeded_and_interleaved():
    real = UtterancePool([_fake_dataset([f"r{i}" for i in range(20)])])
    synthetic = UtterancePool([_fake_dataset([f"s{i}" for i in range(20)])])
    first = [mixed_pool(real, synthetic, 0.5, seed=7)[i].text for i in range(20)]
    same = [mixed_pool(real, synthetic, 0.5, seed=7)[i].text for i in range(20)]
    other = [mixed_pool(real, synthetic, 0.5, seed=8)[i].text for i in range(20)]
    assert first == same and first != other
    # Interleaved, not concatenated: both sources appear in the first stretch.
    head = first[:10]
    assert any(t.startswith("r") for t in head) and any(t.startswith("s") for t in head)


def test_mixture_degenerates_to_a_single_source():
    real = UtterancePool([_fake_dataset(["r0", "r1"])])
    empty = UtterancePool([])
    assert len(mixed_pool(real, empty, 0.75, seed=0)) == 2
    assert len(mixed_pool(empty, real, 0.75, seed=0)) == 2


# -- lazy features ---------------------------------------------------------

def test_lazy_features_match_the_finetune_contract():
    class Processor:
        tokenizer = lambda self, text: SimpleNamespace(input_ids=[1, 2, 3])

        def __call__(self, audio, sampling_rate):
            return SimpleNamespace(input_features=np.zeros((1, 80, 3000), np.float32))

    processor = Processor()
    features = LazyFeatures(_StubPool(["radar contact", ""]), processor)
    assert len(features) == 2
    row = features[np.int64(0)]  # finetune indexes with numpy ints
    assert np.asarray(row["input_features"]).shape == (80, 3000)
    assert isinstance(row["labels"], list) and row["labels"]


# -- stage plumbing --------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch):
    """Replace the heavy stages; record what the recipe handed them."""
    calls = {}

    class FakeProcessor:
        tokenizer = lambda self, text: SimpleNamespace(input_ids=[1, 2, 3])

        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def __call__(self, audio, sampling_rate, **kwargs):
            return SimpleNamespace(input_features=np.zeros((1, 80, 8), np.float32))

        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "preprocessor_config.json").write_text("{}")

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def to(self, device):
            return self

        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "config.json").write_text("{}")

    def fake_load_pool(spec):
        if spec.manifests:
            return UtterancePool([_fake_dataset([f"s{i}" for i in range(12)])])
        return UtterancePool([_fake_dataset([f"r{i}" for i in range(36)])])

    def fake_finetune(model, features, **kwargs):
        calls["sft"] = {"n_features": len(features), **kwargs}
        callback = kwargs.get("on_step")
        if callback:
            for step in range(1, kwargs["steps"] + 1):
                callback(step, 4.0 / step)
        model._ft_losses = [3.0, 2.5, 2.0]
        return model

    def fake_run_grpo(cfg, *, train_pool=None, dev_pool=None):
        calls["grpo"] = {"cfg": cfg, "pool_size": len(train_pool),
                         "dev_size": len(dev_pool) if dev_pool else None}
        return {"best": {"dev_wer": 0.4, "step": 2, "checkpoint": cfg.out + "/best"},
                "last_checkpoint": cfg.out + "/last", "wall_seconds": 1.0}

    def fake_evaluate_dev(model, processor, pool, device, batch_size, max_new_tokens):
        calls["dev"] = len(pool)
        return {"wer": 0.5, "hallucination_rate": 0.0, "samples": len(pool)}

    monkeypatch.setattr(recipe_mod, "load_pool", fake_load_pool)
    monkeypatch.setattr(recipe_mod, "WhisperProcessor", FakeProcessor)
    monkeypatch.setattr(recipe_mod, "WhisperForConditionalGeneration", FakeModel)
    monkeypatch.setattr(recipe_mod, "finetune", fake_finetune)
    monkeypatch.setattr(recipe_mod, "run_grpo", fake_run_grpo)
    monkeypatch.setattr(recipe_mod, "evaluate_dev", fake_evaluate_dev)
    return calls


def _config(tmp_path, arm, **overrides):
    return RecipeConfig(
        arm=arm, out=str(tmp_path / arm), model="openai/whisper-tiny.en",
        real_split="train", real_indices=(0, 36), synth_manifests=["data/synth"],
        dev_split="train", dev_indices=(100, 104),
        sft_steps=3, sft_batch=2, grpo_steps=2, seed=11, device="cpu", **overrides)


@pytest.mark.parametrize("arm,expected_pool", [
    ("real_only", 36), ("synth_only", 12), ("mix", 48),
])
def test_arms_build_the_right_sft_pool(tmp_path, stubbed, arm, expected_pool):
    run = run_recipe(_config(tmp_path, arm))
    # mix: 36 real + 12 synth is already exactly 0.75 real, so neither binds.
    assert stubbed["sft"]["n_features"] == expected_pool
    assert stubbed["sft"]["steps"] == 3 and stubbed["sft"]["seed"] == 11
    assert run["stages"][0]["samples_seen"] == 6  # budget-matched: steps * batch
    assert run["stages"][0]["dev"]["wer"] == 0.5
    assert "grpo" not in stubbed


def test_mix_grpo_chains_the_sft_checkpoint_into_grpo(tmp_path, stubbed):
    run = run_recipe(_config(tmp_path, "mix_grpo"))
    grpo_cfg = stubbed["grpo"]["cfg"]
    assert grpo_cfg.init == str(tmp_path / "mix_grpo" / "sft")
    assert (tmp_path / "mix_grpo" / "sft" / "config.json").exists()
    assert grpo_cfg.steps == 2 and grpo_cfg.seed == 11
    # GRPO trains on the same ratio-controlled mixture, not a uniform union.
    assert stubbed["grpo"]["pool_size"] == 48
    assert stubbed["grpo"]["dev_size"] == 36

    assert [stage["name"] for stage in run["stages"]] == ["sft", "grpo"]
    assert run["final_checkpoint"].endswith("grpo/best")
    saved = json.loads((tmp_path / "mix_grpo" / "run.json").read_text())
    assert saved["arm"] == "mix_grpo" and saved["seed"] == 11
    assert saved["config"]["real_indices"] == [0, 36]


def test_unknown_arm_is_rejected(tmp_path, stubbed):
    with pytest.raises(ValueError, match="unknown arm"):
        run_recipe(_config(tmp_path, "nonsense"))


def test_finetune_on_step_fires_after_every_optimizer_step():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.config = SimpleNamespace(decoder_start_token_id=None, use_cache=True)

        def forward(self, input_features, labels):
            target = input_features.mean()
            return SimpleNamespace(loss=(self.weight - target).square())

    features = [
        {"input_features": np.full((2, 3), value, np.float32), "labels": [1, 2]}
        for value in (0.1, 0.2)
    ]
    seen = []
    finetune(TinyModel(), features, steps=3, batch_size=1, lr=1e-3,
             seed=2, device="cpu", on_step=lambda step, loss: seen.append((step, loss)))
    assert [step for step, _ in seen] == [1, 2, 3]
    assert all(isinstance(loss, float) for _, loss in seen)


def test_recipe_live_tracking_is_best_effort_and_finishes(tmp_path, stubbed, monkeypatch):
    class RecordingRun:
        def __init__(self):
            self.logs = []
            self.finished = False

        def log(self, values, step=None):
            self.logs.append((values, step))

        def finish(self):
            self.finished = True

    recorded = RecordingRun()
    started = {}

    def fake_start_run(**kwargs):
        started.update(kwargs)
        return recorded

    monkeypatch.setattr(recipe_mod, "start_run", fake_start_run)
    cfg = _config(tmp_path, "mix")
    cfg.sft_steps = 10
    run = run_recipe(cfg)
    keys = {key for values, _ in recorded.logs for key in values}
    assert started["project"] == "atcgan-fastcut"
    assert started["tags"] == ("asr", "mix")
    assert {"sft/loss", "sft/wall_seconds", "sft/pool_size",
            "dev/wer", "dev/hallucination_rate", "dev/samples"} <= keys
    assert recorded.finished is True
    assert json.loads((Path(cfg.out) / "run.json").read_text()) == run


def test_device_benchmark_quick_gan_cpu_smoke(tmp_path):
    out = tmp_path / "bench.json"
    result = bench_devices.main([
        "--gan", "--quick", "--device", "cpu", "--out", str(out),
        "--gan-warmup", "1", "--gan-steps", "1",
        "--gan-r1-warmup", "1", "--gan-r1-steps", "1",
        "--gan-batch", "1", "--gan-crop", "32", "--gan-base", "2",
        "--gan-n-res", "1", "--gan-scales", "1",
    ])
    saved = json.loads(out.read_text())
    assert result["results"]["gan"]["status"] == "ok"
    assert saved["results"]["gan"]["ordinary"]["repeats"] == 3
    assert saved["results"]["gan"]["r1"]["seconds"]["min"] >= 0.0
