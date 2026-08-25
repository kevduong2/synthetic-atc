"""Stage-plumbing tests for the L2 recipe orchestrator.

The heavy calls (`finetune`, `run_grpo`, `evaluate_dev`) are stubbed: what is
under test is the glue -- which pool each arm builds, that the mixture honors
`--mix-ratio` deterministically, that the GRPO stage starts from the SFT
checkpoint, and that `run.json` records the budget.
"""

import json

import numpy as np
import pytest

from training import recipe as recipe_mod
from training.grpo import Utterance, UtterancePool
from training.recipe import LazyFeatures, RecipeConfig, mixed_pool, run_recipe

SR = 16000


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
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
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

    def fake_load_pool(spec):
        if spec.manifests:
            return UtterancePool([_fake_dataset([f"s{i}" for i in range(12)])])
        return UtterancePool([_fake_dataset([f"r{i}" for i in range(36)])])

    def fake_finetune(model, features, **kwargs):
        calls["sft"] = {"n_features": len(features), **kwargs}
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
