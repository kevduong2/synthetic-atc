"""Tests for the RL reward harness.

`finetune`/`transcribe` are exercised with a tiny randomly-initialized
Whisper model (no cached checkpoint, no network). `TrueRewardHarness`
plumbing is tested by monkeypatching the heavy calls (`build_dataset`,
`load_manifest`, `finetune`, `transcribe`) so the test only checks
orchestration: the text pool is written once, the baseline WER is cached to
disk and reused, and the reward sign follows the stubbed WERs.
"""

import json

import numpy as np
import pytest
import torch
import yaml
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from atcgen.rl import reward as reward_mod
from atcgen.rl.finetune_lite import finetune, transcribe
from atcgen.rl.reward import TrueRewardHarness
from atcgen.rl.types import RewardResult

MODEL_ID = "openai/whisper-tiny.en"
VOCAB = 200


def _tiny_config(vocab_size: int = VOCAB) -> WhisperConfig:
    return WhisperConfig(
        vocab_size=vocab_size,
        num_mel_bins=80,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_source_positions=1500,
        max_target_positions=448,
        decoder_start_token_id=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=1,
    )


def _random_features(n: int, *, vocab_size: int = VOCAB, min_len: int = 4,
                     max_len: int = 12, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    features = []
    for _ in range(n):
        input_features = rng.standard_normal((80, 3000)).astype(np.float32)
        length = int(rng.integers(min_len, max_len))
        labels = [1] + rng.integers(3, vocab_size, size=length - 1).tolist()
        features.append({"input_features": input_features, "labels": labels})
    return features


def _fresh_model(vocab_size: int = VOCAB) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration(_tiny_config(vocab_size))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return model


def _processor_cached() -> bool:
    try:
        WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any load failure means "not cached"
        return False


# -- finetune_lite --------------------------------------------------------

def test_finetune_runs_and_returns_model_with_loss_curve():
    torch.manual_seed(0)
    model = _fresh_model()
    features = _random_features(8, seed=1)

    trained = finetune(model, features, steps=4, batch_size=2, lr=1e-3, seed=0, device="cpu")

    assert trained is model
    assert len(trained._ft_losses) == 4
    assert all(np.isfinite(loss) for loss in trained._ft_losses)


def test_finetune_batches_are_seed_reproducible():
    features = _random_features(10, seed=2)

    torch.manual_seed(42)
    model_a = _fresh_model()
    torch.manual_seed(42)
    model_b = _fresh_model()

    finetune(model_a, features, steps=5, batch_size=3, lr=1e-3, seed=7, device="cpu")
    finetune(model_b, features, steps=5, batch_size=3, lr=1e-3, seed=7, device="cpu")

    assert model_a._ft_losses == pytest.approx(model_b._ft_losses)


def test_finetune_different_seed_changes_batch_order_and_loss_curve():
    features = _random_features(10, seed=2)

    torch.manual_seed(42)
    model_a = _fresh_model()
    torch.manual_seed(42)
    model_b = _fresh_model()

    finetune(model_a, features, steps=5, batch_size=3, lr=1e-3, seed=7, device="cpu")
    finetune(model_b, features, steps=5, batch_size=3, lr=1e-3, seed=9, device="cpu")

    assert model_a._ft_losses != model_b._ft_losses


def test_transcribe_returns_one_hypothesis_per_feature():
    model = _fresh_model()
    features = _random_features(5, seed=3)

    class _FakeProcessor:
        def batch_decode(self, ids, skip_special_tokens=True):
            return [f"hyp{i}" for i in range(ids.shape[0])]

    hyps = transcribe(model, _FakeProcessor(), features, "cpu", batch_size=2, max_new_tokens=3)
    assert len(hyps) == 5
    assert all(isinstance(h, str) for h in hyps)


@pytest.mark.skipif(not _processor_cached(), reason="whisper-tiny.en processor not cached locally")
def test_finetune_collator_pads_variable_length_labels_with_real_tokenizer():
    processor = WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    cfg = _tiny_config(vocab_size=len(processor.tokenizer))
    cfg.decoder_start_token_id = decoder_start
    cfg.bos_token_id = decoder_start
    cfg.pad_token_id = processor.tokenizer.pad_token_id
    cfg.eos_token_id = processor.tokenizer.eos_token_id

    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    texts = ["cleared to land", "roger", "taxi via alpha hold short runway two seven"]
    rng = np.random.default_rng(3)
    features = []
    for text in texts:
        audio = rng.standard_normal(16000).astype(np.float32)
        input_features = processor.feature_extractor(
            audio, sampling_rate=16000).input_features[0]
        features.append({
            "input_features": input_features,
            "labels": processor.tokenizer(text).input_ids,
        })
    # exercises dynamic padding: label lengths differ across this batch
    assert len({len(f["labels"]) for f in features}) > 1

    trained = finetune(model, features, steps=2, batch_size=3, lr=1e-4, seed=0, device="cpu")
    assert len(trained._ft_losses) == 2
    assert all(np.isfinite(loss) for loss in trained._ft_losses)

    hyps = transcribe(model, processor, features, "cpu", batch_size=3, max_new_tokens=5)
    assert len(hyps) == 3
    assert all(isinstance(h, str) for h in hyps)


# -- TrueRewardHarness plumbing (heavy calls stubbed) ----------------------

class _StubConfig:
    pass


class _StubModel:
    """Stands in for a real Whisper model through the reward call."""

    name_or_path = "stub-model"

    def __init__(self):
        self.config = _StubConfig()

    def to(self, device):
        return self

    def eval(self):
        return self


def _make_harness(tmp_path, monkeypatch, *, dev_texts):
    """A harness whose HF-heavy setup is stubbed: no real model/dataset load."""
    harness = TrueRewardHarness.__new__(TrueRewardHarness)
    harness.work_dir = tmp_path / "work"
    harness.work_dir.mkdir(parents=True, exist_ok=True)
    harness.base_model = "stub/whisper"
    harness.dev_corpus = "stub/corpus"
    harness.dev_split = "train"
    harness.dev_indices = (0, len(dev_texts))
    harness.text_pool_size = 5
    harness.text_seed = 1234
    harness.n_synth = 3
    harness.ft_steps = 1
    harness.ft_batch = 1
    harness.ft_lr = 1e-5
    harness.ft_seed = 0
    harness.keep_audio = True
    harness.gen_seed = reward_mod.GEN_SEED
    harness.device = torch.device("cpu")
    harness.processor = object()  # never touched: transcribe/finetune are stubbed
    harness.pool_path = harness._ensure_text_pool()

    harness._dev_refs = list(dev_texts)
    harness._dev_categories = [None] * len(dev_texts)
    harness._dev_features = [{"input_features": None, "labels": []} for _ in dev_texts]
    harness._baseline_report = None

    monkeypatch.setattr(reward_mod, "WhisperForConditionalGeneration",
                        type("Fake", (), {"from_pretrained": staticmethod(lambda *a, **k: _StubModel())}))
    return harness


def test_text_pool_is_written_once_and_reused(tmp_path, monkeypatch):
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=["a", "b"])
    assert harness.pool_path.exists()
    lines = harness.pool_path.read_text().strip().splitlines()
    assert len(lines) == harness.text_pool_size
    first = json.loads(lines[0])
    assert {"spoken", "transcript", "role", "kind", "category", "weight"} <= first.keys()

    mtime_before = harness.pool_path.stat().st_mtime_ns
    reused = harness._ensure_text_pool()
    assert reused == harness.pool_path
    assert harness.pool_path.stat().st_mtime_ns == mtime_before


def test_baseline_is_cached_to_disk_and_reused(tmp_path, monkeypatch):
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=["cleared to land", "roger"])
    calls = []

    def fake_transcribe(model, processor, features, device, **kwargs):
        calls.append(1)
        return ["cleared to land", "roger"]  # perfect: baseline WER 0

    monkeypatch.setattr(reward_mod, "transcribe", fake_transcribe)

    harness._ensure_baseline()
    assert len(calls) == 1
    assert harness.baseline_report["wer"]["atc_normalized"] == 0.0
    cache_path = harness._baseline_cache_path()
    assert cache_path.exists()

    # a second harness pointed at the same work_dir reuses the cache file
    # without calling transcribe again
    harness2 = _make_harness(tmp_path, monkeypatch, dev_texts=["cleared to land", "roger"])
    harness2._ensure_baseline()
    assert len(calls) == 1
    assert harness2.baseline_report == harness.baseline_report


def test_call_reward_sign_follows_stubbed_wer(tmp_path, monkeypatch):
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=["cleared to land", "roger"])

    # baseline: model gets everything wrong (high WER)
    baseline_hyps = ["wrong wrong wrong", "wrong"]
    # post-finetune: model gets everything right (zero WER) -> reward > 0
    post_hyps = ["cleared to land", "roger"]

    responses = iter([baseline_hyps, post_hyps])
    monkeypatch.setattr(reward_mod, "transcribe",
                        lambda model, processor, features, device, **kw: next(responses))
    monkeypatch.setattr(reward_mod, "build_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(reward_mod, "load_manifest", lambda *a, **kw: object())
    monkeypatch.setattr(reward_mod, "prepare_features", lambda *a, **kw: [])
    monkeypatch.setattr(reward_mod, "finetune",
                        lambda model, features, **kw: model)
    monkeypatch.setattr(reward_mod, "JsonlTextSource", lambda *a, **kw: object())
    monkeypatch.setattr(reward_mod, "load_config", lambda path: object())

    config = {"mode": "procedural", "seed": 999}
    result = harness(config, str(tmp_path / "trial_0"))

    assert isinstance(result, RewardResult)
    assert result.proxy is False
    assert result.wer_baseline > result.wer_after
    assert result.reward > 0
    assert result.reward == pytest.approx(result.wer_baseline - result.wer_after)

    # the config written to trial_dir has the generator seed forced
    written = yaml.safe_load((tmp_path / "trial_0" / "config.yaml").read_text())
    assert written["seed"] == reward_mod.GEN_SEED
