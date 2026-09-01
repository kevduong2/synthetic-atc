"""Tests for the RL reward harness.

`finetune`/`transcribe` are exercised with a tiny randomly-initialized
Whisper model (no cached checkpoint, no network). `TrueRewardHarness`
plumbing is tested by monkeypatching the heavy calls (`build_dataset`,
`load_manifest`, `finetune`, `transcribe`) so the test only checks
orchestration: the text pool is written once, the baseline WER is cached to
disk and reused, and the reward sign follows the stubbed WERs.

The local dev-corpus loader is tested against a hand-built HF dataset rather
than a downloaded one: `load_real_atc`'s Hugging Face branch is three calls
(`load_dataset`, an optional `transcription` rename, the `Audio` cast) and
reproducing those over the same clips is what makes "parity" checkable
offline.
"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from atcgen.dataset.real_atc import (
    REAL_SR,
    is_local_corpus,
    load_local_corpus,
    load_real_atc,
)
from atcgen.rl import reward as reward_mod
from atcgen.rl.finetune_lite import finetune, transcribe
from atcgen.rl.reward import TrueRewardHarness
from atcgen.rl.types import RewardResult
from training.normalize import normalize_atc

MODEL_ID = "openai/whisper-tiny.en"
VOCAB = 200

#: Mixed case, digits and punctuation: everything `normalize_atc` folds away.
#: The loader must not fold any of it -- see `test_local_text_is_not_normalized`.
DEV_TEXTS = [
    "Cleared to land, runway 18.",
    "N123AB hold short RWY 36",
    "",  # a noise-only row: kept, and what the hallucination rate scores
]


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


# -- local dev corpus ------------------------------------------------------

def _sine(path, seconds=0.5, sr=REAL_SR, freq=220.0):
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


@pytest.fixture
def dev_corpus(tmp_path):
    """Three tiny wavs plus a CSV and a JSONL manifest naming them.

    The third clip is written at 8 kHz so the 16 kHz cast has something to do;
    the second manifest row is relative, so path resolution is exercised too.
    """
    clips = tmp_path / "clips"
    clips.mkdir()
    paths = [
        _sine(clips / "a.wav", freq=220.0),
        _sine(clips / "b.wav", freq=330.0),
        _sine(clips / "c.wav", sr=8000, freq=440.0),
    ]
    named = [str(paths[0]), "clips/b.wav", str(paths[2])]  # row 1 is relative

    csv_path = tmp_path / "dev.csv"
    csv_path.write_text(
        "audio,text\n" + "".join(
            f'{name},"{text}"\n' for name, text in zip(named, DEV_TEXTS)),
        encoding="utf-8")

    jsonl_path = tmp_path / "dev.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps({"audio": name, "text": text}) + "\n"
                for name, text in zip(named, DEV_TEXTS)),
        encoding="utf-8")
    return csv_path, jsonl_path, paths


def _hf_style(paths, texts):
    """What `load_real_atc` returns for an HF corpus, built without a download.

    Same three operations in the same order: a dataset with `transcription`,
    the rename to `text`, the cast to 16 kHz `Audio`.
    """
    from datasets import Audio, Dataset

    ds = Dataset.from_dict({"audio": [str(path) for path in paths],
                            "transcription": list(texts)})
    ds = ds.rename_column("transcription", "text")
    return ds.cast_column("audio", Audio(sampling_rate=REAL_SR))


def test_local_corpus_is_detected_by_suffix(tmp_path):
    assert is_local_corpus(tmp_path / "kixd_dev.csv")
    assert is_local_corpus("data/real/kixd/kixd_dev.jsonl")
    assert not is_local_corpus("jacktol/atc-dataset")
    assert not is_local_corpus("Jzuluaga/uwb_atcc")


def test_local_csv_matches_the_hf_path_row_for_row(dev_corpus):
    csv_path, _, paths = dev_corpus
    local = load_local_corpus(csv_path)
    reference = _hf_style(paths, DEV_TEXTS)

    assert local.column_names == reference.column_names == ["audio", "text"]
    assert list(local["text"]) == list(reference["text"]) == DEV_TEXTS
    for row, expected in zip(local, reference):
        assert row["audio"]["sampling_rate"] == expected["audio"]["sampling_rate"]
        assert row["audio"]["array"].dtype == expected["audio"]["array"].dtype
        assert np.allclose(row["audio"]["array"], expected["audio"]["array"])


def test_local_jsonl_matches_local_csv(dev_corpus):
    csv_path, jsonl_path, _ = dev_corpus
    from_csv, from_jsonl = load_local_corpus(csv_path), load_local_corpus(jsonl_path)
    assert list(from_csv["text"]) == list(from_jsonl["text"])
    for left, right in zip(from_csv, from_jsonl):
        assert np.allclose(left["audio"]["array"], right["audio"]["array"])


def test_local_corpus_resamples_and_resolves_relative_paths(dev_corpus):
    csv_path, _, _ = dev_corpus
    dataset = load_local_corpus(csv_path)
    assert len(dataset) == 3
    for row in dataset:
        assert row["audio"]["sampling_rate"] == REAL_SR
        assert len(row["audio"]["array"]) > 0
    # the 8 kHz clip is half a second either way, so the cast doubled its frames
    assert len(dataset[2]["audio"]["array"]) == pytest.approx(REAL_SR * 0.5, abs=2)


def test_local_text_is_not_normalized(dev_corpus):
    """The loader hands references through raw.

    `training.evaluate.build_report` runs `normalize_atc` over references and
    hypotheses together; normalizing here as well would score the model
    against an easier reference than the HF path gives it.
    """
    csv_path, _, _ = dev_corpus
    texts = list(load_local_corpus(csv_path)["text"])
    assert texts == DEV_TEXTS
    assert texts[0] != normalize_atc(texts[0])  # the fixture really does differ


def test_load_real_atc_routes_a_local_path_without_touching_hf(dev_corpus, monkeypatch):
    csv_path, _, _ = dev_corpus

    def fail(*args, **kwargs):
        raise AssertionError("load_real_atc reached Hugging Face for a local path")

    monkeypatch.setattr("datasets.load_dataset", fail)
    dataset = load_real_atc("train", str(csv_path))
    assert list(dataset["text"]) == DEV_TEXTS


def test_missing_local_corpus_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="dev.csv"):
        load_local_corpus(tmp_path / "dev.csv")


def test_local_corpus_rejects_a_manifest_without_a_text_column(tmp_path):
    path = tmp_path / "dev.csv"
    path.write_text("audio,label\n/tmp/a.wav,hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_local_corpus(path)


def test_optional_source_column_is_carried_through(tmp_path):
    """A mixed dev set labels where each row came from; unlabelled has no column."""
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("a.wav", "b.wav"):
        _sine(clips / name)
    labelled = tmp_path / "mixed.csv"
    labelled.write_text(
        f"audio,text,source\n{clips / 'a.wav'},one,kixd\n{clips / 'b.wav'},two,eu\n",
        encoding="utf-8")
    plain = tmp_path / "plain.csv"
    plain.write_text(f"audio,text\n{clips / 'a.wav'},one\n", encoding="utf-8")

    mixed = load_local_corpus(labelled)
    assert "source" in mixed.column_names
    assert list(mixed["source"]) == ["kixd", "eu"]
    assert "source" not in load_local_corpus(plain).column_names


def test_dev_rows_dump_carries_counts_that_re_aggregate(tmp_path, monkeypatch):
    """Per-utterance rows must sum back to the aggregate, not approximate it."""
    refs = ["cleared to land runway one eight", "roger", ""]
    hyps = ["cleared to land runway one eight", "wrong word here", "hello"]
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)

    path = harness.write_dev_rows(tmp_path / "dev_rows.jsonl", hyps)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len(rows) == 3
    assert [row["reference"] for row in rows] == refs
    assert [row["hypothesis"] for row in rows] == hyps
    assert rows[0]["errors"] == 0 and rows[0]["wer"] == 0.0
    assert rows[0]["audio"] == "/clips/0.wav"
    # the empty reference is excluded from WER (every word would be an
    # insertion against nothing) and scored as a hallucination instead
    assert rows[2]["ref_words"] == 0 and rows[2]["wer"] is None
    assert rows[2]["errors"] == 0
    assert rows[2]["hallucinated"] is True
    assert rows[0]["hallucinated"] is None

    corpus_wer = (sum(row["errors"] for row in rows)
                  / sum(row["ref_words"] for row in rows))
    report = harness.dev_report(hyps)
    assert corpus_wer == pytest.approx(report["wer"]["atc_normalized"])
    assert sum(row["ref_words"] for row in rows) == report["wer"]["reference_words"]


def test_bounded_wer_caps_a_looping_row_at_one_row_of_error(tmp_path, monkeypatch):
    """The E0 finding: one repetition row outweighed the whole manipulation.

    whisper-tiny loops until `max_new_tokens`, so a short reference can collect
    dozens of insertions against a denominator it never grows. Bounded WER
    lets that row say "completely wrong" and no more.
    """
    refs = ["cleared to land runway one eight", "roger wilco"]      # 6 + 2 words
    loop = " ".join(["the city of"] * 30)                           # ~90 insertions
    hyps = [loop, "roger wilco"]
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)

    report = harness.dev_report(hyps)
    unbounded = report["wer"]["atc_normalized"]
    bounded = report["wer_bounded"]

    # the loop row alone drives the unbounded number far past 1.0
    assert unbounded > 1.0
    # bounded: the 6-word row contributes at most its own 6 errors, and the
    # second row is perfect, so the corpus WER is exactly 6/8
    assert bounded["atc_normalized"] == pytest.approx(6 / 8)
    assert bounded["reference_words"] == 8
    assert bounded["n_capped_rows"] == 1
    assert bounded["discarded_errors"] > 80
    assert bounded["atc_normalized"] < unbounded


def test_bounded_wer_equals_unbounded_when_nothing_loops(tmp_path, monkeypatch):
    """The cap must be inert on ordinary rows, or it is not a cap but a change."""
    refs = ["cleared to land runway one eight", "roger wilco"]
    hyps = ["cleared to land runway one nine", "roger"]
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)

    report = harness.dev_report(hyps)
    assert report["wer_bounded"]["n_capped_rows"] == 0
    assert report["wer_bounded"]["discarded_errors"] == 0
    assert report["wer_bounded"]["atc_normalized"] == pytest.approx(
        report["wer"]["atc_normalized"])


def test_dev_rows_keep_raw_counts_and_flag_the_capped_ones(tmp_path, monkeypatch):
    """The bootstrap needs the unbounded truth; the flag is how you find loops."""
    refs = ["cleared to land runway one eight", "roger wilco"]
    loop = " ".join(["the city of"] * 30)
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)

    path = harness.write_dev_rows(tmp_path / "rows.jsonl", [loop, "roger wilco"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert rows[0]["errors"] > rows[0]["ref_words"]     # raw, uncapped
    assert rows[0]["wer"] > 1.0
    assert rows[0]["capped"] is True
    assert rows[1]["capped"] is False
    # the raw rows still sum to the *unbounded* aggregate, unchanged
    corpus = (sum(row["errors"] for row in rows)
              / sum(row["ref_words"] for row in rows))
    assert corpus == pytest.approx(
        harness.dev_report([loop, "roger wilco"])["wer"]["atc_normalized"])


def test_reward_is_scored_on_the_bounded_aggregate(tmp_path, monkeypatch):
    """Both sides of the subtraction must be the same metric."""
    refs = ["cleared to land runway one eight", "roger wilco"]
    loop = " ".join(["the city of"] * 30)
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)

    responses = iter([[loop, "roger wilco"],                 # baseline: loops
                      ["cleared to land runway one eight", "roger wilco"]])
    monkeypatch.setattr(reward_mod, "transcribe",
                        lambda model, processor, features, device, **kw: next(responses))
    for name in ("build_dataset", "load_manifest", "prepare_features"):
        monkeypatch.setattr(reward_mod, name, lambda *a, **kw: [])
    monkeypatch.setattr(reward_mod, "finetune", lambda model, features, **kw: model)
    monkeypatch.setattr(reward_mod, "JsonlTextSource", lambda *a, **kw: object())
    monkeypatch.setattr(reward_mod, "load_config", lambda path: object())

    result = harness({"mode": "procedural"}, str(tmp_path / "trial"))

    # baseline 6/8 bounded (not the ~12.0 unbounded), post 0.0 -> reward 0.75
    assert result.wer_baseline == pytest.approx(6 / 8)
    assert result.wer_after == pytest.approx(0.0)
    assert result.reward == pytest.approx(result.wer_baseline - result.wer_after)
    # the unbounded number rides along so a loop-driven divergence is visible
    assert result.metrics["unbounded_wer_after"] == pytest.approx(0.0)
    assert result.metrics["n_capped_rows"] == 0


def test_baseline_cache_without_bounded_wer_is_treated_as_a_miss(tmp_path, monkeypatch):
    """A pre-cap cached baseline must never be subtracted from a bounded WER."""
    refs = ["cleared to land", "roger"]
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)
    cache = harness._baseline_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"wer": {"atc_normalized": 0.9}}))   # stale schema

    calls = []

    def fake_transcribe(model, processor, features, device, **kwargs):
        calls.append(1)
        return list(refs)                                   # perfect: WER 0.0

    monkeypatch.setattr(reward_mod, "transcribe", fake_transcribe)
    harness._ensure_baseline()

    assert len(calls) == 1                                  # recomputed, not reused
    assert harness.baseline_report["wer_bounded"]["atc_normalized"] == 0.0
    assert "wer_bounded" in json.loads(cache.read_text())   # rewritten in place


def test_per_source_breakdown_appears_only_when_labelled(tmp_path, monkeypatch):
    refs = ["cleared to land", "cleared to land"]
    hyps = ["cleared to land", "totally different words"]
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=refs)

    assert "by_source" not in harness.dev_report(hyps)

    harness._dev_sources = ["kixd", "eu"]
    report = harness.dev_report(hyps)
    assert set(report["by_source"]) == {"eu", "kixd"}
    assert report["by_source"]["kixd"]["wer"] == 0.0
    assert report["by_source"]["eu"]["wer"] > 0.0
    assert report["by_source"]["kixd"]["samples"] == 1
    # the halves must reconstruct the aggregate the reward is scored on
    total_words = sum(part["ref_words"] for part in report["by_source"].values())
    assert total_words == report["wer"]["reference_words"]


def test_harness_dev_slice_and_cache_key_follow_the_local_corpus(
        dev_corpus, tmp_path, monkeypatch):
    """The plumbing `--dev-corpus` relies on: slice, refs, and cache identity."""
    csv_path, _, _ = dev_corpus
    harness = _make_harness(tmp_path, monkeypatch, dev_texts=[])
    harness.dev_corpus = str(csv_path)
    harness.dev_indices = (0, 2)
    harness._dev_refs = harness._dev_categories = harness._dev_features = None
    monkeypatch.setattr(reward_mod, "prepare_features",
                        lambda dataset, processor: list(dataset))

    harness._ensure_dev()
    assert harness._dev_refs == DEV_TEXTS[:2]
    assert harness._dev_categories == [None, None]
    assert len(harness._dev_features) == 2

    # the baseline cache is keyed by corpus, so a different dev set cannot
    # silently reuse the previous one's zero-shot WER
    cache = harness._baseline_cache_path()
    assert "dev_csv" in cache.name
    harness.dev_corpus = "jacktol/atc-dataset"
    assert harness._baseline_cache_path() != cache


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
    harness._dev_sources = [None] * len(dev_texts)
    harness._dev_paths = [f"/clips/{index}.wav" for index in range(len(dev_texts))]
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
