"""Tests for the blind A/B verification script.

Every model/network/audio call is stubbed: `WhisperProcessor`/
`WhisperForConditionalGeneration.from_pretrained`, `render_and_finetune`,
`transcribe`, and `load_real_atc` are all monkeypatched. `build_report`,
`paired_bootstrap`, `write_text_pool`, `load_config`, and `config_hash` are
left real -- they are pure computation (jiwer/numpy/YAML), so running them
for real is what actually exercises the script's wiring.
"""

import argparse
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import rl_verify

REFS = ["cleared to land", "roger", "taxi via alpha", "", "wind calm", "contact tower"]
ZERO_SHOT_HYPS = ["static noise", "unclear", "garbled", "thank you", "no response", "static"]
BASE_HYPS = ["cleared to land", "roger", "taxi via bravo", "", "wind calm", "contact ground"]
BEST_HYPS = ["cleared to land", "roger", "taxi via alpha", "", "wind calm", "contact tower"]


# -- small fakes ------------------------------------------------------------

class _FakeConfig:
    pass


class _FakeModel:
    def __init__(self):
        self.config = _FakeConfig()
        self.saved_to = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def save_pretrained(self, path):
        self.saved_to = Path(path)


class _FakeProcessor:
    def save_pretrained(self, path):
        pass


class _FakeDataset:
    """Stands in for a `datasets.Dataset` slice: text-only, no audio/category."""

    def __init__(self, rows):
        self.rows = rows

    def select(self, index_range):
        indices = list(index_range)
        return _FakeDataset([self.rows[i] for i in indices])

    def __len__(self):
        return len(self.rows)

    @property
    def column_names(self):
        return list(self.rows[0].keys()) if self.rows else []

    def __getitem__(self, key):
        return [row[key] for row in self.rows]


def _make_fake_transcribe(sequence):
    calls = []

    def fake(model, processor, features, device, **kwargs):
        calls.append({"model": model, "features": features, "device": device})
        return sequence[len(calls) - 1]

    return fake, calls


def _make_fake_render_and_finetune():
    calls = []

    def fake(config, trial_dir, *, base_model, pool_path, n_synth, ft_steps, ft_batch,
            ft_lr, ft_seed, gen_seed, device, processor):
        trial_dir = Path(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(config)
        payload["seed"] = gen_seed
        (trial_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        calls.append({
            "config": config, "trial_dir": trial_dir, "pool_path": pool_path,
            "n_synth": n_synth, "ft_steps": ft_steps, "ft_batch": ft_batch,
            "gen_seed": gen_seed, "device": device,
        })
        return _FakeModel()

    return fake, calls


# -- unit-level tests ---------------------------------------------------------

def test_indices_parses_lo_hi():
    assert rl_verify._indices("0:500") == (0, 500)
    assert rl_verify._indices("120:900") == (120, 900)


def test_indices_rejects_malformed_input():
    with pytest.raises(argparse.ArgumentTypeError):
        rl_verify._indices("500")


def test_speech_subset_filters_empty_references():
    refs = ["a b", "", "c d"]
    hyps_by_arm = {"x": ["a b", "z", "c d"], "y": ["a b", "z", "d c"]}
    speech_refs, speech_hyps = rl_verify._speech_subset(refs, hyps_by_arm)
    assert speech_refs == ["a b", "c d"]
    assert speech_hyps == {"x": ["a b", "c d"], "y": ["a b", "d c"]}


def test_pairwise_deltas_calls_bootstrap_on_speech_only_for_every_pair(monkeypatch):
    refs = ["a b", "", "c d"]
    hyps_by_arm = {
        "zero_shot": ["x y", "noise", "c d"],
        "base": ["a b", "", "c d"],
        "best": ["a b", "", "c d"],
    }
    calls = []

    def fake_bootstrap(speech_refs, hyps_a, hyps_b, *, n_boot, seed):
        calls.append((speech_refs, hyps_a, hyps_b))
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

    monkeypatch.setattr(rl_verify, "paired_bootstrap", fake_bootstrap)
    result = rl_verify.pairwise_deltas(refs, hyps_by_arm)

    assert set(result) == {"zero_shot_vs_base", "zero_shot_vs_best", "base_vs_best"}
    assert len(calls) == 3
    for speech_refs, _, _ in calls:
        assert speech_refs == ["a b", "c d"]  # the empty-reference row is dropped


def test_format_table_handles_missing_callsign_accuracy():
    verify_report = {
        "arms": {
            name: {"report": {
                "wer": {"atc_normalized": 0.1, "raw": 0.2},
                "callsign": {"accuracy": None},
                "hallucination": {"rate": None},
            }}
            for name in rl_verify.ARM_ORDER
        },
        "pairwise_bootstrap": {
            pair: {"delta": 0.05, "ci_low": -0.01, "ci_high": 0.11, "p_value": 0.2}
            for pair in rl_verify.PAIR_ORDER
        },
    }
    table = rl_verify.format_table(verify_report)
    assert "n/a" in table
    for name in rl_verify.ARM_ORDER:
        assert name in table
    for pair in rl_verify.PAIR_ORDER:
        assert pair in table


# -- full orchestration, stub-based ------------------------------------------

def test_main_end_to_end_with_stubs(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_config.yaml").write_text(
        yaml.safe_dump({"mode": "procedural", "seed": 0, "qc": {"enabled": False}}))
    base_config_path = tmp_path / "base_config.yaml"
    base_config_path.write_text(yaml.safe_dump({"mode": "procedural", "seed": 0}))
    out_dir = tmp_path / "out"

    dataset_rows = [{"text": ref} for ref in REFS]
    monkeypatch.setattr(rl_verify, "load_real_atc", lambda split, corpus: _FakeDataset(dataset_rows))
    monkeypatch.setattr(rl_verify, "prepare_features", lambda dataset, processor: [None] * len(dataset))

    monkeypatch.setattr(rl_verify.WhisperProcessor, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeProcessor()))
    monkeypatch.setattr(rl_verify.WhisperForConditionalGeneration, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeModel()))
    monkeypatch.setattr(rl_verify, "pick_device", lambda arg: torch.device("cpu"))

    fake_transcribe, transcribe_calls = _make_fake_transcribe(
        [ZERO_SHOT_HYPS, BASE_HYPS, BEST_HYPS])
    monkeypatch.setattr(rl_verify, "transcribe", fake_transcribe)

    fake_render, render_calls = _make_fake_render_and_finetune()
    monkeypatch.setattr(rl_verify, "render_and_finetune", fake_render)

    result = rl_verify.main([
        "--run", str(run_dir),
        "--base-config", str(base_config_path),
        "--out", str(out_dir),
        "--test-corpus", "fake/corpus",
        "--test-split", "test",
        "--test-indices", "0:6",
        "--n-synth", "3",
        "--ft-steps", "2",
        "--ft-batch", "2",
        "--text-pool", "5",
        "--text-seed", "999",
    ])

    # three transcribe calls: zero-shot, base, best -- in that order
    assert len(transcribe_calls) == 3
    # both config arms rendered against the identical fresh pool and gen seed
    assert len(render_calls) == 2
    assert render_calls[0]["pool_path"] == render_calls[1]["pool_path"]
    assert render_calls[0]["gen_seed"] == rl_verify.GEN_SEED
    assert render_calls[1]["gen_seed"] == rl_verify.GEN_SEED
    assert render_calls[0]["n_synth"] == 3
    assert render_calls[1]["ft_steps"] == 2

    # the fresh pool was written with the given size/seed, distinct from any
    # search-harness pool
    pool_path = out_dir / "text_pool.jsonl"
    assert pool_path.exists()
    assert len(pool_path.read_text().strip().splitlines()) == 5

    # best is perfect, base has two substitutions, zero-shot is worse than both
    zero_shot_wer = result["arms"]["zero_shot"]["report"]["wer"]["atc_normalized"]
    base_wer = result["arms"]["base"]["report"]["wer"]["atc_normalized"]
    best_wer = result["arms"]["best"]["report"]["wer"]["atc_normalized"]
    assert best_wer == pytest.approx(0.0)
    assert base_wer > best_wer
    assert zero_shot_wer > base_wer

    # distinct configs hash differently
    assert result["arms"]["base"]["config_hash"] != result["arms"]["best"]["config_hash"]

    # bootstrap: base beats zero-shot, best beats both
    boot = result["pairwise_bootstrap"]
    assert boot["zero_shot_vs_base"]["delta"] > 0
    assert boot["zero_shot_vs_best"]["delta"] > 0
    assert boot["base_vs_best"]["delta"] > 0

    # report written to disk and matches the returned dict
    report_path = out_dir / "verify_report.json"
    assert report_path.exists()
    written = json.loads(report_path.read_text())
    assert written == result

    # auditable: hypotheses for every arm are present
    for name in rl_verify.ARM_ORDER:
        assert len(result["arms"][name]["hypotheses"]) == len(REFS)

    # table printing doesn't crash
    table = rl_verify.format_table(result)
    assert "zero_shot" in table and "best" in table


def test_main_respects_save_models_flag(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_config.yaml").write_text(yaml.safe_dump({"mode": "procedural", "seed": 0}))
    base_config_path = tmp_path / "base_config.yaml"
    base_config_path.write_text(yaml.safe_dump({"mode": "procedural", "seed": 0}))
    out_dir = tmp_path / "out"

    dataset_rows = [{"text": ref} for ref in REFS]
    monkeypatch.setattr(rl_verify, "load_real_atc", lambda split, corpus: _FakeDataset(dataset_rows))
    monkeypatch.setattr(rl_verify, "prepare_features", lambda dataset, processor: [None] * len(dataset))
    monkeypatch.setattr(rl_verify.WhisperProcessor, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeProcessor()))
    monkeypatch.setattr(rl_verify.WhisperForConditionalGeneration, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeModel()))
    monkeypatch.setattr(rl_verify, "pick_device", lambda arg: torch.device("cpu"))
    fake_transcribe, _ = _make_fake_transcribe([ZERO_SHOT_HYPS, BASE_HYPS, BEST_HYPS])
    monkeypatch.setattr(rl_verify, "transcribe", fake_transcribe)
    fake_render, _ = _make_fake_render_and_finetune()
    monkeypatch.setattr(rl_verify, "render_and_finetune", fake_render)

    rl_verify.main([
        "--run", str(run_dir),
        "--base-config", str(base_config_path),
        "--out", str(out_dir),
        "--test-indices", "0:6",
        "--n-synth", "3", "--ft-steps", "2", "--text-pool", "5",
        "--save-models",
    ])

    assert (out_dir / "models" / "base").exists()
    assert (out_dir / "models" / "best").exists()
