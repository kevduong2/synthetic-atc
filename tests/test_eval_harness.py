"""Fast regression tests for the versioned evaluation harness.

Embedding and probe runners are injected fakes: these tests exercise harness
wiring and verdict semantics without loading WavLM/CLAP or using the network.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

from atcgen.eval.harness import (build_verdict, format_diff, run_evaluation)

SR = 16000
NOW = datetime(2026, 8, 23, 12, 34, 56, tzinfo=timezone.utc)


def _wav(freq: float, seed: int) -> np.ndarray:
    t = np.arange(SR // 4) / SR
    noise = np.random.default_rng(seed).normal(0.0, 0.002, len(t))
    return (0.15 * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)


def _fake_run(tmp_path, *, discard_rate=0.0, syn_freq=440.0, ref_freq=440.0):
    run = tmp_path / "candidate"
    wavs = run / "wavs"
    ref = tmp_path / "reference"
    wavs.mkdir(parents=True)
    ref.mkdir()
    rows = []
    for i in range(6):
        name = f"{i:03d}.wav"
        sf.write(wavs / name, _wav(syn_freq, i), SR)
        sf.write(ref / name, _wav(ref_freq, i), SR)
        rows.append({"audio": f"wavs/{name}", "text": f"sample {i}"})
    (run / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    (run / "stats.json").write_text(json.dumps({
        "config_hash": "abc123",
        "qc": {"total": 6, "kept": 6, "discarded": 0,
               "discard_rate": discard_rate, "reasons": {}},
    }))
    return run, ref


def _fake_embeddings(synthetic, real):
    assert len(synthetic) == len(real) == 6
    return {"synthetic_dir": str(synthetic), "real_dir": str(real), "families": {
        "wavlm": {"kid": 0.01, "frechet": 1.5},
        "clap": {"kid": 0.02, "frechet": 2.5},
    }}


def _probe(accuracy):
    def runner(synthetic, real):
        assert len(synthetic) == len(real) == 6
        return {"synthetic_dir": str(synthetic), "real_dir": str(real),
                "balanced_accuracy": accuracy,
                "null_balanced_accuracy": 0.5}
    return runner


def test_harness_writes_schema_and_passing_verdict(tmp_path):
    run, ref = _fake_run(tmp_path)
    artifacts = run_evaluation(
        run, ref, out_dir=tmp_path / "versions",
        embedding_compare=_fake_embeddings, probe_runner=_probe(0.65), now=NOW,
    )

    written = json.loads(artifacts.json_path.read_text())
    assert written == artifacts.result
    assert artifacts.json_path.name == "abc123_20260823T123456.000000Z.json"
    assert written["schema_version"] == 1
    assert written["config_hash"] == "abc123"
    assert written["inputs"]["manifest_samples"] == 6
    assert written["tier0"]["stats"]["discard_rate"] == 0.0
    assert written["tier1"]["channel"]["comparison"]["all_medians_in_range"]
    assert set(written["tier1"]["embeddings"]["result"]["families"]) == {
        "wavlm", "clap"}
    assert written["tier1"]["embeddings"]["result"]["synthetic_dir"] == \
        str((run / "wavs").resolve())
    assert written["tier2"]["probe"]["result"]["real_dir"] == str(ref.resolve())
    assert written["tier2"]["probe"]["result"]["null_balanced_accuracy"] == 0.5
    assert written["verdict"]["overall"] == {
        "all_tier0_to_2_criteria_covered": True,
        "note": "Full train-ready status is undetermined until Tier 3 passes.",
        "pass": True,
        "scope": "Tier 0-2 train-ready criteria only",
        "status": "pass",
        "tier3_covered": False,
        "train_ready": None,
    }
    assert written["verdict"]["criteria"]["tier3_downstream_wer"]["status"] == \
        "not_covered"
    assert "python" in written["tool_versions"]


def test_verdict_reports_each_failure_and_overall_failure(tmp_path):
    run, ref = _fake_run(tmp_path, discard_rate=0.15,
                         syn_freq=2400.0, ref_freq=300.0)
    artifacts = run_evaluation(
        run, ref, out_dir=tmp_path / "versions",
        embedding_compare=_fake_embeddings, probe_runner=_probe(0.6501), now=NOW,
    )
    verdict = artifacts.result["verdict"]

    assert verdict["criteria"]["tier0_discard_rate"]["status"] == "fail"
    assert verdict["criteria"]["tier1_channel_medians"]["status"] == "fail"
    assert verdict["criteria"]["tier1_channel_medians"]["value"][
        "failed_statistics"]
    assert verdict["criteria"]["tier2_probe_accuracy"]["status"] == "fail"
    assert verdict["overall"]["status"] == "fail"
    assert verdict["overall"]["pass"] is False


def test_skip_embeddings_never_calls_model_runners_and_is_incomplete(tmp_path):
    run, ref = _fake_run(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("model-backed runner must not be called")

    artifacts = run_evaluation(
        run, ref, out_dir=tmp_path / "versions", skip_embeddings=True,
        embedding_compare=forbidden, probe_runner=forbidden, now=NOW,
    )
    result = artifacts.result
    assert result["tier1"]["embeddings"]["status"] == "skipped"
    assert result["tier2"]["probe"]["status"] == "skipped"
    assert result["verdict"]["criteria"]["tier2_probe_accuracy"]["pass"] is None
    assert result["verdict"]["overall"]["status"] == "incomplete"


def test_wavs_only_skips_tier0_and_uses_run_name_for_filename(tmp_path):
    run, ref = _fake_run(tmp_path)
    bare = tmp_path / "bare-wavs"
    bare.mkdir()
    for wav in sorted((run / "wavs").glob("*.wav")):
        (bare / wav.name).write_bytes(wav.read_bytes())

    artifacts = run_evaluation(
        bare, ref, out_dir=tmp_path / "versions", wavs_only=True,
        skip_embeddings=True, n_max=3, now=NOW,
    )
    result = artifacts.result
    assert artifacts.json_path.name == "bare-wavs_20260823T123456.000000Z.json"
    assert result["config_hash"] is None
    assert result["inputs"]["synthetic_samples_evaluated"] == 3
    assert result["inputs"]["reference_samples_evaluated"] == 3
    assert result["tier0"]["status"] == "not_evaluated"


def test_diff_prints_compact_metric_deltas(tmp_path):
    run, ref = _fake_run(tmp_path)
    old = run_evaluation(
        run, ref, out_dir=tmp_path / "old", embedding_compare=_fake_embeddings,
        probe_runner=_probe(0.60), now=NOW,
    ).result
    new = deepcopy(old)
    new["tier0"]["stats"]["discard_rate"] = 0.05
    new["tier1"]["embeddings"]["result"]["families"]["wavlm"]["kid"] = 0.04
    new["tier2"]["probe"]["result"]["balanced_accuracy"] = 0.64

    table = format_diff(old, new)
    assert "metric" in table and "old" in table and "delta" in table
    assert "tier0.discard_rate" in table and "0.05" in table
    assert "tier1.embeddings.wavlm.kid" in table and "0.03" in table
    assert "tier2.probe.balanced_accuracy" in table and "0.04" in table


def test_build_verdict_treats_exact_thresholds_as_documented():
    channel = {"stats": {"rms_db": {"median_in_range": True}}}
    tier0 = {"status": "recorded", "stats": {"discard_rate": 0.15}}
    verdict = build_verdict(tier0, channel, {"balanced_accuracy": 0.65})
    assert verdict["criteria"]["tier0_discard_rate"]["pass"] is False
    assert verdict["criteria"]["tier2_probe_accuracy"]["pass"] is True
    assert verdict["overall"]["pass"] is False
