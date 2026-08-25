"""Toy-scale regression tests for the hardened residual trainer."""

import json
import os
from pathlib import Path

os.environ.setdefault("ATCGAN_TRACKING", "off")

import numpy as np
import pytest
import soundfile as sf
import torch

from atcgen.channel.learned import residual_train as rt
from atcgen.channel.learned.preset import BAND_EDGES, Preset, band_centers, write_presets

SR = 16000


def _tone(seed: int, seconds: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    wav = (0.16 * np.sin(2 * np.pi * (150 + seed) * t)
           + 0.06 * np.sin(2 * np.pi * 700 * t)
           + 0.005 * rng.standard_normal(len(t)))
    return wav.astype(np.float32)


def _preset() -> Preset:
    centres = np.asarray(band_centers(BAND_EDGES))
    gains = np.where((centres >= 250) & (centres <= 3200), 0.0, -40.0)
    return Preset(clip_id="fit", station="TOWER",
                  band_gains_db=[float(value) for value in gains], drive=1.1,
                  poly=[0.0, 0.0], agc_tau_ms=60.0, agc_strength=0.1,
                  noise_gain=0.01, snr_est=24.0, fit_loss=1.0,
                  passband_hz=[250.0, 3200.0])


@pytest.fixture
def toy_data(tmp_path):
    clips = tmp_path / "clips"
    train_tts = tmp_path / "tts_train"
    val_tts = tmp_path / "tts_val"
    clips.mkdir()
    train_tts.mkdir()
    val_tts.mkdir()
    rows = []
    for index in range(6):
        path = clips / f"clip_{index}.wav"
        sf.write(path, _tone(index), SR)
        split = "channel_train" if index < 4 else "channel_val"
        rows.append({"clip_id": f"clip_{index}", "path": f"clips/{path.name}",
                     "split": split, "block_id": f"fold_{index % 2}"})
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for index in range(2):
        sf.write(train_tts / f"train_{index}.wav", _tone(20 + index), SR)
        sf.write(val_tts / f"val_{index}.wav", _tone(30 + index), SR)
    presets = write_presets(tmp_path / "presets.jsonl", [_preset()])
    return {"root": tmp_path, "corpus": corpus, "presets": presets,
            "train_tts": train_tts, "val_tts": val_tts}


def _args(data, out: Path, steps: int = 1, *extra: str) -> list[str]:
    return ["--corpus", str(data["corpus"]), "--tts-dir", str(data["train_tts"]),
            "--presets", str(data["presets"]), "--out", str(out),
            "--device", "cpu", "--toy", "--steps", str(steps),
            "--base", "4", "--n-res", "1", "--batch-size", "1",
            "--num-patches", "8", "--r1-gamma", "0", "--save-every", "1",
            "--log-every", "1", *extra]


def test_zero_padded_shift_has_no_wraparound():
    x = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 1, 6)
    right = rt.zero_padded_shift(x, torch.tensor([2]))
    left = rt.zero_padded_shift(x, torch.tensor([-2]))
    assert torch.equal(right.flatten(), torch.tensor([0., 0., 1., 2., 3., 4.]))
    assert torch.equal(left.flatten(), torch.tensor([3., 4., 5., 6., 0., 0.]))


def test_physical_gain_matches_linear_magnitude_identity():
    x = torch.tensor([[[[0.0, 0.1, 0.4]]]])
    gain = torch.tensor([[[[1.1]]]])
    out = rt.physical_gain(x, gain)
    expected = torch.log1p(torch.expm1(x * rt.SPEC_SCALE) * gain) / rt.SPEC_SCALE
    assert torch.allclose(out, expected)
    assert out[..., 0].item() == 0.0


def test_patchnce_keys_are_detached_in_training(toy_data, tmp_path, monkeypatch):
    seen = []
    original = rt.PatchNCE.forward

    def checked(self, feats_q, feats_k):
        seen.append(all(not feature.requires_grad for feature in feats_k))
        return original(self, feats_q, feats_k)

    monkeypatch.setattr(rt.PatchNCE, "forward", checked)
    rt.main(_args(toy_data, tmp_path / "detached"))
    assert seen and all(seen)


def test_identity_mode_runs_and_logs_positive_identity_loss(toy_data, tmp_path):
    summary = rt.main(_args(toy_data, tmp_path / "identity", 1,
                            "--nce-mode", "source+identity"))
    train_rows = [row for row in summary["history"] if "g" in row]
    assert train_rows[0]["idt"] > 0.0


def test_resume_round_trip_continues_step_and_log(toy_data, tmp_path):
    out = tmp_path / "resume"
    rt.main(_args(toy_data, out, 1))
    first = torch.load(out / "state_latest.pt", weights_only=False)
    rt.main(_args(toy_data, out, 2, "--resume"))
    second = torch.load(out / "state_latest.pt", weights_only=False)
    rows = [json.loads(line) for line in
            (out / "train_log.jsonl").read_text().splitlines()]
    assert first["step"] == 1 and second["step"] == 2
    assert [row["step"] for row in rows if "g" in row] == [1, 2]
    assert set(second) >= {"ema", "best", "stale", "rng"}


def test_cache_fingerprint_mismatch_raises(toy_data, tmp_path):
    channel = rt.dsp_channel(toy_data["presets"], None)
    cache = tmp_path / "cache"
    rt.render_domain_a(toy_data["train_tts"], channel, cache, renders=1, seed=0)
    source = sorted(toy_data["train_tts"].glob("*.wav"))[0]
    sf.write(source, _tone(99), SR)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        rt.render_domain_a(toy_data["train_tts"], channel, cache,
                           renders=1, seed=0)


def test_validation_clips_never_enter_training_datasets(toy_data, tmp_path,
                                                        monkeypatch):
    captured = []
    original = rt.SpecCrops

    class CapturingCrops(original):
        def __init__(self, paths, *args, **kwargs):
            captured.append([Path(path).name for path in paths])
            super().__init__(paths, *args, **kwargs)

    class NoEval:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(rt, "SpecCrops", CapturingCrops)
    monkeypatch.setattr(rt, "KidTracker", NoEval)
    rt.main(_args(toy_data, tmp_path / "isolated", 1,
                  "--val-tts-dir", str(toy_data["val_tts"]),
                  "--eval-every", "2"))
    assert len(captured) == 2
    assert all(not name.startswith("clip_4") and not name.startswith("clip_5")
               for name in captured[1])


def test_nonfinite_loss_saves_state_and_names_step(toy_data, tmp_path, monkeypatch):
    original = rt.lsgan_loss
    calls = 0

    def nonfinite(preds, is_real):
        nonlocal calls
        calls += 1
        if calls == 1:
            return preds[0].sum() * 0 + float("nan")
        return original(preds, is_real)

    monkeypatch.setattr(rt, "lsgan_loss", nonfinite)
    out = tmp_path / "nonfinite"
    with pytest.raises(RuntimeError, match="step 1"):
        rt.main(_args(toy_data, out))
    assert (out / "state_latest.pt").exists()


def test_eval_selection_uses_earliest_candidate_within_best_se(toy_data, tmp_path,
                                                               monkeypatch):
    scripted = [(0.10, 0.03), (0.08, 0.03), (0.20, 0.01)]

    class ScriptedTracker:
        def __init__(self, *args, **kwargs):
            self.index = 0

        def __call__(self, translator):
            kid_mean, kid_se = scripted[self.index]
            self.index += 1
            result = {"gates_ok": True, "gates": [{"ok": True}],
                      "residual_sat": 0.0, "folds": {"a": kid_mean},
                      "kid_mean": kid_mean, "kid_se": kid_se}
            return result, [np.zeros(1000, dtype=np.float32)] * 2

    monkeypatch.setattr(rt, "KidTracker", ScriptedTracker)
    out = tmp_path / "selection"
    rt.main(_args(toy_data, out, 3,
                  "--val-tts-dir", str(toy_data["val_tts"]),
                  "--eval-every", "1"))
    report = json.loads((out / "validation_report.json").read_text())
    assert len(report["evaluations"]) == 3
    assert report["selection"]["step"] == 1
    assert report["selection"]["rule"] == "lexicographic_v1"
    assert (out / "G_selected.pt").exists()
