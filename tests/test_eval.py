import json

import numpy as np
import pytest
import soundfile as sf

from atcgen.eval import QCConfig, QCTally, qc_sample
from atcgen.eval.channel_stats import (SCALAR_KEYS, clip_stats, compare,
                                       compute_stats, main)

SR = 16000
NO_ASR = QCConfig(asr_gate=False)


def _speech_like(sec=2.0, sr=SR, amp=0.2, rate=4.0, seed=0):
    """Amplitude-modulated harmonic buzz: rough stand-in for speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * sec)) / sr
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / (i + 1)
            for i, f in enumerate([200, 400, 800, 1600]))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)
    return (amp * env * x / np.abs(x).max()).astype(np.float32)


def _band_noise(low, high, sec=3.0, sr=SR, amp=0.2, seed=0):
    from scipy import signal

    x = np.random.default_rng(seed).standard_normal(int(sr * sec))
    sos = signal.butter(8, [low, high], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)
    return (amp * y / np.abs(y).max()).astype(np.float32)


# --- Tier 0 QC gates ---------------------------------------------------------

def test_qc_passes_good_sample():
    r = qc_sample(_speech_like(), SR, config=NO_ASR)
    assert r.ok and r.reason is None
    assert r.metrics["duration"] == pytest.approx(2.0)


def test_qc_rejects_clipping():
    x = _speech_like(amp=1.0)
    x[: int(0.05 * len(x))] = 1.0  # 5% of samples at full scale
    assert qc_sample(x, SR, config=NO_ASR).reason == "clipping"


def test_qc_accepts_rare_clipping():
    x = _speech_like(amp=0.5)
    x[:10] = 1.0  # well under the 1% budget
    assert qc_sample(x, SR, config=NO_ASR).ok


def test_qc_rejects_silence():
    assert qc_sample(np.zeros(SR, np.float32), SR, config=NO_ASR).reason == "silence"


def test_qc_rejects_nonfinite():
    x = _speech_like()
    x[100] = np.nan
    assert qc_sample(x, SR, config=NO_ASR).reason == "nonfinite"
    y = _speech_like()
    y[5] = np.inf
    assert qc_sample(y, SR, config=NO_ASR).reason == "nonfinite"


def test_qc_rejects_duration_out_of_bounds():
    cfg = QCConfig(min_duration=1.0, max_duration=5.0, asr_gate=False)
    assert qc_sample(_speech_like(sec=0.5), SR, config=cfg).reason == "duration"
    assert qc_sample(_speech_like(sec=6.0), SR, config=cfg).reason == "duration"
    assert qc_sample(_speech_like(sec=2.0), SR, config=cfg).ok


def test_qc_rejects_level_outside_window():
    quiet = qc_sample(_speech_like(amp=0.01), SR, config=NO_ASR)  # audible, but too quiet
    assert quiet.reason == "level" and quiet.metrics["rms_db"] > NO_ASR.silence_rms_db
    loud = QCConfig(max_rms_db=-30.0, asr_gate=False)
    assert qc_sample(_speech_like(amp=0.5), SR, config=loud).reason == "level"


# --- Tier 0 ASR round-trip gate (fake transcriber; no model loads) -----------

def test_asr_gate_passes_matching_transcript():
    r = qc_sample(_speech_like(), SR, text="cleared to land runway two seven",
                  transcriber=lambda w, sr: "cleared to land runway 27")
    assert r.ok and r.metrics["wer"] == 0.0


def test_asr_gate_discards_high_wer():
    r = qc_sample(_speech_like(), SR, text="cleared to land runway two seven",
                  transcriber=lambda w, sr: "and uh")
    assert r.reason == "asr_wer" and r.metrics["wer"] > 0.5


def test_asr_gate_skipped_without_reference_text():
    calls = []

    def fake(w, sr):
        calls.append(1)
        return "hallucinated"

    assert qc_sample(_speech_like(), SR, text="", transcriber=fake).ok
    assert not calls  # noise-only samples have no reference to score against


def test_asr_gate_threshold_configurable():
    cfg = QCConfig(max_wer=0.9)
    r = qc_sample(_speech_like(), SR, text="one two three four",
                  config=cfg, transcriber=lambda w, sr: "one two three nine")
    assert r.ok and r.metrics["wer"] == pytest.approx(0.25)


def test_tally_counts_reasons_and_rate():
    tally = QCTally()
    tally.add(qc_sample(_speech_like(), SR, config=NO_ASR))
    tally.add(qc_sample(np.zeros(SR, np.float32), SR, config=NO_ASR))
    tally.add(qc_sample(_speech_like(sec=0.1), SR, config=NO_ASR))
    s = tally.summary()
    assert s == {"total": 3, "kept": 1, "discarded": 2, "discard_rate": 0.6667,
                 "reasons": {"silence": 1, "duration": 1},
                 "reason_rates": {"silence": 0.3333, "duration": 0.3333}}


# --- Tier 1 channel statistics ----------------------------------------------

def test_spectral_edge_tracks_band_limit():
    for cutoff in (1500, 2400, 3400):
        edge = clip_stats(_band_noise(300, cutoff), SR)["spectral_edge_hz"]
        assert abs(edge - cutoff) < 0.12 * cutoff, (cutoff, edge)


def test_spectral_low_edge_tracks_highpass():
    assert abs(clip_stats(_band_noise(500, 3000), SR)["spectral_low_hz"] - 500) < 120


def test_snr_estimate_matches_constructed_mixture():
    rng = np.random.default_rng(0)
    speech = _speech_like(sec=4.0, amp=0.3, seed=1)
    # gate the "speech" so the clip has genuine pauses to measure the floor in
    gate = np.ones_like(speech)
    gate[: SR // 2] = 0.0
    gate[-SR // 2:] = 0.0
    speech = speech * gate
    active_pow = float(np.mean(speech[SR // 2:-SR // 2] ** 2))
    for target in (10.0, 20.0, 30.0):
        noise = rng.standard_normal(len(speech)).astype(np.float32)
        noise *= np.sqrt(active_pow / 10 ** (target / 10)) / np.std(noise)
        est = clip_stats(speech + noise, SR)["snr_db"]
        assert abs(est - target) < 5.0, (target, est)


def test_loudness_and_peak():
    x = (0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    s = clip_stats(x, SR)
    assert s["peak_db"] == pytest.approx(-6.0, abs=0.1)   # 0.5 full scale
    assert s["rms_db"] == pytest.approx(-9.0, abs=0.2)    # sine: peak - 3 dB
    assert s["duration"] == pytest.approx(1.0)


def test_modulation_energy_peaks_at_syllable_rate():
    in_band = clip_stats(_speech_like(sec=4.0, rate=4.0), SR)["mod_4hz"]
    out_band = clip_stats(_speech_like(sec=4.0, rate=15.0), SR)["mod_4hz"]
    assert in_band > 0.5 > out_band


def test_compute_stats_from_dir_and_json(tmp_path):
    for i in range(6):
        sf.write(tmp_path / f"{i:03d}.wav", _band_noise(300, 2400, seed=i), SR)
    stats = compute_stats(tmp_path)
    assert stats["n"] == 6 and len(stats["clips"]) == 6
    assert set(SCALAR_KEYS) <= set(stats["summary"])
    assert len(stats["ltas_db_mean"]) == len(stats["ltas_hz"]) == 32
    json.dumps(stats)  # must stay serializable


def test_compute_stats_from_arrays():
    stats = compute_stats([_band_noise(300, 2400, seed=i) for i in range(3)], sr=SR)
    assert stats["n"] == 3
    assert stats["clips"][0]["name"] == "clip_000000"


def test_compare_identical_sets_is_zero():
    a = compute_stats([_band_noise(300, 2400, seed=i) for i in range(5)], sr=SR)
    c = compare(a, a)
    assert c["all_medians_in_range"]
    assert c["ltas_l1_db"] == 0.0
    assert all(v["wasserstein"] == 0.0 for v in c["stats"].values())


def test_compare_shifted_sets_flags_mismatch():
    real = compute_stats([_band_noise(300, 2400, seed=i) for i in range(5)], sr=SR)
    syn = compute_stats([_band_noise(300, 6000, amp=0.02, seed=i) for i in range(5)],
                        sr=SR)
    c = compare(syn, real)
    edge = c["stats"]["spectral_edge_hz"]
    assert edge["wasserstein"] > 500 and not edge["median_in_range"]
    assert not c["stats"]["rms_db"]["median_in_range"]
    assert not c["all_medians_in_range"]
    assert c["ltas_l1_db"] > 0


def test_cli_writes_json(tmp_path, capsys):
    wav_dir = tmp_path / "wavs"
    ref_dir = tmp_path / "ref"
    for d, seed0 in ((wav_dir, 0), (ref_dir, 100)):
        d.mkdir()
        for i in range(4):
            sf.write(d / f"{i:03d}.wav", _band_noise(300, 2400, seed=seed0 + i), SR)
    out = tmp_path / "stats.json"
    main([str(wav_dir), "--ref", str(ref_dir), "--out", str(out)])
    written = json.loads(out.read_text())
    assert written["n"] == 4 and "comparison" in written
    assert "spectral_edge_hz" in capsys.readouterr().out


# --- report ------------------------------------------------------------------

def test_report_renders_html_with_plots_and_players(tmp_path):
    pytest.importorskip("matplotlib")
    from atcgen.eval.report import build_report

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    paths = []
    for i in range(4):
        p = wav_dir / f"{i:03d}.wav"
        sf.write(p, _band_noise(300, 2400, seed=i), SR)
        paths.append(p)
    syn = compute_stats(wav_dir)
    out = build_report(tmp_path / "report" / "index.html", syn, real=syn,
                       comparison=compare(syn, syn),
                       audition=[("syn 0", paths[0]), ("real 1", paths[1])],
                       qc_summary=QCTally().summary())
    html = out.read_text()
    assert "data:image/png;base64," in html
    assert '<audio controls preload="none" src="../wavs/000.wav">' in html
    assert "spectral_edge_hz" in html
