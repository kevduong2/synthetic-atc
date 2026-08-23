"""M2.2: per-clip channel fitting and the preset format it writes.

Fits here are deliberately tiny — 1.5 s clips, a handful of bands' worth of
structure, 50 steps — so the whole file stays a couple of seconds.  Nothing
touches the network, a GPU, or anything under `runs/`.
"""

import json

import numpy as np
import pytest
import torch

from atcgen.channel.learned.channel_fit import (FittedChannel, _qc, _speech_rms,
                                                active_span, fit_clip, fit_corpus,
                                                measure_snr, probe_batch,
                                                synthetic_probe)
from atcgen.channel.learned.preset import (BAND_EDGES, Preset, apply_preset,
                                           band_centers, fir_taps, load_presets,
                                           passband_edges, speech_rms, write_presets)

SR = 16000
CENTRES = np.asarray(band_centers(BAND_EDGES))


def _preset(low=300.0, high=2600.0, drive=1.0, snr_db=25.0, **kwargs) -> Preset:
    gains = np.where((CENTRES >= low) & (CENTRES <= high), 0.0, -60.0)
    fields = {"clip_id": "t", "station": "TEST", "fit_loss": 0.0} | kwargs
    return Preset(band_gains_db=[float(v) for v in gains], drive=drive,
                  poly=[0.0, 0.0], agc_tau_ms=50.0, agc_strength=0.3,
                  noise_gain=10.0 ** (-snr_db / 20.0), snr_est=snr_db, **fields)


def _padded_probe(seconds: float, seed: int, lead: float = 0.5) -> np.ndarray:
    """A probe framed by real silence, so `active_span` has something to trim."""
    n, pad = int(seconds * SR), int(lead * SR)
    x = np.zeros(n + 2 * pad, dtype=np.float32)
    x[pad:pad + n] = synthetic_probe(n, np.random.default_rng(seed))
    return x


def _band_db(x: np.ndarray, low: float, high: float) -> float:
    """Power in [low, high) Hz, in dB relative to the clip's total."""
    spectrum = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    inside = (freqs >= low) & (freqs < high)
    return float(10.0 * np.log10(spectrum[inside].sum() / spectrum.sum() + 1e-20))


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #

def test_torch_and_numpy_chains_agree():
    """The fit optimizes in torch and generation evaluates in numpy: same audio."""
    n = SR
    rng = np.random.default_rng(0)
    probe = synthetic_probe(n, rng)
    probe = probe / speech_rms(probe, SR)
    noise = rng.standard_normal(n).astype(np.float32)

    model = FittedChannel(n, SR)
    with torch.no_grad():
        model.drive_raw.fill_(1.0)
        model.poly_raw.copy_(torch.tensor([0.3, -0.2]))
        model.strength_raw.fill_(0.5)
        model.tau_raw.fill_(-0.5)
        model.set_snr(22.0)
        expected = model(torch.from_numpy(probe).reshape(1, -1),
                         torch.from_numpy(noise).reshape(1, -1))[0].numpy()

    preset = model.to_preset(clip_id="x", station="TEST", fit_loss=0.0)
    assert preset.snr_est == pytest.approx(22.0, abs=0.05)
    actual = apply_preset(probe, SR, preset, noise=noise, filter_noise=True)

    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 0.02 * np.abs(expected).max()
    assert np.corrcoef(actual, expected)[0, 1] > 0.999


def test_apply_preset_shapes_the_band_and_holds_the_input_level():
    preset = _preset(low=300.0, high=2000.0, snr_db=40.0)
    rng = np.random.default_rng(1)
    probe = synthetic_probe(2 * SR, rng) * 0.1

    out = apply_preset(probe, SR, preset, filter_noise=True)
    assert _band_db(out, 3000.0, 8000.0) < _band_db(probe, 3000.0, 8000.0) - 25.0
    # the chain is level-preserving: post/loudness stages own absolute gain
    assert speech_rms(out, SR) == pytest.approx(speech_rms(probe, SR), rel=0.25)


def test_noise_gain_lands_at_the_requested_snr():
    preset = _preset(low=100.0, high=8000.0, drive=0.1, snr_db=20.0)
    quiet = _preset(low=100.0, high=8000.0, drive=0.1, snr_db=40.0)
    probe = _padded_probe(2.0, seed=2)
    noise = np.random.default_rng(3).standard_normal(len(probe))

    loud = apply_preset(probe, SR, preset, noise=noise, filter_noise=False)
    quiet_out = apply_preset(probe, SR, quiet, noise=noise, filter_noise=False)
    # 20 dB more SNR is 20 dB less noise: read it off the silence around the probe
    lo, _ = active_span(probe, SR)
    assert lo > 0
    gap_loud = float(np.sqrt(np.mean(loud[:lo] ** 2)))
    gap_quiet = float(np.sqrt(np.mean(quiet_out[:lo] ** 2)))
    assert 20.0 * np.log10(gap_loud / gap_quiet) == pytest.approx(20.0, abs=3.0)


def test_fir_taps_realize_the_requested_band_gains():
    gains = np.where((CENTRES >= 400) & (CENTRES <= 2400), 0.0, -70.0)
    taps = fir_taps(gains, BAND_EDGES, SR)
    response = np.abs(np.fft.rfft(taps, 4096))
    freqs = np.fft.rfftfreq(4096, 1.0 / SR)

    def at(hz):
        return 20.0 * np.log10(response[np.argmin(np.abs(freqs - hz))] + 1e-12)

    assert at(1000.0) == pytest.approx(0.0, abs=1.0)
    assert at(6000.0) < -60.0
    assert at(120.0) < -50.0


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("low,high,drive", [(300.0, 2600.0, 1.0), (200.0, 1500.0, 6.0)])
def test_fit_recovers_a_known_chain(low, high, drive):
    """Push a probe through a known chain, fit it back, check what came out."""
    n = int(1.5 * SR)
    truth = _preset(low=low, high=high, drive=drive)
    target = apply_preset(probe_batch(n, 1, np.random.default_rng(11))[0], SR, truth)

    probes = probe_batch(n, 2, np.random.default_rng(3))
    model, history = fit_clip(target, SR, probes, steps=50, seed=0,
                              snr_db=measure_snr(target, SR))
    fitted = model.to_preset(clip_id="f", station="F", fit_loss=history[-1])

    assert history[-1] < 0.75 * history[0]
    truth_low, truth_high = passband_edges(truth.band_gains_db, BAND_EDGES)
    # tolerance in *grid steps*, because that is the resolution a preset has;
    # the low edge also sits systematically high, since 513 taps cannot realize
    # an instant 60 dB step and the realized -6 dB point slides up the skirt
    step = float(BAND_EDGES[1] / BAND_EDGES[0])
    for fitted_hz, truth_hz in zip(fitted.passband_hz, (truth_low, truth_high)):
        assert abs(np.log(fitted_hz / truth_hz)) < 3.0 * np.log(step)


def test_fitted_drive_orders_with_the_true_drive():
    """`drive` is only weakly identified by statistics (04 §4 risk 1) — the fit
    is expected to rank two chains correctly, not to recover the value."""
    n = int(1.5 * SR)
    probes = probe_batch(n, 2, np.random.default_rng(3))
    drives = []
    for drive in (1.0, 8.0):
        target = apply_preset(probe_batch(n, 1, np.random.default_rng(11))[0], SR,
                              _preset(drive=drive))
        model, history = fit_clip(target, SR, probes, steps=50, seed=0,
                                  snr_db=measure_snr(target, SR))
        drives.append(float(model.drive.detach()))
    assert drives[1] > drives[0] * 1.3


def test_measure_snr_reads_an_injected_floor():
    """Speech bursts with silence between them, plus a known noise floor."""
    rng = np.random.default_rng(5)
    speech = synthetic_probe(3 * SR, rng)
    speech[int(0.9 * SR):int(1.4 * SR)] = 0.0
    speech[int(2.0 * SR):int(2.5 * SR)] = 0.0
    level = speech_rms(speech, SR)
    for snr in (12.0, 24.0):
        noisy = speech + rng.standard_normal(len(speech)).astype(np.float32) * (
            level * 10.0 ** (-snr / 20.0))
        assert measure_snr(noisy, SR) == pytest.approx(snr, abs=4.0)


def test_active_span_trims_leading_and_trailing_silence():
    x = np.zeros(3 * SR, dtype=np.float32)
    x[SR:2 * SR] = synthetic_probe(SR, np.random.default_rng(6))
    lo, hi = active_span(x, SR, margin_ms=100.0)
    assert 0.8 * SR <= lo <= SR
    assert 2 * SR <= hi <= 2.2 * SR


def test_probe_batch_reads_a_directory_and_normalizes(tmp_path):
    import soundfile as sf

    for index in range(3):
        clip = np.zeros(2 * SR, dtype=np.float32)
        clip[SR // 2:] = synthetic_probe(len(clip) - SR // 2,
                                         np.random.default_rng(index))
        sf.write(tmp_path / f"tts_{index}.wav", clip, SR)

    batch = probe_batch(SR, 4, np.random.default_rng(0), tmp_path)
    assert batch.shape == (4, SR)
    for row in batch:
        assert speech_rms(row, SR) == pytest.approx(1.0, rel=1e-3)


def test_speech_rms_ignores_padding():
    x = np.zeros(3 * SR, dtype=np.float32)
    x[SR:2 * SR] = 0.5
    assert speech_rms(x, SR) == pytest.approx(0.5, rel=1e-3)
    assert float(_speech_rms(torch.from_numpy(x).reshape(1, -1), SR)) == pytest.approx(
        0.5, rel=1e-3)


# --------------------------------------------------------------------------- #
# the preset file
# --------------------------------------------------------------------------- #

def test_presets_jsonl_round_trip(tmp_path):
    presets = [_preset(low=250.0 + 50 * i, clip_id=f"c{i}", station=f"S{i % 2}")
               for i in range(4)]
    for index, preset in enumerate(presets):
        preset.clip_id, preset.station = f"c{index}", f"S{index % 2}"

    path = write_presets(tmp_path / "presets.jsonl", presets)
    loaded = load_presets(path)
    assert [p.as_dict() for p in loaded] == [p.as_dict() for p in presets]
    assert all(len(p.band_gains_db) == len(BAND_EDGES) - 1 for p in loaded)


def test_loading_rejects_unknown_fields(tmp_path):
    path = tmp_path / "presets.jsonl"
    row = _preset().as_dict() | {"mystery": 1}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unknown preset field"):
        load_presets(path)


def test_qc_drops_fit_loss_outliers():
    presets = [_preset(clip_id=f"c{i}") for i in range(10)]
    for index, preset in enumerate(presets):
        preset.clip_id = f"c{index}"
        preset.fit_loss = 1.0 + 0.05 * index
    presets[3].fit_loss = 50.0
    presets[7].fit_loss = float("nan")

    kept, dropped = _qc(presets, k=5.0)
    assert {p.clip_id for p in dropped} == {"c3", "c7"}
    assert len(kept) == 8


def test_qc_keeps_everything_when_the_fits_agree():
    presets = [_preset(clip_id=f"c{i}") for i in range(6)]
    for index, preset in enumerate(presets):
        preset.clip_id, preset.fit_loss = f"c{index}", 2.0 + 0.01 * index
    kept, dropped = _qc(presets, k=5.0)
    assert len(kept) == 6 and not dropped


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #

def test_fit_corpus_writes_presets_and_stats(tmp_path):
    import soundfile as sf

    clips = tmp_path / "clips"
    clips.mkdir()
    rows = []
    for index, (station, low, high) in enumerate([("ALPHA", 300.0, 2600.0),
                                                  ("BRAVO", 250.0, 1500.0)]):
        source = probe_batch(int(1.6 * SR), 1, np.random.default_rng(20 + index))[0]
        wav = apply_preset(source, SR, _preset(low=low, high=high), filter_noise=True)
        sf.write(clips / f"{station}_c{index}.wav", wav, SR)
        rows.append({"clip_id": f"{station}_c{index}",
                     "path": f"clips/{station}_c{index}.wav",
                     "station": station, "split": "train"})
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = tmp_path / "presets.jsonl"
    summary = fit_corpus(manifest, out, steps=30, n_probes=1, seed=0)

    assert summary["kept"] == 2 and summary["dropped"] == 0
    presets = load_presets(out)
    assert {p.station for p in presets} == {"ALPHA", "BRAVO"}
    assert all(p.ltas_l1_db > 0 for p in presets)
    assert set(summary["stations"]) == {"ALPHA", "BRAVO"}
    assert json.loads((out.parent / "presets_stats.json").read_text())["kept"] == 2
    # the narrower station must come back with the narrower passband
    by_station = {p.station: p for p in presets}
    assert by_station["BRAVO"].passband_hz[1] < by_station["ALPHA"].passband_hz[1]
