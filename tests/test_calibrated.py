"""M2.3: the CalibratedChannel backend and its wiring into the builder.

Everything runs off fixtures written into `tmp_path` — a handful of synthetic
presets and a synthetic noise bank — so the suite never depends on the fitted
artifacts under `runs/`, on Kokoro, or on a GPU.
"""

import json
import random
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from atcgen.channel.learned.backend import (COCHANNEL_PROB, PTT_PROB,
                                            CalibratedChannel, StationNoise)
from atcgen.channel.learned.preset import BAND_EDGES, Preset, band_centers, write_presets
from atcgen.config import DistSpec, PostEffectsConfig, load_config
from atcgen.dataset.build import build_dataset, make_backend

SR = 16000
CENTRES = np.asarray(band_centers(BAND_EDGES))
STATIONS = {"ALPHA_TOWER": (300.0, 2600.0, 22.0), "BRAVO_CENTER": (250.0, 1600.0, 9.0)}


def _preset(clip_id: str, station: str, low: float, high: float,
            snr_db: float) -> Preset:
    gains = np.where((CENTRES >= low) & (CENTRES <= high), 0.0, -60.0)
    return Preset(clip_id=clip_id, station=station,
                  band_gains_db=[float(v) for v in gains], drive=1.5,
                  poly=[0.0, 0.0], agc_tau_ms=60.0, agc_strength=0.2,
                  noise_gain=10.0 ** (-snr_db / 20.0), snr_est=snr_db, fit_loss=1.0,
                  passband_hz=[low, high])


@pytest.fixture
def calibration(tmp_path):
    """A presets file and a station-keyed noise bank, as M2.1/M2.2 would write them."""
    presets = [_preset(f"{station}_{index}", station, low, high, snr)
               for station, (low, high, snr) in STATIONS.items()
               for index in range(4)]
    presets_path = write_presets(tmp_path / "presets.jsonl", presets)

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for station in STATIONS:
        for _ in range(5):
            segment = rng.standard_normal(int(0.3 * SR)).astype(np.float32) * 0.01
            sf.write(noise_dir / f"{len(rows):04d}.wav", segment, SR)
            rows.append({"source_clip": "c", "station": station, "duration": 0.3,
                         "rms_db": -40.0, "ltas_centroid_hz": 1000.0,
                         "squelch_gated": len(rows) % 10 == 0})
    (noise_dir / "noise_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return presets_path, noise_dir, presets


def _speech(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Band-limited noise bursts with pauses — enough shape for the whole chain."""
    from atcgen.channel.learned.channel_fit import synthetic_probe

    return synthetic_probe(int(seconds * SR), np.random.default_rng(seed)) * 0.2


def _channel(calibration, **kwargs) -> CalibratedChannel:
    _, noise_dir, presets = calibration
    kwargs.setdefault("post_effects", PostEffectsConfig())
    return CalibratedChannel(presets, StationNoise(noise_dir), **kwargs)


def _no_codec() -> PostEffectsConfig:
    """Post-effects with the codec off, so a spectrum reads the fitted band."""
    effects = PostEffectsConfig()
    effects.codec.prob = 0.0
    return effects


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #

def test_produces_plausible_16k_audio(calibration):
    channel = _channel(calibration)
    speech = _speech()
    out, record = channel(speech, SR, random.Random(0))

    assert out.dtype == np.float32
    assert len(out) > len(speech)                      # the chain's own padding
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(out.astype(np.float64) ** 2)) + 1e-12)
    assert -45.0 < rms_db < -3.0
    assert record.hops == 1 and record.snr_db is not None


def test_resamples_a_non_16k_input(calibration):
    channel = _channel(calibration)
    speech = _speech()
    out, _ = channel(speech, 24000, random.Random(0))
    assert len(out) == pytest.approx(len(speech) * SR / 24000 + 0.3 * SR, rel=0.02)


def test_record_carries_the_preset_it_used(calibration):
    channel = _channel(calibration)
    _, record = channel(_speech(), SR, random.Random(3))

    steps = [s for s in record.steps if s["primitive"] == "calibrated_preset"]
    assert len(steps) == 1
    assert steps[0]["clip_id"].startswith(steps[0]["station"])
    assert steps[0]["noise_station"] in STATIONS
    assert record.as_dict()["snr_db"] == steps[0]["snr_db"]


def test_station_mix_is_honored(calibration):
    channel = _channel(calibration, station_mix={"BRAVO_CENTER": 1.0})
    rng = random.Random(0)
    stations = Counter(channel.draw_preset(rng).station for _ in range(200))
    assert set(stations) == {"BRAVO_CENTER"}

    channel = _channel(calibration, station_mix={"ALPHA_TOWER": 0.75,
                                                 "BRAVO_CENTER": 0.25})
    stations = Counter(channel.draw_preset(rng).station for _ in range(2000))
    assert stations["ALPHA_TOWER"] / 2000 == pytest.approx(0.75, abs=0.05)


def test_default_station_mix_is_the_preset_pool_empirical(calibration):
    _, _, presets = calibration
    extra = presets + [_preset("ALPHA_TOWER_x", "ALPHA_TOWER", 300.0, 2600.0, 20.0)] * 8
    channel = CalibratedChannel(extra, None, post_effects=PostEffectsConfig())
    rng = random.Random(1)
    stations = Counter(channel.draw_preset(rng).station for _ in range(4000))
    assert stations["ALPHA_TOWER"] / 4000 == pytest.approx(12 / 16, abs=0.05)


def test_station_mix_rejects_unknown_stations(calibration):
    with pytest.raises(ValueError, match="no presets"):
        _channel(calibration, station_mix={"CHARLIE_GROUND": 1.0})


def test_snr_jitter_stays_inside_its_range(calibration):
    channel = _channel(calibration, snr_jitter=DistSpec.parse({"uniform": [-3, 3]}),
                       post_effects=PostEffectsConfig())
    fitted = {station: snr for station, (_, _, snr) in STATIONS.items()}
    seen = []
    for seed in range(60):
        _, record = channel(_speech(1.0), SR, random.Random(seed))
        step = next(s for s in record.steps if s["primitive"] == "calibrated_preset")
        offset = step["snr_db"] - fitted[step["station"]]
        assert -3.0 <= offset <= 3.0
        seen.append(offset)
    assert max(seen) - min(seen) > 3.0          # the jitter is actually applied


def test_noise_comes_from_the_presets_own_station(calibration):
    channel = _channel(calibration, cross_station_prob=0.0)
    for seed in range(30):
        _, record = channel(_speech(1.0), SR, random.Random(seed))
        step = next(s for s in record.steps if s["primitive"] == "calibrated_preset")
        assert step["noise_station"] == step["station"]

    channel = _channel(calibration, cross_station_prob=1.0)
    crossed = 0
    for seed in range(40):
        _, record = channel(_speech(1.0), SR, random.Random(seed))
        step = next(s for s in record.steps if s["primitive"] == "calibrated_preset")
        crossed += step["noise_station"] != step["station"]
    assert crossed > 5


def test_stations_produce_audibly_different_spectra(calibration):
    """The point of a preset pool: a Center channel is not a Tower channel."""
    speech = _speech(2.0)
    edges = {}
    for station in STATIONS:
        channel = _channel(calibration, station_mix={station: 1.0},
                           post_effects=_no_codec())
        out, _ = channel(speech, SR, random.Random(0))
        spectrum = np.abs(np.fft.rfft(out.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(out), 1.0 / SR)
        cumulative = np.cumsum(spectrum) / spectrum.sum()
        edges[station] = float(freqs[np.searchsorted(cumulative, 0.98)])
    assert edges["BRAVO_CENTER"] < edges["ALPHA_TOWER"] - 300.0


def test_relay_hop_adds_a_second_preset(calibration):
    channel = _channel(calibration)
    _, record = channel(_speech(), SR, random.Random(0), hops=2)
    steps = [s for s in record.steps if s["primitive"] == "calibrated_preset"]
    assert record.hops == 2 and len(steps) == 2
    assert [s["hop"] for s in steps] == [0, 1]


def test_post_effect_probabilities_are_respected(calibration):
    effects = PostEffectsConfig()
    effects.squelch.prob = 1.0
    effects.dropouts.prob = 1.0
    effects.codec.prob = 0.0
    channel = _channel(calibration, post_effects=effects)
    _, record = channel(_speech(), SR, random.Random(0))
    applied = {step["primitive"] for step in record.steps}
    assert {"squelch_gate", "squelch_clicks", "dropouts"} <= applied
    assert "codec_roundtrip" not in applied

    effects.squelch.prob = 0.0
    effects.dropouts.prob = 0.0
    _, record = _channel(calibration, post_effects=effects)(
        _speech(), SR, random.Random(0))
    assert {"squelch_gate", "dropouts"}.isdisjoint(
        {step["primitive"] for step in record.steps})


def test_gated_floor_probability_splits_the_squelch_floor(calibration):
    effects = PostEffectsConfig()
    effects.squelch.prob = 1.0
    effects.squelch.gated_floor_prob = 1.0
    channel = _channel(calibration, post_effects=effects)
    floors = []
    for seed in range(20):
        _, record = channel(_speech(1.0), SR, random.Random(seed))
        floors.append(next(s for s in record.steps
                           if s["primitive"] == "squelch_gate")["floor_db"])
    assert max(floors) <= -55.0

    effects.squelch.gated_floor_prob = 0.0
    channel = _channel(calibration, post_effects=effects)
    floors = [next(s for s in channel(_speech(1.0), SR, random.Random(seed))[1].steps
                   if s["primitive"] == "squelch_gate")["floor_db"]
              for seed in range(20)]
    assert min(floors) >= -45.0


def test_cochannel_only_fires_when_interference_is_supplied(calibration):
    channel = _channel(calibration)
    other = _speech(2.0, seed=7)
    hits = 0
    for seed in range(120):
        _, record = channel(_speech(1.0), SR, random.Random(seed), interference=other)
        hits += any(s["primitive"] == "cochannel_mix" for s in record.steps)
    assert 0 < hits < 120 * COCHANNEL_PROB * 4

    for seed in range(30):
        _, record = channel(_speech(1.0), SR, random.Random(seed))
        assert not any(s["primitive"] == "cochannel_mix" for s in record.steps)


def test_ptt_truncation_fires_at_roughly_its_probability(calibration):
    channel = _channel(calibration)
    hits = sum(any(s["primitive"] == "ptt_truncation"
                   for s in channel(_speech(1.0), SR, random.Random(seed))[1].steps)
               for seed in range(200))
    assert abs(hits / 200 - PTT_PROB) < 0.1


def test_is_deterministic_for_a_given_seed(calibration):
    channel = _channel(calibration)
    speech = _speech()
    first, _ = channel(speech, SR, random.Random(11))
    second, _ = channel(speech, SR, random.Random(11))
    assert np.array_equal(first, second)


def test_runs_without_a_noise_bank(calibration):
    _, _, presets = calibration
    channel = CalibratedChannel(presets, None, post_effects=PostEffectsConfig())
    out, record = channel(_speech(), SR, random.Random(0))
    assert np.isfinite(out).all()
    step = next(s for s in record.steps if s["primitive"] == "calibrated_preset")
    assert step["noise_station"] is None


def test_empty_preset_pool_is_rejected():
    with pytest.raises(ValueError, match="at least one preset"):
        CalibratedChannel([], None)


# --------------------------------------------------------------------------- #
# the noise bank
# --------------------------------------------------------------------------- #

def test_station_noise_groups_by_station_and_stitches(calibration):
    _, noise_dir, _ = calibration
    bank = StationNoise(noise_dir)
    assert set(bank.stations) == set(STATIONS)
    assert bank.gated_fraction == pytest.approx(0.1, abs=1e-6)

    bed = bank.sample(5 * SR, random.Random(0), "ALPHA_TOWER")
    assert bed.shape == (5 * SR,)
    # stitched from many independent 0.3 s draws, so no half repeats the other
    half = len(bed) // 2
    assert abs(np.corrcoef(bed[:half], bed[half:2 * half])[0, 1]) < 0.2


def test_station_noise_falls_back_when_a_station_has_none(calibration):
    _, noise_dir, _ = calibration
    bank = StationNoise(noise_dir)
    bed = bank.sample(SR, random.Random(0), "NOT_A_STATION")
    assert bed.shape == (SR,) and np.isfinite(bed).all()


def test_station_noise_needs_a_stats_file(tmp_path):
    with pytest.raises(ValueError, match="noise_stats.jsonl"):
        StationNoise(tmp_path)


# --------------------------------------------------------------------------- #
# builder wiring
# --------------------------------------------------------------------------- #

class _FakeTTS:
    sample_rate = SR

    def synthesize(self, text, rng, voice=None, speed=None):
        return _speech(1.2, seed=abs(hash(text)) % 1000)


def _config(tmp_path, calibration):
    presets_path, noise_dir, _ = calibration
    text = (tmp_path / "mode2.yaml")
    text.write_text(f"""
mode: calibrated
seed: 3
dataset: {{noise_only_frac: 0.1, pilot_double_hop_prob: 0.5}}
calibrated:
  calibration:
    presets: {presets_path}
    noise_bank: {noise_dir}
    snr_jitter_db: {{uniform: [-3, 3]}}
  residual: {{enabled: false}}
  post_effects:
    squelch: {{prob: 0.8, gated_floor_prob: 0.05}}
    codec: {{prob: 0.5, kind: mp3, quality: {{uniform: [0.8, 0.95]}}}}
""")
    return load_config(text)


def test_make_backend_builds_the_calibrated_channel(tmp_path, calibration):
    backend = make_backend(_config(tmp_path, calibration))
    assert isinstance(backend, CalibratedChannel)
    assert backend.noise is not None
    assert backend.cross_station_prob == pytest.approx(0.1)
    assert backend.post.squelch.gated_floor_prob == pytest.approx(0.05)


def test_make_backend_needs_a_calibrated_section():
    config = load_config.__globals__["GeneratorConfig"](mode="calibrated")
    config.calibrated = None
    with pytest.raises(ValueError, match="requires a calibrated section"):
        make_backend(config)


def test_build_dataset_end_to_end(tmp_path, calibration):
    config = _config(tmp_path, calibration)
    out = tmp_path / "set"
    manifest = build_dataset(config, out, 12, text_source="grammar", tts=_FakeTTS())

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    assert len(rows) == 12
    assert {row["gen"]["mode"] for row in rows} == {"calibrated"}
    for row in rows:
        audio, file_sr = sf.read(out / row["audio"], dtype="float32")
        assert file_sr == SR and len(audio) > 0
        channel = row["gen"]["channel"]
        assert channel["snr_db"] is not None
        used = [s for s in channel["steps"] if s["primitive"] == "calibrated_preset"]
        assert used and used[0]["clip_id"].startswith(used[0]["station"])

    stats = json.loads((out / "stats.json").read_text())
    assert stats["mode"] == "calibrated" and stats["snr_db"]["n"] == 12
