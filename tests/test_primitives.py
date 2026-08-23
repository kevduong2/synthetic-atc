import random

import numpy as np
import pytest

from atcgen.channel import primitives as P
from atcgen.channel.primitives import TARGET_SR, NoiseBank, resample

SR = TARGET_SR


def _tone(f=1000.0, sec=1.0, amp=0.5, sr=SR):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _band_power(x, lo, hi, sr=SR):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


def test_registry_signatures():
    for name, fn in P.PRIMITIVES.items():
        params = list(__import__("inspect").signature(fn).parameters)
        assert params[:3] == ["x", "sr", "rng"], name


def test_resample_roundtrip_length_and_identity():
    x = _tone(sec=0.5)
    assert len(resample(x, SR, 8000)) == len(x) // 2
    assert np.array_equal(resample(x, SR, SR), x)


def test_narrowband_roundtrip_kills_above_nyquist():
    x = _tone(f=3500) + _tone(f=1000)
    y = P.narrowband_roundtrip(x, SR, random.Random(0), narrow_sr=6000)
    assert len(y) == len(x)
    assert _band_power(y, 3300, 3700) < 0.01 * _band_power(x, 3300, 3700)
    assert _band_power(y, 900, 1100) > 0.5 * _band_power(x, 900, 1100)


def test_narrowband_roundtrip_noop_when_not_narrower():
    x = _tone()
    assert np.array_equal(P.narrowband_roundtrip(x, SR, random.Random(0), narrow_sr=SR), x)


def test_bandpass_kills_out_of_band_tones():
    rng = random.Random(0)
    low, mid, high = _tone(f=80), _tone(f=1200), _tone(f=6000)
    args = dict(low=300.0, high=3400.0)
    assert np.mean(P.bandpass(low, SR, rng, **args) ** 2) < 0.01 * np.mean(low ** 2)
    assert np.mean(P.bandpass(high, SR, rng, **args) ** 2) < 0.01 * np.mean(high ** 2)
    assert np.mean(P.bandpass(mid, SR, rng, **args) ** 2) > 0.4 * np.mean(mid ** 2)


def test_agc_wander_modulates_envelope_only_when_on():
    x = _tone()
    assert np.array_equal(P.agc_wander(x, SR, random.Random(0), strength=0.0), x)
    y = P.agc_wander(x, SR, random.Random(0), strength=0.6)
    # slow gain wander: envelope varies but stays within +-strength*0.3
    ratio = np.abs(y[np.abs(x) > 0.4]) / np.abs(x[np.abs(x) > 0.4])
    assert ratio.max() <= 1.0 + 0.6 * 0.3 + 1e-5
    assert ratio.min() >= 1.0 - 0.6 * 0.3 - 1e-5
    assert ratio.max() - ratio.min() > 0.01


def test_am_distortion_adds_harmonics():
    x = _tone(f=500)
    y = P.am_distortion(x, SR, random.Random(0), depth=0.25)
    # x + depth*|x|*x is odd, so it grows the odd harmonics
    assert _band_power(y, 1400, 1600) > 100 * _band_power(x, 1400, 1600)
    assert np.array_equal(P.am_distortion(x, SR, random.Random(0), depth=0.0), x)


def test_soft_clip_compresses_peaks():
    x = _tone(amp=1.0)
    y = P.soft_clip(x, SR, random.Random(0), drive=4.0)
    assert np.abs(y).max() <= 1.0 + 1e-6
    # crest factor drops: peaks squashed relative to rms
    assert np.abs(y).max() / y.std() < np.abs(x).max() / x.std()
    assert np.allclose(P.soft_clip(x, SR, random.Random(0), drive=1.0), np.tanh(x) / np.tanh(1.0))


def test_dropouts_attenuate_a_contiguous_span():
    x = np.ones(SR, np.float32)
    y = P.dropouts(x, SR, random.Random(3), dropout_prob=1.0)
    quiet = np.flatnonzero(y < 0.5)
    assert len(quiet) > 0
    assert 160 <= len(quiet) <= 4 * 800          # 1-4 drops of 10-50 ms
    assert y.max() == pytest.approx(1.0)         # untouched elsewhere
    assert np.array_equal(P.dropouts(x, SR, random.Random(3), dropout_prob=0.0), x)


@pytest.mark.parametrize("color", ["white", "pink"])
@pytest.mark.parametrize("snr_db", [3.0, 15.0, 25.0])
def test_additive_noise_hits_target_snr(color, snr_db):
    x = _tone(sec=2.0)
    y = P.additive_noise(x, SR, random.Random(1), snr_db=snr_db, color=color)
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert measured == pytest.approx(snr_db, abs=1.0)


def test_additive_noise_snr_measured_on_unpadded_core():
    pad = int(SR * 0.15)
    x = np.concatenate([np.zeros(pad, np.float32), _tone(), np.zeros(pad, np.float32)])
    y = P.additive_noise(x, SR, random.Random(1), snr_db=20.0, color="white", pad=pad)
    core_power = np.mean(x[pad:len(x) - pad] ** 2)
    measured = 10 * np.log10(core_power / np.mean((y - x) ** 2))
    assert measured == pytest.approx(20.0, abs=1.0)
    assert np.std(y[:pad]) > 1e-4                # noise floor fills the padding


def test_pink_noise_is_tilted_down():
    n = P.pink_noise(SR, np.random.default_rng(0))
    assert len(n) == SR
    # power density falls with frequency (white noise would be flat)
    assert _band_power(n, 100, 500) / 400 > 4 * _band_power(n, 2000, 4000) / 2000


def test_hum_adds_mains_tone():
    x = np.zeros(SR, np.float32)
    y = P.hum(x, SR, random.Random(0), amp=0.01)
    assert _band_power(y, 45, 65) > 100 * _band_power(y, 500, 1000)
    assert np.array_equal(P.hum(x, SR, random.Random(0), amp=0.0), x)


def test_crackle_rate_controls_event_count():
    x = np.zeros(SR * 2, np.float32)
    few = P.crackle(x, SR, random.Random(0), rate=0.5)
    many = P.crackle(x, SR, random.Random(0), rate=20.0)
    assert np.count_nonzero(many) > np.count_nonzero(few)
    assert np.array_equal(P.crackle(x, SR, random.Random(0), rate=0.0), x)


def test_heterodyne_adds_a_single_in_band_whine():
    x = np.zeros(SR, np.float32)
    y = P.heterodyne(x, SR, random.Random(0), f_low=1000.0, f_high=1000.0)
    assert 0.01 <= np.abs(y).max() <= 0.04 + 1e-6
    assert _band_power(y, 900, 1100) > 100 * _band_power(y, 1500, 3000)


def test_squelch_clicks_only_touch_the_ends():
    x = np.zeros(SR, np.float32)
    y = P.squelch_clicks(x, SR, random.Random(0))
    assert np.count_nonzero(y[200:-200]) == 0
    assert np.abs(y[:200]).max() > 0.05
    assert np.abs(y[-200:]).max() > 0.05


def test_cochannel_mix_adds_scaled_interference():
    x = np.zeros(SR, np.float32)
    other = _tone(f=700, sec=0.5)
    y = P.cochannel_mix(x, SR, random.Random(0), level=0.2, interference=other)
    assert np.abs(y).max() == pytest.approx(0.2 * np.abs(other).max(), abs=1e-6)
    assert np.count_nonzero(y[len(other):]) == 0          # short interference is padded
    assert np.array_equal(P.cochannel_mix(x, SR, random.Random(0), level=0.2), x)
    assert np.array_equal(P.cochannel_mix(x, SR, random.Random(0), level=0.0,
                                          interference=other), x)


def test_codec_roundtrip_preserves_length_and_signal():
    x = _tone(f=1000)
    y = P.codec_roundtrip(x, SR, random.Random(0), compression_level=0.9)
    assert len(y) == len(x)
    assert np.isfinite(y).all()
    assert np.std(y) > 1e-3
    assert _band_power(y, 900, 1100) > 0.5 * _band_power(x, 900, 1100)


def test_primitives_do_not_mutate_input():
    x = _tone(amp=0.9)
    for name, fn in P.PRIMITIVES.items():
        before = x.copy()
        fn(x, SR, random.Random(0))
        assert np.array_equal(x, before), name


def test_noise_bank_crops_and_tiles(tmp_path):
    import soundfile as sf

    bed = np.random.default_rng(0).standard_normal(SR // 2).astype(np.float32) * 0.05
    sf.write(tmp_path / "noise_00000.wav", bed, SR)
    bank = NoiseBank(tmp_path)

    assert len(bank.sample(SR * 2, random.Random(0))) == SR * 2      # tiled
    assert len(bank.sample(SR // 4, random.Random(0))) == SR // 4    # cropped

    x = _tone(sec=1.0)
    y = P.additive_noise(x, SR, random.Random(0), snr_db=10.0, noise_bank=bank, bed_prob=1.0)
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert measured == pytest.approx(10.0, abs=1.0)


def test_noise_bank_rejects_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        NoiseBank(tmp_path)
