import random

import numpy as np
import pytest
from scipy import signal

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


PAD = int(SR * 0.15)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if len(x) else 0.0


def _gated_fixture(gap_sec=0.0, speech_sec=1.0, noise_amp=0.02, sr=SR):
    """Padded clip: noise everywhere, speech (a 4 Hz-modulated tone) in the middle,
    optionally with a silent-but-for-noise gap in the speech."""
    speech = _tone(f=800, sec=speech_sec, amp=0.5)
    t = np.arange(len(speech)) / sr
    speech *= (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)).astype(np.float32)
    if gap_sec > 0:
        mid = len(speech) // 2
        speech[mid:mid + int(sr * gap_sec)] = 0.0
    x = np.concatenate([np.zeros(PAD, np.float32), speech, np.zeros(PAD, np.float32)])
    noise = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32) * noise_amp
    return (x + noise).astype(np.float32)


def test_squelch_gate_drops_the_pad_to_the_floor():
    x = _gated_fixture()
    floor_db = -40.0
    y = P.squelch_gate(x, SR, random.Random(0), floor_db=floor_db, attack_ms=10.0,
                       release_ms=30.0, tail_burst_prob=0.0, pad=PAD)
    assert len(y) == len(x)
    # measured clear of the release ramp at each end of the padding
    drop_db = 20 * np.log10(_rms(y[PAD // 4:PAD - 200]) / _rms(x[PAD // 4:PAD - 200]))
    assert drop_db <= floor_db + 6.0                    # ramp/interpolation tolerance
    assert _rms(y[PAD:len(x) - PAD]) > 0.5 * _rms(x[PAD:len(x) - PAD])   # speech survives


@pytest.mark.parametrize("floor_db", [-20.0, -60.0])
def test_squelch_gate_floor_depth_follows_the_parameter(floor_db):
    x = _gated_fixture()
    y = P.squelch_gate(x, SR, random.Random(0), floor_db=floor_db,
                       tail_burst_prob=0.0, pad=PAD)
    quiet = slice(PAD // 4, PAD - 200)
    assert 20 * np.log10(_rms(y[quiet]) / _rms(x[quiet])) == pytest.approx(floor_db, abs=6.0)


def test_squelch_gate_closes_on_intra_speech_silence():
    x = _gated_fixture(gap_sec=0.3)
    y = P.squelch_gate(x, SR, random.Random(0), floor_db=-40.0, hold_ms=20.0,
                       threshold_db=-20.0, tail_burst_prob=0.0, pad=PAD)
    mid = PAD + int(SR * 0.5)                            # inside the 300 ms gap
    gap = slice(mid + 1600, mid + 3200)                  # clear of the release ramp
    assert _rms(y[gap]) < 0.2 * _rms(x[gap])
    # the same clip at a threshold below its noise floor keeps the gate open
    loose = P.squelch_gate(x, SR, random.Random(0), floor_db=-40.0, hold_ms=20.0,
                           threshold_db=-40.0, tail_burst_prob=0.0, pad=PAD)
    assert _rms(loose[gap]) > 0.9 * _rms(x[gap])


def test_squelch_gate_tail_burst_is_added_at_the_close():
    x = _gated_fixture()
    args = dict(floor_db=-60.0, pad=PAD)
    quiet = P.squelch_gate(x, SR, random.Random(0), tail_burst_prob=0.0, **args)
    burst = P.squelch_gate(x, SR, random.Random(0), tail_burst_prob=1.0,
                           tail_burst_amp=0.5, **args)
    added = burst - quiet
    assert np.count_nonzero(added[:len(x) - PAD]) == 0    # only after the speech ends
    assert _rms(added[len(x) - PAD:]) > 5 * _rms(quiet[len(x) - PAD // 2:])
    assert np.abs(added).max() > 0.1                      # amp 0.5 x the envelope peak


def test_ptt_truncation_shortens_speech_but_not_the_array():
    x = _gated_fixture(noise_amp=0.0)
    y = P.ptt_truncation(x, SR, random.Random(0), head_ms=120.0, tail_ms=60.0, pad=PAD)
    assert len(y) == len(x)
    head, tail = int(SR * 0.120), int(SR * 0.060)
    assert _rms(y[PAD:PAD + head]) == 0.0
    assert _rms(y[len(x) - PAD - tail:len(x) - PAD]) == 0.0
    assert _rms(x[PAD:PAD + head]) > 0.1                 # there was speech there
    middle = slice(PAD + head + 200, len(x) - PAD - tail - 200)
    assert np.array_equal(y[middle], x[middle])          # untouched between the cuts


def test_ptt_truncation_anchors_on_the_first_audible_sample():
    """TTS leading silence must not absorb the cut -- phonemes have to go."""
    lead = int(SR * 0.4)
    speech = _tone(f=800, sec=1.0, amp=0.5)
    x = np.concatenate([np.zeros(PAD + lead, np.float32), speech,
                        np.zeros(PAD + lead, np.float32)])
    y = P.ptt_truncation(x, SR, random.Random(0), head_ms=100.0, tail_ms=100.0, pad=PAD)
    cut = int(SR * 0.100)
    assert _rms(y[PAD + lead:PAD + lead + cut]) == 0.0        # speech, not silence
    assert _rms(x[PAD + lead:PAD + lead + cut]) > 0.1
    assert _rms(y[len(x) - PAD - lead - cut:len(x) - PAD - lead]) == 0.0
    assert _rms(y[PAD + lead + cut + 200:len(x) - PAD - lead - cut - 200]) > 0.1


def test_ptt_truncation_defaults_and_caps_are_noops_or_bounded():
    x = _gated_fixture(noise_amp=0.0)
    assert np.array_equal(P.ptt_truncation(x, SR, random.Random(0), pad=PAD), x)
    # a cut longer than the speech is clamped to a third of the extent, not fatal
    y = P.ptt_truncation(x, SR, random.Random(0), head_ms=9000.0, pad=PAD)
    assert len(y) == len(x) and np.isfinite(y).all()
    assert _rms(y[PAD:len(x) - PAD]) > 0.0


@pytest.mark.parametrize("tilt_db", [-4.0, 4.0])
def test_mic_coloration_tilt_sign_moves_the_spectral_slope(tilt_db):
    x = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.1
    y = P.mic_coloration(x, SR, random.Random(0), tilt_db=tilt_db, peaks=0)
    ratio = _band_power(y, 2000, 4000) / _band_power(y, 200, 500)
    base = _band_power(x, 2000, 4000) / _band_power(x, 200, 500)
    assert (ratio > 1.4 * base) if tilt_db > 0 else (ratio < 0.7 * base)


def test_mic_coloration_peaks_are_bounded_and_local():
    x = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.1
    y = P.mic_coloration(x, SR, random.Random(2), tilt_db=0.0, peaks=2, peak_gain_db=6.0)
    assert len(y) == len(x) and np.isfinite(y).all()
    # +-6 dB peaking filters cannot move the broadband level far
    assert abs(20 * np.log10(_rms(y) / _rms(x))) < 6.0
    assert not np.allclose(y, x)
    assert np.array_equal(P.mic_coloration(x, SR, random.Random(0), tilt_db=0.0, peaks=0), x)


@pytest.mark.parametrize("rate_hz", [0.5, 2.0])
def test_fading_modulates_the_envelope_at_the_configured_rate(rate_hz):
    x = _tone(f=1000, sec=8.0)
    y = P.fading(x, SR, random.Random(0), rate_hz=rate_hz, depth_db=6.0)
    gain = np.abs(y[np.abs(x) > 0.4]) / np.abs(x[np.abs(x) > 0.4])
    # peak-to-trough matches depth_db; both bounds land within a tolerance
    assert 20 * np.log10(gain.max() / gain.min()) == pytest.approx(6.0, abs=0.5)
    env = np.abs(signal.hilbert(y.astype(np.float64)))
    spec = np.abs(np.fft.rfft(env - env.mean()))
    freqs = np.fft.rfftfreq(len(env), 1.0 / SR)
    assert freqs[int(np.argmax(spec))] == pytest.approx(rate_hz, abs=0.2)
    assert np.array_equal(P.fading(x, SR, random.Random(0), depth_db=0.0), x)


def test_agc_attack_surges_at_the_start_of_the_transmission():
    x = _gated_fixture(noise_amp=0.0)
    y = P.agc_attack(x, SR, random.Random(0), attack_ms=100.0, surge_db=6.0, pad=PAD)
    onset = slice(PAD, PAD + int(SR * 0.05))
    late = slice(PAD + int(SR * 0.6), PAD + int(SR * 0.9))
    assert 20 * np.log10(_rms(y[onset]) / _rms(x[onset])) == pytest.approx(5.0, abs=1.5)
    assert _rms(y[late]) == pytest.approx(_rms(x[late]), rel=0.02)
    assert np.array_equal(y[:PAD], x[:PAD])              # nothing before squelch open
    assert np.array_equal(P.agc_attack(x, SR, random.Random(0), surge_db=0.0), x)


def test_resample_chain_folds_content_back_into_the_band():
    x = _tone(f=3000, sec=1.0)
    y = P.resample_chain(x, SR, random.Random(0), narrow_sr=5000, alias=True)
    assert len(y) == len(x)
    # 3 kHz sampled at 5 kHz aliases to 5000 - 3000 = 2000 Hz
    assert _band_power(y, 1900, 2100) > 10 * _band_power(y, 2900, 3100)
    clean = P.resample_chain(x, SR, random.Random(0), narrow_sr=5000, alias=False)
    assert _band_power(clean, 1900, 2100) < 0.01 * _band_power(y, 1900, 2100)
    assert np.array_equal(P.resample_chain(x, SR, random.Random(0), narrow_sr=SR), x)


@pytest.mark.parametrize("bitrate_kbps", [16, 23, 32, 64])
def test_codec_roundtrip_bitrate_tiers_encode_and_decode(bitrate_kbps):
    x = _tone(f=1000, sec=2.0)
    y = P.codec_roundtrip(x, SR, random.Random(0), bitrate_kbps=bitrate_kbps)
    assert len(y) == len(x)
    assert np.isfinite(y).all() and np.std(y) > 1e-3
    assert _band_power(y, 900, 1100) > 0.5 * _band_power(x, 900, 1100)


def test_codec_roundtrip_encoded_size_scales_with_bitrate():
    import io

    import soundfile as sf

    x = _tone(f=1000, sec=4.0)
    sizes = []
    for kbps in (16, 32, 64):
        buf = io.BytesIO()
        sf.write(buf, x, SR, format="MP3", bitrate_mode="CONSTANT",
                 compression_level=P.MP3_CBR_COMPRESSION[kbps])
        sizes.append(buf.getbuffer().nbytes * 8 / 4.0 / 1000)
    assert sizes == [pytest.approx(kbps, rel=0.05) for kbps in (16, 32, 64)]
    # 23 kbps is not in the MPEG-2 Layer III table; it snaps to LAME's 24
    assert (P.codec_roundtrip(x, SR, random.Random(0), bitrate_kbps=23)
            == pytest.approx(P.codec_roundtrip(x, SR, random.Random(0), bitrate_kbps=24)))


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

    assert len(bank.sample(SR * 2, random.Random(0))) == SR * 2      # stitched
    assert len(bank.sample(SR // 4, random.Random(0))) == SR // 4    # cropped

    x = _tone(sec=1.0)
    y = P.additive_noise(x, SR, random.Random(0), snr_db=10.0, noise_bank=bank, bed_prob=1.0)
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert measured == pytest.approx(10.0, abs=1.0)


def test_noise_bank_stitches_short_beds_without_looping_one(tmp_path):
    """Short harvested beds must not be repeated bit-for-bit across a clip."""
    import soundfile as sf

    rng_np = np.random.default_rng(0)
    short = SR // 4                                       # 0.25 s, like the real bank
    for index in range(8):
        sf.write(tmp_path / f"noise_{index:05d}.wav",
                 rng_np.standard_normal(short).astype(np.float32) * 0.05, SR)
    bank = NoiseBank(tmp_path)

    n = SR * 5
    for seed in range(20):
        bed = bank.sample(n, random.Random(seed))
        assert len(bed) == n
        # no lag that is a whole number of bed lengths repeats exactly
        for lag in range(short, n, short):
            assert not np.array_equal(bed[:n - lag], bed[lag:]), (seed, lag)


def test_noise_bank_rejects_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        NoiseBank(tmp_path)
