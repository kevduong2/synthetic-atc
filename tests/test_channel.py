import random

import numpy as np

from atcgen.channel.dsp import ChannelParams, RadioChannelSim, TARGET_SR


def _tone(sr=24000, sec=1.0, f=440.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_output_sr_and_range():
    sim = RadioChannelSim()
    rng = random.Random(0)
    for seed in range(10):
        wav, params = sim(_tone(), 24000, random.Random(seed))
        assert wav.dtype == np.float32
        assert np.abs(wav).max() <= 1.0
        assert len(wav) > TARGET_SR  # 1s + padding
        assert np.isfinite(wav).all()


def test_noise_added_at_low_snr():
    sim = RadioChannelSim()
    p = ChannelParams.sample(random.Random(1))
    p.snr_db = 3.0
    p.squelch_click = False
    wav, _ = sim(_tone(f=1000), 24000, random.Random(1), params=p)
    # leading pad region should contain noise floor, not silence
    pad = int(TARGET_SR * 0.1)
    assert np.std(wav[:pad]) > 1e-4


def test_deterministic_given_seed_and_params():
    sim = RadioChannelSim()
    p = ChannelParams.sample(random.Random(2))
    a, _ = sim(_tone(), 24000, random.Random(3), params=p)
    b, _ = sim(_tone(), 24000, random.Random(3), params=p)
    assert np.array_equal(a, b)


def test_bandpass_removes_out_of_band_energy():
    sim = RadioChannelSim()
    p = ChannelParams.sample(random.Random(4))
    p.snr_db = 60.0  # nearly clean so we can measure the filter
    p.crackle_rate = 0.0
    p.squelch_click = False
    p.hum_amp = 0.0
    p.heterodyne = False
    p.dropout_prob = 0.0
    low, _ = sim(_tone(f=100), 24000, random.Random(4), params=p)   # below band
    mid, _ = sim(_tone(f=1000), 24000, random.Random(4), params=p)  # in band
    assert np.mean(mid**2) > 10 * np.mean(low**2)


def test_interference_mixed_in():
    sim = RadioChannelSim()
    p = ChannelParams.sample(random.Random(5))
    p.cochannel_level = 0.2
    interference = np.random.default_rng(0).standard_normal(TARGET_SR).astype(np.float32) * 0.3
    wav, _ = sim(_tone(), 24000, random.Random(5), params=p, interference=interference)
    assert np.isfinite(wav).all()
