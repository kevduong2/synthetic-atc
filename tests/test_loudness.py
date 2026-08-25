import numpy as np
import pytest

from atcgen.channel.loudness import BLOCK_SEC, integrated_lufs, normalize_lufs
from atcgen.channel.primitives import TARGET_SR


def _sine(sec=3.0, f=440.0, amp=0.2, sr=TARGET_SR):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _rms_db(x):
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))))


def test_sine_normalizes_to_the_target():
    at_minus_14 = normalize_lufs(_sine(), TARGET_SR, target_lufs=-14.0)
    assert integrated_lufs(at_minus_14, TARGET_SR) == pytest.approx(-14.0, abs=0.5)
    for target in (-23.0, -20.0, -30.0):
        out = normalize_lufs(at_minus_14, TARGET_SR, target_lufs=target)
        assert integrated_lufs(out, TARGET_SR) == pytest.approx(target, abs=0.5)
        assert out.dtype == np.float32 and np.isfinite(out).all()


def test_short_clip_falls_back_to_rms():
    short = _sine(sec=BLOCK_SEC / 2)
    assert integrated_lufs(short, TARGET_SR) == float("-inf")
    out = normalize_lufs(short, TARGET_SR, target_lufs=-23.0)
    assert _rms_db(out) == pytest.approx(-23.0, abs=0.1)


def test_silence_passes_through():
    silence = np.zeros(TARGET_SR, dtype=np.float32)
    assert np.array_equal(normalize_lufs(silence, TARGET_SR), silence)
    assert normalize_lufs(np.zeros(0, np.float32), TARGET_SR).size == 0


def test_peak_ceiling_holds():
    out = normalize_lufs(_sine(amp=0.02), TARGET_SR, target_lufs=0.0, peak_ceiling=0.5)
    assert np.abs(out).max() == pytest.approx(0.5, abs=1e-6)


def test_non_finite_input_never_leaks_out():
    dirty = _sine()
    dirty[:10] = [np.nan, np.inf, -np.inf] + [0.0] * 7
    out = normalize_lufs(dirty, TARGET_SR)
    assert np.isfinite(out).all()
    assert integrated_lufs(out, TARGET_SR) == pytest.approx(-23.0, abs=0.5)


def test_a_clip_under_r128s_gate_falls_back_to_rms():
    """R128's absolute gate reads -inf below -70 LUFS; RMS still has an answer."""
    quiet = _sine(amp=1e-9)
    assert integrated_lufs(quiet, TARGET_SR) == float("-inf")
    assert _rms_db(normalize_lufs(quiet, TARGET_SR, target_lufs=-23.0)) == pytest.approx(
        -23.0, abs=0.1)
