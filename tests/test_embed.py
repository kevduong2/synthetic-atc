"""Tier 1 embedding distances + Tier 2 probe.

No network and no model downloads: every test injects a fake embedder that is a
cheap deterministic function of the waveform, so what is exercised here is the
distance maths, the fold plumbing and the directory/array wiring -- not WavLM
or CLAP themselves.
"""

import json

import numpy as np
import pytest
import soundfile as sf

from atcgen.eval import embed_dist, probe as probe_mod
from atcgen.eval.embed_dist import (compare, compare_dirs, embed_clips, frechet,
                                    kid)
from atcgen.eval.probe import layer_sweep, null_control, probe, probe_dirs

SR = 16000
DIM = 8


def fake_embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """Deterministic 8-d descriptor of a waveform (band energies + level)."""
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    spec = np.abs(np.fft.rfft(x, n=1024)) ** 2
    bands = np.array([spec[i * 64:(i + 1) * 64].sum() for i in range(7)])
    bands = np.log10(bands / (bands.sum() + 1e-20) + 1e-6)
    return np.concatenate([bands, [np.log10(np.mean(x ** 2) + 1e-12)]])


def fake_factory(**_kwargs):
    return fake_embed


def _tone(freq=440.0, sec=0.5, amp=0.2, sr=SR, seed=0):
    t = np.arange(int(sr * sec)) / sr
    noise = np.random.default_rng(seed).standard_normal(len(t)) * 0.01
    return (amp * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)


def _write_clips(path, n, freq, seed0=0):
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sf.write(path / f"c{i:03d}.wav", _tone(freq, seed=seed0 + i), SR)
    return path


def _gauss(n, dim=DIM, shift=0.0, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)) + shift


# --- KID ---------------------------------------------------------------------

def test_kid_near_zero_on_identical_sets():
    # Comparing a pool against itself leaves an O(1/pool) negative residue:
    # the two subsets overlap, so the cross term picks up k(x, x) values the
    # within terms drop.  It shrinks with the pool and stays far below the
    # separation the shifted sets show below.
    x = _gauss(400, seed=1)
    r = kid(x, x.copy(), subsets=50, subset_size=30)
    assert abs(r["kid"]) < 0.06
    assert abs(r["kid"]) < 0.05 * kid(_gauss(400, shift=1.0, seed=2), x,
                                      subsets=50, subset_size=30)["kid"]
    assert r["subset_size"] == 30 and r["subsets"] == 50


def test_kid_near_zero_for_same_distribution():
    r = kid(_gauss(80, seed=1), _gauss(80, seed=2), subsets=50, subset_size=30)
    assert abs(r["kid"]) < 0.1


def test_kid_positive_and_monotone_in_the_shift():
    ref = _gauss(80, seed=1)
    small = kid(_gauss(80, shift=0.5, seed=2), ref, subsets=50, subset_size=30)
    large = kid(_gauss(80, shift=2.0, seed=3), ref, subsets=50, subset_size=30)
    assert 0.0 < small["kid"] < large["kid"]


def test_kid_unbiased_at_small_n():
    """The point of the unbiased estimator (05 §1): over repeated draws of two
    n=15 same-distribution sets KID averages to ~0, while the Frechet distance
    is inflated on every single draw."""
    kids, fads = [], []
    for s in range(40):
        a, b = _gauss(15, seed=2 * s), _gauss(15, seed=2 * s + 1)
        kids.append(kid(a, b, subsets=30, subset_size=10)["kid"])
        fads.append(frechet(a, b))
    assert abs(float(np.mean(kids))) < 0.05
    assert min(fads) > 1.5


def test_kid_subset_size_clamped_to_the_smaller_set():
    r = kid(_gauss(12, seed=1), _gauss(40, seed=2), subsets=20, subset_size=50)
    assert r["subset_size"] == 12


def test_kid_rejects_mismatched_dims_and_tiny_sets():
    with pytest.raises(ValueError, match="size mismatch"):
        kid(_gauss(10, dim=4), _gauss(10, dim=6))
    with pytest.raises(ValueError, match="at least 2"):
        kid(_gauss(1), _gauss(1))


def test_kid_is_deterministic_for_a_seed():
    a, b = _gauss(40, seed=1), _gauss(40, shift=1.0, seed=2)
    assert kid(a, b, seed=7)["kid"] == kid(a, b, seed=7)["kid"]


# --- Frechet -----------------------------------------------------------------

def test_frechet_zero_on_identical_sets():
    x = _gauss(200, seed=1)
    assert frechet(x, x.copy()) == pytest.approx(0.0, abs=1e-3)


def test_frechet_grows_with_the_mean_shift():
    ref = _gauss(300, seed=1)
    d1 = frechet(_gauss(300, shift=0.5, seed=2), ref)
    d2 = frechet(_gauss(300, shift=2.0, seed=3), ref)
    assert 0.0 < d1 < d2
    # a pure mean shift of s contributes s^2 per dimension
    assert d2 == pytest.approx(4.0 * DIM, rel=0.25)


def test_compare_reports_both_and_flags_small_n_frechet():
    small = compare(_gauss(10, seed=1), _gauss(10, seed=2), subsets=20,
                    subset_size=5)
    big = compare(_gauss(50, seed=1), _gauss(50, seed=2), subsets=20)
    assert small["frechet_reliable"] is False      # n=10 < 2*8 dims
    assert big["frechet_reliable"] is True
    assert small["dim"] == DIM and small["n_real"] == 10
    json.dumps(small)                               # JSON-serializable


# --- embedding a set of clips ------------------------------------------------

def test_embed_clips_from_arrays_and_from_a_directory(tmp_path):
    clips = [_tone(440, seed=i) for i in range(4)]
    names, x = embed_clips(clips, fake_embed, sr=SR)
    assert x.shape == (4, DIM) and len(names) == 4

    d = _write_clips(tmp_path / "wavs", 4, 440.0)
    dir_names, dir_x = embed_clips(d, fake_embed)
    assert dir_names == ["c000.wav", "c001.wav", "c002.wav", "c003.wav"]
    assert dir_x == pytest.approx(x, abs=1e-3)   # 16-bit PCM round-trip


def test_embed_clips_stacks_multi_layer_embedders():
    def three_layers(wav, sr):
        return np.stack([fake_embed(wav, sr) * (i + 1) for i in range(3)])

    _, x = embed_clips([_tone(seed=i) for i in range(5)], three_layers, sr=SR)
    assert x.shape == (5, 3, DIM)


def test_embed_clips_rejects_an_empty_source(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no clips"):
        embed_clips(tmp_path / "empty", fake_embed)


def test_compare_dirs_runs_every_family(tmp_path, monkeypatch):
    monkeypatch.setitem(embed_dist.EMBEDDERS, "wavlm", fake_factory)
    monkeypatch.setitem(embed_dist.EMBEDDERS, "clap", fake_factory)
    syn = _write_clips(tmp_path / "syn", 8, 900.0)
    real = _write_clips(tmp_path / "real", 8, 300.0, seed0=100)

    res = compare_dirs(syn, real, subsets=20, subset_size=5)
    assert set(res["families"]) == {"wavlm", "clap"}
    assert res["families"]["wavlm"]["layer"] == embed_dist.WAVLM_LAYER
    assert res["families"]["wavlm"]["kid"] > 0.0     # 900 Hz vs 300 Hz tones
    json.dumps(res)

    with pytest.raises(ValueError, match="unknown embedding family"):
        compare_dirs(syn, real, families=("vggish",))


# --- Tier 2 probe ------------------------------------------------------------

def test_probe_separates_trivially_separable_embeddings():
    r = probe(_gauss(60, shift=6.0, seed=1), _gauss(60, seed=2))
    assert r["balanced_accuracy"] > 0.95
    assert r["verdict"] == "separable" and r["passes_gate"] is False


def test_probe_is_near_chance_on_one_distribution():
    r = probe(_gauss(80, seed=1), _gauss(80, seed=2))
    assert 0.3 < r["balanced_accuracy"] < 0.7
    assert r["passes_gate"] is True


def test_probe_reports_fold_plumbing():
    r = probe(_gauss(50, shift=6.0, seed=1), _gauss(50, seed=2), folds=5)
    assert len(r["fold_accuracies"]) == 5
    assert r["folds"] == 5 and r["dim"] == DIM
    assert r["accuracy_std"] >= 0.0
    json.dumps(r)


def test_probe_balances_the_classes_by_default():
    r = probe(_gauss(90, seed=1), _gauss(30, seed=2))
    assert r["n_synthetic"] == r["n_real"] == 30
    unbalanced = probe(_gauss(90, seed=1), _gauss(30, seed=2), balance=False)
    assert unbalanced["n_synthetic"] == 90 and unbalanced["n_real"] == 30


def test_probe_mlp_variant_also_separates():
    r = probe(_gauss(60, shift=6.0, seed=1), _gauss(60, seed=2), hidden=16)
    assert r["balanced_accuracy"] > 0.95 and r["classifier"] == "mlp16"


def test_probe_rejects_bad_inputs():
    with pytest.raises(ValueError, match="size mismatch"):
        probe(_gauss(20, dim=4), _gauss(20, dim=6))
    with pytest.raises(ValueError, match="5-fold"):
        probe(_gauss(3), _gauss(3))


def test_stratified_folds_partition_and_balance_the_classes():
    y = np.concatenate([np.ones(20), np.zeros(20)])
    folds = probe_mod._stratified_folds(y, 5, seed=0)
    assert len(folds) == 5
    assert sorted(np.concatenate(folds)) == list(range(40))   # a partition
    for f in folds:
        assert len(f) == 8 and y[f].sum() == 4                # 4 of each class


def test_null_control_sits_near_chance_even_on_separable_data():
    """Half-splitting one set has to look like chance whatever the set is --
    that is what makes it a usable floor for the real-vs-synthetic number."""
    x = np.vstack([_gauss(40, shift=6.0, seed=1), _gauss(40, seed=2)])
    assert null_control(x)["balanced_accuracy"] < 0.7
    assert null_control(_gauss(9)) is None      # too small to split k-fold


def test_probe_dirs_and_layer_sweep_wire_through(tmp_path, monkeypatch):
    syn = _write_clips(tmp_path / "syn", 20, 900.0)
    real = _write_clips(tmp_path / "real", 20, 300.0, seed0=100)
    monkeypatch.setattr(probe_mod, "wavlm_embedder",
                        lambda layer=0, device=None: (
                            fake_embed if isinstance(layer, int) else
                            (lambda w, s: np.stack([fake_embed(w, s)] * len(layer)))))

    r = probe_dirs(syn, real, layer=6)
    assert r["layer"] == 6 and r["balanced_accuracy"] > 0.95
    assert r["null_balanced_accuracy"] < 0.9      # real split against itself

    sweep = layer_sweep(syn, real, layers=(4, 6, 8))
    assert sweep["layers"] == [4, 6, 8]
    assert set(sweep["per_layer"]) == {"4", "6", "8"}
    assert sweep["per_layer"]["4"]["null_balanced_accuracy"] is not None
    assert sweep["best_layer"] in (4, 6, 8)
    assert sweep["best_balanced_accuracy"] > 0.95
    json.dumps(sweep)
