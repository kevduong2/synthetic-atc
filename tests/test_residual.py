"""M2.4: the residual CUT translator — model, training loop, inference, wiring.

Everything is toy-scale and runs on CPU in seconds: a four-channel generator, a
handful of 1.5 s synthetic fixtures, ten training steps.  The point is that the
whole loop *executes* — data, DiffAugment, R1, patchNCE, EMA, checkpointing —
and that the pieces with a stated contract (the residual clamp, phase reuse,
the backend's apply-probability) honour it.  Nothing here downloads a model,
touches the network or computes KID; the real run happens on the 5080 and the
gates are judged there.
"""

import json
import random
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy import signal

from atcgen.channel.gan.model import wav_to_spec as gan_wav_to_spec
from atcgen.channel.learned.backend import CalibratedChannel
from atcgen.channel.learned.preset import BAND_EDGES, Preset, band_centers, write_presets
from atcgen.channel.learned.residual import (ResidualGenerator, ResidualTranslator,
                                             default_nce_layers, encoder_end,
                                             load_generator, save_generator,
                                             spec_to_wav, wav_to_spec)
from atcgen.channel.learned.residual_train import (Ema, MultiResDiscriminator,
                                                   PatchNCE, corpus_clips,
                                                   diff_augment, dsp_channel,
                                                   lsgan_loss, main, r1_penalty,
                                                   render_domain_a)
from atcgen.config import (CalibratedConfig, CalibrationConfig, CodecEffectConfig,
                           DropoutsEffectConfig, PostEffectsConfig, ResidualConfig,
                           SquelchEffectConfig)

SR = 16000
CENTRES = np.asarray(band_centers(BAND_EDGES))


def _speechish(seconds: float, seed: int) -> np.ndarray:
    """A voiced-sounding stand-in: harmonics under a syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = rng.uniform(110, 190)
    wav = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t + rng.uniform(0, 6))
    return (0.3 * wav * envelope).astype(np.float32)


def _radioish(seconds: float, seed: int) -> np.ndarray:
    """The same, band-limited with a noise floor — a stand-in for a real clip."""
    rng = np.random.default_rng(seed + 1000)
    x = _speechish(seconds, seed)
    sos = signal.butter(4, [300.0, 2600.0], btype="bandpass", fs=SR, output="sos")
    y = signal.sosfilt(sos, x) + 0.01 * rng.standard_normal(len(x))
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)


def _preset(clip_id: str, station: str) -> Preset:
    gains = np.where((CENTRES >= 300.0) & (CENTRES <= 2600.0), 0.0, -60.0)
    return Preset(clip_id=clip_id, station=station,
                  band_gains_db=[float(v) for v in gains], drive=1.5,
                  poly=[0.0, 0.0], agc_tau_ms=60.0, agc_strength=0.2,
                  noise_gain=10.0 ** (-20.0 / 20.0), snr_est=20.0, fit_loss=1.0,
                  passband_hz=[300.0, 2600.0])


@pytest.fixture
def corpus(tmp_path):
    """A calibration corpus, a presets file and a folder of clean TTS wavs."""
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = []
    for index in range(6):
        name = f"ALPHA_TOWER_{index:03d}"
        sf.write(clips / f"{name}.wav", _radioish(1.5, index), SR)
        rows.append({"clip_id": name, "path": f"clips/{name}.wav",
                     "station": "ALPHA_TOWER", "duration": 1.5,
                     "split": "train" if index < 5 else "holdout"})
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    presets = write_presets(tmp_path / "presets.jsonl",
                            [_preset(f"p{i}", "ALPHA_TOWER") for i in range(3)])

    tts = tmp_path / "tts"
    tts.mkdir()
    for index in range(4):
        sf.write(tts / f"tts_{index:03d}.wav", _speechish(1.5, 100 + index), SR)
    return {"corpus": corpus_path, "presets": presets, "tts": tts, "root": tmp_path}


def _toy_generator(seed: int = 0) -> ResidualGenerator:
    torch.manual_seed(seed)
    return ResidualGenerator(base=4, n_res=1, residual_scale_max=0.3)


def _no_post_effects() -> PostEffectsConfig:
    return PostEffectsConfig(SquelchEffectConfig(prob=0.0, gated_floor_prob=0.0),
                             DropoutsEffectConfig(prob=0.0),
                             CodecEffectConfig(prob=0.0))


# --- the spectrogram convention ---------------------------------------------

def test_stft_matches_the_gan_module():
    """The convention is restated in residual.py, not imported. Pin them together."""
    wav = torch.from_numpy(_radioish(1.0, 7))
    assert torch.allclose(wav_to_spec(wav)[0], gan_wav_to_spec(wav)[0])


def test_spec_roundtrip_reconstructs():
    wav = torch.from_numpy(_radioish(1.0, 8))
    spec, phase = wav_to_spec(wav)
    out = spec_to_wav(spec, phase, length=len(wav))
    error = torch.mean((out - wav) ** 2) / torch.mean(wav ** 2)
    assert error < 1e-3


# --- the generator -----------------------------------------------------------

def test_generator_emits_a_bounded_non_negative_residual():
    model = _toy_generator()
    x = torch.rand(2, 1, 256, 64) * 0.5
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y - x).abs().max() <= model.residual_scale_max + 1e-5
    assert y.min() >= 0.0                      # log1p magnitudes cannot go negative


def test_default_nce_layers_stay_inside_the_encoder():
    for n_res in (1, 2, 4, 6, 8):
        assert max(default_nce_layers(n_res)) <= encoder_end(n_res)
    assert default_nce_layers(6) == (-1, 0, 1, 2, 5)     # CUT's own choice


def test_feature_extraction_returns_one_tensor_per_layer():
    model = _toy_generator()
    layers = default_nce_layers(1)
    feats = model(torch.rand(2, 1, 256, 64), features=layers)
    assert len(feats) == len(layers)
    assert feats[0].shape[1] == 1                        # -1 is the input itself
    assert all(torch.isfinite(f).all() for f in feats)


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    model = _toy_generator()
    x = torch.rand(1, 1, 256, 64) * 0.5
    save_generator(tmp_path / "G.pt", model, extra={"step": 3})
    reloaded, payload = load_generator(tmp_path / "G.pt")
    assert payload["step"] == 3
    assert payload["stft"]["n_fft"] == 512
    assert torch.allclose(model(x), reloaded(x), atol=1e-6)


def test_config_scale_can_only_tighten_the_clamp(tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())         # trained at 0.3
    assert load_generator(tmp_path / "G.pt", residual_scale_max=0.05
                          )[0].residual_scale_max == 0.05
    assert load_generator(tmp_path / "G.pt", residual_scale_max=9.0
                          )[0].residual_scale_max == 0.3


# --- training pieces ---------------------------------------------------------

def test_diff_augment_preserves_shape_and_flows_gradient():
    x = (torch.rand(4, 1, 64, 32) * 0.5).requires_grad_(True)
    torch.manual_seed(0)
    y = diff_augment(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_diff_augment_never_flips_frequency():
    """A mirrored spectrum is not a signal any receiver produces (00 §4)."""
    x = torch.linspace(0, 1, 64)[None, None, :, None].expand(2, 1, 64, 32).contiguous()
    torch.manual_seed(0)
    y = diff_augment(x)
    # every column is either masked to zero or still ascending in frequency
    columns = y[0, 0].T
    for column in columns:
        assert column.abs().sum() == 0 or torch.all(torch.diff(column) >= -1e-6)


def test_r1_penalty_is_finite_and_non_negative():
    torch.manual_seed(0)
    d = MultiResDiscriminator(base=4, scales=(1, 2))
    penalty = r1_penalty(d, torch.rand(2, 1, 256, 64))
    assert torch.isfinite(penalty) and penalty >= 0


def test_multires_discriminator_scores_every_resolution():
    d = MultiResDiscriminator(base=4, scales=(1, 2, 4))
    outs = d(torch.rand(2, 1, 256, 128))                 # the default crop
    assert len(outs) == 3
    assert all(out.shape[0] == 2 for out in outs)
    assert outs[0].shape[-1] > outs[-1].shape[-1]         # coarser view, fewer patches
    assert torch.isfinite(lsgan_loss(outs, True))


def test_patchnce_prefers_the_identity_mapping():
    """Content kept in place must cost less than content scrambled."""
    torch.manual_seed(0)
    feats = [torch.randn(2, 8, 16, 16)]
    nce = PatchNCE([8], n_patches=32)
    torch.manual_seed(1)
    same = nce(feats, feats)
    torch.manual_seed(1)
    shuffled = nce([torch.randn(2, 8, 16, 16)], feats)
    assert same < shuffled


def test_ema_lags_behind_the_live_weights():
    model = _toy_generator()
    ema = Ema(model, decay=0.9)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update(model)
    live = torch.cat([p.flatten() for p in model.parameters()])
    shadow = torch.cat([p.flatten() for p in ema.shadow.parameters()])
    assert not torch.allclose(live, shadow)
    assert torch.isfinite(shadow).all()


# --- data --------------------------------------------------------------------

def test_corpus_clips_honours_the_split(corpus):
    assert len(corpus_clips(corpus["corpus"], "train")) == 5
    assert len(corpus_clips(corpus["corpus"], "holdout")) == 1
    with pytest.raises(ValueError):
        corpus_clips(corpus["corpus"], "nonesuch")


def test_domain_a_is_rendered_through_the_fitted_chain(corpus, tmp_path):
    channel = dsp_channel(corpus["presets"], None)
    cache = tmp_path / "a"
    paths = render_domain_a(corpus["tts"], channel, cache, renders=2, seed=0)
    assert len(paths) == 8 and all(path.exists() for path in paths)
    rendered, sr = sf.read(paths[0], dtype="float32")
    assert sr == SR and np.isfinite(rendered).all()
    # band-limited by the preset's EQ, unlike the clean source it came from
    clean, _ = sf.read(sorted(corpus["tts"].glob("*.wav"))[0], dtype="float32")
    top = lambda x: np.abs(np.fft.rfft(x))[int(len(x) * 4000 / SR):].mean()
    assert top(rendered) < top(clean)
    # a second call reuses the cache rather than re-rendering
    assert render_domain_a(corpus["tts"], channel, cache, renders=2, seed=0) == paths


# --- the training loop --------------------------------------------------------

@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """Ten toy steps on synthetic fixtures — the whole loop, once."""
    root = tmp_path_factory.mktemp("smoke")
    clips = root / "clips"
    clips.mkdir()
    rows = []
    for index in range(6):
        name = f"ALPHA_TOWER_{index:03d}"
        sf.write(clips / f"{name}.wav", _radioish(1.5, index), SR)
        rows.append({"clip_id": name, "path": f"clips/{name}.wav",
                     "station": "ALPHA_TOWER", "duration": 1.5, "split": "train"})
    (root / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    write_presets(root / "presets.jsonl",
                  [_preset(f"p{i}", "ALPHA_TOWER") for i in range(3)])
    tts = root / "tts"
    tts.mkdir()
    for index in range(4):
        sf.write(tts / f"tts_{index:03d}.wav", _speechish(1.5, 100 + index), SR)

    out = root / "run"
    summary = main([
        "--corpus", str(root / "corpus.jsonl"), "--tts-dir", str(tts),
        "--presets", str(root / "presets.jsonl"), "--out", str(out),
        "--device", "cpu", "--toy", "--steps", "10", "--base", "4", "--n-res", "1",
        "--r1-every", "2", "--ema-decay", "0.9", "--log-every", "1", "--seed", "0",
    ])
    return {"summary": summary, "out": out}


def test_smoke_run_produces_finite_losses(smoke):
    steps = [row for row in smoke["summary"]["history"] if "g" in row]
    assert len(steps) == 10
    for row in steps:
        assert all(np.isfinite(row[key]) for key in ("g", "gan", "nce", "d", "r1"))
    assert any(row["r1"] > 0 for row in steps)             # lazy R1 actually fired
    # ten steps is not convergence; it is enough to prove the losses are moving
    assert steps[0]["g"] != steps[-1]["g"]
    assert steps[0]["d"] != steps[-1]["d"]


def test_smoke_run_writes_the_documented_artifacts(smoke):
    out = smoke["out"]
    for name in ("G_ema.pt", "G_latest.pt", "state_latest.pt", "train_log.jsonl"):
        assert (out / name).exists(), name
    logged = [json.loads(line) for line in
              (out / "train_log.jsonl").read_text().splitlines() if line.strip()]
    assert [row["step"] for row in logged] == list(range(1, 11))
    assert (out / "domain_a").is_dir()


def test_smoke_checkpoint_reloads_and_ema_differs_from_raw(smoke):
    ema, payload = load_generator(smoke["out"] / "G_ema.pt")
    raw, _ = load_generator(smoke["out"] / "G_latest.pt")
    assert payload["kid"] is None                          # KID off in the smoke
    ema_flat = torch.cat([p.flatten() for p in ema.parameters()])
    raw_flat = torch.cat([p.flatten() for p in raw.parameters()])
    assert torch.isfinite(ema_flat).all()
    assert not torch.allclose(ema_flat, raw_flat)


# --- inference ---------------------------------------------------------------

def test_translator_preserves_length_and_dtype(tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu")
    wav = _radioish(1.5, 3)
    out = translator(wav, SR)
    assert out.shape == wav.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0


def test_translator_honours_the_residual_clamp(tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu",
                                         residual_scale_max=0.1)
    spec, _ = wav_to_spec(torch.from_numpy(_radioish(1.5, 4)))
    out = translator.translate_spec(spec)
    assert out.shape == spec.shape
    assert (out - spec).abs().max() <= 0.1 + 1e-5
    # values are log1p(mag) / SPEC_SCALE, so a clamp of 0.1 there is exactly a
    # factor exp(0.1 * 4) either way on (1 + magnitude) -- ~3.5 dB here, ~12 dB
    # at the 0.35 default, and nothing at all can be zeroed out
    ratio = torch.exp((out - spec) * 4.0)
    assert ratio.max() <= np.exp(0.4) + 1e-3
    assert ratio.min() >= np.exp(-0.4) - 1e-3


def test_translator_leaves_speech_recognizably_intact(tmp_path):
    """The clamp exists so the translator cannot destroy content (ROSE guard)."""
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu")
    wav = _radioish(2.0, 5)
    out = translator(wav, SR)
    correlation = float(np.corrcoef(wav, out)[0, 1])
    assert correlation > 0.5


def test_translator_passes_through_clips_shorter_than_a_frame(tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu")
    tiny = np.zeros(100, dtype=np.float32)
    assert translator(tiny, SR).shape == tiny.shape


# --- backend wiring ----------------------------------------------------------

def _channel(presets_path, translator=None, prob=0.0) -> CalibratedChannel:
    from atcgen.channel.learned.preset import load_presets

    return CalibratedChannel(load_presets(presets_path), None,
                             post_effects=_no_post_effects(),
                             residual=translator, residual_prob=prob)


def test_backend_applies_and_records_the_residual(corpus, tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu")
    channel = _channel(corpus["presets"], translator, prob=1.0)
    out, record = channel(_speechish(1.5, 11), SR, random.Random(0))
    assert np.isfinite(out).all() and len(out) > 0
    assert record.residual_applied is True
    assert "residual_translate" in record.applied()
    step = next(s for s in record.steps if s["primitive"] == "residual_translate")
    assert step["residual_applied"] is True


def test_backend_without_a_translator_records_nothing(corpus):
    channel = _channel(corpus["presets"])
    _, record = channel(_speechish(1.5, 12), SR, random.Random(0))
    assert record.residual_applied is False
    assert "residual_translate" not in record.applied()


def test_apply_prob_zero_never_fires(corpus, tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu")
    channel = _channel(corpus["presets"], translator, prob=0.0)
    rng = random.Random(0)
    assert not any(channel(_speechish(0.5, i), SR, rng)[1].residual_applied
                   for i in range(5))


def test_residual_runs_before_the_post_effects(corpus, tmp_path):
    """Training only ever saw the fitted chain's output — order is the contract."""
    save_generator(tmp_path / "G.pt", _toy_generator())
    translator = ResidualTranslator.load(tmp_path / "G.pt", device="cpu")
    from atcgen.channel.learned.preset import load_presets

    channel = CalibratedChannel(
        load_presets(corpus["presets"]), None,
        post_effects=PostEffectsConfig(SquelchEffectConfig(prob=1.0),
                                       DropoutsEffectConfig(prob=1.0),
                                       CodecEffectConfig(prob=0.0)),
        residual=translator, residual_prob=1.0)
    _, record = channel(_speechish(1.5, 13), SR, random.Random(0))
    order = record.applied()
    assert order.index("residual_translate") < order.index("squelch_gate")
    assert order.index("residual_translate") < order.index("dropouts")


def _calibrated_config(corpus, checkpoint) -> CalibratedConfig:
    return CalibratedConfig(
        calibration=CalibrationConfig(presets=str(corpus["presets"]),
                                      noise_bank=str(corpus["root"] / "missing")),
        residual=ResidualConfig(enabled=True, checkpoint=str(checkpoint),
                                apply_prob=1.0, residual_scale_max=0.2),
        post_effects=_no_post_effects())


def test_from_config_loads_an_enabled_checkpoint(corpus, tmp_path):
    save_generator(tmp_path / "G.pt", _toy_generator())
    channel = CalibratedChannel.from_config(_calibrated_config(corpus,
                                                               tmp_path / "G.pt"))
    assert channel.residual is not None
    assert channel.residual.residual_scale_max == 0.2
    assert channel.residual_prob == 1.0
    _, record = channel(_speechish(1.0, 14), SR, random.Random(0))
    assert record.residual_applied is True


def test_from_config_warns_and_falls_back_when_untrained(corpus, tmp_path):
    config = _calibrated_config(corpus, tmp_path / "nope.pt")
    with pytest.warns(UserWarning, match="no checkpoint"):
        channel = CalibratedChannel.from_config(config)
    assert channel.residual is None
    _, record = channel(_speechish(1.0, 15), SR, random.Random(0))
    assert record.residual_applied is False


def test_disabled_residual_never_imports_the_translator(corpus):
    config = _calibrated_config(corpus, "runs/does/not/matter.pt")
    config.residual.enabled = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        channel = CalibratedChannel.from_config(config)
    assert channel.residual is None
