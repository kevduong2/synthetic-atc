#!/usr/bin/env python
"""Train the bounded residual CUT translator on isolated channel splits.

Domain A is clean TTS rendered through the train-derived fitted channel,
including default post-effects. The residual is moving after those effects at
inference, so Domain A and real Domain B differ only by the residual gap.
Training uses ``channel_train``; selection uses independent ``channel_val``
real clips and independently rendered validation TTS probes.

S1 compares source with source+identity PatchNCE in paired 1k--2k-update pilots;
S2 gives the winner one 5k-update run. Immutable EMA candidates are scored
serially about every 2k updates. Resume restores models, optimizers, EMA,
selection state, and CPU RNG state; it is coarse, not bitwise-identical on MPS.
Selection applies hard waveform gates, minimizes mean fold-level WavLM KID,
then chooses the earliest candidate within one standard error of the best.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ...config import DistSpec, PostEffectsConfig
from ...tracking import log_audio, start_run
from ..primitives import TARGET_SR, resample
from .backend import CalibratedChannel, StationNoise
from .preset import load_presets
from .residual import (N_FFT, N_FREQ, SPEC_SCALE, ResidualGenerator,
                       ResidualTranslator, default_nce_layers, encoder_end,
                       load_generator, pick_device, save_generator, wav_to_spec)

CROP_FRAMES = 128
MIN_D_FRAMES = 32
NCE_TEMPERATURE = 0.07
SHIFT_FRACTION = 0.125
MASK_FRACTION = 0.125
GAIN_RANGE = 0.1
SELECTION_RULE = "lexicographic_v1.1_fold_paired_tiebreak"


def _read_wav(path: str | Path, sr: int = TARGET_SR) -> np.ndarray:
    wav, file_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = resample(wav, file_sr, sr)
    return wav.astype(np.float32)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    item = Path(path).resolve()
    return {"path": str(item), "sha256": _sha256(item) if item.is_file() else None}


def corpus_rows(corpus: str | Path, split: str | None) -> list[dict[str, Any]]:
    root = Path(corpus).resolve().parent
    rows: list[dict[str, Any]] = []
    for line in Path(corpus).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split and row.get("split") != split:
            continue
        path = root / row["path"]
        if path.exists():
            rows.append({**row, "_path": path})
    if not rows:
        raise ValueError(f"no clips in {corpus} for split {split!r}")
    return rows


def corpus_clips(corpus: str | Path,
                 split: str | None = "channel_train") -> list[Path]:
    """Domain B paths for exactly one channel split."""
    return [row["_path"] for row in corpus_rows(corpus, split)]


def _validation_folds(rows: Sequence[dict[str, Any]],
                      min_clips: int = 3) -> dict[str, list[Path]]:
    """Block-level folds, with sub-`min_clips` blocks pooled into one.

    `kid()` needs at least two clips per reference set (its unbiased MMD has
    no n=1 estimate), and a two-clip fold is all noise anyway; capture blocks
    are often a single stray transmission, so tiny blocks score together.
    """
    if not any(row.get("block_id") is not None for row in rows):
        return {"all": [Path(row["_path"]) for row in rows]}
    folds: dict[str, list[Path]] = {}
    for row in rows:
        folds.setdefault(str(row.get("block_id", "unassigned")), []).append(
            Path(row["_path"]))
    small = [name for name, paths in folds.items() if len(paths) < min_clips]
    if small:
        pooled = [path for name in small for path in folds.pop(name)]
        if len(pooled) >= 2:
            folds["pooled_small_blocks"] = pooled
        elif folds:
            next(iter(folds.values())).extend(pooled)
        else:
            folds["all"] = pooled
    return folds


def dsp_channel(presets: str | Path, noise_bank: str | Path | None
                ) -> CalibratedChannel:
    """Train-derived fitted channel with default post-effects enabled.

    Real Domain B contains squelch, dropout, and codec events, and inference is
    moving the residual after them. Domain A therefore includes the same
    default effects so the residual learns only the remaining A/B gap.
    """
    noise_path = Path(noise_bank).resolve() if noise_bank else None
    noise = StationNoise(noise_path) if noise_path and noise_path.is_dir() else None
    channel = CalibratedChannel(
        load_presets(presets), noise, post_effects=PostEffectsConfig(),
        snr_jitter=DistSpec.parse({"uniform": [-3, 3]}))
    channel._cache_presets_path = Path(presets).resolve()
    channel._cache_noise_stats_path = (
        noise_path / "noise_stats.jsonl" if noise_path else None)
    channel._cache_post_effects_on = True
    return channel


def _cache_expectation(tts_dir: str | Path, channel: CalibratedChannel,
                       renders: int, seed: int,
                       limit: int | None) -> tuple[list[Path], str]:
    sources = sorted(Path(tts_dir).glob("*.wav"))[:limit]
    if not sources:
        raise ValueError(f"no wavs in {tts_dir}")
    presets = getattr(channel, "_cache_presets_path", None)
    noise_stats = getattr(channel, "_cache_noise_stats_path", None)
    payload = {
        "sources": [{"path": str(path.resolve()), "sha256": _sha256(path)}
                    for path in sources],
        "presets_sha256": _sha256(presets) if presets and presets.is_file() else None,
        "noise_stats_sha256": (_sha256(noise_stats)
                               if noise_stats and noise_stats.is_file() else None),
        "render": {"renders": renders, "seed": seed, "limit": limit,
                   "post_effects_on": bool(getattr(
                       channel, "_cache_post_effects_on", False))},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sources, hashlib.sha256(encoded).hexdigest()


def render_domain_a(tts_dir: str | Path, channel: CalibratedChannel,
                    cache: str | Path, renders: int = 4, seed: int = 0,
                    limit: int | None = None) -> list[Path]:
    """Render or fail-closed reuse of one fingerprinted Domain-A cache."""
    out = Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    sources, fingerprint = _cache_expectation(tts_dir, channel, renders, seed, limit)
    expected = [out / f"a_{index:06d}.wav"
                for index in range(len(sources) * renders)]
    meta_path = out / "cache_meta.json"
    existing_wavs = sorted(out.glob("*.wav"))
    if existing_wavs and not meta_path.exists():
        raise RuntimeError(f"stale Domain-A cache {out}: wavs exist without "
                           "cache_meta.json; delete the directory and render again")
    if meta_path.exists():
        try:
            actual = json.loads(meta_path.read_text()).get("fingerprint")
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(f"stale Domain-A cache {out}: unreadable metadata; "
                               "delete the directory and render again") from error
        if actual != fingerprint:
            raise RuntimeError(f"Domain-A cache fingerprint mismatch for {out}; "
                               "delete the directory and render again")
        if existing_wavs != expected:
            raise RuntimeError(f"stale Domain-A cache {out}: cached wav set is "
                               "incomplete or unexpected; delete it and render again")
        return expected
    rng = random.Random(seed)
    index = 0
    for source in sources:
        wav = _read_wav(source)
        for _ in range(renders):
            degraded, _ = channel(wav, TARGET_SR, rng)
            sf.write(expected[index], degraded, TARGET_SR)
            index += 1
    meta_path.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
    return expected


def _cache_identity(cache: Path) -> dict[str, Any]:
    meta = json.loads((cache / "cache_meta.json").read_text())
    return {"path": str(cache.resolve()), "fingerprint": meta["fingerprint"]}


class SpecCrops(Dataset):
    def __init__(self, paths: Sequence[str | Path], crop_frames: int = CROP_FRAMES,
                 sr: int = TARGET_SR):
        self.crop_frames = crop_frames
        self.paths = [Path(path) for path in paths]
        self.specs: list[torch.Tensor] = []
        for path in self.paths:
            wav = _read_wav(path, sr)
            if len(wav) < N_FFT:
                continue
            spec, _ = wav_to_spec(torch.from_numpy(wav))
            if spec.shape[-1] < crop_frames:
                spec = F.pad(spec, (0, crop_frames - spec.shape[-1]))
            self.specs.append(spec)
        if not self.specs:
            raise ValueError("no usable clips (all shorter than one STFT frame)")

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> torch.Tensor:
        spec = self.specs[index]
        start = random.randrange(0, spec.shape[-1] - self.crop_frames + 1)
        return spec[..., start:start + self.crop_frames]


def infinite(loader: DataLoader):
    while True:
        yield from loader


def lsgan_loss(preds: list[torch.Tensor], is_real: bool) -> torch.Tensor:
    total = preds[0].new_zeros(())
    for pred in preds:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        total = total + F.mse_loss(pred, target)
    return total / len(preds)


def zero_padded_shift(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Shift each item in time, filling vacated log-magnitude cells with zero."""
    b, c, f, t = x.shape
    source = torch.arange(t, device=x.device)[None, :] - offsets[:, None]
    valid = (source >= 0) & (source < t)
    index = source.clamp(0, t - 1)
    shifted = torch.gather(x, 3, index[:, None, None, :].expand(b, c, f, t))
    return shifted * valid[:, None, None, :].to(x.dtype)


def physical_gain(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Apply linear-magnitude gain to normalized log1p magnitudes."""
    return torch.log1p(torch.expm1(x * SPEC_SCALE) * gain) / SPEC_SCALE


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Zero-padded time shift, time mask, and physical magnitude gain."""
    b, _, _, t = x.shape
    shift = int(max(1, round(t * SHIFT_FRACTION)))
    offsets = torch.randint(-shift, shift + 1, (b,), device=x.device)
    x = zero_padded_shift(x, offsets)
    width = int(max(1, round(t * MASK_FRACTION)))
    starts = torch.randint(0, max(t - width + 1, 1), (b,), device=x.device)
    grid = torch.arange(t, device=x.device)[None, :]
    mask = ((grid < starts[:, None]) | (grid >= starts[:, None] + width))
    x = x * mask[:, None, None, :].to(x.dtype)
    gain = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * GAIN_RANGE
    return physical_gain(x, gain)


def r1_penalty(discriminator: nn.Module, real: torch.Tensor) -> torch.Tensor:
    real = real.detach().requires_grad_(True)
    total = sum(output.sum() for output in discriminator(real))
    grad, = torch.autograd.grad(total, real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()


class PatchDiscriminator(nn.Module):
    def __init__(self, base: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 4, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1))

    def forward(self, x):
        return self.net(x)


class MultiResDiscriminator(nn.Module):
    def __init__(self, base: int = 48, scales: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.scales = list(scales)
        self.nets = nn.ModuleList([PatchDiscriminator(base) for _ in self.scales])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [net(x if scale == 1 else F.avg_pool2d(x, scale))
                for net, scale in zip(self.nets, self.scales)]


class PatchNCE(nn.Module):
    def __init__(self, dims: Sequence[int], n_patches: int = 256, dim: int = 256,
                 temperature: float = NCE_TEMPERATURE):
        super().__init__()
        self.n_patches = n_patches
        self.temperature = temperature
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(c, dim), nn.ReLU(), nn.Linear(dim, dim))
            for c in dims])

    def forward(self, feats_q: Sequence[torch.Tensor],
                feats_k: Sequence[torch.Tensor]) -> torch.Tensor:
        total = feats_q[0].new_zeros(())
        for mlp, q, k in zip(self.mlps, feats_q, feats_k):
            b, _, h, w = q.shape
            count = min(self.n_patches, h * w)
            index = torch.randperm(h * w, device=q.device)[:count]
            qs = q.flatten(2).transpose(1, 2)[:, index]
            ks = k.flatten(2).transpose(1, 2)[:, index]
            qs = F.normalize(mlp(qs), dim=-1)
            ks = F.normalize(mlp(ks), dim=-1)
            positive = (qs * ks).sum(-1, keepdim=True)
            negative = qs @ ks.transpose(1, 2)
            eye = torch.eye(count, device=q.device, dtype=torch.bool)
            negative = negative.masked_fill(eye[None], -1e4)
            logits = torch.cat([positive, negative], dim=-1) / self.temperature
            target = torch.zeros(b * count, dtype=torch.long, device=q.device)
            total = total + F.cross_entropy(logits.flatten(0, 1), target)
        return total / len(self.mlps)


class Ema:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow, live in zip(self.shadow.parameters(), model.parameters()):
            shadow.mul_(self.decay).add_(live.detach(), alpha=1.0 - self.decay)
        for shadow, live in zip(self.shadow.buffers(), model.buffers()):
            shadow.copy_(live)


# --- held-out evaluation -----------------------------------------------------

def _make_embedder(device: torch.device):
    from ...eval.embed_dist import wavlm_embedder
    return wavlm_embedder(device=str(device))


def _embed_clips(source, embedder, sr: int | None = None):
    from ...eval.embed_dist import embed_clips
    return embed_clips(source, embedder, sr=sr)


def _kid_score(fake: np.ndarray, real: np.ndarray) -> float:
    from ...eval.embed_dist import kid
    return float(kid(fake, real)["kid"])


@torch.no_grad()
def translate_wav(translator: ResidualTranslator, wav: np.ndarray) -> np.ndarray:
    """Translate through exactly the inference object generation uses."""
    return translator(wav, TARGET_SR, alpha=1.0)


@torch.no_grad()
def _residual_saturation(model: ResidualGenerator, wav: np.ndarray,
                         device: torch.device, window: int) -> float:
    spec, _ = wav_to_spec(torch.from_numpy(
        np.asarray(wav, dtype=np.float32)).to(device))
    frames = spec.shape[-1]
    last = max(0, frames - window)
    starts = list(range(0, last + 1, max(1, window // 2)))
    if not starts or starts[-1] != last:
        starts.append(last)
    saturated = 0
    cells = 0
    for start in starts:
        crop = spec[..., start:min(start + window, frames)]
        pad = (-crop.shape[-1]) % 4
        if pad:
            crop = F.pad(crop, (0, pad))
        raw = crop.unsqueeze(0)
        for block in model.blocks:
            raw = block(raw)
        squashed = torch.tanh(raw)
        saturated += int((squashed.abs() > 0.99).sum().item())
        cells += squashed.numel()
    return saturated / max(cells, 1)


def _gate_probe(source: np.ndarray, output: np.ndarray) -> dict[str, Any]:
    reasons: list[str] = []
    if output.size == 0:
        reasons.append("empty")
    if output.shape != source.shape:
        reasons.append("length")
    finite = bool(np.isfinite(output).all())
    if not finite:
        reasons.append("non_finite")
    clip_fraction = (float(np.mean(np.abs(output) >= 0.999))
                     if output.size and finite else math.inf)
    if clip_fraction > 0.01:
        reasons.append("clipping")
    in_rms = float(np.sqrt(np.mean(np.square(source, dtype=np.float64))))
    out_rms = (float(np.sqrt(np.mean(np.square(output, dtype=np.float64))))
               if output.size and finite else 0.0)
    rms_delta_db = 20.0 * math.log10((out_rms + 1e-12) / (in_rms + 1e-12))
    if abs(rms_delta_db) > 6.0:
        reasons.append("rms")
    return {"ok": not reasons, "reasons": reasons,
            "clip_fraction": clip_fraction, "rms_delta_db": rms_delta_db}


class KidTracker:
    """Fixed validation probes and cached per-fold real WavLM embeddings."""

    def __init__(self, real_folds: dict[str, Sequence[Path]],
                 probe_paths: Sequence[Path], device: torch.device,
                 clips: int = 64, seed: int = 1, crop_frames: int = CROP_FRAMES):
        rng = random.Random(seed)
        self.device = device
        self.crop_frames = crop_frames
        self.probe = [Path(path) for path in rng.sample(
            list(probe_paths), min(clips, len(probe_paths)))]
        self.embedder = _make_embedder(device)
        self.real: dict[str, np.ndarray] = {}
        for name, paths in real_folds.items():
            _, embeddings = _embed_clips([str(path) for path in paths], self.embedder)
            self.real[name] = embeddings

    @torch.no_grad()
    def __call__(self, translator: ResidualTranslator
                 ) -> tuple[dict[str, Any], list[np.ndarray]]:
        sources = [_read_wav(path) for path in self.probe]
        rendered = [translate_wav(translator, wav) for wav in sources]
        gates = [_gate_probe(source, output)
                 for source, output in zip(sources[:8], rendered[:8])]
        saturation = float(np.mean([
            _residual_saturation(translator.model, wav, self.device,
                                 self.crop_frames) for wav in sources[:8]
        ])) if sources else 0.0
        embeddable = all(output.size and np.isfinite(output).all()
                         for output in rendered)
        if embeddable:
            _, fake = _embed_clips(rendered, self.embedder, sr=TARGET_SR)
            folds = {name: _kid_score(fake, real)
                     for name, real in self.real.items()}
            values = np.asarray(list(folds.values()), dtype=np.float64)
            mean: float | None = float(values.mean())
            se: float | None = (float(values.std(ddof=1) / math.sqrt(len(values)))
                                if len(values) > 1 else 0.0)
        else:
            # Invalid outputs are already a hard-gate failure.  Do not hand
            # NaNs or empties to WavLM; record the unavailable score and keep
            # training so a later candidate can recover.
            folds = {name: None for name in self.real}
            mean = None
            se = None
        return ({"gates_ok": bool(gates) and all(gate["ok"] for gate in gates),
                 "gates": gates, "residual_sat": saturation,
                 "folds": folds, "kid_mean": mean, "kid_se": se}, rendered[:2])


def build_models(args, device: torch.device
                 ) -> tuple[ResidualGenerator, MultiResDiscriminator, PatchNCE]:
    coarsest = args.crop_frames // max(args.scales)
    if coarsest < MIN_D_FRAMES:
        raise ValueError(
            f"--scales {tuple(args.scales)} leaves the coarsest discriminator "
            f"{coarsest} frames of a {args.crop_frames}-frame crop; it needs "
            f"{MIN_D_FRAMES}. Widen --crop-frames or drop the coarsest scale.")
    if not args.nce_layers:
        args.nce_layers = list(default_nce_layers(args.n_res))
    deepest = max(args.nce_layers)
    if deepest > encoder_end(args.n_res):
        raise ValueError(
            f"--nce-layers reaches block {deepest}, past the encoder's last "
            f"({encoder_end(args.n_res)}) for --n-res {args.n_res}")
    generator = ResidualGenerator(args.base, args.n_res,
                                  args.residual_scale_max).to(device)
    discriminator = MultiResDiscriminator(args.base, args.scales).to(device)
    with torch.no_grad():
        probe = torch.zeros(1, 1, N_FREQ, args.crop_frames, device=device)
        dims = [feat.shape[1]
                for feat in generator(probe, features=args.nce_layers)]
    return generator, discriminator, PatchNCE(dims, args.num_patches).to(device)


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def _safe_log(run, values: dict[str, Any], step: int) -> None:
    try:
        run.log(values, step=step)
    except Exception:
        return


def _select(out: Path, evaluations: list[dict[str, Any]], ema: Ema,
            step: int, crop_frames: int, eval_enabled: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"rule": SELECTION_RULE, "evaluations": evaluations}
    if not eval_enabled:
        save_generator(out / "G_ema.pt", ema.shadow,
                       extra={"step": step, "crop_frames": crop_frames,
                              "selection": None})
        report["selection"] = {"status": "skipped", "reason": "eval_every=0"}
    else:
        eligible = [row for row in evaluations
                    if row["gates_ok"] and row["kid_mean"] is not None]
        if not eligible:
            save_generator(out / "G_ema.pt", ema.shadow,
                           extra={"step": step, "crop_frames": crop_frames,
                                  "selection": None})
            report["selection"] = {"status": "no_eligible_candidate"}
        else:
            best = min(eligible, key=lambda row: row["kid_mean"])

            def within_paired_se(row: dict[str, Any]) -> bool:
                # The marginal kid_se measures between-fold (station) spread,
                # not candidate uncertainty; two candidates scored on the
                # same folds compare by their paired per-fold differences.
                shared = [name for name, value in best["folds"].items()
                          if value is not None
                          and row["folds"].get(name) is not None]
                if not shared:
                    return row is best
                diffs = [row["folds"][name] - best["folds"][name]
                         for name in shared]
                mean = sum(diffs) / len(diffs)
                if len(diffs) < 2:
                    return mean <= 0.0
                var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
                return mean <= math.sqrt(var / len(diffs))

            selected = min((row for row in eligible if within_paired_se(row)),
                           key=lambda row: row["step"])
            candidate = out / "checkpoints" / f"step_{selected['step']:06d}.pt"
            model, _ = load_generator(candidate)
            selection = {"step": selected["step"],
                         "kid_mean": selected["kid_mean"],
                         "kid_se": selected["kid_se"], "rule": SELECTION_RULE}
            for name in ("G_ema.pt", "G_selected.pt"):
                save_generator(out / name, model,
                               extra={"step": selected["step"],
                                      "crop_frames": crop_frames,
                                      "selection": selection})
            report["selection"] = {
                **selection, "status": "selected",
                "sha256": _sha256(out / "G_selected.pt")}
    (out / "validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report["selection"]


def train(args) -> dict[str, Any]:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.a_cache) if args.a_cache else out / "domain_a"
    val_cache = (Path(str(cache) + "_val")
                 if args.a_cache else out / "domain_a_val")
    noise_bank = (Path(args.noise_bank) if args.noise_bank
                  else Path(args.corpus).parent / "noise")
    channel = dsp_channel(args.presets, noise_bank if noise_bank.is_dir() else None)
    a_paths = render_domain_a(args.tts_dir, channel, cache, args.a_renders,
                              args.seed, args.max_tts)
    b_paths = corpus_clips(args.corpus, args.split)

    val_paths: list[Path] = []
    val_rows: list[dict[str, Any]] = []
    if args.eval_every:
        val_paths = render_domain_a(args.val_tts_dir, channel, val_cache,
                                    args.a_renders, args.seed + 1, None)
        val_rows = corpus_rows(args.corpus, args.val_split)

    noise_stats = noise_bank / "noise_stats.jsonl"
    inputs = {
        "corpus": _file_identity(args.corpus),
        "presets": _file_identity(args.presets),
        "noise_stats": _file_identity(noise_stats),
        "domain_a_cache": _cache_identity(cache),
        "domain_a_val_cache": (_cache_identity(val_cache)
                               if args.eval_every else None),
        "args": vars(args),
    }
    (out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    input_hashes = {name: item.get("sha256", item.get("fingerprint"))
                    for name, item in inputs.items() if isinstance(item, dict)}
    def loader(dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          drop_last=len(dataset) >= args.batch_size)

    train_a = SpecCrops(a_paths, args.crop_frames)
    train_b = SpecCrops(b_paths, args.crop_frames)
    it_a = infinite(loader(train_a))
    it_b = infinite(loader(train_b))
    generator, discriminator, nce = build_models(args, device)
    ema = Ema(generator, args.ema_decay)
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(nce.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                             betas=(0.5, 0.999))
    evaluator = (KidTracker(_validation_folds(val_rows), val_paths, device,
                            args.eval_clips, args.seed + 1, args.crop_frames)
                 if args.eval_every else None)

    best: dict[str, Any] = {"kid_mean": None, "step": -1}
    stale = 0
    evaluations: list[dict[str, Any]] = []
    start_step = 1
    if args.resume:
        state_path = out / "state_latest.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"--resume requires {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=False)
        generator.load_state_dict(state["G"])
        discriminator.load_state_dict(state["D"])
        nce.load_state_dict(state["nce"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        ema.shadow.load_state_dict(state["ema"])
        best = state["best"]
        stale = int(state["stale"])
        evaluations = list(state.get("evaluations", []))
        start_step = int(state["step"]) + 1
        _restore_rng(state["rng"])

    log_path = out / "train_log.jsonl"
    history = ([json.loads(line) for line in log_path.read_text().splitlines()
                if line.strip()] if args.resume and log_path.exists() else [])
    log = log_path.open("a" if args.resume else "w")
    run = start_run(project="atcgan-fastcut", name=out.name,
                    config={**vars(args), "input_hashes": input_hashes},
                    tags=("gan-s1",))
    started = time.time()
    step = start_step - 1
    r1 = 0.0

    def save_state(at_step: int, candidate: bool = False) -> None:
        extra = {"step": at_step, "crop_frames": args.crop_frames}
        save_generator(out / "G_latest.pt", generator, extra=extra)
        if candidate:
            save_generator(out / "checkpoints" / f"step_{at_step:06d}.pt",
                           ema.shadow, extra=extra)
        torch.save({"step": at_step, "G": generator.state_dict(),
                    "D": discriminator.state_dict(), "nce": nce.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "ema": ema.shadow.state_dict(), "best": best, "stale": stale,
                    "evaluations": evaluations, "rng": _rng_state()},
                   out / "state_latest.pt")

    def abort_nonfinite(side: str, at_step: int) -> None:
        save_state(at_step)
        raise RuntimeError(f"non-finite {side} loss at step {at_step}")

    try:
        for step in range(start_step, args.steps + 1):
            real_a = next(it_a).to(device)
            real_b = next(it_b).to(device)
            fake_b = generator(real_a)

            opt_d.zero_grad(set_to_none=True)
            aug_real = diff_augment(real_b)
            loss_d = (lsgan_loss(discriminator(aug_real), True)
                      + lsgan_loss(discriminator(diff_augment(fake_b.detach())),
                                   False)) * 0.5
            if not bool(torch.isfinite(loss_d)):
                abort_nonfinite("discriminator", step)
            loss_d.backward()
            if args.r1_gamma > 0 and step % args.r1_every == 0:
                penalty = r1_penalty(discriminator, aug_real)
                if not bool(torch.isfinite(penalty)):
                    abort_nonfinite("discriminator", step)
                (0.5 * args.r1_gamma * penalty * args.r1_every).backward()
                r1 = float(penalty.item())
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            loss_gan = lsgan_loss(discriminator(diff_augment(fake_b)), True)
            with torch.no_grad():
                feats_k = generator(real_a, features=args.nce_layers)
            feats_q = generator(fake_b, features=args.nce_layers)
            loss_nce = nce(feats_q, feats_k)
            loss_idt = loss_nce.new_zeros(())
            if args.nce_mode == "source+identity":
                idt = generator(real_b)
                with torch.no_grad():
                    idt_k = generator(real_b, features=args.nce_layers)
                idt_q = generator(idt, features=args.nce_layers)
                loss_idt = nce(idt_q, idt_k)
            lambda_idt = (args.lambda_nce if args.lambda_idt is None
                          else args.lambda_idt)
            loss_g = (args.lambda_gan * loss_gan + args.lambda_nce * loss_nce
                      + lambda_idt * loss_idt)
            if not bool(torch.isfinite(loss_g)):
                abort_nonfinite("generator", step)
            loss_g.backward()
            opt_g.step()
            ema.update(generator)

            if step % args.log_every == 0 or step == start_step:
                elapsed = max(time.time() - started, 1e-9)
                row = {"step": step, "g": float(loss_g.item()),
                       "gan": float(loss_gan.item()), "nce": float(loss_nce.item()),
                       "idt": float(loss_idt.item()), "d": float(loss_d.item()),
                       "r1": r1,
                       "steps_per_sec": (step - start_step + 1) / elapsed}
                history.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                _safe_log(run, {f"train/{key}": value for key, value in row.items()
                                if key != "step"}, step)

            if step % args.save_every == 0:
                save_state(step, candidate=True)

            if evaluator and step % args.eval_every == 0:
                translator = ResidualTranslator(
                    ema.shadow, device, crop_frames=args.crop_frames)
                result, auditions = evaluator(translator)
                row = {"step": step, "type": "validation", **result}
                evaluations.append(row)
                history.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                metrics = {"val/gates_ok": float(result["gates_ok"]),
                           "val/residual_sat": result["residual_sat"]}
                if result["kid_mean"] is not None:
                    metrics.update({"val/kid_mean": result["kid_mean"],
                                    "val/kid_se": result["kid_se"]})
                    metrics.update({f"val/kid_fold/{name}": value
                                    for name, value in result["folds"].items()})
                _safe_log(run, metrics, step)
                for index, audio in enumerate(auditions):
                    log_audio(run, f"val/audition_{index}", audio,
                              sample_rate=TARGET_SR,
                              caption=f"fixed validation probe {index}", step=step)
                score = result["kid_mean"]
                if score is not None and (best["kid_mean"] is None
                                          or score < best["kid_mean"]):
                    best = {"kid_mean": score, "step": step}
                    stale = 0
                else:
                    stale += 1
                save_state(step)
                if args.patience and stale >= args.patience:
                    break

        save_state(step)
        selection = _select(out, evaluations, ema, step, args.crop_frames,
                            bool(args.eval_every))
        summary = {"out": str(out), "steps": step, "best": best,
                   "selection": selection, "history": history,
                   "seconds": round(time.time() - started, 1)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        log.close()
        try:
            run.finish()
        except Exception:
            pass


TOY = {"base": 8, "n_res": 2, "batch_size": 2, "crop_frames": 64,
       "scales": (1, 2), "num_patches": 32, "a_renders": 1, "max_tts": 24,
       "eval_every": 0, "log_every": 5, "save_every": 25}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="channel-split corpus.jsonl")
    ap.add_argument("--split", default="channel_train")
    ap.add_argument("--val-split", default="channel_val")
    ap.add_argument("--tts-dir", required=True, help="train Domain-A clean wavs")
    ap.add_argument("--val-tts-dir",
                    help="validation-probe clean wavs (required with evaluation)")
    ap.add_argument("--presets", required=True)
    ap.add_argument("--noise-bank", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a-cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--crop-frames", type=int, default=CROP_FRAMES)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--n-res", type=int, default=6)
    ap.add_argument("--scales", type=int, nargs="+", default=(1, 2, 4))
    ap.add_argument("--nce-layers", type=int, nargs="+", default=None)
    ap.add_argument("--num-patches", type=int, default=256)
    ap.add_argument("--nce-mode", choices=("source", "source+identity"),
                    default="source")
    ap.add_argument("--lambda-nce", type=float, default=10.0)
    ap.add_argument("--lambda-idt", type=float, default=None)
    ap.add_argument("--lambda-gan", type=float, default=1.0)
    ap.add_argument("--r1-gamma", type=float, default=1.0)
    ap.add_argument("--r1-every", type=int, default=16)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--residual-scale-max", type=float, default=0.20)
    ap.add_argument("--a-renders", type=int, default=4)
    ap.add_argument("--max-tts", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=2000,
                    help="held-out evaluation interval; 0 disables")
    ap.add_argument("--eval-clips", type=int, default=64)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    if ap.parse_known_args(argv)[0].toy:
        ap.set_defaults(**TOY)
    args = ap.parse_args(argv)
    if args.save_every <= 0 or args.eval_every < 0:
        ap.error("--save-every must be positive and --eval-every non-negative")
    if args.eval_every and not args.val_tts_dir:
        ap.error("--val-tts-dir is required when --eval-every > 0")
    if args.eval_every and args.eval_every % args.save_every:
        ap.error("--eval-every must be a multiple of --save-every")
    return args


def main(argv=None) -> dict[str, Any]:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
