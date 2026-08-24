#!/usr/bin/env python
"""Train the residual CUT translator (04 §2.4) — FastCUT, clean+DSP -> real.

Domain A is *not* clean TTS: it is clean TTS already pushed through the fitted
DSP sampling model (presets + real noise beds, post-effects off), so the only
thing left for the translator to learn is the gap that fit does not close.
Domain B is the real calibration corpus.  One-sided by construction: nothing in
this project ever needs radio -> clean, so CUT's patchNCE replaces CycleGAN's
cycle loss, which on an information-destroying mapping like a radio channel
invites the steganography failure (1712.02950) and costs twice the compute
(2007.15651).

The small-data stack is the one 00 §4 settles on, all of it live here:
DiffAugment on real and fake patches (time shift, time mask, gain -- *no*
frequency flips: a mirrored spectrum is not a signal any receiver produces),
lazy R1 on the discriminator with gamma swept log-scale, a generator EMA, a
multi-resolution discriminator, batch 8-16, and KID against the real set every
N steps with best-checkpoint selection, because small-data GANs peak early and
then degrade.

    THE 5080 RUN (this box is CPU/MPS; nothing below has been run for real)

      uv run python -m atcgen.channel.learned.residual_train \\
          --corpus runs/calib_v1/corpus.jsonl \\
          --tts-dir runs/p2_smoke/s0_tts_matched \\
          --presets runs/calib_v2/presets.jsonl \\
          --noise-bank runs/calib_v1/noise \\
          --out runs/cut_v1 --device cuda \\
          --steps 60000 --batch-size 12 --r1-gamma 1.0 --ema-decay 0.9995 \\
          --kid-every 1000 --kid-clips 64 --patience 8

    Roughly 0.5-1 day.  Sweep gamma log-scale over four runs (04 §2.4 / 00 §4),
    changing nothing else and keeping --seed fixed:

      for g in 0.01 0.1 1 10; do ... --r1-gamma $g --out runs/cut_v1_g$g ; done

    Reading the `r1` column while a run is going: it is the raw penalty, summed
    over the patch, so it starts in the hundreds or thousands on a fresh
    discriminator and should fall by an order of magnitude within a few hundred
    steps.  Not falling means gamma is too low for this D; `d` collapsing toward
    zero while `gan` climbs means it is too high, or that D has won outright.

    Pick the run with the lowest best-KID, then judge it on the ship gates
    before flipping `residual.enabled` in the config.  Verbatim from 04 §2.4:

      **Gates to ship** (05): channel-probe accuracy improves vs DSP-only;
      ASR round-trip WER delta <= 2% absolute vs its input (ROSE guard);
      Tier 3 WER not worse.

    Failing them, the feature stays flagged off and the failure is documented --
    the DSP backbone is the floor, not this.

Outputs in `--out`: `G_ema.pt` (the checkpoint generation loads: best-by-KID
when KID tracking is on, otherwise the final EMA), `G_latest.pt` (raw weights,
for resuming or comparison), `state_latest.pt` (optimizers, for resuming) and
`train_log.jsonl`, one row per logged step.
"""

import argparse
import copy
import json
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

from ...config import (CodecEffectConfig, DistSpec, DropoutsEffectConfig,
                       PostEffectsConfig, SquelchEffectConfig)
from ..primitives import TARGET_SR, resample
from .backend import CalibratedChannel, StationNoise
from .preset import load_presets
from .residual import (DOWNSAMPLE, N_FFT, N_FREQ, ResidualGenerator,
                       default_nce_layers, encoder_end, pick_device,
                       save_generator, spec_to_wav, wav_to_spec)

CROP_FRAMES = 128          # ~1 s at hop 128 / 16 kHz
MIN_D_FRAMES = 32          # three stride-2 layers then two 4x4 kernels
NCE_TEMPERATURE = 0.07
SHIFT_FRACTION = 0.125     # DiffAugment: time shift up to +/- this much of a crop
MASK_FRACTION = 0.125      # ... and mask up to this much of it
GAIN_RANGE = 0.1           # ... and scale log-magnitudes by 1 +/- this


# --- data --------------------------------------------------------------------

def _read_wav(path: str | Path, sr: int = TARGET_SR) -> np.ndarray:
    wav, file_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = resample(wav, file_sr, sr)
    return wav.astype(np.float32)


def corpus_clips(corpus: str | Path, split: str | None = "train") -> list[Path]:
    """Domain B: the real clips of one split, as absolute paths."""
    root = Path(corpus).parent
    paths = []
    for line in Path(corpus).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split and row.get("split") != split:
            continue
        path = root / row["path"]
        if path.exists():
            paths.append(path)
    if not paths:
        raise ValueError(f"no clips in {corpus} for split {split!r}")
    return paths


def dsp_channel(presets: str | Path, noise_bank: str | Path | None
                ) -> CalibratedChannel:
    """The fitted sampling model with its post-effects turned off.

    Squelch, dropouts and the codec are events the *shared primitives* stamp on
    afterwards, at probabilities the config sets.  If they were in domain A the
    translator would be asked to learn a distribution over them too, and it
    would learn it from a few dozen real clips; leaving them out keeps the
    residual a residual.  PTT truncation stays: the backend applies it before
    anything else and it only trims the ends, which random crops mostly miss,
    and real clips are truncated the same way.
    """
    off = PostEffectsConfig(SquelchEffectConfig(prob=0.0, gated_floor_prob=0.0),
                            DropoutsEffectConfig(prob=0.0),
                            CodecEffectConfig(prob=0.0))
    noise = (StationNoise(noise_bank)
             if noise_bank and Path(noise_bank).is_dir() else None)
    return CalibratedChannel(load_presets(presets), noise, post_effects=off,
                             snr_jitter=DistSpec.parse({"uniform": [-3, 3]}))


def render_domain_a(tts_dir: str | Path, channel: CalibratedChannel,
                    cache: str | Path, renders: int = 4, seed: int = 0,
                    limit: int | None = None) -> list[Path]:
    """Precompute domain A: every TTS wav through `renders` sampled presets.

    Rendering on the fly would give unlimited diversity but costs an FIR
    convolution per clip per step, which dominates a CPU step and starves a GPU
    one.  A cache of `renders` x |TTS| clips is drawn once, reused across the
    run and reused again across a gamma sweep; raise `--a-renders` for more
    channel diversity rather than lowering it for speed.
    """
    out = Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path(tts_dir).glob("*.wav"))[:limit]
    if not sources:
        raise ValueError(f"no wavs in {tts_dir}")
    expected = [out / f"a_{index:06d}.wav"
                for index in range(len(sources) * renders)]
    if all(path.exists() for path in expected):
        return expected
    rng = random.Random(seed)
    index = 0
    for source in sources:
        wav = _read_wav(source)
        for _ in range(renders):
            path = expected[index]
            index += 1
            if path.exists():
                continue
            degraded, _ = channel(wav, TARGET_SR, rng)
            sf.write(path, degraded, TARGET_SR)
    return expected


class SpecCrops(Dataset):
    """Random fixed-width crops of log-magnitude spectrograms from a wav list.

    Spectrograms are computed once and held in memory -- re-running the STFT
    every epoch would be the single largest cost in a CPU step.  A 4 s clip is
    ~0.5 MB of float32 here, so the default domain A (4 renders x ~200 TTS
    clips) costs a few hundred MB; `--a-renders` is the knob if that matters.
    """

    def __init__(self, paths: Sequence[str | Path], crop_frames: int = CROP_FRAMES,
                 sr: int = TARGET_SR):
        self.crop_frames = crop_frames
        self.specs: list[torch.Tensor] = []
        for path in paths:
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


# --- losses ------------------------------------------------------------------

def lsgan_loss(preds: list[torch.Tensor], is_real: bool) -> torch.Tensor:
    """Least-squares GAN loss, averaged over the discriminator's resolutions."""
    total = preds[0].new_zeros(())
    for pred in preds:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        total = total + F.mse_loss(pred, target)
    return total / len(preds)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """DiffAugment for spectrogram patches: time shift, time mask, gain.

    Differentiable and applied to real *and* fake in both the D and G steps,
    which is what stops the discriminator memorizing ~80 real clips
    (2006.10738).  Every transform is one a real transmission could differ by:
    the utterance starts a little later, a chunk of it is lost, the receiver's
    level sits a little higher.  Frequency flips are deliberately absent -- a
    mirrored spectrum has formants running the wrong way and teaching D to
    accept one teaches it nothing about radios.
    """
    b, c, f, t = x.shape
    device = x.device

    shift = int(max(1, round(t * SHIFT_FRACTION)))
    offsets = torch.randint(-shift, shift + 1, (b,), device=device)
    index = (torch.arange(t, device=device)[None, :] - offsets[:, None]) % t
    x = torch.gather(x, 3, index[:, None, None, :].expand(b, c, f, t))

    width = int(max(1, round(t * MASK_FRACTION)))
    starts = torch.randint(0, max(t - width, 1), (b,), device=device)
    grid = torch.arange(t, device=device)[None, :]
    mask = ((grid < starts[:, None]) | (grid >= (starts + width)[:, None])).float()
    x = x * mask[:, None, None, :]

    gain = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * GAIN_RANGE
    return x * gain


def r1_penalty(discriminator: nn.Module, real: torch.Tensor) -> torch.Tensor:
    """E[||grad_x D(x)||^2] on real inputs (1801.04406).

    Computed on the augmented tensor D actually sees, summed over the
    discriminator's resolutions.
    """
    real = real.detach().requires_grad_(True)
    outputs = discriminator(real)
    total = sum(output.sum() for output in outputs)
    grad, = torch.autograd.grad(total, real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()


class PatchDiscriminator(nn.Module):
    """70x70-ish PatchGAN over a log-magnitude patch."""

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
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class MultiResDiscriminator(nn.Module):
    """One PatchGAN per spectrogram resolution (2103.05236, adapted).

    The published multi-resolution STFT discriminator re-analyses the *waveform*
    at several window sizes.  This model never produces a waveform during
    training -- it edits magnitudes and the phase is the source's -- so the
    resolutions are made by average-pooling the log-magnitude patch instead:
    scale 2 is the view a window twice as long and a hop twice as coarse would
    give, up to the smoothing the pooling adds.  The point survives the
    adaptation: a fine-grained D polices harmonic texture, a coarse one polices
    the band shape and the noise floor, and a generator that games one is caught
    by the other.
    """

    def __init__(self, base: int = 48, scales: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.scales = list(scales)
        self.nets = nn.ModuleList([PatchDiscriminator(base) for _ in self.scales])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [net(x if scale == 1 else F.avg_pool2d(x, scale))
                for net, scale in zip(self.nets, self.scales)]


class PatchNCE(nn.Module):
    """CUT's patchwise contrastive loss over the generator's encoder features.

    For each feature layer the same spatial locations are read out of the input's
    features and the output's; a location's own input feature is its positive and
    the other sampled locations in the same clip are its negatives.  That is what
    keeps content in place without a cycle: the translated patch at 1.2 kHz,
    frame 40 must still look more like *that* patch than like any other in the
    clip, however the channel recoloured it.
    """

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
    """Generator EMA (1806.04498) — the weights generation actually loads."""

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


# --- KID tracking ------------------------------------------------------------

class KidTracker:
    """KID of translated domain-A clips against the real set (1801.01401).

    KID rather than FID because n ~ 100 (FID's small-n bias would swamp the
    signal), on WavLM embeddings because its layers encode channel information
    (2501.05310).  `transformers` is imported here and nowhere else, so training
    without `--kid-every` -- and the whole test suite -- never touches it.
    """

    def __init__(self, real_paths: Sequence[Path], probe_paths: Sequence[Path],
                 device: torch.device, clips: int = 64, seed: int = 0):
        from ...eval.embed_dist import embed_clips, wavlm_embedder

        rng = random.Random(seed)
        self.embedder = wavlm_embedder(device=str(device))
        self.probe = [Path(p) for p in rng.sample(
            list(probe_paths), min(clips, len(probe_paths)))]
        _, self.real = embed_clips(
            [str(p) for p in rng.sample(list(real_paths),
                                        min(clips, len(real_paths)))],
            self.embedder)

    @torch.no_grad()
    def __call__(self, model: ResidualGenerator, device: torch.device) -> float:
        from ...eval.embed_dist import embed_clips, kid

        rendered = [translate_wav(model, _read_wav(path), device)
                    for path in self.probe]
        _, fake = embed_clips(rendered, self.embedder, sr=TARGET_SR)
        return float(kid(fake, self.real)["kid"])


@torch.no_grad()
def translate_wav(model: ResidualGenerator, wav: np.ndarray,
                  device: torch.device) -> np.ndarray:
    """One wav through the generator, phase reused — the inference path."""
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(device)
    spec, phase = wav_to_spec(tensor)
    frames = spec.shape[-1]
    pad = (-frames) % DOWNSAMPLE
    if pad:
        spec = F.pad(spec, (0, pad))
    out = model(spec.unsqueeze(0)).squeeze(0)[..., :frames]
    return spec_to_wav(out, phase, length=len(wav)).cpu().numpy().astype(np.float32)


# --- the loop ----------------------------------------------------------------

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


def train(args) -> dict[str, Any]:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.a_cache or out / "domain_a")
    channel = dsp_channel(args.presets, args.noise_bank or
                          Path(args.corpus).parent / "noise")
    a_paths = render_domain_a(args.tts_dir, channel, cache, args.a_renders,
                              args.seed, args.max_tts)
    b_paths = corpus_clips(args.corpus, args.split)
    print(f"domain A: {len(a_paths)} rendered clips in {cache}")
    print(f"domain B: {len(b_paths)} real clips ({args.split} split)")

    def loader(dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          drop_last=len(dataset) >= args.batch_size)

    it_a = infinite(loader(SpecCrops(a_paths, args.crop_frames)))
    it_b = infinite(loader(SpecCrops(b_paths, args.crop_frames)))

    generator, discriminator, nce = build_models(args, device)
    ema = Ema(generator, args.ema_decay)
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(nce.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                             betas=(0.5, 0.999))

    tracker = None
    if args.kid_every:
        tracker = KidTracker(b_paths, a_paths, device, args.kid_clips, args.seed)

    log_path = out / "train_log.jsonl"
    log = log_path.open("a")
    best: dict[str, Any] = {"kid": None, "step": -1}
    stale = 0
    step = 0
    started = time.time()
    history: list[dict[str, Any]] = []
    r1 = 0.0                   # lazy: the logged value is the most recent one

    for step in range(1, args.steps + 1):
        real_a = next(it_a).to(device)
        real_b = next(it_b).to(device)

        fake_b = generator(real_a)

        # -- discriminator -------------------------------------------------- #
        opt_d.zero_grad(set_to_none=True)
        aug_real = diff_augment(real_b)
        loss_d = (lsgan_loss(discriminator(aug_real), True)
                  + lsgan_loss(discriminator(diff_augment(fake_b.detach())),
                               False)) * 0.5
        loss_d.backward()
        if args.r1_gamma > 0 and step % args.r1_every == 0:
            # lazy regularization: the penalty every `r1_every` steps, scaled by
            # the same factor, costs a fraction of a per-step penalty and is
            # indistinguishable in effect (1912.04958)
            penalty = r1_penalty(discriminator, aug_real)
            (0.5 * args.r1_gamma * penalty * args.r1_every).backward()
            r1 = float(penalty.item())
        opt_d.step()

        # -- generator ------------------------------------------------------ #
        # D was just stepped, so the generator is judged by the updated critic;
        # `fake_b`'s own graph is untouched by that and is reused here
        opt_g.zero_grad(set_to_none=True)
        loss_gan = lsgan_loss(discriminator(diff_augment(fake_b)), True)
        feats_k = generator(real_a, features=args.nce_layers)
        feats_q = generator(fake_b, features=args.nce_layers)
        loss_nce = nce(feats_q, feats_k)
        loss_g = args.lambda_gan * loss_gan + args.lambda_nce * loss_nce
        loss_g.backward()
        opt_g.step()
        ema.update(generator)

        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "g": round(float(loss_g.item()), 4),
                   "gan": round(float(loss_gan.item()), 4),
                   "nce": round(float(loss_nce.item()), 4),
                   "d": round(float(loss_d.item()), 4), "r1": round(r1, 4),
                   "sec": round(time.time() - started, 1)}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"step {step:6d}  G {row['g']:7.3f}  (gan {row['gan']:6.3f} "
                  f"nce {row['nce']:6.3f})  D {row['d']:6.3f}  r1 {row['r1']:8.4f}")

        if tracker and step % args.kid_every == 0:
            score = tracker(ema.shadow, device)
            improved = best["kid"] is None or score < best["kid"]
            row = {"step": step, "kid": round(score, 6), "best": improved}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"step {step:6d}  KID {score:.6f}" + ("  (best)" if improved else ""))
            if improved:
                best = {"kid": score, "step": step}
                stale = 0
                save_generator(out / "G_ema.pt", ema.shadow,
                               extra={"step": step, "kid": score})
            else:
                stale += 1
                if args.patience and stale >= args.patience:
                    print(f"early stop: {stale} evaluations without improvement")
                    break

        if step % args.save_every == 0 or step == args.steps:
            save_generator(out / "G_latest.pt", generator, extra={"step": step})
            torch.save({"step": step, "G": generator.state_dict(),
                        "D": discriminator.state_dict(), "nce": nce.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()},
                       out / "state_latest.pt")

    save_generator(out / "G_latest.pt", generator, extra={"step": step})
    if best["step"] < 0:
        # no KID evaluation ever selected one, so the final EMA is what ships
        save_generator(out / "G_ema.pt", ema.shadow,
                       extra={"step": step, "kid": None})
    log.close()
    summary = {"out": str(out), "steps": step, "best": best,
               "history": history, "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


TOY = {"base": 8, "n_res": 2, "batch_size": 2, "crop_frames": 64, "scales": (1, 2),
       "num_patches": 32, "a_renders": 1, "max_tts": 24, "kid_every": 0,
       "log_every": 5, "save_every": 25}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Train the residual CUT translator (04 §2.4).")
    ap.add_argument("--corpus", required=True, help="corpus.jsonl (domain B)")
    ap.add_argument("--tts-dir", required=True, help="clean TTS wavs (domain A source)")
    ap.add_argument("--presets", required=True, help="presets.jsonl from channel_fit")
    ap.add_argument("--noise-bank", default=None,
                    help="noise bank dir (default: <corpus dir>/noise)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--a-cache", default=None,
                    help="where rendered domain A lives (default: <out>/domain_a)")
    ap.add_argument("--split", default="train", help="corpus split for domain B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--toy", action="store_true",
                    help="tiny model and batch, for smoke runs on CPU")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--crop-frames", type=int, default=CROP_FRAMES)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--n-res", type=int, default=6)
    ap.add_argument("--scales", type=int, nargs="+", default=(1, 2, 4),
                    help="discriminator resolutions (pooling factors)")
    ap.add_argument("--nce-layers", type=int, nargs="+", default=None,
                    help="encoder block indices (default: derived from --n-res)")
    ap.add_argument("--num-patches", type=int, default=256)
    ap.add_argument("--lambda-nce", type=float, default=10.0, help="FastCUT: 10")
    ap.add_argument("--lambda-gan", type=float, default=1.0)
    ap.add_argument("--r1-gamma", type=float, default=1.0,
                    help="sweep log-scale: 0.01 0.1 1 10")
    ap.add_argument("--r1-every", type=int, default=16, help="lazy R1 interval")
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--residual-scale-max", type=float, default=0.35)
    ap.add_argument("--a-renders", type=int, default=4,
                    help="DSP renders per clean TTS clip")
    ap.add_argument("--max-tts", type=int, default=None,
                    help="cap the number of TTS source clips")
    ap.add_argument("--kid-every", type=int, default=1000, help="0 disables KID")
    ap.add_argument("--kid-clips", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8,
                    help="KID evaluations without improvement before stopping")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    # --toy lowers the defaults; anything named explicitly on the command line
    # still wins, so a smoke run can be steered without editing TOY
    if ap.parse_known_args(argv)[0].toy:
        ap.set_defaults(**TOY)
    return ap.parse_args(argv)


def main(argv=None) -> dict[str, Any]:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
