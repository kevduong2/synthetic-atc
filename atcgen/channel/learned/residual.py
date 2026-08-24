"""Residual CUT translator: generator, and the inference path Mode 2 uses (04 §2.4).

Stage 2 of the calibrated backend.  The fitted DSP chain (`preset.py`) already
reproduces what a real receiver does to speech as a *stationary* process; this
translator learns only what that fit misses — the residual gap between
`clean TTS -> fitted chain` and a real clip.  It is trained by
`residual_train.py` (FastCUT) and applied here at probability `residual.apply_prob`,
so the pure-DSP path stays represented in any generated corpus.

Two properties keep the thing from becoming another black-box channel:

*Magnitudes only, phase reused.*  The generator reshapes a log-magnitude STFT
and the source signal's own phase is put back for the inverse transform — no
vocoder.  Clean-trained vocoders fail on noisy spectrograms (2305.12460), which
is the same reason the v1 CycleGAN in `channel/gan/` works this way.

*A bounded residual, not a free mapping.*  The network's output is
`tanh`-squashed to `+/- residual_scale_max` in normalized log-magnitude units and
*added* to the input.  At the default 0.35 that is at most ~12 dB of change in
any one time-frequency cell, so the translator can colour, add a floor and smear,
but cannot delete the speech it was handed.  The clamp is the mechanical half of
the ROSE guard (2312.06118): a learned component that degrades ASR is worse than
no learned component, so it gets a leash and an ASR gate.

The STFT convention (n_fft 512, hop 128, 256 bins, log1p/4 scaling) is the same
one `channel/gan/model.py` uses, and `tests/test_residual.py` asserts the two
agree numerically.  It is restated here rather than imported because the v1
CycleGAN is a comparison baseline scheduled for retirement (04 §1) and Mode 2
must not go with it.
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..primitives import TARGET_SR, resample

N_FFT = 512
HOP = 128
N_FREQ = 256          # 257 minus the Nyquist bin, so shapes stay conv-friendly
SPEC_SCALE = 4.0      # log1p magnitudes divided by this to land near [-1, 1]
DOWNSAMPLE = 4        # the generator halves the grid twice


def encoder_end(n_res: int) -> int:
    """Index of the generator's last encoder block: stem, 2 downs, `n_res` res."""
    return 2 + n_res


def default_nce_layers(n_res: int) -> tuple[int, ...]:
    """Input, stem, both downsamplings, a mid residual block.

    Encoder-only by construction — patchNCE compares *content* representations,
    and past the bottleneck the features are already committed to the output.
    Derived from `n_res` rather than hard-coded so a toy model does not silently
    end up sampling its decoder.  At the default n_res=6 this is CUT's own
    (-1, 0, 1, 2, 5); -1 is the input itself, which CUT counts as a layer.
    """
    return (-1, 0, 1, 2, 2 + max(1, (n_res + 1) // 2))


def wav_to_spec(wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """wav (T,) -> (logmag (1, 256, frames), phase (257, frames))."""
    window = torch.hann_window(N_FFT, device=wav.device)
    st = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)
    mag, phase = st.abs(), st.angle()
    logmag = torch.log1p(mag[:N_FREQ]) / SPEC_SCALE
    return logmag.unsqueeze(0), phase


def spec_to_wav(logmag: torch.Tensor, phase: torch.Tensor,
                length: int | None = None) -> torch.Tensor:
    """(1, 256, frames) + phase (257, frames) -> wav (T,), reusing the phase."""
    mag = torch.expm1((logmag.squeeze(0) * SPEC_SCALE).clamp(min=0, max=12))
    frames = min(mag.shape[-1], phase.shape[-1])
    full = torch.zeros(N_FREQ + 1, frames, dtype=torch.complex64, device=logmag.device)
    full[:N_FREQ] = mag[..., :frames] * torch.exp(1j * phase[:N_FREQ, :frames])
    window = torch.hann_window(N_FFT, device=logmag.device)
    return torch.istft(full, N_FFT, HOP, window=window, length=length)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualGenerator(nn.Module):
    """ResNet encoder-decoder emitting a bounded additive log-magnitude residual.

    Held as a `ModuleList` rather than a `Sequential` because patchNCE needs the
    encoder's intermediate features: `forward(x, features=[...])` returns the
    activations at those block indices instead of an output, stopping as soon as
    the deepest requested one is reached.  Index -1 means the input itself, which
    CUT counts as a feature layer (it anchors the identity of the content).

    Block indices, for `--nce-layers`: 0 stem conv, 1-2 the two downsamplings,
    3..3+n_res-1 the residual stack, then two upsamplings and the output conv.
    """

    def __init__(self, base: int = 48, n_res: int = 6,
                 residual_scale_max: float = 0.35):
        super().__init__()
        self.base, self.n_res = base, n_res
        self.residual_scale_max = float(residual_scale_max)
        blocks: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(1, base, 7, padding=3),
                          nn.InstanceNorm2d(base), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
                          nn.InstanceNorm2d(base * 2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
                          nn.InstanceNorm2d(base * 4), nn.ReLU(inplace=True)),
        ]
        blocks += [ResBlock(base * 4) for _ in range(n_res)]
        blocks += [
            nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 3, stride=2,
                                             padding=1, output_padding=1),
                          nn.InstanceNorm2d(base * 2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(base * 2, base, 3, stride=2,
                                             padding=1, output_padding=1),
                          nn.InstanceNorm2d(base), nn.ReLU(inplace=True)),
            nn.Conv2d(base, 1, 7, padding=3),
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor,
                features: Sequence[int] | None = None
                ) -> torch.Tensor | list[torch.Tensor]:
        if features is not None:
            wanted = sorted(features)
            deepest = wanted[-1]
            out = [x] if wanted[0] == -1 else []
            h = x
            for index, block in enumerate(self.blocks):
                h = block(h)
                if index in wanted:
                    out.append(h)
                if index >= deepest:
                    break
            return out
        h = x
        for block in self.blocks:
            h = block(h)
        return self.apply_residual(x, h)

    def apply_residual(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """`x` plus the squashed residual, floored at 0 (log1p magnitudes are >= 0)."""
        return (x + self.residual_scale_max * torch.tanh(raw)).clamp(min=0.0)


def save_generator(path: str | Path, model: ResidualGenerator,
                   extra: dict[str, Any] | None = None) -> Path:
    """Checkpoint the architecture alongside the weights, so `load` needs no flags."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "residual_cut_g",
        "arch": {"base": model.base, "n_res": model.n_res,
                 "residual_scale_max": model.residual_scale_max},
        "stft": {"n_fft": N_FFT, "hop": HOP, "n_freq": N_FREQ,
                 "spec_scale": SPEC_SCALE},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    payload.update(extra or {})
    torch.save(payload, out)
    return out


def load_generator(checkpoint: str | Path, device: torch.device | str = "cpu",
                   residual_scale_max: float | None = None
                   ) -> tuple[ResidualGenerator, dict[str, Any]]:
    """Rebuild a generator from a `save_generator` payload."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{checkpoint} is not a residual-CUT checkpoint")
    arch = dict(payload.get("arch") or {})
    if residual_scale_max is not None:
        # the config's clamp wins over the trained one: it can only tighten a
        # leash at generation time, never loosen what training was bounded by
        arch["residual_scale_max"] = min(float(residual_scale_max),
                                         float(arch.get("residual_scale_max", 1e9)))
    model = ResidualGenerator(**arch)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResidualTranslator:
    """A trained residual generator, applied to one wav at a time.

    Callable with the same `(wav, sr, rng)` shape as the DSP primitives, so
    `CalibratedChannel` can drop it into its chain.  `rng` is accepted and
    ignored — the mapping is deterministic; the randomness is *whether* it is
    applied, which the backend draws.
    """

    def __init__(self, model: ResidualGenerator, device: torch.device | str = "cpu",
                 target_sr: int = TARGET_SR):
        self.model = model
        self.device = torch.device(device)
        self.target_sr = target_sr

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | None = None,
             residual_scale_max: float | None = None,
             target_sr: int = TARGET_SR) -> "ResidualTranslator":
        dev = pick_device(device)
        model, _ = load_generator(checkpoint, dev, residual_scale_max)
        return cls(model, dev, target_sr)

    @property
    def residual_scale_max(self) -> float:
        return self.model.residual_scale_max

    @torch.no_grad()
    def translate_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """(1, 256, frames) log-magnitudes in, the same shape out.

        Frames are padded up to a multiple of the generator's downsampling and
        cropped back, so any clip length round-trips exactly.
        """
        frames = spec.shape[-1]
        pad = (-frames) % DOWNSAMPLE
        if pad:
            spec = F.pad(spec, (0, pad))
        out = self.model(spec.unsqueeze(0).to(self.device)).squeeze(0)
        return out[..., :frames]

    @torch.no_grad()
    def __call__(self, wav: np.ndarray, sr: int | None = None,
                 rng: Any = None) -> np.ndarray:
        """Wav in (float32 mono), the translated wav out at the same length."""
        x = np.asarray(wav, dtype=np.float32).reshape(-1)
        if len(x) < N_FFT:
            return x
        if sr is not None and sr != self.target_sr:
            x = resample(x, sr, self.target_sr)
        t = torch.from_numpy(x).to(self.device)
        spec, phase = wav_to_spec(t)
        out = spec_to_wav(self.translate_spec(spec), phase, length=len(x))
        y = out.detach().cpu().numpy().astype(np.float32)
        peak = float(np.abs(y).max())
        if peak > 1.0:
            y = (y / peak * 0.98).astype(np.float32)
        return y


def load_translator(checkpoint: str | Path, residual_scale_max: float | None = None,
                    device: str | None = None) -> ResidualTranslator | None:
    """`ResidualTranslator.load`, or None when the checkpoint is not there yet.

    `residual.enabled` is a config statement of intent; a missing checkpoint is
    the normal state before M2.4 has been trained, and generation falls back to
    the pure-DSP path rather than failing.
    """
    if not Path(checkpoint).exists():
        return None
    return ResidualTranslator.load(checkpoint, device, residual_scale_max)
