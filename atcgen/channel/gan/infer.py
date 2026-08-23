"""Apply a trained CycleGAN channel generator to clean TTS audio."""

import random

import numpy as np
import torch
import torch.nn.functional as F

from ..primitives import TARGET_SR, resample
from .model import Generator, spec_to_wav, wav_to_spec


class GanChannel:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"))
        self.G_ab = Generator().to(self.device).eval()
        self.G_ab.load_state_dict(torch.load(checkpoint, map_location=self.device))

    @torch.no_grad()
    def __call__(self, wav: np.ndarray, sr: int, rng: random.Random | None = None) -> np.ndarray:
        """Clean wav (float32 mono at sr) -> radio-fied 16 kHz wav."""
        x = wav.astype(np.float32)
        if sr != TARGET_SR:
            x = resample(x, sr, TARGET_SR)
        t = torch.from_numpy(x).to(self.device)
        spec, phase = wav_to_spec(t)
        frames = spec.shape[-1]
        pad = (-frames) % 4
        if pad:
            spec = F.pad(spec, (0, pad))
        fake = self.G_ab(spec.unsqueeze(0)).squeeze(0)[..., :frames]
        out = spec_to_wav(fake, phase, length=len(x)).cpu().numpy()
        peak = np.abs(out).max()
        if peak > 1.0:
            out = out / peak * 0.98
        return out.astype(np.float32)
