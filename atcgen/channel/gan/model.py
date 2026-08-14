"""CycleGAN over log-magnitude STFT spectrograms.

Domain A: clean TTS speech. Domain B: real ATC radio audio.
G_ab learns the channel (clean -> radio); phase is reused from the source
signal at inference so no vocoder is needed — the generator only reshapes
spectral magnitudes (band-limiting, noise floor, distortion coloration).

Spectrogram convention: n_fft=512, hop=128, 16 kHz; 257 freq bins cropped to
256 (drops Nyquist bin) so shapes stay conv-friendly. Values are
log1p-magnitudes scaled to roughly [-1, 1].
"""

import torch
import torch.nn as nn

N_FFT = 512
HOP = 128
N_FREQ = 256  # 257 minus the Nyquist bin
SPEC_SCALE = 4.0  # log1p magnitudes divided by this to land near [-1, 1]


def wav_to_spec(wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """wav (T,) -> (logmag (1, 256, frames), phase (257, frames))."""
    window = torch.hann_window(N_FFT, device=wav.device)
    st = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)
    mag, phase = st.abs(), st.angle()
    logmag = torch.log1p(mag[:N_FREQ]) / SPEC_SCALE
    return logmag.unsqueeze(0), phase


def spec_to_wav(logmag: torch.Tensor, phase: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """(1, 256, frames) + phase (257, frames) -> wav (T,)."""
    mag = torch.expm1((logmag.squeeze(0) * SPEC_SCALE).clamp(min=0, max=12))
    frames = min(mag.shape[-1], phase.shape[-1])
    full = torch.zeros(N_FREQ + 1, frames, dtype=torch.complex64, device=logmag.device)
    full[:N_FREQ] = mag[..., :frames] * torch.exp(1j * phase[:N_FREQ, :frames])
    window = torch.hann_window(N_FFT, device=logmag.device)
    return torch.istft(full, N_FFT, HOP, window=window, length=length)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet encoder-decoder, 2x downsampling, 6 residual blocks."""

    def __init__(self, base=48, n_res=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 7, padding=3), nn.InstanceNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.InstanceNorm2d(base * 4), nn.ReLU(inplace=True),
            *[ResBlock(base * 4) for _ in range(n_res)],
            nn.ConvTranspose2d(base * 4, base * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 7, padding=3),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """70x70-ish PatchGAN."""

    def __init__(self, base=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 4, 4, stride=1, padding=1), nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)
