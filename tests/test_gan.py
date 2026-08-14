import numpy as np
import torch

from atcgen.channel.gan.model import (
    Discriminator, Generator, spec_to_wav, wav_to_spec,
)


def test_spec_roundtrip_preserves_signal():
    t = torch.arange(16000) / 16000
    wav = (0.5 * torch.sin(2 * np.pi * 700 * t)).float()
    spec, phase = wav_to_spec(wav)
    assert spec.shape[0] == 1 and spec.shape[1] == 256
    out = spec_to_wav(spec, phase, length=len(wav))
    # identity spec -> near-perfect reconstruction (Nyquist bin dropped)
    err = torch.mean((out - wav) ** 2) / torch.mean(wav ** 2)
    assert err < 1e-3


def test_generator_and_discriminator_shapes():
    g, d = Generator(base=16, n_res=2), Discriminator(base=16)
    x = torch.randn(2, 1, 256, 128)
    y = g(x)
    assert y.shape == x.shape
    p = d(y)
    assert p.shape[0] == 2 and p.shape[1] == 1


def test_one_training_step_runs():
    g = Generator(base=8, n_res=1)
    d = Discriminator(base=8)
    opt = torch.optim.Adam(list(g.parameters()) + list(d.parameters()), lr=1e-4)
    a = torch.randn(2, 1, 256, 64)
    fake = g(a)
    loss = torch.mean((d(fake) - 1) ** 2) + torch.nn.functional.l1_loss(fake, a)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
