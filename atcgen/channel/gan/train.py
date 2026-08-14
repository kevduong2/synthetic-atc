#!/usr/bin/env python
"""Train the CycleGAN channel model.

Domain A: clean TTS wavs (generate with `scripts/generate_dataset.py --channel clean`).
Domain B: real ATC wavs (export with `atcgen.dataset.real_atc.export_gan_domain_audio`).

Run on the CUDA box:
  uv run python -m atcgen.channel.gan.train --domain-a data/clean/wavs \\
      --domain-b data/real_atc --out runs/cyclegan --device cuda --epochs 40

Checkpoints: G_ab_epNN.pt (the channel generator used by inference) plus a
full state for resuming. Audition samples are written each epoch.
"""

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import Discriminator, Generator, spec_to_wav, wav_to_spec

CROP_FRAMES = 128  # ~1 s at hop 128 / 16 kHz


class SpecFolder(Dataset):
    """Random fixed-size crops of log-magnitude spectrograms from a wav folder."""

    def __init__(self, wav_dir: str, sr: int = 16000):
        self.paths = sorted(Path(wav_dir).glob("*.wav"))
        if not self.paths:
            raise ValueError(f"no wavs in {wav_dir}")
        self.sr = sr

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = sf.read(self.paths[idx], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        spec, _ = wav_to_spec(torch.from_numpy(wav))
        frames = spec.shape[-1]
        if frames < CROP_FRAMES:
            spec = F.pad(spec, (0, CROP_FRAMES - frames))
        else:
            start = random.randrange(0, frames - CROP_FRAMES + 1)
            spec = spec[..., start:start + CROP_FRAMES]
        return spec


def lsgan_loss(pred, is_real: bool):
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ds_a = SpecFolder(args.domain_a)
    ds_b = SpecFolder(args.domain_b)
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    dl_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    G_ab, G_ba = Generator().to(device), Generator().to(device)
    D_a, D_b = Discriminator().to(device), Discriminator().to(device)

    opt_g = torch.optim.Adam(itertools.chain(G_ab.parameters(), G_ba.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(D_a.parameters(), D_b.parameters()), lr=args.lr, betas=(0.5, 0.999))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        for k, m in [("G_ab", G_ab), ("G_ba", G_ba), ("D_a", D_a), ("D_b", D_b)]:
            m.load_state_dict(state[k])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        g_losses, d_losses = [], []
        for real_a, real_b in zip(dl_a, dl_b):
            real_a, real_b = real_a.to(device), real_b.to(device)

            # generators
            opt_g.zero_grad(set_to_none=True)
            fake_b = G_ab(real_a)
            fake_a = G_ba(real_b)
            loss_gan = lsgan_loss(D_b(fake_b), True) + lsgan_loss(D_a(fake_a), True)
            loss_cyc = F.l1_loss(G_ba(fake_b), real_a) + F.l1_loss(G_ab(fake_a), real_b)
            loss_idt = F.l1_loss(G_ab(real_b), real_b) + F.l1_loss(G_ba(real_a), real_a)
            loss_g = loss_gan + args.lambda_cyc * loss_cyc + args.lambda_idt * loss_idt
            loss_g.backward()
            opt_g.step()

            # discriminators
            opt_d.zero_grad(set_to_none=True)
            loss_d = (
                lsgan_loss(D_b(real_b), True) + lsgan_loss(D_b(fake_b.detach()), False)
                + lsgan_loss(D_a(real_a), True) + lsgan_loss(D_a(fake_a.detach()), False)
            ) * 0.5
            loss_d.backward()
            opt_d.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        print(f"epoch {epoch}: G {np.mean(g_losses):.3f}  D {np.mean(d_losses):.3f}")

        torch.save(G_ab.state_dict(), out / "G_ab_latest.pt")
        torch.save({
            "epoch": epoch, "G_ab": G_ab.state_dict(), "G_ba": G_ba.state_dict(),
            "D_a": D_a.state_dict(), "D_b": D_b.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        }, out / "state_latest.pt")

        # audition sample: run one full domain-A file through the channel
        _write_audition(G_ab, ds_a.paths[epoch % len(ds_a.paths)], out / f"audition_ep{epoch:03d}.wav", device)


@torch.no_grad()
def _write_audition(G_ab, wav_path, out_path, device):
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    spec, phase = wav_to_spec(torch.from_numpy(wav).to(device))
    frames = spec.shape[-1]
    pad = (-frames) % 4  # generator downsamples twice
    if pad:
        spec = F.pad(spec, (0, pad))
    fake = G_ab(spec.unsqueeze(0)).squeeze(0)[..., :frames]
    out = spec_to_wav(fake.cpu(), phase.cpu(), length=len(wav))
    sf.write(out_path, out.numpy(), sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain-a", required=True, help="folder of clean TTS wavs")
    ap.add_argument("--domain-b", required=True, help="folder of real ATC wavs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-cyc", type=float, default=10.0)
    ap.add_argument("--lambda-idt", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="path to state_latest.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
