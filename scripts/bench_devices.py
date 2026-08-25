#!/usr/bin/env python
"""Idle-device timing benchmark for GAN, WavLM, TTS, DSP, and Whisper SFT."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from atcgen.channel.learned.residual import N_FREQ, ResidualGenerator, pick_device
from atcgen.channel.learned.residual_train import (
    MultiResDiscriminator,
    diff_augment,
    lsgan_loss,
    r1_penalty,
)
from atcgen.channel.primitives import TARGET_SR
from atcgen.tracking import start_run


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(action: Callable[[], None], *, warmup: int, measured: int,
           repeats: int, device: torch.device) -> dict:
    for _ in range(warmup):
        action()
    _sync(device)
    seconds = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        for _ in range(measured):
            action()
        _sync(device)
        seconds.append(time.perf_counter() - started)
    rates = [value / measured for value in seconds]
    return {
        "warmup": warmup,
        "measured": measured,
        "repeats": repeats,
        "seconds": {
            "median": statistics.median(seconds),
            "min": min(seconds),
            "max": max(seconds),
            "all": seconds,
        },
        "seconds_per_unit": {
            "median": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
        },
    }


def _memory_begin(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_peak(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def benchmark_gan(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    generator = ResidualGenerator(args.gan_base, args.gan_n_res).to(device)
    discriminator = MultiResDiscriminator(args.gan_base, args.gan_scales).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def ordinary_step() -> None:
        real_a = torch.rand(args.gan_batch, 1, N_FREQ, args.gan_crop, device=device)
        real_b = torch.rand_like(real_a)
        fake_b = generator(real_a)
        opt_d.zero_grad(set_to_none=True)
        loss_d = 0.5 * (
            lsgan_loss(discriminator(diff_augment(real_b)), True)
            + lsgan_loss(discriminator(diff_augment(fake_b.detach())), False))
        loss_d.backward()
        opt_d.step()
        opt_g.zero_grad(set_to_none=True)
        loss_g = lsgan_loss(discriminator(diff_augment(fake_b)), True)
        loss_g.backward()
        opt_g.step()

    def r1_step() -> None:
        discriminator.zero_grad(set_to_none=True)
        real = torch.rand(args.gan_batch, 1, N_FREQ, args.gan_crop, device=device)
        penalty = r1_penalty(discriminator, diff_augment(real))
        penalty.backward()

    _memory_begin(device)
    ordinary = _timed(ordinary_step, warmup=args.gan_warmup,
                      measured=args.gan_steps, repeats=3, device=device)
    r1 = _timed(r1_step, warmup=max(1, args.gan_r1_warmup),
                measured=args.gan_r1_steps, repeats=3, device=device)
    return {
        "status": "ok",
        "batch": args.gan_batch,
        "crop_frames": args.gan_crop,
        "base": args.gan_base,
        "n_res": args.gan_n_res,
        "scales": args.gan_scales,
        "ordinary": ordinary,
        "r1": r1,
        "peak_memory_bytes": _memory_peak(device),
    }


def benchmark_wavlm(args, device: torch.device) -> dict:
    from atcgen.eval.embed_dist import embed_clips, kid, wavlm_embedder

    rng = np.random.default_rng(args.seed)
    clips = [rng.standard_normal(4 * TARGET_SR).astype(np.float32) * 0.03
             for _ in range(args.wavlm_clips)]
    embedder = wavlm_embedder(device=str(device))
    embedder(clips[0], TARGET_SR)

    def action() -> None:
        _, embeddings = embed_clips(clips, embedder, sr=TARGET_SR)
        midpoint = len(embeddings) // 2
        kid(embeddings[:midpoint], embeddings[midpoint:], subsets=1,
            subset_size=midpoint, seed=args.seed)

    _memory_begin(device)
    timing = _timed(action, warmup=0, measured=1, repeats=3, device=device)
    return {"status": "ok", "clips": len(clips), "embedding_and_kid": timing,
            "peak_memory_bytes": _memory_peak(device)}


def benchmark_tts(args, device: torch.device) -> dict:
    del device
    try:
        from atcgen.tts import KokoroTTS

        engine = KokoroTTS()
    except Exception as error:
        return {"status": "skipped", "reason": f"Kokoro unavailable: {error}"}
    utterances = [
        "alpha one cleared to land",
        "turn left heading two seven zero",
        "contact tower one one eight point three",
        "descend and maintain five thousand",
        "radar contact",
    ]
    counter = 0

    def action() -> None:
        nonlocal counter
        text = utterances[counter % len(utterances)]
        engine.synthesize(text, random.Random(f"{args.seed}:tts:{counter}"))
        counter += 1

    try:
        timing = _timed(action, warmup=args.tts_warmup, measured=args.tts_clips,
                        repeats=3, device=torch.device("cpu"))
    except Exception as error:
        return {"status": "skipped", "reason": f"Kokoro render failed: {error}"}
    return {"status": "ok", "renders": args.tts_clips, "timing": timing}


def benchmark_dsp(args, device: torch.device) -> dict:
    del device
    if not args.presets or not args.noise_bank:
        return {"status": "skipped", "reason": "--presets and --noise-bank are required"}
    if not Path(args.presets).exists() or not Path(args.noise_bank).is_dir():
        return {"status": "skipped", "reason": "preset or noise-bank path is absent"}
    try:
        from atcgen.channel.learned.residual_train import dsp_channel

        channel = dsp_channel(args.presets, args.noise_bank)
    except Exception as error:
        return {"status": "skipped", "reason": f"calibrated DSP unavailable: {error}"}
    rng_np = np.random.default_rng(args.seed)
    clips = [rng_np.standard_normal(4 * TARGET_SR).astype(np.float32) * 0.03
             for _ in range(args.dsp_clips)]
    counter = 0

    def action() -> None:
        nonlocal counter
        index = counter % len(clips)
        channel(clips[index], TARGET_SR,
                random.Random(f"{args.seed}:dsp:{counter}"))
        counter += 1

    timing = _timed(action, warmup=args.dsp_warmup, measured=args.dsp_clips,
                    repeats=3, device=torch.device("cpu"))
    return {"status": "ok", "clips": args.dsp_clips, "timing": timing}


def benchmark_sft(args, device: torch.device) -> dict:
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(args.sft_model).to(device)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    mel_bins = int(getattr(model.config, "num_mel_bins", 80))
    vocab = int(model.config.vocab_size)

    def action() -> None:
        features = torch.randn(args.sft_batch, mel_bins, args.sft_frames, device=device)
        labels = torch.randint(4, vocab, (args.sft_batch, args.sft_label_length),
                               device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_features=features, labels=labels).loss
        loss.backward()
        optimizer.step()

    _memory_begin(device)
    timing = _timed(action, warmup=args.sft_warmup, measured=args.sft_steps,
                    repeats=3, device=device)
    return {"status": "ok", "model": args.sft_model, "batch": args.sft_batch,
            "optimizer_steps": timing, "peak_memory_bytes": _memory_peak(device)}


def _budget(value: int, quick: bool) -> int:
    return max(1, value // 10) if quick else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    for section in ("gan", "wavlm", "tts", "dsp", "sft"):
        parser.add_argument(f"--{section}", action="store_true")

    parser.add_argument("--gan-warmup", type=int, default=100)
    parser.add_argument("--gan-steps", type=int, default=300)
    parser.add_argument("--gan-r1-warmup", type=int, default=3)
    parser.add_argument("--gan-r1-steps", type=int, default=10)
    parser.add_argument("--gan-batch", type=int, default=12)
    parser.add_argument("--gan-crop", type=int, default=128)
    parser.add_argument("--gan-base", type=int, default=48)
    parser.add_argument("--gan-n-res", type=int, default=6)
    parser.add_argument("--gan-scales", type=int, nargs="+", default=[1, 2, 4])

    parser.add_argument("--wavlm-clips", type=int, default=64)
    parser.add_argument("--tts-clips", type=int, default=20)
    parser.add_argument("--tts-warmup", type=int, default=2)
    parser.add_argument("--presets")
    parser.add_argument("--noise-bank")
    parser.add_argument("--dsp-clips", type=int, default=50)
    parser.add_argument("--dsp-warmup", type=int, default=5)

    parser.add_argument("--sft-model", default="openai/whisper-tiny.en")
    parser.add_argument("--sft-batch", type=int, default=8)
    parser.add_argument("--sft-warmup", type=int, default=10)
    parser.add_argument("--sft-steps", type=int, default=30)
    parser.add_argument("--sft-frames", type=int, default=3000)
    parser.add_argument("--sft-label-length", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected = [name for name in ("gan", "wavlm", "tts", "dsp", "sft")
                if getattr(args, name)]
    if not selected:
        build_parser().error("select at least one section: --gan/--wavlm/--tts/--dsp/--sft")
    device = pick_device(args.device)
    for name in ("gan_warmup", "gan_steps", "gan_r1_warmup", "gan_r1_steps",
                 "wavlm_clips", "tts_clips", "tts_warmup", "dsp_clips",
                 "dsp_warmup", "sft_warmup", "sft_steps"):
        setattr(args, name, _budget(getattr(args, name), args.quick))
    if args.wavlm_clips < 4:
        args.wavlm_clips = 4

    out = Path(args.out or f"runs/bench/{device}.json")
    payload = {
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "args": vars(args).copy(),
        "results": {},
    }
    run = start_run(project="atcgan-fastcut", name=f"bench-{device}",
                    config=payload["args"], tags=("bench",))
    functions = {
        "gan": benchmark_gan,
        "wavlm": benchmark_wavlm,
        "tts": benchmark_tts,
        "dsp": benchmark_dsp,
        "sft": benchmark_sft,
    }
    try:
        for name in selected:
            payload["results"][name] = functions[name](args, device)
            result = payload["results"][name]
            try:
                run.log({f"bench/{name}": result})
            except Exception:
                pass
    finally:
        try:
            run.finish()
        except Exception:
            pass

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
