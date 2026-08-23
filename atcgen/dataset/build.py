"""End-to-end synthetic dataset builder.

text source -> TTS -> channel degradation (dsp | gan | mix) -> wav + manifest.jsonl

Manifest lines: {"audio": "wavs/000001.wav", "text": ..., "role": ..., "kind": ...,
"channel": "dsp"|"gan", "snr_db": ..., "duration": ..., "hops": ...}
A small fraction are noise-only samples with text "" (Whisper hallucination
control); pilot utterances are sometimes double-hopped (ground relay).
Loadable with `datasets.load_dataset("json", data_files=manifest)` or the
helper `load_manifest` below.
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..channel.dsp import ChannelParams, NoiseBank, RadioChannelSim, TARGET_SR
from ..text.sources import TextSource, make_text_source

PILOT_DOUBLE_HOP_PROB = 0.5   # pilot audio relayed through a ground station


def build_dataset(
    out_dir: str | Path,
    n_samples: int,
    text_source: TextSource | str = "grammar",
    channel: str = "dsp",          # "dsp" | "gan" | "mix" | "clean"
    gan_checkpoint: str | None = None,
    seed: int = 0,
    tts=None,
    noise_dir: str | None = None,  # real noise beds (real_atc.export_noise_beds)
    noise_only_frac: float = 0.03,  # noise-only samples with empty transcript
) -> Path:
    """Generate n_samples utterances. Returns path to manifest.jsonl."""
    out = Path(out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if isinstance(text_source, str):
        text_source = make_text_source(text_source)
    if tts is None:
        from ..tts import KokoroTTS
        tts = KokoroTTS()
    sim = RadioChannelSim(noise_bank=NoiseBank(noise_dir) if noise_dir else None)

    gan = None
    if channel in ("gan", "mix"):
        from ..channel.gan.infer import GanChannel
        gan = GanChannel(gan_checkpoint)

    manifest_path = out / "manifest.jsonl"
    prev_wav = None  # reused as co-channel interference material
    with open(manifest_path, "w") as mf:
        for i in tqdm(range(n_samples), desc=f"generating ({channel})"):
            mode = channel
            if channel == "mix":
                mode = "gan" if rng.random() < 0.5 else "dsp"

            # noise-only sample with empty transcript (anti-hallucination)
            if channel != "clean" and rng.random() < noise_only_frac:
                silence = np.zeros(int(TARGET_SR * rng.uniform(2.0, 6.0)), np.float32)
                wav, params = sim(silence, TARGET_SR, rng)
                rel = f"wavs/{i:06d}.wav"
                sf.write(out / rel, wav, TARGET_SR)
                mf.write(json.dumps({
                    "audio": rel, "text": "", "role": "none", "kind": "noise",
                    "channel": "dsp", "duration": round(len(wav) / TARGET_SR, 3),
                    "snr_db": round(params.snr_db, 1),
                }) + "\n")
                continue

            utt = text_source.sample(rng)
            clean = tts.synthesize(utt.spoken, rng)

            meta = {}
            if mode == "clean":
                from ..channel.dsp import _resample
                wav = _resample(clean, tts.sample_rate, TARGET_SR)
            elif mode == "gan":
                wav = gan(clean, tts.sample_rate, rng)
                # GAN alone is deterministic (one learned radio); a mild DSP
                # pass restores per-sample SNR/band/codec diversity
                wav, params = sim(wav, TARGET_SR, rng, params=ChannelParams.mild(rng))
                meta = {"snr_db": round(params.snr_db, 1)}
            else:
                hops = 2 if utt.role == "pilot" and rng.random() < PILOT_DOUBLE_HOP_PROB else 1
                wav, params = sim(clean, tts.sample_rate, rng,
                                  interference=prev_wav, hops=hops)
                meta = {"snr_db": round(params.snr_db, 1), "hops": hops}

            rel = f"wavs/{i:06d}.wav"
            sf.write(out / rel, wav, TARGET_SR)
            prev_wav = wav

            record = {
                "audio": rel,
                "text": utt.transcript,
                "role": utt.role,
                "kind": utt.kind,
                "channel": mode,
                "duration": round(len(wav) / TARGET_SR, 3),
                **meta,
            }
            mf.write(json.dumps(record) + "\n")
    return manifest_path


def load_manifest(manifest_path: str | Path):
    """Load a built dataset as a HF Dataset with an Audio column."""
    from datasets import Audio, Dataset

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    records = [json.loads(l) for l in open(manifest_path) if l.strip()]
    for r in records:
        r["audio"] = str(root / r["audio"])
    ds = Dataset.from_list(records)
    return ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
