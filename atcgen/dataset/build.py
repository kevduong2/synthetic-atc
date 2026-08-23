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
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..channel.chain import ProceduralChannel, UtteranceMeta, mild_chain
from ..channel.primitives import TARGET_SR, NoiseBank, resample
from ..config import load_config
from ..text.sources import TextSource, make_text_source

PILOT_DOUBLE_HOP_PROB = 0.5   # pilot audio relayed through a ground station
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mode1_default.yaml"


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
    channel_cfg = load_config(DEFAULT_CONFIG).channel
    sim = ProceduralChannel.from_config(
        channel_cfg, noise_bank=NoiseBank(noise_dir) if noise_dir else None)

    gan = mild = None
    if channel in ("gan", "mix"):
        from ..channel.gan.infer import GanChannel
        gan = GanChannel(gan_checkpoint)
        mild = ProceduralChannel(mild_chain())

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
                wav, rec = sim(silence, TARGET_SR, rng, UtteranceMeta(kind="noise"))
                rel = f"wavs/{i:06d}.wav"
                sf.write(out / rel, wav, TARGET_SR)
                mf.write(json.dumps({
                    "audio": rel, "text": "", "role": "none", "kind": "noise",
                    "channel": "dsp", "duration": round(len(wav) / TARGET_SR, 3),
                    "snr_db": rec.snr_db,
                }) + "\n")
                continue

            utt = text_source.sample(rng)
            clean = tts.synthesize(utt.spoken, rng)

            utt_meta = UtteranceMeta(role=utt.role, kind=utt.kind)
            meta = {}
            if mode == "clean":
                wav = resample(clean, tts.sample_rate, TARGET_SR)
            elif mode == "gan":
                wav = gan(clean, tts.sample_rate, rng)
                # GAN alone is deterministic (one learned radio); a mild DSP
                # pass restores per-sample SNR/band/codec diversity
                wav, rec = mild(wav, TARGET_SR, rng, utt_meta)
                meta = {"snr_db": rec.snr_db}
            else:
                hops = 2 if utt.role == "pilot" and rng.random() < PILOT_DOUBLE_HOP_PROB else 1
                wav, rec = sim(clean, tts.sample_rate, rng, utt_meta,
                               interference=prev_wav, hops=hops)
                meta = {"snr_db": rec.snr_db, "hops": hops}

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
