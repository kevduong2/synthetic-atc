"""End-to-end synthetic dataset builder.

text source -> TTS -> channel degradation (dsp | gan | mix) -> wav + manifest.jsonl

Manifest lines: {"audio": "wavs/000001.wav", "text": ..., "role": ..., "kind": ...,
"channel": "dsp"|"gan", "snr_db": ..., "duration": ...}
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

from ..channel.dsp import RadioChannelSim, TARGET_SR
from ..text.sources import TextSource, make_text_source


def build_dataset(
    out_dir: str | Path,
    n_samples: int,
    text_source: TextSource | str = "grammar",
    channel: str = "dsp",          # "dsp" | "gan" | "mix" | "clean"
    gan_checkpoint: str | None = None,
    seed: int = 0,
    tts=None,
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
    sim = RadioChannelSim()

    gan = None
    if channel in ("gan", "mix"):
        from ..channel.gan.infer import GanChannel
        gan = GanChannel(gan_checkpoint)

    manifest_path = out / "manifest.jsonl"
    prev_wav = None  # reused as co-channel interference material
    with open(manifest_path, "w") as mf:
        for i in tqdm(range(n_samples), desc=f"generating ({channel})"):
            utt = text_source.sample(rng)
            clean = tts.synthesize(utt.spoken, rng)

            mode = channel
            if channel == "mix":
                mode = "gan" if rng.random() < 0.5 else "dsp"

            meta = {}
            if mode == "clean":
                from ..channel.dsp import _resample
                wav = _resample(clean, tts.sample_rate, TARGET_SR)
            elif mode == "gan":
                wav = gan(clean, tts.sample_rate, rng)
            else:
                wav, params = sim(clean, tts.sample_rate, rng, interference=prev_wav)
                meta = {"snr_db": round(params.snr_db, 1)}

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
