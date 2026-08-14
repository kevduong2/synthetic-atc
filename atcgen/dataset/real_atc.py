"""Download/prepare real ATC corpora from Hugging Face.

Used for (a) CycleGAN domain-B training audio and (b) real-WER evaluation.

Corpora:
  - jacktol/atc-dataset: ATCO2 1-h test subset + UWB-ATCC, segmented pairs
  - Jzuluaga/uwb_atcc: UWB-ATCC with train/test splits
"""

from pathlib import Path

import numpy as np
import soundfile as sf

REAL_SR = 16000


def load_real_atc(split: str = "test", corpus: str = "jacktol/atc-dataset"):
    """Return a HF Dataset with 'audio' (16 kHz) and 'text' columns."""
    from datasets import Audio, load_dataset

    ds = load_dataset(corpus, split=split)
    # normalize column names across corpora
    if "transcription" in ds.column_names:
        ds = ds.rename_column("transcription", "text")
    ds = ds.cast_column("audio", Audio(sampling_rate=REAL_SR))
    return ds


def export_gan_domain_audio(out_dir: str | Path, max_clips: int = 2000,
                            corpus: str = "jacktol/atc-dataset", split: str = "train",
                            min_sec: float = 1.5, max_sec: float = 12.0) -> int:
    """Dump real ATC clips as wavs for CycleGAN domain-B training. Returns count."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_real_atc(split=split, corpus=corpus)
    n = 0
    for ex in ds:
        wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
        dur = len(wav) / REAL_SR
        if not (min_sec <= dur <= max_sec):
            continue
        peak = np.abs(wav).max()
        if peak < 1e-4:
            continue
        sf.write(out / f"real_{n:05d}.wav", wav / peak * 0.9, REAL_SR)
        n += 1
        if n >= max_clips:
            break
    return n
