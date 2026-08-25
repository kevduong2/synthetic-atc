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


def load_real_atc(split: str = "test", corpus: str = "jacktol/atc-dataset",
                  cast_audio: bool = True):
    """Return a HF Dataset with 'audio' (16 kHz) and 'text' columns.

    `cast_audio=False` skips the resampling cast for transcript-only work
    (vocab harvesting, split bookkeeping), where decoding the clips is waste.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(corpus, split=split)
    # normalize column names across corpora
    if "transcription" in ds.column_names:
        ds = ds.rename_column("transcription", "text")
    if cast_audio:
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


def export_noise_beds(out_dir: str | Path, max_clips: int = 500,
                      corpus: str = "jacktol/atc-dataset", split: str = "train",
                      win_sec: float = 0.6) -> int:
    """Harvest speech-free noise beds (static/carrier hiss) from real ATC clips.

    Takes the quietest `win_sec` window of each clip when it is clearly below
    the clip's overall level (i.e. between transmissions, not during speech).
    Output feeds `atcgen.channel.primitives.NoiseBank`. Returns count written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_real_atc(split=split, corpus=corpus)
    win = int(REAL_SR * win_sec)
    n = 0
    for ex in ds:
        wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
        if len(wav) < 2 * win:
            continue
        clip_rms = float(np.sqrt(np.mean(wav ** 2)))
        # rolling mean power over win-sized windows
        power = np.convolve(wav ** 2, np.ones(win) / win, mode="valid")
        start = int(np.argmin(power))
        bed_rms = float(np.sqrt(power[start]))
        # quiet relative to speech, but not digital silence
        if not (0.02 * clip_rms < bed_rms < 0.35 * clip_rms):
            continue
        sf.write(out / f"noise_{n:05d}.wav", wav[start:start + win], REAL_SR)
        n += 1
        if n >= max_clips:
            break
    return n
