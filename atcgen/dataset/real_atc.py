"""Download/prepare real ATC corpora from Hugging Face, or load a local one.

Used for (a) CycleGAN domain-B training audio and (b) real-WER evaluation.

Corpora:
  - jacktol/atc-dataset: ATCO2 1-h test subset + UWB-ATCC, segmented pairs
  - Jzuluaga/uwb_atcc: UWB-ATCC with train/test splits
  - any local `audio,text` CSV or JSONL manifest (see `load_local_corpus`)

Both paths return the same object -- a `datasets.Dataset` with an `audio`
column cast to `Audio(sampling_rate=16000)` and a `text` column -- so callers
(`training/evaluate.py`, `atcgen.rl.reward.TrueRewardHarness`,
`scripts/rl_verify.py`) never branch on where the corpus came from, and the
decoding, resampling and text handling are literally the same code.
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

REAL_SR = 16000

#: A corpus name with one of these suffixes is a local manifest, not an HF id.
LOCAL_SUFFIXES = {".csv", ".jsonl"}

#: Column aliases, in preference order. `transcription` mirrors the rename
#: `load_real_atc` applies to the HF corpora.
_AUDIO_KEYS = ("audio", "path", "file", "filename")
_TEXT_KEYS = ("text", "transcript", "transcription")

#: Optional third column naming where a row came from. A mixed dev set (say
#: local KIXD rows beside public EU rows) carries it so the reward can report
#: a per-source WER breakdown and a divergence between the two is visible
#: rather than averaged away.
SOURCE_KEY = "source"


def is_local_corpus(corpus: str | Path) -> bool:
    """True when `corpus` names a local manifest file rather than an HF dataset."""
    return Path(corpus).suffix.lower() in LOCAL_SUFFIXES


def _first(row, keys, path, index):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError(
        f"{path} row {index} has none of {keys} (keys: {sorted(map(str, row))})")


def _local_rows(path: Path) -> dict[str, list[str]]:
    """Parse a local manifest into aligned columns.

    Always returns `audio` and `text`; returns `source` as well when at least
    one row declares it (missing values become `""`), so a mixed-provenance dev
    set can be sliced by where its rows came from.

    Relative audio paths resolve against the manifest's own directory, which
    is what a manifest written next to its `wavs/` expects; absolute paths are
    left alone.  Empty text is kept rather than dropped -- noise-only rows are
    what the hallucination metric scores.
    """
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no rows")

    audio, text, source = [], [], []
    for index, row in enumerate(rows):
        clip = Path(str(_first(row, _AUDIO_KEYS, path, index)))
        if not clip.is_absolute():
            clip = path.parent / clip
        value = _first(row, _TEXT_KEYS, path, index)
        audio.append(str(clip))
        text.append("" if value is None else str(value))
        label = row.get(SOURCE_KEY)
        source.append("" if label is None else str(label))

    columns = {"audio": audio, "text": text}
    if any(source):
        columns[SOURCE_KEY] = source
    return columns


def load_local_corpus(path: str | Path, cast_audio: bool = True):
    """Return a HF Dataset over a local `audio,text` CSV or JSONL manifest.

    The rows keep their file order, so `--dev-indices lo:hi` selects the same
    clips on every run, and the text is passed through verbatim: ATC
    normalization is `training.normalize.normalize_atc`, applied downstream by
    `training.evaluate.build_report` to references and hypotheses alike, and
    pre-normalizing here would score a different (easier) reference than the
    HF path does.
    """
    from datasets import Audio, Dataset

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"local corpus manifest not found: {path}")
    return Dataset.from_dict(_local_rows(path)).cast_column(
        "audio", Audio(sampling_rate=REAL_SR, decode=cast_audio))


def load_real_atc(split: str = "test", corpus: str = "jacktol/atc-dataset",
                  cast_audio: bool = True):
    """Return a HF Dataset with 'audio' (16 kHz) and 'text' columns.

    `corpus` is either a Hugging Face dataset id or a path to a local
    `audio,text` CSV/JSONL manifest; `split` is ignored for the latter, which
    is one file, not a split registry.

    `cast_audio=False` skips the resampling cast for transcript-only work
    (vocab harvesting, split bookkeeping), where decoding the clips is waste.
    """
    from datasets import Audio, load_dataset

    if is_local_corpus(corpus):
        return load_local_corpus(corpus, cast_audio=cast_audio)

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
