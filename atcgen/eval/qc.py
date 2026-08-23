"""Tier 0 per-sample QC gates (see docs/plans/05-evaluation-plan.md §2).

Cheap checks meant to run inside generation and block bad samples before they
reach the manifest: NaN/Inf, digital clipping, all-silence, duration and
loudness bounds, plus an ASR round-trip gate — a pretrained (non-fine-tuned)
Whisper transcribes the degraded audio and the sample is discarded when the
normalized WER against the reference transcript exceeds a threshold.

The transcriber is injectable (`Callable[[wav, sr], str]`) so tests can pass a
fake; `default_transcriber()` lazily loads `openai/whisper-small.en` only when
the gate first fires.  `QCTally` accumulates discard reasons/rates for a run.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Transcriber = Callable[[np.ndarray, int], str]

try:  # the training/ scripts are not part of the installed wheel
    from training.normalize import normalize_atc
except ImportError:  # pragma: no cover - fallback: case/punctuation folding only
    def normalize_atc(text: str) -> str:
        return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


@dataclass
class QCConfig:
    min_duration: float = 0.5      # seconds
    max_duration: float = 30.0
    clip_level: float = 0.999      # |x| at or above this counts as clipped
    max_clip_frac: float = 0.01    # >1% clipped samples -> discard
    silence_rms_db: float = -60.0  # below this the clip is treated as silent
    min_rms_db: float = -40.0      # loudness window
    max_rms_db: float = -8.0
    max_wer: float = 0.5           # ASR round-trip discard threshold
    asr_gate: bool = True          # run the round-trip gate at all


@dataclass
class QCResult:
    ok: bool
    reason: str | None            # None | nonfinite | clipping | silence | duration | level | asr_wer
    metrics: dict


def _rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20))


def qc_sample(wav: np.ndarray, sr: int, text: str | None = None,
              config: QCConfig | None = None,
              transcriber: Transcriber | None = None) -> QCResult:
    """Run the Tier 0 gates on one sample. First failing gate wins.

    The ASR round-trip gate runs only when `text` normalizes to something
    non-empty (noise-only anti-hallucination samples have no reference) and
    `config.asr_gate` is set; `transcriber` defaults to `default_transcriber()`.
    """
    cfg = config or QCConfig()
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    metrics: dict = {"duration": float(len(x) / sr) if sr else 0.0}

    if len(x) == 0 or not np.isfinite(x).all():
        return QCResult(False, "nonfinite" if len(x) else "silence", metrics)

    peak = float(np.abs(x).max())
    clip_frac = float(np.mean(np.abs(x) >= cfg.clip_level))
    rms_db = _rms_db(x)
    metrics.update({"peak": peak, "clip_frac": clip_frac, "rms_db": rms_db})

    if clip_frac > cfg.max_clip_frac:
        return QCResult(False, "clipping", metrics)
    if rms_db < cfg.silence_rms_db or peak == 0.0:
        return QCResult(False, "silence", metrics)
    if not cfg.min_duration <= metrics["duration"] <= cfg.max_duration:
        return QCResult(False, "duration", metrics)
    if not cfg.min_rms_db <= rms_db <= cfg.max_rms_db:
        return QCResult(False, "level", metrics)

    ref = normalize_atc(text) if text else ""
    if cfg.asr_gate and ref:
        transcriber = transcriber or default_transcriber()
        hyp = normalize_atc(transcriber(x, sr))
        metrics["wer"] = _wer(ref, hyp)
        if metrics["wer"] > cfg.max_wer:
            return QCResult(False, "asr_wer", metrics)

    return QCResult(True, None, metrics)


def _wer(ref: str, hyp: str) -> float:
    import jiwer

    return float(jiwer.wer(ref, hyp))


@dataclass
class QCTally:
    """Discard reasons/rates for a generation run (`stats.json`, 05 §2)."""

    total: int = 0
    kept: int = 0
    reasons: Counter = field(default_factory=Counter)

    def add(self, result: QCResult) -> bool:
        """Record one QC outcome; returns whether the sample was kept."""
        self.total += 1
        if result.ok:
            self.kept += 1
        else:
            self.reasons[result.reason or "unknown"] += 1
        return result.ok

    @property
    def discard_rate(self) -> float:
        return (self.total - self.kept) / self.total if self.total else 0.0

    def summary(self) -> dict:
        n = self.total or 1
        return {
            "total": self.total,
            "kept": self.kept,
            "discarded": self.total - self.kept,
            "discard_rate": round(self.discard_rate, 4),
            "reasons": dict(self.reasons),
            "reason_rates": {k: round(v / n, 4) for k, v in self.reasons.items()},
        }


def whisper_transcriber(model_name: str = "openai/whisper-small.en",
                        device=None) -> Transcriber:
    """A `Transcriber` that loads the pretrained Whisper on its first call."""
    state: dict = {}

    def transcribe(wav: np.ndarray, sr: int) -> str:
        if "pipe" not in state:
            from transformers import pipeline

            state["pipe"] = pipeline("automatic-speech-recognition",
                                     model=model_name, device=device)
        out = state["pipe"]({"raw": np.asarray(wav, np.float32), "sampling_rate": sr})
        return out["text"]

    return transcribe


_DEFAULT: list[Transcriber] = []


def default_transcriber() -> Transcriber:
    """Process-wide lazy Whisper transcriber (loaded on first transcription)."""
    if not _DEFAULT:
        _DEFAULT.append(whisper_transcriber())
    return _DEFAULT[0]
