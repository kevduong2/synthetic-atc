"""Frozen ASR teachers for the verification gate (research-findings §4.4, D4).

The gate asks a question the student cannot be trusted to answer about its own
training data: *is this label provably what the audio says?*  So the pool is
deliberately not the student (`whisper-tiny.en`) and deliberately not one
family — a seq2seq decoder and a CTC decoder fail in different directions, and
D4's judge-diversity argument is that agreement across architectures is worth
more than agreement across checkpoints of one:

* ``openai/whisper-base.en`` — encoder-decoder, language-modelled.  Reads
  fluent English out of bad audio, which is its strength and its failure mode:
  it will happily invent a plausible clearance, and it hallucinates on silence.
* ``facebook/wav2vec2-base-960h`` — CTC, frame-synchronous, no decoder LM.
  Transcribes what it hears letter by letter and goes quiet (or to letter soup)
  rather than confabulating, which is exactly the second opinion the seq2seq
  model cannot give.

Both are frozen: no fine-tuning, ever.  A teacher that has seen this
generator's output is no longer independent evidence about it.

Everything is batched — the gate runs over whole datasets, and per-clip
inference is the difference between a twenty-minute pass and an afternoon.
`Teacher.transcribe` takes a list of waveforms and returns a list of raw
(un-normalized) hypotheses; normalization is the gate's job, so the raw text
stays in the manifest for eyeballing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

WHISPER_TEACHER = "openai/whisper-base.en"
CTC_TEACHER = "facebook/wav2vec2-base-960h"
MODEL_SR = 16000              # both teachers are 16 kHz models


class Teacher(Protocol):
    """A frozen ASR judge: a name and batched transcription."""

    name: str

    def transcribe(self, waves: list[np.ndarray], sr: int) -> list[str]: ...


def pick_device(prefer: str | None = None) -> str:
    """`prefer`, or MPS when this Mac has it, else CPU."""
    if prefer:
        return prefer
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resample(waves: list[np.ndarray], sr: int) -> list[np.ndarray]:
    """Both teachers are 16 kHz-only; a set built at another rate still gates."""
    if sr == MODEL_SR:
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
    import librosa

    return [librosa.resample(np.asarray(w, dtype=np.float32).reshape(-1),
                             orig_sr=sr, target_sr=MODEL_SR) for w in waves]


@dataclass
class WhisperTeacher:
    """Batched greedy decode from a frozen Whisper checkpoint.

    Whisper's feature extractor pads every clip to the same 30 s log-mel, so a
    batch needs no attention mask and no length bookkeeping — the same
    property `atcgen.rl.finetune_lite.transcribe` leans on.
    """

    model_name: str = WHISPER_TEACHER
    device: str | None = None
    max_new_tokens: int = 128
    name: str = ""
    _state: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or self.model_name.split("/")[-1]
        self.device = pick_device(self.device)

    def _load(self) -> tuple:
        if "model" not in self._state:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            processor = WhisperProcessor.from_pretrained(self.model_name)
            model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            model.to(torch.device(self.device)).eval()
            self._state.update(processor=processor, model=model)
        return self._state["processor"], self._state["model"]

    def transcribe(self, waves: list[np.ndarray], sr: int) -> list[str]:
        if not waves:
            return []
        import torch

        processor, model = self._load()
        arrays = _resample(waves, sr)
        features = processor.feature_extractor(
            arrays, sampling_rate=MODEL_SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(features.to(torch.device(self.device)),
                                 max_new_tokens=self.max_new_tokens,
                                 num_beams=1, do_sample=False)
        return [text.strip()
                for text in processor.batch_decode(ids, skip_special_tokens=True)]


@dataclass
class CTCTeacher:
    """Batched wav2vec2 CTC greedy decode.

    Padding a CTC batch is not free the way padding Whisper is: the model sees
    the pad frames.  `facebook/wav2vec2-base-960h` was trained without an
    attention mask, so one is passed only when its own processor asks for it,
    and clips are batched as given — the gate's batches are short ATC
    transmissions of similar length, so the padding waste is small.
    """

    model_name: str = CTC_TEACHER
    device: str | None = None
    name: str = ""
    _state: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or self.model_name.split("/")[-1]
        self.device = pick_device(self.device)

    def _load(self) -> tuple:
        if "model" not in self._state:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            model.to(torch.device(self.device)).eval()
            self._state.update(processor=processor, model=model)
        return self._state["processor"], self._state["model"]

    def transcribe(self, waves: list[np.ndarray], sr: int) -> list[str]:
        if not waves:
            return []
        import torch

        processor, model = self._load()
        arrays = _resample(waves, sr)
        batch = processor(arrays, sampling_rate=MODEL_SR, return_tensors="pt",
                          padding=True)
        device = torch.device(self.device)
        kwargs = {}
        if "attention_mask" in batch:
            kwargs["attention_mask"] = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(batch["input_values"].to(device), **kwargs).logits
        predicted = torch.argmax(logits, dim=-1)
        return [text.strip() for text in processor.batch_decode(predicted)]


def default_teachers(device: str | None = None) -> list[Teacher]:
    """The frozen pool: one seq2seq judge and one CTC judge.

    Both land on MPS, which is not a foregone conclusion for a 94 M-parameter
    CTC model doing one forward pass — but it measures that way here.  On 24
    smoke clips (M-series, batch 8):

        whisper-base.en     mps 16.4 clips/s   cpu 5.6
        wav2vec2-base-960h  mps 29.8 clips/s   cpu 6.3

    Batch 8 also beats batch 16 for both, because ATC transmissions are short
    and a bigger batch mostly buys more padding.  Aggregate over the pair is
    ~10.6 clips/s, so the two judges together gate 4k clips in about six
    minutes.
    """
    return [WhisperTeacher(device=device), CTCTeacher(device=device)]


@dataclass
class Throughput:
    """Wall-clock accounting for a gate pass, in clips per second."""

    clips: int = 0
    seconds: float = 0.0
    per_teacher: dict = field(default_factory=dict)

    def add(self, teacher: str, clips: int, seconds: float) -> None:
        entry = self.per_teacher.setdefault(teacher, {"clips": 0, "seconds": 0.0})
        entry["clips"] += clips
        entry["seconds"] += seconds

    def summary(self) -> dict:
        return {
            "clips": self.clips,
            "seconds": round(self.seconds, 2),
            "clips_per_sec": round(self.clips / self.seconds, 3) if self.seconds else 0.0,
            "per_teacher": {
                name: {**entry, "seconds": round(entry["seconds"], 2),
                       "clips_per_sec": round(entry["clips"] / entry["seconds"], 3)
                       if entry["seconds"] else 0.0}
                for name, entry in self.per_teacher.items()
            },
        }


def timed(fn, *args, **kwargs):
    """Call `fn` and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
