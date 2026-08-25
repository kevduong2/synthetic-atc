"""Shared contracts for the RL generation-tuning loop.

The outer loop optimizes generator-config knobs against a downstream-ASR
reward: candidate config -> generate synthetic batch -> short whisper-tiny
fine-tune -> WER on a fixed real ATC dev set.  These dataclasses are the seam
between the optimizer side (space/policy/loop) and the reward side
(reward/finetune_lite): both depend on this module and not on each other's
internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RewardResult:
    """Outcome of one candidate evaluation.

    ``reward`` is the maximized scalar: baseline dev WER minus post-fine-tune
    dev WER (positive = the synthetic batch helped), both ATC-normalized.
    ``proxy`` distinguishes a cheap screening score from a true fine-tune
    reward so the trial log stays honest about which is which.
    """

    reward: float
    wer_after: float
    wer_baseline: float
    hallucination_rate: float | None = None
    proxy: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


class RewardFn(Protocol):
    """Evaluate one candidate. ``config`` is a raw YAML-style mapping ready for
    ``atcgen.config`` parsing; ``trial_dir`` receives generated audio and any
    per-trial artifacts and is unique per call."""

    def __call__(self, config: Mapping[str, Any], trial_dir: str) -> RewardResult: ...


@dataclass(frozen=True)
class Trial:
    """One row of the loop's JSONL log."""

    index: int
    vector: list[float]              # position in the unit-cube search space
    overrides: dict[str, Any]        # human-readable knob values applied
    result: RewardResult
    wall_time_sec: float
