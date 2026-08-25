"""Verification gate: multi-teacher proof that a sample's label is right.

`gate_dataset` is the whole-dataset entry point; `evaluate_row` is the
per-sample verdict for callers that already hold hypotheses; `select_tiers`
assembles a training mix under the adversarial cap.  See `gate.py` for the
tier definitions and D8's reject-never-relabel rule.
"""

from .gate import (
                   TIERS,
                   TRAINABLE_TIERS,
                   GateConfig,
                   audio_checks,
                   evaluate_row,
                   gate_dataset,
                   gate_stats,
                   hypothesis_entities,
                   load_gated,
                   repeat_score,
                   retier,
                   select_tiers,
                   verify_entities,
)
from .teachers import (
                   CTC_TEACHER,
                   WHISPER_TEACHER,
                   CTCTeacher,
                   Teacher,
                   Throughput,
                   WhisperTeacher,
                   default_teachers,
                   pick_device,
)

__all__ = [
                   "CTC_TEACHER",
                   "TIERS",
                   "TRAINABLE_TIERS",
                   "WHISPER_TEACHER",
                   "CTCTeacher",
                   "GateConfig",
                   "Teacher",
                   "Throughput",
                   "WhisperTeacher",
                   "audio_checks",
                   "default_teachers",
                   "evaluate_row",
                   "gate_dataset",
                   "gate_stats",
                   "hypothesis_entities",
                   "load_gated",
                   "pick_device",
                   "repeat_score",
                   "retier",
                   "select_tiers",
                   "verify_entities",
]
