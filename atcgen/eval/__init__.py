"""Synthetic-audio evaluation: Tier 0 QC gates and Tier 1 channel statistics.

See docs/plans/05-evaluation-plan.md. The Tier 0 API is re-exported here;
`channel_stats` (Tier 1) and `report` (HTML, needs matplotlib from the
`[eval]` extra) are imported from their modules so that
`python -m atcgen.eval.channel_stats` stays warning-free.
"""

from .qc import (QCConfig, QCResult, QCTally, Transcriber, default_transcriber,
                 qc_sample, whisper_transcriber)

__all__ = [
    "QCConfig", "QCResult", "QCTally", "Transcriber", "qc_sample",
    "default_transcriber", "whisper_transcriber",
]
