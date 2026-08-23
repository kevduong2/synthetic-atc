"""Synthetic-audio evaluation: Tier 0 QC gates, Tier 1 distribution match,
Tier 2 real-vs-synthetic probe.

See docs/plans/05-evaluation-plan.md. The Tier 0 API is re-exported here;
`channel_stats` (Tier 1 statistics), `embed_dist` (Tier 1 embedding distances),
`probe` (Tier 2) and `report` (HTML, needs matplotlib from the `[eval]` extra)
are imported from their modules so that `python -m atcgen.eval.channel_stats`
and friends stay warning-free.
"""

from .qc import (QCConfig, QCResult, QCTally, Transcriber, default_transcriber,
                 qc_sample, whisper_transcriber)

__all__ = [
    "QCConfig", "QCResult", "QCTally", "Transcriber", "qc_sample",
    "default_transcriber", "whisper_transcriber",
]
