"""Deterministic, disjoint slices of the real ATC corpus (D11).

Single source of truth for *which rows are allowed where*. Every consumer --
grammar vocab harvesting, gate teachers, the RL reward's dev set, model
selection, the final report -- names a split from `SPLITS` instead of
open-coding `train[0:200]`, so overlap is a registry-level property that
`_check_disjoint` enforces at import rather than a fact someone has to
remember.

Why the discipline matters
--------------------------
Anything the generator or the search loop is allowed to look at stops being a
measurement of generalization: vocabulary harvested from a slice leaks its
carriers and waypoints into synthetic references, and a reward computed on a
slice makes that slice a training signal even though no gradient touched it.
So the corpus is partitioned once:

    real_train    train[0:8000]      generator vocab anchor, SFT arms, gate
                                     calibration -- freely used upstream
    reward_val    train[8000:9000]   the RL/bandit reward's dev set; burned
                                     continuously by the search loop
    model_select  train[9000:10000]  checkpoint/arm picking -- read often
                                     enough that it too is "spent"
    train_tail    train[10000:]      unassigned reserve
    locked_test   test[500:2500]     TOUCH ONLY FOR FINAL REPORTS
    spent_test    test[0:500]        burned by runs/rl_v1 verification;
                                     never reuse it for a headline number

`locked_test` is the only slice that has not informed a single decision, which
is the entire reason it can be quoted. Reading it during development converts
it into another `model_select` and there is no third test split behind it.

PoC caveat (accepted, documented, not fixed here)
-------------------------------------------------
`jacktol/atc-dataset` is *utterance*-segmented: its own train and test splits
can share speakers, sectors and callsigns, so index-disjointness here is not
speaker-disjointness. Numbers off `locked_test` are therefore optimistic
relative to a truly held-out airfield. This is prohibited in production
(research-findings §4.8) and accepted for the proof of concept; the user's own
transcribed set replaces this registry's corpus later, at which point the
splits should be cut by recording session/speaker, not by row index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .real_atc import load_real_atc

#: The corpus every split in this registry indexes into.
CORPUS = "jacktol/atc-dataset"


@dataclass(frozen=True)
class SplitSpec:
    """One named half-open row range `[start, stop)` of a corpus split."""

    name: str
    source_split: str
    start: int
    stop: int | None
    purpose: str
    policy: str

    @property
    def slice_str(self) -> str:
        """HF-style slice notation, e.g. ``train[8000:9000]``."""
        return f"{self.source_split}[{self.start}:{'' if self.stop is None else self.stop}]"

    def dataset_name(self, corpus: str = CORPUS) -> str:
        """Identifier written into report JSON (`dataset` field)."""
        return f"{corpus}:{self.slice_str}"

    def indices(self, total: int) -> range:
        """Row indices this split selects from a corpus split of `total` rows."""
        stop = total if self.stop is None else min(self.stop, total)
        return range(min(self.start, stop), stop)

    def overlaps(self, other: SplitSpec) -> bool:
        """Whether two specs would ever hand out the same corpus row."""
        if self.source_split != other.source_split:
            return False
        left = max(self.start, other.start)
        right = min(math.inf if self.stop is None else self.stop,
                    math.inf if other.stop is None else other.stop)
        return left < right

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "corpus": CORPUS,
            "source_split": self.source_split,
            "start": self.start,
            "stop": self.stop,
            "slice": self.slice_str,
            "purpose": self.purpose,
            "policy": self.policy,
        }


SPLITS: dict[str, SplitSpec] = {
    spec.name: spec for spec in (
        SplitSpec(
            "real_train", "train", 0, 8000,
            purpose="generator vocab anchor, real-SFT arms, gate calibration",
            policy="free to use upstream; never quote a number off it",
        ),
        SplitSpec(
            "reward_val", "train", 8000, 9000,
            purpose="RL/bandit reward dev set",
            policy="spent by the search loop; not a reportable metric",
        ),
        SplitSpec(
            "model_select", "train", 9000, 10000,
            purpose="checkpoint and arm selection, A0 development baseline",
            policy="read often; treat as spent, quote only as a dev number",
        ),
        SplitSpec(
            "train_tail", "train", 10000, None,
            purpose="unassigned reserve",
            policy="assign a role here before using it, then add it above",
        ),
        SplitSpec(
            "locked_test", "test", 500, 2500,
            purpose="final report numbers for every arm",
            policy="TOUCH ONLY FOR FINAL REPORTS -- one read per arm, no tuning",
        ),
        SplitSpec(
            "spent_test", "test", 0, 500,
            purpose="historical: runs/rl_v1 A/B verification",
            policy="burned; never reuse for a headline number",
        ),
    )
}

SPLIT_NAMES: tuple[str, ...] = tuple(SPLITS)


def _check_disjoint() -> None:
    """Fail loudly at import if the registry ever grows an overlap."""
    specs = list(SPLITS.values())
    for i, left in enumerate(specs):
        for right in specs[i + 1:]:
            if left.overlaps(right):
                raise ValueError(
                    f"split registry is not disjoint: {left.slice_str} "
                    f"({left.name}) overlaps {right.slice_str} ({right.name})")


_check_disjoint()


def split_spec(name: str) -> SplitSpec:
    """The `SplitSpec` called `name`."""
    try:
        return SPLITS[name]
    except KeyError:
        raise KeyError(
            f"unknown split {name!r}; known splits: {', '.join(SPLIT_NAMES)}"
        ) from None


def load_split(name: str, cast_audio: bool = True, corpus: str = CORPUS):
    """Load the rows of split `name` as a HF Dataset ('audio' + 'text').

    Normalization (column renames, 16 kHz audio cast) is `load_real_atc`'s, so
    a split loads exactly like the corpus does everywhere else. Pass
    `cast_audio=False` when only the transcripts are needed -- decoding 2000
    clips to look at text is pure waste.
    """
    spec = split_spec(name)
    dataset = load_real_atc(split=spec.source_split, corpus=corpus,
                            cast_audio=cast_audio)
    return dataset.select(spec.indices(len(dataset)))


def describe() -> dict:
    """JSON-ready registry dump, for embedding in reports and run logs."""
    return {name: spec.to_dict() for name, spec in SPLITS.items()}
