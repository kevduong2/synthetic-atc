"""Pluggable text sources for the dataset builder.

A TextSource is anything with `sample(rng) -> Utterance`. The built-in grammar
is one implementation; teammates' generators can plug in via JsonlTextSource
(one {"spoken": ..., "transcript": ...} object per line — "text" is accepted
as an alias filling both fields) or by implementing the protocol directly.

Records may also carry `weight` and `category` (02 §5). A source that exposes
its records as a pool (`records`) is sampled through `WeightedSampler`, which
honours those weights and tops categories up to `dataset.category_quotas`;
streaming sources keep being asked for one utterance at a time.
"""

import json
import random
from pathlib import Path
from typing import Protocol

from .grammar import Utterance, generate_utterance

DEFAULT_CATEGORY = "routine"


class TextSource(Protocol):
    def sample(self, rng: random.Random) -> Utterance: ...


class GrammarTextSource:
    """Built-in FAA/ICAO phraseology grammar."""

    def sample(self, rng: random.Random) -> Utterance:
        return generate_utterance(rng)


class JsonlTextSource:
    """Reads utterances from a JSONL file produced by any external script.

    Each line: {"spoken": str, "transcript": str, "role"?: str, "kind"?: str,
    "weight"?: float, "category"?: str} or simply {"text": str}. Sampled
    uniformly with replacement unless the builder wraps it in a
    `WeightedSampler`.
    """

    def __init__(self, path: str | Path):
        self.records: list[Utterance] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text")
                spoken = obj.get("spoken", text)
                transcript = obj.get("transcript", spoken)
                if not spoken:
                    raise ValueError(f"line missing 'spoken'/'text': {line[:80]}")
                weight = obj.get("weight", 1.0)
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                    raise ValueError(f"'weight' must be a positive number: {line[:80]}")
                self.records.append(Utterance(
                    spoken=spoken,
                    transcript=transcript,
                    role=obj.get("role", "unknown"),
                    kind=obj.get("kind", "external"),
                    weight=float(weight),
                    category=obj.get("category", DEFAULT_CATEGORY),
                ))
        if not self.records:
            raise ValueError(f"no utterances found in {path}")

    def sample(self, rng: random.Random) -> Utterance:
        return rng.choice(self.records)


class WeightedSampler:
    """Category-aware sampling over a pool of utterances (02 §5).

    Two independent draws per sample: first a category, then an utterance
    within it by `weight`. Categories named in `quotas` are drawn at their
    target fraction; the remaining probability mass goes to the other
    categories in proportion to their natural weight share, so an empty
    `quotas` reduces to plain weighted sampling. Quotas for categories the
    source does not provide are dropped and their mass redistributed — the
    top-up is best-effort, and `achieved()` reports what actually happened.
    """

    def __init__(self, records: list[Utterance], quotas: dict[str, float] | None = None):
        if not records:
            raise ValueError("WeightedSampler needs at least one utterance")
        self.pools: dict[str, list[Utterance]] = {}
        for record in records:
            self.pools.setdefault(record.category or DEFAULT_CATEGORY, []).append(record)
        self.cum_weights = {
            name: _cumulative([max(float(r.weight), 0.0) for r in pool])
            for name, pool in self.pools.items()
        }
        self.quotas = {name: frac for name, frac in (quotas or {}).items()
                       if name in self.pools}
        self.dropped_quotas = sorted(set(quotas or {}) - set(self.quotas))
        self.categories, self.category_weights = self._category_distribution()
        self.counts: dict[str, int] = {name: 0 for name in self.pools}

    @classmethod
    def for_source(cls, source: TextSource,
                   quotas: dict[str, float] | None = None) -> "WeightedSampler | None":
        """Wrap `source` when it exposes a record pool, else return None."""
        records = getattr(source, "records", None)
        if not records:
            return None
        return cls(list(records), quotas)

    def _category_distribution(self) -> tuple[list[str], list[float]]:
        natural = {name: sum(max(float(r.weight), 0.0) for r in pool)
                   for name, pool in self.pools.items()}
        quota_total = sum(self.quotas.values())
        scale = 1.0 / quota_total if quota_total > 1.0 else 1.0   # over-subscribed
        weights = {name: frac * scale for name, frac in self.quotas.items()}
        remaining = max(0.0, 1.0 - sum(weights.values()))
        rest = [name for name in self.pools if name not in self.quotas]
        rest_total = sum(natural[name] for name in rest)
        for name in rest:
            share = natural[name] / rest_total if rest_total > 0 else 1.0 / len(rest)
            weights[name] = remaining * share
        total = sum(weights.values()) or 1.0
        weights = {name: value / total for name, value in weights.items()}
        names = sorted(weights)
        return names, [weights[name] for name in names]

    def sample(self, rng: random.Random) -> Utterance:
        category = rng.choices(self.categories, weights=self.category_weights)[0]
        pool = self.pools[category]
        utterance = rng.choices(pool, cum_weights=self.cum_weights[category])[0]
        self.counts[category] += 1
        return utterance

    def achieved(self) -> dict[str, float]:
        """Fraction of samples drawn per category so far."""
        total = sum(self.counts.values())
        return {name: (count / total if total else 0.0)
                for name, count in sorted(self.counts.items())}


def _cumulative(weights: list[float]) -> list[float]:
    total = 0.0
    out = []
    for weight in weights:
        total += weight
        out.append(total)
    return out if total > 0 else [float(i + 1) for i in range(len(weights))]


def make_text_source(spec: str) -> TextSource:
    """'grammar' -> built-in; anything else is treated as a JSONL path."""
    if spec == "grammar":
        return GrammarTextSource()
    return JsonlTextSource(spec)
