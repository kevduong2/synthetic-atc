"""Pluggable text sources for the dataset builder.

A TextSource is anything with `sample(rng) -> Utterance`. The built-in grammar
is one implementation; teammates' generators can plug in via JsonlTextSource
(one {"spoken": ..., "transcript": ...} object per line — "text" is accepted
as an alias filling both fields) or by implementing the protocol directly.
"""

import json
import random
from pathlib import Path
from typing import Protocol

from .grammar import Utterance, generate_utterance


class TextSource(Protocol):
    def sample(self, rng: random.Random) -> Utterance: ...


class GrammarTextSource:
    """Built-in FAA/ICAO phraseology grammar."""

    def sample(self, rng: random.Random) -> Utterance:
        return generate_utterance(rng)


class JsonlTextSource:
    """Reads utterances from a JSONL file produced by any external script.

    Each line: {"spoken": str, "transcript": str, "role"?: str, "kind"?: str}
    or simply {"text": str}. Samples uniformly with replacement.
    """

    def __init__(self, path: str | Path):
        self.records = []
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
                self.records.append(Utterance(
                    spoken=spoken,
                    transcript=transcript,
                    role=obj.get("role", "unknown"),
                    kind=obj.get("kind", "external"),
                ))
        if not self.records:
            raise ValueError(f"no utterances found in {path}")

    def sample(self, rng: random.Random) -> Utterance:
        return rng.choice(self.records)


def make_text_source(spec: str) -> TextSource:
    """'grammar' -> built-in; anything else is treated as a JSONL path."""
    if spec == "grammar":
        return GrammarTextSource()
    return JsonlTextSource(spec)
