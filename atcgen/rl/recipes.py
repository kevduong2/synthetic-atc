"""Named data recipes: the discrete action space of the L3 bandit.

research-findings §4.7 gives L3 an action space over (scenario class, accent,
voice, rate, generator branch, SNR, channel condition, interference, entity
type, difficulty tier).  The full product is far larger than a loop whose pull
costs a minute of generation plus a minute of ASR can explore, so it is
collapsed to twelve *named buckets*.  Each bucket moves one or two of those
axes away from ``configs/mode1_matched.yaml`` and leaves the rest of the
fitted profile alone — the same discipline `atcgen.rl.space` applies to the
config search, for the same reason: the matched profile was fitted against
measured statistics and is not the loop's to relitigate.

A recipe is two things:

*   **Config overrides** — dotted paths into the raw profile mapping.  A path
    beginning ``chain.`` addresses a chain step by primitive name
    (``chain.additive_noise.snr_db``) rather than by list index, so a chain
    reorder does not silently retarget a recipe.  Values replace the whole
    node, which lets a recipe swap an entire distribution spec.
*   **A text source** — a `make_text_source` spec string, optionally narrowed
    by rejection sampling to a set of utterance kinds or categories.  The
    scenario knobs (region, readback errors, confusable callsigns, phonetic
    respelling) are the scenario-class and difficulty axes; the kind filter is
    the entity-type axis, which the grammar cannot express through
    ``dataset.category_quotas`` because a streaming source has no record pool
    for `WeightedSampler` to weight.

Every bucket is meant to stay inside the measured envelope
(`configs/real_envelope.json`); `check_recipe` reports where one does not.
That check warns and never fails — §4.3's cap is a report, and a bucket that
deliberately probes the edge of intelligibility is exactly what the hardness
window is there to filter.
"""

from __future__ import annotations

import copy
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..channel.envelope import check_profile, load_envelope
from ..config import load_config
from ..text.grammar import Utterance
from ..text.sources import TextSource, make_text_source

#: Utterance kinds whose transcripts are mostly numbers — the entity-type axis.
NUMERIC_KINDS = (
    "climb", "climb_readback", "descend", "descend_readback",
    "altitude", "altitude_readback", "heading", "heading_readback",
    "speed", "speed_readback", "qnh", "qnh_readback",
    "altimeter", "altimeter_readback", "squawk", "squawk_readback",
    "freq_change", "freq_change_readback", "level_report",
)


class FilteredTextSource:
    """Rejection-samples an inner source down to a set of kinds/categories.

    The grammar is a streaming source: it hands back one utterance drawn from a
    freshly generated exchange, so there is no pool to filter up front and no
    way to ask it for a particular template.  Rejection sampling is the honest
    way to bias it, and it is cheap — composing an exchange is pure Python.

    After ``max_tries`` misses the last draw is returned anyway rather than
    raising: a bucket that starves is a bucket the bandit should learn to stop
    pulling, not a crash mid-run.  ``rejected``/``accepted`` record how often
    that happened so a starving filter shows up in the pull log.

    ``max_tries`` is set for the *rarest* bucket rather than the typical one.
    The emergency/ILS templates carry a combined weight of about 2% in
    `EU_EXCHANGES`/`US_EXCHANGES`, so 64 tries would leak an off-target
    utterance 27% of the time; 512 brings that to 3e-5.  It is affordable
    because composing an exchange costs ~265 us — a whole pull's worth of
    rejection sampling is under a second against a minute of TTS.
    """

    def __init__(self, inner: TextSource, *, kinds: Sequence[str] = (),
                 categories: Sequence[str] = (), max_tries: int = 512) -> None:
        if not kinds and not categories:
            raise ValueError("FilteredTextSource needs kinds or categories")
        self.inner = inner
        self.kinds = frozenset(kinds)
        self.categories = frozenset(categories)
        self.max_tries = int(max_tries)
        self.accepted = 0
        self.rejected = 0

    def matches(self, utterance: Utterance) -> bool:
        if self.kinds and utterance.kind in self.kinds:
            return True
        return bool(self.categories) and utterance.category in self.categories

    def sample(self, rng: random.Random) -> Utterance:
        utterance = self.inner.sample(rng)
        for _ in range(self.max_tries):
            if self.matches(utterance):
                self.accepted += 1
                return utterance
            self.rejected += 1
            utterance = self.inner.sample(rng)
        return utterance

    @property
    def hit_rate(self) -> float:
        total = self.accepted + self.rejected
        return self.accepted / total if total else 0.0


# --------------------------------------------------------------------------
# override plumbing
# --------------------------------------------------------------------------


def _chain_step(config: Mapping[str, Any], primitive: str) -> dict:
    """The ``channel.chain`` entry with this primitive, addressed by name."""
    channel = config.get("channel")
    chain = channel.get("chain") if isinstance(channel, Mapping) else None
    if not isinstance(chain, list):
        raise KeyError("config has no channel.chain list")
    for step in chain:
        if isinstance(step, Mapping) and step.get("primitive") == primitive:
            return step  # type: ignore[return-value]
    present = sorted(str(s.get("primitive")) for s in chain if isinstance(s, Mapping))
    raise KeyError(f"channel.chain has no {primitive!r} step; present: {present}")


def set_path(config: dict, path: str, value: Any) -> None:
    """Write ``value`` at a dotted ``path``, mutating ``config`` in place.

    ``chain.<primitive>.<param>`` addresses a chain step by primitive name;
    every other path walks plain mappings.  Intermediate keys must already
    exist — a recipe that invents a config section is a typo, not a feature.
    """
    parts = path.split(".")
    if parts[0] == "chain":
        if len(parts) < 3:
            raise KeyError(f"chain path needs primitive and param: {path!r}")
        node: Any = _chain_step(config, parts[1])
        parts = parts[2:]
    else:
        node = config
    for depth, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config path {'.'.join(parts[:depth + 1])!r} is missing "
                           f"(override path {path!r})")
        node = node[part]
    if not isinstance(node, dict):
        raise KeyError(f"config path {path!r} does not resolve inside a mapping")
    node[parts[-1]] = value


@dataclass(frozen=True)
class Recipe:
    """One bucket: config overrides plus the text source that feeds them."""

    name: str
    axis: str            # which §4.7 axis this bucket moves, for the log
    description: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    text_spec: str = "grammar"
    kinds: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()

    def apply(self, base_config: Mapping[str, Any]) -> dict:
        """Deep-copy ``base_config`` and apply the overrides; base untouched."""
        config = copy.deepcopy(dict(base_config))
        for path, value in self.overrides.items():
            set_path(config, path, copy.deepcopy(value))
        return config

    def text_source(self) -> TextSource:
        """A fresh source per call — `FilteredTextSource` carries counters."""
        source = make_text_source(self.text_spec)
        if self.kinds or self.categories:
            return FilteredTextSource(source, kinds=self.kinds,
                                      categories=self.categories)
        return source


# --------------------------------------------------------------------------
# the buckets
# --------------------------------------------------------------------------
#
# SNR bands are quoted in *config* space and checked against the envelope's
# config-space bounds (real p10-p90 shifted by the rule's measured offset:
# [0.1, 22.6] dB with 5 dB of slack, so [-4.9, 27.6] draws clean).  Bandpass
# corners likewise: [2792, 3700] Hz +/- 300.

_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "eu_routine", "scenario",
        "ICAO/European phraseology at the matched profile — the anchor bucket.",
        text_spec="grammar:region=eu",
    ),
    Recipe(
        "eu_fast_speech", "rate",
        "European phraseology read fast; rate is the talker-side control that "
        "most reliably moves ASR error.",
        overrides={"tts.speed": {"uniform": [1.30, 1.55]}},
        text_spec="grammar:region=eu",
    ),
    Recipe(
        "eu_readback_errors", "difficulty",
        "Pilot reads a value back wrong and the controller corrects it; the "
        "label follows the audio, so the wrong readback is labelled as spoken.",
        text_spec="grammar:region=eu,readback_error_prob=0.25",
    ),
    Recipe(
        "eu_confusable_callsigns", "difficulty",
        "Two callsigns one digit apart inside one transmission.",
        text_spec="grammar:region=eu,confusable_callsign_prob=0.35",
    ),
    Recipe(
        "us_routine", "scenario",
        "FAA tower/approach phraseology — the accent/phraseology gap "
        "(WhisperATC degrades 13.5% -> 30.3% crossing it).",
        text_spec="grammar:region=us",
    ),
    Recipe(
        "mixed_phonetic_respell", "difficulty",
        "Radio variants (niner/tree/fife/fower) and grouped digits every time, "
        "both regions.",
        text_spec="grammar:region=mixed,phonetic_respelling_prob=1.0",
    ),
    Recipe(
        "low_snr", "snr",
        "Noise floor skewed hard: injected SNR 0-14 dB, still inside the "
        "measured envelope.",
        overrides={"chain.additive_noise.snr_db": {"beta_scaled": [2.0, 2.5, 0, 14]}},
    ),
    Recipe(
        "high_snr_clean", "snr",
        "The easy end: injected SNR 15-27 dB with receiver artifacts backed "
        "off. The control bucket — it should mostly fall below the window.",
        overrides={
            "chain.additive_noise.snr_db": {"beta_scaled": [2.0, 1.5, 15, 27]},
            "chain.crackle.prob": 0.15,
            "chain.dropouts.prob": 0.02,
            "chain.cochannel_mix.prob": 0.0,
            "chain.am_distortion.depth": {"uniform": [0.0, 0.06]},
            "chain.soft_clip.drive": {"uniform": [1.0, 1.6]},
            "chain.codec_roundtrip.prob": 0.4,
        },
    ),
    Recipe(
        "dense_numerics", "entity_type",
        "Number-heavy transmissions only (levels, headings, speeds, squawks, "
        "frequencies) — where a single substitution is operationally serious.",
        text_spec="grammar:region=mixed",
        kinds=NUMERIC_KINDS,
    ),
    Recipe(
        "noise_heavy_channel", "channel",
        "Receiver-side artifacts up: crackle, dropouts, co-channel, hum, "
        "fading, heterodyne. SNR left at matched so this isolates the "
        "artifact axis from the noise-floor one.",
        overrides={
            "chain.crackle.prob": 0.95,
            "chain.crackle.rate": {"uniform": [1.0, 6.0]},
            "chain.dropouts.prob": 0.40,
            "chain.cochannel_mix.prob": 0.25,
            "chain.cochannel_mix.level": {"uniform": [0.08, 0.20]},
            "chain.hum.prob": 0.60,
            "chain.fading.prob": 0.35,
            "chain.heterodyne.prob": 0.20,
        },
    ),
    Recipe(
        "narrowband_codec", "channel",
        "Delivery-path degradation: low-bitrate codec every clip, extra "
        "resample hops, band tightened toward the low edge of the envelope.",
        overrides={
            "chain.codec_roundtrip.prob": 1.0,
            "chain.codec_roundtrip.bitrate_kbps": {"choice": [23, 23, 32]},
            "chain.resample_chain.prob": 0.35,
            "chain.narrowband_roundtrip.prob": 1.0,
            "chain.narrowband_roundtrip.narrow_sr": {"choice": [6000, 6000, 8000]},
            "chain.bandpass.high": {"uniform": [2900, 3200]},
        },
    ),
    Recipe(
        "rare_and_emergency", "scenario",
        "ILS/approach vocabulary and emergency exchanges — the rare-entity end "
        "of §4.7's curriculum.",
        text_spec="grammar:region=mixed",
        categories=("rare_vocab", "emergency"),
    ),
)

RECIPES: dict[str, Recipe] = {recipe.name: recipe for recipe in _RECIPES}


def write_config(raw: Mapping[str, Any], path: str | Path):
    """Dump a raw config mapping to ``path`` and load it back, validated.

    The same write-then-load audit trail `render_and_finetune` keeps: the file
    left behind is exactly the config that generated the clips next to it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(raw), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return load_config(path)


def check_recipe(recipe: Recipe, base_config: Mapping[str, Any],
                 envelope: Mapping[str, Any] | None = None) -> list[str]:
    """Envelope findings for a recipe's config — a report, never a failure.

    Parses the config through `load_config` on the way, so a bucket that writes
    a malformed distribution is caught here rather than a minute into a pull.
    """
    with tempfile.TemporaryDirectory() as tmp:
        parsed = write_config(recipe.apply(base_config), Path(tmp) / "recipe.yaml")
    envelope = envelope if envelope is not None else load_envelope()
    if envelope is None:
        return []
    return check_profile(parsed, envelope)


__all__ = ["NUMERIC_KINDS", "RECIPES", "FilteredTextSource", "Recipe",
           "check_recipe", "set_path", "write_config"]
