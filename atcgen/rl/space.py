"""Declarative search space: the unit cube [0,1]^d -> generator-config mutations.

The optimizer never sees YAML.  It proposes points in a fixed-dimension unit
cube and this module turns each point into a raw config mapping that
``atcgen.config.load_config`` will accept.  Two properties matter:

*   **The base profile stays authoritative.**  ``to_config`` deep-copies the
    hand-tuned profile and edits only the knobbed leaves, so everything the
    search does not cover (voice list, chain order, noise beds) keeps its
    curated value instead of being re-derived from scratch.
*   **The hand-tuned config is a point in the cube.**  ``default_vector``
    inverts each knob against the base profile, which lets the loop evaluate
    the human's setting as trial 0 and lets the optimizer start there rather
    than in the middle of a range nobody chose.

Knob ranges are deliberately narrow.  The reward is a whisper-tiny fine-tune
run, so the budget is tens of evaluations for ~19 dimensions; a wide range
spends that budget rediscovering that the profile was roughly right.  Every
range below brackets the base profile's value rather than the physically
defensible extreme.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_KINDS = {"linear", "log", "prob"}

# Which entries of a distribution payload are the (lower, upper) bounds pair.
# ``beta_scaled`` is [alpha, beta, low, high]; ``uniform`` is [low, high].
_BOUND_INDEX = {"uniform": (0, 1), "beta_scaled": (2, 3)}


@dataclass(frozen=True)
class Knob:
    """One search dimension: a unit value in [0,1] mapped onto ``[lo, hi]``.

    ``kind`` selects the mapping: ``linear`` interpolates, ``log`` interpolates
    geometrically (both bounds must be positive), and ``prob`` is linear but
    marks the target as a probability so logs and clamps can treat it as one.

    ``apply`` writes the concrete value into a config mapping; ``read`` pulls
    the current value back out for ``SearchSpace.default_vector`` and returns
    ``None`` when the base profile does not carry that leaf.
    """

    name: str
    lo: float
    hi: float
    kind: str
    apply: Callable[[dict, float], None]
    read: Callable[[Mapping[str, Any]], float | None] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"knob {self.name}: kind must be one of {sorted(_KINDS)}")
        if self.hi <= self.lo:
            raise ValueError(f"knob {self.name}: hi must exceed lo")
        if self.kind == "log" and self.lo <= 0:
            raise ValueError(f"knob {self.name}: log knobs need a positive lower bound")

    def value(self, unit: float) -> float:
        """Concrete value for a unit-cube coordinate (clipped into [0,1])."""
        unit = min(1.0, max(0.0, float(unit)))
        if self.kind == "log":
            return float(self.lo * (self.hi / self.lo) ** unit)
        return float(self.lo + unit * (self.hi - self.lo))

    def unit(self, value: float) -> float:
        """Inverse of :meth:`value`, clipped so out-of-range bases still map in."""
        value = float(value)
        if self.kind == "log":
            if value <= 0:
                return 0.0
            raw = math.log(value / self.lo) / math.log(self.hi / self.lo)
        else:
            raw = (value - self.lo) / (self.hi - self.lo)
        return float(min(1.0, max(0.0, raw)))


# --------------------------------------------------------------------------
# path plumbing
# --------------------------------------------------------------------------


def _walk(config: Mapping[str, Any], path: str) -> tuple[dict, str]:
    """Resolve a dotted path to its (parent mapping, leaf key)."""
    parts = path.split(".")
    node: Any = config
    for depth, part in enumerate(parts[:-1]):
        if not isinstance(node, Mapping) or part not in node:
            prefix = ".".join(parts[: depth + 1])
            raise KeyError(f"config path {prefix!r} is missing (knob path {path!r})")
        node = node[part]
    if not isinstance(node, Mapping):
        raise KeyError(f"config path {path!r} does not resolve inside a mapping")
    return node, parts[-1]  # type: ignore[return-value]


def _chain_step(config: Mapping[str, Any], primitive: str) -> dict:
    """Find the ``channel.chain`` entry whose ``primitive`` is ``primitive``.

    The chain is a list, not a mapping, so steps are addressed by name here;
    that keeps knob declarations stable when the chain order changes.
    """
    channel = config.get("channel")
    chain = channel.get("chain") if isinstance(channel, Mapping) else None
    if not isinstance(chain, list):
        raise KeyError("config has no channel.chain list")
    for step in chain:
        if isinstance(step, Mapping) and step.get("primitive") == primitive:
            return step  # type: ignore[return-value]
    present = sorted(
        str(step.get("primitive")) for step in chain if isinstance(step, Mapping)
    )
    raise KeyError(
        f"channel.chain has no step with primitive {primitive!r}; present: {present}"
    )


def _dist_payload(spec: Any, where: str) -> tuple[str, list]:
    """Return the (kind, payload list) of a ``{prob?, uniform|beta_scaled}`` dict.

    A chain param may be a bare scalar, a ``{"uniform": [...]}`` mapping, or the
    same mapping with a ``prob`` gate sitting next to the payload key, so the
    payload has to be looked up by kind rather than positionally.
    """
    if not isinstance(spec, Mapping):
        raise KeyError(f"{where} is not a distribution mapping")
    for kind in _BOUND_INDEX:
        payload = spec.get(kind)
        if isinstance(payload, list):
            return kind, payload
    raise KeyError(
        f"{where} has no uniform/beta_scaled payload (keys: {sorted(map(str, spec))})"
    )


def _set_bound(spec: Any, index: int, value: float, where: str) -> None:
    """Write one bound of a distribution payload, keeping lower <= upper.

    A knob that pushes the lower bound past the upper one would make
    ``DistSpec.parse`` reject the config, so the pair is simply sorted after the
    write.  The optimizer sees a squashed range rather than a crash, which is
    the right failure mode when the two bounds are separate search dimensions.
    """
    kind, payload = _dist_payload(spec, where)
    if not 0 <= index < len(payload):
        raise KeyError(f"{where}.{kind} has no index {index}")
    payload[index] = float(value)
    low_i, high_i = _BOUND_INDEX[kind]
    if payload[low_i] > payload[high_i]:
        payload[low_i], payload[high_i] = payload[high_i], payload[low_i]


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _reader(fn: Callable[[Mapping[str, Any]], Any]) -> Callable[[Mapping[str, Any]], float | None]:
    """Wrap a lookup so a base profile missing that leaf reports ``None``."""

    def read(config: Mapping[str, Any]) -> float | None:
        try:
            return _numeric(fn(config))
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    return read


# --------------------------------------------------------------------------
# knob constructors
# --------------------------------------------------------------------------


def scalar_knob(name: str, path: str, lo: float, hi: float, kind: str = "linear") -> Knob:
    """Knob over a plain scalar at a dotted path, e.g. ``dataset.noise_only_frac``."""

    def apply(config: dict, value: float) -> None:
        parent, leaf = _walk(config, path)
        parent[leaf] = float(value)

    def get(config: Mapping[str, Any]) -> Any:
        parent, leaf = _walk(config, path)
        return parent[leaf]

    return Knob(name, lo, hi, kind, apply, _reader(get))


def dist_bound_knob(
    name: str, path: str, index: int, lo: float, hi: float, kind: str = "linear"
) -> Knob:
    """Knob over one bound of a distribution at a dotted path (e.g. ``tts.speed``)."""

    def apply(config: dict, value: float) -> None:
        parent, leaf = _walk(config, path)
        _set_bound(parent.get(leaf), index, value, path)

    def get(config: Mapping[str, Any]) -> Any:
        parent, leaf = _walk(config, path)
        return _dist_payload(parent.get(leaf), path)[1][index]

    return Knob(name, lo, hi, kind, apply, _reader(get))


def dist_prob_knob(name: str, path: str, lo: float = 0.0, hi: float = 1.0) -> Knob:
    """Knob over the ``prob`` gate of a distribution mapping at a dotted path."""

    def apply(config: dict, value: float) -> None:
        parent, leaf = _walk(config, path)
        spec = parent.get(leaf)
        if not isinstance(spec, dict):
            raise KeyError(f"{path} is not a distribution mapping; cannot set prob")
        spec["prob"] = float(value)

    def get(config: Mapping[str, Any]) -> Any:
        parent, leaf = _walk(config, path)
        spec = parent.get(leaf)
        return spec.get("prob", 1.0) if isinstance(spec, Mapping) else None

    return Knob(name, lo, hi, "prob", apply, _reader(get))


def chain_prob_knob(name: str, primitive: str, lo: float = 0.0, hi: float = 1.0) -> Knob:
    """Knob over ``channel.chain[primitive].prob`` — how often the step fires."""

    def apply(config: dict, value: float) -> None:
        _chain_step(config, primitive)["prob"] = float(value)

    return Knob(
        name, lo, hi, "prob", apply,
        _reader(lambda c: _chain_step(c, primitive).get("prob", 1.0)),
    )


def chain_param_knob(
    name: str, primitive: str, param: str, index: int, lo: float, hi: float,
    kind: str = "linear",
) -> Knob:
    """Knob over one bound of a chain step's distribution parameter.

    ``index`` is the raw payload index: 0/1 for ``uniform``, 2/3 for the
    ``beta_scaled`` low/high pair.
    """

    def apply(config: dict, value: float) -> None:
        step = _chain_step(config, primitive)
        _set_bound(step.get(param), index, value, f"channel.chain[{primitive}].{param}")

    def get(config: Mapping[str, Any]) -> Any:
        step = _chain_step(config, primitive)
        where = f"channel.chain[{primitive}].{param}"
        return _dist_payload(step.get(param), where)[1][index]

    return Knob(name, lo, hi, kind, apply, _reader(get))


def chain_scalar_knob(
    name: str, primitive: str, param: str, lo: float, hi: float, kind: str = "linear"
) -> Knob:
    """Knob over a chain step's scalar parameter, e.g. ``additive_noise.bed_prob``."""

    def apply(config: dict, value: float) -> None:
        _chain_step(config, primitive)[param] = float(value)

    return Knob(
        name, lo, hi, kind, apply,
        _reader(lambda c: _chain_step(c, primitive).get(param)),
    )


def chain_center_knob(
    name: str, primitive: str, param: str, lo: float, hi: float
) -> Knob:
    """Knob that slides a uniform range's midpoint while holding its width.

    Used where the *position* of a band edge matters but its jitter width was
    already chosen deliberately — spending two dimensions on the two bounds
    would let the optimizer collapse the jitter as a side effect.
    """

    where = f"channel.chain[{primitive}].{param}"

    def bounds(spec: Any) -> tuple[list, int, int]:
        kind, payload = _dist_payload(spec, where)
        low_i, high_i = _BOUND_INDEX[kind]
        return payload, low_i, high_i

    def apply(config: dict, value: float) -> None:
        payload, low_i, high_i = bounds(_chain_step(config, primitive).get(param))
        half = (payload[high_i] - payload[low_i]) / 2.0
        payload[low_i] = float(value) - half
        payload[high_i] = float(value) + half

    def get(config: Mapping[str, Any]) -> Any:
        payload, low_i, high_i = bounds(_chain_step(config, primitive).get(param))
        return (payload[low_i] + payload[high_i]) / 2.0

    return Knob(name, lo, hi, "linear", apply, _reader(get))


def tts_speed_knobs() -> list[Knob]:
    """The speaking-rate range: a slow edge and a fast edge, searched separately.

    Rate is the one talker-side control that reliably moves ASR error, and the
    two edges do different jobs — the slow edge sets how much easy audio the
    fine-tune sees, the fast edge how much of the hard tail.
    """
    return [
        dist_bound_knob("tts.speed_lo", "tts.speed", 0, 0.9, 1.2),
        dist_bound_knob("tts.speed_hi", "tts.speed", 1, 1.2, 1.6),
    ]


# --------------------------------------------------------------------------
# the space
# --------------------------------------------------------------------------


class SearchSpace:
    """An ordered list of :class:`Knob` addressing a fixed-dimension unit cube."""

    def __init__(self, knobs: Sequence[Knob]):
        names = [knob.name for knob in knobs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate knob names: {duplicates}")
        self.knobs: list[Knob] = list(knobs)

    @property
    def dim(self) -> int:
        return len(self.knobs)

    def _values(self, vector: np.ndarray | Sequence[float]) -> list[float]:
        array = np.asarray(vector, dtype=float).reshape(-1)
        if array.size != self.dim:
            raise ValueError(f"vector has {array.size} entries, space has {self.dim}")
        return [knob.value(unit) for knob, unit in zip(self.knobs, array)]

    def to_config(self, base: Mapping[str, Any], vector: np.ndarray | Sequence[float]) -> dict:
        """Deep-copy ``base`` and apply every knob; ``base`` is never mutated."""
        config = copy.deepcopy(dict(base))
        for knob, value in zip(self.knobs, self._values(vector)):
            knob.apply(config, value)
        return config

    def describe(self, vector: np.ndarray | Sequence[float]) -> dict[str, float]:
        """Knob name -> concrete value, for the ``Trial.overrides`` log."""
        return dict(zip((knob.name for knob in self.knobs), self._values(vector)))

    def default_vector(self, base: Mapping[str, Any]) -> np.ndarray:
        """Best-effort inverse: where the hand-tuned ``base`` sits in the cube.

        Knobs whose leaf is absent from ``base`` fall back to 0.5 rather than
        failing, so a partially-specified profile still yields a usable anchor.
        """
        units = []
        for knob in self.knobs:
            current = knob.read(base) if knob.read is not None else None
            units.append(0.5 if current is None else knob.unit(current))
        return np.asarray(units, dtype=float)


def default_atc_space() -> SearchSpace:
    """The curated space for ``configs/mode1_matched.yaml``.

    Nineteen knobs across the four stages that plausibly move downstream WER:
    how noisy and how band-limited the channel is, how hard the receiver-side
    artifacts hit, how fast and how varied the talker is, and what fraction of
    the batch is non-speech or multi-hop.  Everything else in the profile —
    chain order, voice list, noise beds, the fitted squelch threshold — is left
    alone; those were fitted against measured statistics in Tier 1 and are not
    the loop's to relitigate.
    """
    return SearchSpace([
        # -- channel noise floor ------------------------------------------
        # The single strongest lever: the SNR beta's low and high edges set how
        # much of the batch is genuinely hard.
        chain_param_knob("additive_noise.snr_lo", "additive_noise", "snr_db", 2, 0.0, 15.0),
        chain_param_knob("additive_noise.snr_hi", "additive_noise", "snr_db", 3, 15.0, 35.0),
        # Harvested real beds vs synthetic pink/white.
        chain_scalar_knob("additive_noise.bed_prob", "additive_noise", "bed_prob", 0.0, 1.0,
                          kind="prob"),

        # -- band limiting -------------------------------------------------
        chain_param_knob("bandpass.high_hi", "bandpass", "high", 1, 2400.0, 3400.0),
        chain_center_knob("bandpass.low_center", "bandpass", "low", 180.0, 380.0),

        # -- receiver-side artifacts --------------------------------------
        chain_prob_knob("codec_roundtrip.prob", "codec_roundtrip", 0.3, 1.0),
        chain_prob_knob("squelch_gate.prob", "squelch_gate", 0.3, 1.0),
        chain_param_knob("am_distortion.depth_hi", "am_distortion", "depth", 1, 0.0, 0.35),
        chain_param_knob("soft_clip.drive_hi", "soft_clip", "drive", 1, 1.5, 4.0),
        chain_prob_knob("crackle.prob", "crackle", 0.0, 1.0),
        chain_prob_knob("dropouts.prob", "dropouts", 0.0, 0.4),
        chain_prob_knob("cochannel_mix.prob", "cochannel_mix", 0.0, 0.25),

        # -- talker --------------------------------------------------------
        *tts_speed_knobs(),
        dist_prob_knob("voice_augment.pitch_prob", "voice_augment.pitch_semitones", 0.0, 0.8),
        dist_prob_knob("voice_augment.tempo_prob", "voice_augment.tempo", 0.0, 0.8),

        # -- batch composition ---------------------------------------------
        scalar_knob("dataset.noise_only_frac", "dataset.noise_only_frac", 0.0, 0.10,
                    kind="prob"),
        scalar_knob("dataset.pilot_double_hop_prob", "dataset.pilot_double_hop_prob",
                    0.0, 0.8, kind="prob"),
        scalar_knob("channel.clean_arm_prob", "channel.clean_arm_prob", 0.0, 0.10,
                    kind="prob"),
    ])
