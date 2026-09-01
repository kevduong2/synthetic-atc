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

_KINDS = {"linear", "log", "prob", "choice"}

# Which entries of a distribution payload are the (lower, upper) bounds pair.
# ``beta_scaled`` is [alpha, beta, low, high]; ``uniform`` is [low, high].
_BOUND_INDEX = {"uniform": (0, 1), "beta_scaled": (2, 3)}


@dataclass(frozen=True)
class Knob:
    """One search dimension: a unit value in [0,1] mapped onto ``[lo, hi]``.

    ``kind`` selects the mapping: ``linear`` interpolates, ``log`` interpolates
    geometrically (both bounds must be positive), ``prob`` is linear but marks
    the target as a probability so logs and clamps can treat it as one, and
    ``choice`` snaps the cube coordinate onto one of ``values`` -- a
    categorical arm rather than a continuous range, for a knob whose
    interesting settings are "off" and "what the profile already says" with
    nothing meaningful in between.

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
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"knob {self.name}: kind must be one of {sorted(_KINDS)}")
        if self.kind == "choice":
            if self.values is None or len(set(self.values)) < 2:
                raise ValueError(
                    f"knob {self.name}: choice knobs need two or more distinct values")
        elif self.values is not None:
            raise ValueError(f"knob {self.name}: values is only for choice knobs")
        if self.hi <= self.lo:
            raise ValueError(f"knob {self.name}: hi must exceed lo")
        if self.kind == "log" and self.lo <= 0:
            raise ValueError(f"knob {self.name}: log knobs need a positive lower bound")

    def value(self, unit: float) -> float:
        """Concrete value for a unit-cube coordinate (clipped into [0,1])."""
        unit = min(1.0, max(0.0, float(unit)))
        if self.kind == "choice":
            index = min(int(unit * len(self.values)), len(self.values) - 1)
            return float(self.values[index])
        if self.kind == "log":
            return float(self.lo * (self.hi / self.lo) ** unit)
        return float(self.lo + unit * (self.hi - self.lo))

    def unit(self, value: float) -> float:
        """Inverse of :meth:`value`, clipped so out-of-range bases still map in."""
        value = float(value)
        if self.kind == "choice":
            # the centre of the winning cell, so `default_vector` lands on the
            # arm the profile already sets rather than on a cell boundary
            index = min(range(len(self.values)),
                        key=lambda i: abs(self.values[i] - value))
            return (index + 0.5) / len(self.values)
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


def _dist_prob_accessors(path: str):
    """The (apply, read) pair for a distribution's ``prob`` gate at ``path``."""

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

    return apply, _reader(get)


def dist_prob_knob(name: str, path: str, lo: float = 0.0, hi: float = 1.0) -> Knob:
    """Knob over the ``prob`` gate of a distribution mapping at a dotted path."""
    apply, read = _dist_prob_accessors(path)
    return Knob(name, lo, hi, "prob", apply, read)


def dist_prob_choice_knob(name: str, path: str, values: Sequence[float]) -> Knob:
    """Categorical knob over a distribution's ``prob`` gate: one of ``values``.

    For a gate whose interesting settings are a short list rather than a range
    — "off" against "whatever the profile says" — where a continuous sweep
    would spend the budget resolving a difference the reward cannot see.
    """
    apply, read = _dist_prob_accessors(path)
    values = tuple(float(value) for value in values)
    return Knob(name, min(values), max(values), "choice", apply, read, values)


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
        # The matched profile's cascade corners were re-fitted to 3200-3400 Hz
        # in P4, which put this knob's ceiling exactly on the profile value —
        # the search could only ever narrow the band.  Re-ranged to bracket it
        # instead; the top sits on the envelope's own config-space p90 for
        # `spectral_edge_hz` (3700 Hz), so the widest draw is still in-envelope.
        chain_param_knob("bandpass.high_hi", "bandpass", "high", 1, 2700.0, 3700.0),
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


#: The `voice_augment.pitch_semitones` gate the shipped profiles carry, and the
#: fallback arm for `talker_only_space` when a base profile does not declare one.
DEFAULT_PITCH_PROB = 0.5


def mode2_safe_space(base: Mapping[str, Any] | None = None) -> SearchSpace:
    """The full curated space for a Mode 2 (`calibrated`) profile.

    ``default_atc_space`` is Mode 1 only: twelve of its knobs address
    ``channel.chain`` steps and a thirteenth ``channel.clean_arm_prob``, none of
    which a calibrated profile has, so resolving it against one raises before
    a single clip is rendered.  Every knob here resolves *and* renders on a
    Mode 2 config; ``tests/test_rl_space.py`` proves that by rendering from
    sampled vectors rather than by asserting the paths exist.

    Fifteen dimensions is too many for a twenty-five-trial budget -- use
    ``talker_only_space`` for a short run.  This space is for a search with
    room to move, against calibration artifacts that have actually been fitted:
    the residual knobs below are inert while ``calibrated.residual.enabled`` is
    false, and reading them as dead dimensions is the expected outcome, not a
    bug.

    The band shape, drive, AGC and noise floor are not knobs at all in this
    mode -- they are one real receiver's *measured* values, drawn per utterance
    out of the preset pool, and re-deriving them from a search would throw away
    the thing that makes Mode 2 Mode 2.  What is left to search is the three
    places the profile still guesses:

    *   **How far the drawn SNR may wander from the measured one**
        (``snr_jitter_db``).  This is the one lever over the fitted channel,
        and it is the Mode 2 analogue of Mode 1's ``additive_noise.snr_*``
        pair: how much of the batch is genuinely hard.
    *   **The receiving station's event artifacts** -- squelch, dropouts, the
        delivery codec.  A preset cannot produce these (they are events, not
        stationary channel properties), so ``post_effects`` declares them at
        probabilities copied from ``mode1_matched.yaml`` rather than fitted.
    *   **The learned residual's dose** -- how often it fires and how far it
        may move the waveform.  Both are policy, not measurement: the FastCUT
        S2 run trained at ``--residual-scale-max 0.20`` while the config-side
        default is 0.35, and ``apply_prob`` deliberately keeps a pure-DSP
        share in every corpus (04 §2.4).

    Plus the four talker and two batch-composition knobs that are
    backend-agnostic and already work in both modes.

    Not searched: ``cross_station_prob``, which is a no-op on a single-station
    calibration such as KIXD (there is no other station's hiss to borrow), and
    ``station_mix``, which is a mapping rather than a scalar.
    """
    return SearchSpace([
        # -- the fitted channel's one degree of freedom ---------------------
        # Added to each preset's own measured `snr_est`; the two edges are
        # searched separately for the same reason Mode 1's SNR beta's are.
        dist_bound_knob("calibration.snr_jitter_lo",
                        "calibrated.calibration.snr_jitter_db", 0, -9.0, -1.0),
        dist_bound_knob("calibration.snr_jitter_hi",
                        "calibrated.calibration.snr_jitter_db", 1, 1.0, 9.0),

        # -- receiving-station artifacts (the post-effects block) -----------
        scalar_knob("post_effects.squelch.prob",
                    "calibrated.post_effects.squelch.prob", 0.3, 1.0, kind="prob"),
        # The measured gated fraction is 0.045; the range brackets it rather
        # than reaching for the physically defensible extreme.
        scalar_knob("post_effects.squelch.gated_floor_prob",
                    "calibrated.post_effects.squelch.gated_floor_prob",
                    0.0, 0.20, kind="prob"),
        scalar_knob("post_effects.dropouts.prob",
                    "calibrated.post_effects.dropouts.prob", 0.0, 0.40, kind="prob"),
        scalar_knob("post_effects.codec.prob",
                    "calibrated.post_effects.codec.prob", 0.3, 1.0, kind="prob"),
        # libsndfile's compression_level runs 0 (best) to 1 (worst): the
        # profile's [0.75, 0.95] is roughly 32 down to 16 kbps, and lowering
        # the clean edge is what lets the search buy a cleaner delivery.
        dist_bound_knob("post_effects.codec.quality_lo",
                        "calibrated.post_effects.codec.quality", 0, 0.50, 0.85),

        # -- learned residual dose ------------------------------------------
        scalar_knob("residual.apply_prob", "calibrated.residual.apply_prob",
                    0.0, 1.0, kind="prob"),
        scalar_knob("residual.scale_max",
                    "calibrated.residual.residual_scale_max", 0.05, 0.35),

        # -- talker (backend-agnostic) --------------------------------------
        *tts_speed_knobs(),
        dist_prob_knob("voice_augment.pitch_prob", "voice_augment.pitch_semitones",
                       0.0, 0.8),
        dist_prob_knob("voice_augment.tempo_prob", "voice_augment.tempo", 0.0, 0.8),

        # -- batch composition (backend-agnostic) ---------------------------
        scalar_knob("dataset.noise_only_frac", "dataset.noise_only_frac", 0.0, 0.10,
                    kind="prob"),
        scalar_knob("dataset.pilot_double_hop_prob", "dataset.pilot_double_hop_prob",
                    0.0, 0.8, kind="prob"),
    ])


def talker_only_space(base: Mapping[str, Any] | None = None) -> SearchSpace:
    """Four talker knobs that resolve and render on *either* generator mode.

    The space for a short run.  A cross-entropy method over fifteen dimensions
    needs far more than the twenty-five trials a night of seven-minute
    evaluations buys; four dimensions it can actually move.  Nothing here
    touches a channel backend, so it is valid against a procedural profile and
    a calibrated one alike -- which is also what lets the same knobs be read
    against tonight's Mode 1 base and tomorrow's calibrated one.

    **The omissions are the argument**, and they are about *when* a knob is
    worth searching, not whether it matters:

    *   **Calibrated and residual knobs get re-derived at calibration.**
        Searching a residual's ``apply_prob`` and ``residual_scale_max``
        against an *untrained* residual measures nothing -- the config fields
        exist, but ``calibrated.residual.enabled`` is false, so the reward
        cannot see them and the optimizer spends its budget on dead
        dimensions.  ``mode2_safe_space`` carries them for the run that
        happens after the artifacts are fitted.
    *   **The batch-composition knobs are cheap to set and expensive to
        search.**  ``noise_only_frac`` and ``pilot_double_hop_prob`` change
        what fraction of the batch is which kind of row; at a few hundred
        clips per trial the reward's resolution is well above the difference
        they make.

    What is left is the talker.  Rate is the one talker-side control that
    reliably moves ASR error, and the two speed edges do different jobs -- the
    slow edge sets how much easy audio the fine-tune sees, the fast edge how
    much of the hard tail.

    ``voice_augment.pitch_prob`` is a **categorical arm**, not a range:
    ``{0.0, the base profile's own value}``.  The open question about pitch is
    whether to do it at all -- ``configs/mode2_fastcut.yaml``'s header records
    that pitch shifting costs Mode 2 a 2.6x WavLM KID regression through
    resampling artifacts, while it is also what buys speaker diversity for the
    downstream task.  That is an on/off decision the reward can answer in a
    handful of trials; sweeping it continuously would spend the budget
    resolving intermediate settings nobody would ship.

    Pass ``base`` to anchor that second arm on the profile actually being
    searched; without it the arm is ``DEFAULT_PITCH_PROB``.
    """
    pitch_prob = DEFAULT_PITCH_PROB
    if base is not None:
        current = dist_prob_knob("probe", "voice_augment.pitch_semitones").read(base)
        if current is not None and current > 0.0:
            pitch_prob = float(current)

    return SearchSpace([
        *tts_speed_knobs(),
        dist_prob_knob("voice_augment.tempo_prob", "voice_augment.tempo", 0.0, 0.8),
        dist_prob_choice_knob("voice_augment.pitch_prob",
                              "voice_augment.pitch_semitones", (0.0, pitch_prob)),
    ])


#: `--space` choices for `scripts/rl_loop.py`, name -> constructor. Every
#: constructor takes the base profile so a space may anchor a knob on it.
SPACES = {
    "default": lambda base=None: default_atc_space(),
    "mode2_safe": mode2_safe_space,
    "talker_only": talker_only_space,
}

#: Generator modes each space can search. `talker_only` is mode-agnostic (it
#: touches only TTS and voice augment); `default` addresses `channel.chain` and
#: `mode2_safe` addresses `calibrated.*`, so each is confined to its own mode.
SPACE_MODES = {
    "default": {"procedural"},
    "mode2_safe": {"calibrated"},
    "talker_only": {"procedural", "calibrated"},
}

