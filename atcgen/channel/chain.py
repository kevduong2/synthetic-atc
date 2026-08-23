"""Mode 1 channel backend: a config-declared chain of `primitives`.

Each `ChainStep` is applied in order with per-sample draws: the step is skipped
when `rng.random() >= step.prob`, and every parameter is drawn from its
`DistSpec`.  `ChannelRecord` keeps what was actually applied, for the manifest.

Hop structure (ported from `dsp.py`, extended in P1): the chain runs in three
stages, matching where in the physical path each effect happens.  `SOURCE_ONCE`
primitives belong to the talker — the handset's timbre, the moment they pressed
PTT — so they run once on the source audio however many radios follow.  Everything
else is transmit/path-side and runs once per hop.  `RECEIVER_END` primitives
belong to the receiving station — the co-channel sum arriving at its antenna, its
AGC and squelch, the delivery codec — so they run once after the last hop.
`hops=2` models pilot audio relayed through a ground station: the second hop
draws fresh parameters, with the noise floor forced to `HOP2_SNR_DB` as dsp.py
did.  Stage membership is a module constant, not config: it is a property of
where the effect physically happens, not of a profile.

The clean arm (`clean_arm_prob`, 03 §1's MTR zero-effects arm) bypasses the
chain apart from `CLEAN_ARM_KEEP` (bandpass) and the 16 kHz resample.
"""

import inspect
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ..config import ChainStep, ChannelConfig, DistSpec
from .primitives import PRIMITIVES, TARGET_SR, NoiseBank, resample

PAD_SEC = 0.15                # silence framing the speech, so clicks/noise have room
SOURCE_ONCE = {"mic_coloration", "ptt_truncation"}
RECEIVER_END = {"cochannel_mix", "agc_attack", "squelch_gate", "squelch_clicks",
                "codec_roundtrip"}
CLEAN_ARM_KEEP = {"bandpass"}
HOP2_SNR_DB = (10.0, 25.0)    # relay hop: quieter noise floor than the first radio


@dataclass
class UtteranceMeta:
    """What the channel needs to know about the utterance it is degrading."""

    role: str = "none"        # pilot | controller | none -- double-hop eligibility
    kind: str = ""
    category: str = ""


@dataclass
class ChannelRecord:
    """Provenance blob for the manifest's `gen.channel` field."""

    hops: int = 1
    clean_arm: bool = False
    snr_db: float | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"hops": self.hops, "clean_arm": self.clean_arm,
                "snr_db": self.snr_db, "steps": self.steps}

    def applied(self) -> list[str]:
        return [step["primitive"] for step in self.steps]


@lru_cache(maxsize=None)
def _accepts(fn) -> frozenset:
    return frozenset(inspect.signature(fn).parameters)


class ProceduralChannel:
    """Randomized primitive chain: clean speech in, one radio's output out."""

    def __init__(self, chain_steps: list[ChainStep], noise_bank: NoiseBank | None = None,
                 target_sr: int = TARGET_SR, clean_arm_prob: float = 0.0,
                 shuffle_groups: list[list[str]] | None = None):
        unknown = sorted({s.primitive for s in chain_steps} - set(PRIMITIVES))
        if unknown:
            raise ValueError(f"unknown channel primitive(s): {', '.join(unknown)}")
        self.steps = list(chain_steps)
        self.noise_bank = noise_bank
        self.target_sr = target_sr
        self.clean_arm_prob = clean_arm_prob
        self.shuffle_groups = [list(g) for g in (shuffle_groups or [])]

    @classmethod
    def from_config(cls, channel_cfg: ChannelConfig,
                    noise_bank: NoiseBank | None = None,
                    target_sr: int = TARGET_SR) -> "ProceduralChannel":
        return cls(channel_cfg.chain, noise_bank, target_sr,
                   channel_cfg.clean_arm_prob, channel_cfg.shuffle_groups)

    def __call__(self, wav: np.ndarray, sr: int, rng: random.Random,
                 meta: UtteranceMeta | None = None,
                 interference: np.ndarray | None = None,
                 hops: int = 1) -> tuple[np.ndarray, ChannelRecord]:
        """Degrade `wav` (float32 mono at `sr`). Returns (16 kHz wav, record)."""
        x = np.asarray(wav, dtype=np.float32)
        if sr != self.target_sr:
            x = resample(x, sr, self.target_sr)
        pad = int(self.target_sr * PAD_SEC)
        x = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])

        record = ChannelRecord(hops=hops, clean_arm=rng.random() < self.clean_arm_prob)
        source = [s for s in self.steps if s.primitive in SOURCE_ONCE]
        per_hop = [s for s in self.steps
                   if s.primitive not in SOURCE_ONCE and s.primitive not in RECEIVER_END]
        tail = [s for s in self.steps if s.primitive in RECEIVER_END]
        if record.clean_arm:
            source = [s for s in source if s.primitive in CLEAN_ARM_KEEP]
            per_hop = [s for s in per_hop if s.primitive in CLEAN_ARM_KEEP]
            tail = []
            record.hops = hops = 1

        for step in self._ordered(source, rng):
            x = self._apply(step, x, rng, record, hop=0, pad=pad)
        for hop in range(hops):
            for step in self._ordered(per_hop, rng):
                x = self._apply(step, x, rng, record, hop=hop, pad=pad)
        for step in self._ordered(tail, rng):
            x = self._apply(step, x, rng, record, hop=0, pad=pad,
                            interference=interference)

        peak = np.abs(x).max()
        if peak > 1.0:
            x = x / peak * 0.98
        return x.astype(np.float32), record

    def _ordered(self, steps: list[ChainStep], rng: random.Random) -> list[ChainStep]:
        """Steps in declared order, shuffled within each configured group."""
        if not self.shuffle_groups:
            return steps
        out = list(steps)
        for group in self.shuffle_groups:
            slots = [i for i, s in enumerate(out) if s.primitive in group]
            picked = [out[i] for i in slots]
            rng.shuffle(picked)
            for slot, step in zip(slots, picked):
                out[slot] = step
        return out

    def _apply(self, step: ChainStep, x: np.ndarray, rng: random.Random,
               record: ChannelRecord, hop: int, pad: int,
               interference: np.ndarray | None = None) -> np.ndarray:
        if rng.random() >= step.prob:
            return x
        fn = PRIMITIVES[step.primitive]
        drawn = {name: spec.sample(rng) for name, spec in step.params.items()}
        drawn = {name: value for name, value in drawn.items() if value is not None}
        if step.primitive == "additive_noise":
            if hop > 0:
                drawn["snr_db"] = rng.uniform(*HOP2_SNR_DB)
            if record.snr_db is None:
                record.snr_db = round(float(drawn.get("snr_db", 20.0)), 2)

        params = dict(drawn)
        for name, value in (("pad", pad), ("noise_bank", self.noise_bank),
                            ("interference", interference)):
            if value is not None and name in _accepts(fn):
                params[name] = value
        record.steps.append({"primitive": step.primitive, "hop": hop, **drawn})
        return fn(x, self.target_sr, rng, **params)


def mild_chain() -> list[ChainStep]:
    """Light-touch chain for post-GAN diversification: the GAN already supplies
    spectral realism, this only varies SNR/band/codec per sample."""
    spec = DistSpec.parse
    return [
        ChainStep("narrowband_roundtrip", 1.0,
                  {"narrow_sr": spec({"choice": [8000, 11025, 16000]})}),
        ChainStep("bandpass", 1.0, {"low": spec({"uniform": [200, 320]}),
                                    "high": spec({"uniform": [3000, 3800]})}),
        ChainStep("agc_wander", 1.0, {"strength": spec({"uniform": [0.0, 0.2]})}),
        ChainStep("soft_clip", 1.0, {"drive": spec({"uniform": [1.0, 1.5]})}),
        ChainStep("additive_noise", 1.0, {"snr_db": spec({"uniform": [15, 30]}),
                                          "color": spec({"choice": ["white", "pink"]})}),
        ChainStep("crackle", 1.0, {"rate": spec({"uniform": [0.0, 1.0]})}),
        ChainStep("squelch_clicks", 0.8, {}),
        ChainStep("codec_roundtrip", 0.5,
                  {"compression_level": spec({"uniform": [0.75, 0.95]})}),
    ]
