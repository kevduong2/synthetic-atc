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

Bandpass re-application (`reapply_bandpass`, research-findings §4.3)
-------------------------------------------------------------------
A real link band-limits more than once, and every filter is downstream of
something that splatters.  Declaring `bandpass` once in the chain models only
the transmitter's audio filter: the steps after it — clipping and AM
distortion products, broadband static, crackle, the mains hum below the
passband, the rectangular gating of a dropout, a co-channel signal arriving at
the antenna — all put energy where no receiver could pass it.  So the *same*
drawn filter (no re-draw: one link, one passband) is re-applied wherever the
signal crosses a real filter:

*   **Receiver front end.**  Everything the RF path adds is upstream of the
    receiving radio's IF/audio filter, and so is the co-channel sum at its
    antenna.  The re-application therefore lands *after* `cochannel_mix` and
    *before* the first step that models the receiver's own processing
    (`AFTER_RECEIVER_FILTER`: AGC, the squelch gate, the delivery codec).  A
    relay hop has its own receiver, so hops before the last flush at the hop
    boundary instead.
*   **Delivery band limit.**  `squelch_gate`'s tail burst and `squelch_clicks`
    are generated in the receiver's audio stage, *downstream* of that filter,
    so they are not covered by it — but they still pass the audio output stage
    on the way to the encoder.  Mode 2 measured what skipping this costs:
    a raw click leaves ~10 dB more energy at 5-7 kHz than the real clips carry
    (`learned/backend.py::_band_limited`).  The pre-`codec_roundtrip` flush is
    that filter; `codec_roundtrip` itself is the delivery format, so nothing is
    filtered after it.

`squelch_gate` is in both sets on purpose: it sits downstream of the IF filter
(pending RF splatter is flushed before it) and its tail burst dirties the band
again afterwards.  Membership is by primitive, not by outcome — a step whose
draw happened to inject nothing still triggers a flush.  That is deliberate:
an extra pass of the same filter can only remove out-of-band energy, and
outcome-tracking would couple this module to each primitive's internals.

Set `channel.reapply_bandpass: false` to get the pre-P4 single-filter chain,
for an ablation.  It is physics, so it defaults to true.
"""

import inspect
import random
import warnings
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

# Steps that leave energy outside the drawn passband (see the module docstring):
# broadband additions, sub-passband hum, and the harmonics/splatter of a
# nonlinearity or a rectangular gate.
BAND_SPLATTER = {"am_distortion", "soft_clip", "dropouts", "additive_noise", "hum",
                 "crackle", "cochannel_mix", "squelch_gate", "squelch_clicks"}
# Steps that model receiver-side processing, i.e. that physically happen after
# the receiver's filter -- pending splatter is filtered out before they run.
AFTER_RECEIVER_FILTER = {"agc_wander", "agc_attack", "squelch_gate", "codec_roundtrip"}


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
    residual_alpha: float | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"hops": self.hops, "clean_arm": self.clean_arm,
                "snr_db": self.snr_db, "residual_alpha": self.residual_alpha,
                "steps": self.steps}

    def applied(self) -> list[str]:
        return [step["primitive"] for step in self.steps]


@lru_cache(maxsize=None)
def _accepts(fn) -> frozenset:
    return frozenset(inspect.signature(fn).parameters)


@lru_cache(maxsize=None)
def _defaults(fn) -> dict[str, Any]:
    """A primitive's own keyword defaults, for params a config step omits."""
    return {name: param.default
            for name, param in inspect.signature(fn).parameters.items()
            if param.default is not inspect.Parameter.empty}


@dataclass
class _Band:
    """The passband in force, and whether anything has splattered outside it."""

    cutoffs: tuple[float, float] | None = None
    pending: bool = False


def _warn_out_of_envelope(channel_cfg: ChannelConfig) -> None:
    """Warn (never fail) when a profile randomizes past the measured real envelope.

    research-findings §4.3's "capped domain randomization": ranges are meant to
    stay inside what the real corpus shows, because unlimited distortion
    manufactures audio whose transcript is no longer recoverable — mislabeled
    training data.  `wide` explores past the cap on purpose, so this documents
    by how much instead of blocking the run.  All findings go into one warning:
    the default warning filter shows a given call site once, so separate calls
    would hide everything after the first.
    """
    from .envelope import check_profile, load_envelope

    envelope = load_envelope()
    problems = check_profile(channel_cfg, envelope) if envelope else []
    if problems:
        warnings.warn(f"channel profile {channel_cfg.profile!r} randomizes outside the "
                      "measured real envelope:\n  " + "\n  ".join(problems), stacklevel=3)


class ProceduralChannel:
    """Randomized primitive chain: clean speech in, one radio's output out."""

    def __init__(self, chain_steps: list[ChainStep], noise_bank: NoiseBank | None = None,
                 target_sr: int = TARGET_SR, clean_arm_prob: float = 0.0,
                 shuffle_groups: list[list[str]] | None = None,
                 reapply_bandpass: bool = True):
        unknown = sorted({s.primitive for s in chain_steps} - set(PRIMITIVES))
        if unknown:
            raise ValueError(f"unknown channel primitive(s): {', '.join(unknown)}")
        self.steps = list(chain_steps)
        self.noise_bank = noise_bank
        self.target_sr = target_sr
        self.clean_arm_prob = clean_arm_prob
        self.shuffle_groups = [list(g) for g in (shuffle_groups or [])]
        self.reapply_bandpass = reapply_bandpass

    @classmethod
    def from_config(cls, channel_cfg: ChannelConfig,
                    noise_bank: NoiseBank | None = None,
                    target_sr: int = TARGET_SR) -> "ProceduralChannel":
        _warn_out_of_envelope(channel_cfg)
        return cls(channel_cfg.chain, noise_bank, target_sr,
                   channel_cfg.clean_arm_prob, channel_cfg.shuffle_groups,
                   channel_cfg.reapply_bandpass)

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

        band = _Band()
        for step in self._ordered(source, rng):
            x = self._run(step, x, rng, record, band, hop=0, pad=pad)
        for hop in range(hops):
            for step in self._ordered(per_hop, rng):
                x = self._run(step, x, rng, record, band, hop=hop, pad=pad)
            if hop < hops - 1:
                # a relay demodulated this hop through its own receiver before
                # keying it out again; the last hop's filter waits for the
                # co-channel sum arriving at the final antenna
                x = self._refilter(x, rng, record, band, hop, "relay")
        for step in self._ordered(tail, rng):
            x = self._run(step, x, rng, record, band, hop=0, pad=pad,
                          interference=interference)
        x = self._refilter(x, rng, record, band, 0, "chain_end")

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

    def _run(self, step: ChainStep, x: np.ndarray, rng: random.Random,
             record: ChannelRecord, band: _Band, hop: int, pad: int,
             interference: np.ndarray | None = None) -> np.ndarray:
        """One step, plus the bandpass bookkeeping the module docstring describes."""
        if band.pending and step.primitive in AFTER_RECEIVER_FILTER:
            x = self._refilter(x, rng, record, band, hop, f"before:{step.primitive}")
        x, drawn = self._apply(step, x, rng, record, hop=hop, pad=pad,
                               interference=interference)
        if drawn is None:
            return x
        if step.primitive == "bandpass":
            defaults = _defaults(PRIMITIVES["bandpass"])
            band.cutoffs = (float(drawn.get("low", defaults["low"])),
                            float(drawn.get("high", defaults["high"])))
            band.pending = False
        elif step.primitive in BAND_SPLATTER:
            band.pending = True
        return x

    def _refilter(self, x: np.ndarray, rng: random.Random, record: ChannelRecord,
                  band: _Band, hop: int, reason: str) -> np.ndarray:
        """Re-apply the drawn passband, logging why, when anything is pending."""
        if not (self.reapply_bandpass and band.pending) or band.cutoffs is None:
            return x
        low, high = band.cutoffs
        band.pending = False
        record.steps.append({"primitive": "bandpass", "hop": hop, "low": low,
                             "high": high, "reapply": reason})
        return PRIMITIVES["bandpass"](x, self.target_sr, rng, low=low, high=high)

    def _apply(self, step: ChainStep, x: np.ndarray, rng: random.Random,
               record: ChannelRecord, hop: int, pad: int,
               interference: np.ndarray | None = None
               ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Apply one step; returns the audio and the drawn params (None if skipped)."""
        if rng.random() >= step.prob:
            return x, None
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
        return fn(x, self.target_sr, rng, **params), drawn


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
