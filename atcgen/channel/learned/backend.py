"""Mode 2 channel backend: fitted presets + real noise + shared primitives (04 §2.3).

Same interface as `ProceduralChannel` — `(wav, sr, rng, meta, interference, hops)
-> (wav16k, ChannelRecord)` — so the builder treats the two backends
identically and everything downstream (manifest, stats, eval) is common.

What a preset can and cannot supply decides the split.  The fitted chain
reproduces the *transmission*: band shape, transmitter clipping, receiver gain
control and the noise floor riding under the speech, all measured off one real
clip.  It cannot produce anything that is not a stationary property of the
channel — the click when the carrier opens, the gate slamming the floor shut
between transmissions, a dropout, the delivery codec.  Those come from the
shared Mode 1 primitives, at probabilities `post_effects` declares.

Draws are correlated on purpose: a preset comes with a noise bed harvested from
*its own station*, at an SNR jittered around the one measured in that station's
clips.  Mixing a Center receiver's hiss into a Tower's channel is possible but
rare (`cross_station_prob`), because the combination is not one any real
receiver produces.
"""

import random
from collections import Counter, defaultdict
from json import loads
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ...config import CalibratedConfig, DistSpec, PostEffectsConfig
from ..chain import PAD_SEC, ChannelRecord, UtteranceMeta
from ..primitives import TARGET_SR, codec_roundtrip, dropouts, resample
from ..primitives import cochannel_mix, ptt_truncation, squelch_clicks, squelch_gate
from .preset import Preset, apply_preset, fir_taps, load_presets

# Probabilities and ranges the Mode 2 config schema has no field for; values
# mirror `configs/mode1_matched.yaml`, which was fitted against the same clips.
PTT_PROB = 0.2
PTT_MS = (20.0, 100.0)
COCHANNEL_PROB = 0.08
COCHANNEL_LEVEL = (0.05, 0.15)
SQUELCH_FLOOR_GATED_DB = (-70.0, -55.0)     # carrier drops to near digital silence
SQUELCH_FLOOR_OPEN_DB = (-45.0, -25.0)      # ... or merely ducks
SQUELCH_THRESHOLD_DB = (-42.0, -30.0)
SQUELCH_ATTACK_MS = (5.0, 40.0)
SQUELCH_RELEASE_MS = (10.0, 50.0)
RELAY_SNR_DB = (25.0, 40.0)                 # the relayed hop is the quieter one
NOISE_FADE_MS = 5.0


class StationNoise:
    """The M2.1 noise bank, keyed by the station each segment was harvested from.

    `NoiseBank` serves one undifferentiated pool; Mode 2 needs to pair a preset
    with its own receiver's floor, so segments are grouped and `noise_stats.jsonl`
    (which carries the station and the squelch-gated flag) is the index.
    """

    def __init__(self, noise_dir: str | Path, sr: int = TARGET_SR):
        directory = Path(noise_dir)
        stats = directory / "noise_stats.jsonl"
        if not stats.exists():
            raise ValueError(f"no noise_stats.jsonl in {directory}")
        self.sr = sr
        self.by_station: dict[str, list[np.ndarray]] = defaultdict(list)
        gated = 0
        rows = [loads(line) for line in stats.read_text().splitlines() if line.strip()]
        for index, row in enumerate(rows):
            path = directory / f"{index:04d}.wav"
            if not path.exists():
                continue
            wav, file_sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != sr:
                wav = resample(wav, file_sr, sr)
            if len(wav):
                self.by_station[row.get("station", "unknown")].append(wav)
                gated += bool(row.get("squelch_gated"))
        self.stations = sorted(self.by_station)
        if not self.stations:
            raise ValueError(f"no usable noise segments in {directory}")
        self.gated_fraction = gated / max(len(rows), 1)

    def sample(self, n: int, rng: random.Random, station: str | None = None
               ) -> np.ndarray:
        """`n` samples of bed from `station` (any station when it has none).

        Harvested segments are short — a few hundred milliseconds — so a bed is
        stitched from independent draws crossfaded together, never by looping
        one segment, which would stamp a periodic pattern on the clip.  Same
        reasoning as `NoiseBank.sample`, over a per-station pool.
        """
        clips = self.by_station.get(station or "") or self.by_station[
            rng.choice(self.stations)]
        fade = max(1, int(self.sr * NOISE_FADE_MS / 1000))
        bed = np.zeros(0, dtype=np.float32)
        while len(bed) < n:
            clip = rng.choice(clips)
            if len(bed) >= fade and len(clip) > fade:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                join = bed[-fade:] * (1.0 - ramp) + clip[:fade] * ramp
                bed = np.concatenate([bed[:-fade], join, clip[fade:]])
            else:
                bed = np.concatenate([bed, clip])
        start = rng.randrange(0, len(bed) - n + 1) if len(bed) > n else 0
        return bed[start:start + n]


class CalibratedChannel:
    """Clean speech in, one real receiver's output out — sampled per utterance."""

    def __init__(self, presets: list[Preset], noise: StationNoise | None = None,
                 station_mix: dict[str, float] | None = None,
                 snr_jitter: DistSpec | None = None, cross_station_prob: float = 0.1,
                 post_effects: PostEffectsConfig | None = None,
                 target_sr: int = TARGET_SR):
        if not presets:
            raise ValueError("CalibratedChannel needs at least one preset")
        self.presets = list(presets)
        self.noise = noise
        self.snr_jitter = snr_jitter
        self.cross_station_prob = cross_station_prob
        self.post = post_effects or PostEffectsConfig()
        self.target_sr = target_sr

        self.by_station: dict[str, list[Preset]] = defaultdict(list)
        for preset in self.presets:
            self.by_station[preset.station].append(preset)
        counts = Counter(preset.station for preset in self.presets)
        if station_mix is None:
            station_mix = {name: count / len(self.presets)
                           for name, count in counts.items()}
        unknown = sorted(set(station_mix) - set(counts))
        if unknown:
            raise ValueError("station_mix names stations with no presets: "
                             + ", ".join(unknown))
        total = sum(station_mix.values())
        if total <= 0:
            raise ValueError("station_mix weights must sum to more than zero")
        self.stations = sorted(station_mix)
        self.weights = [station_mix[name] / total for name in self.stations]
        # taps are a pure function of a preset's band gains, and a preset is
        # redrawn every utterance out of a pool of ~1k: build each set once
        self._taps: dict[int, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: CalibratedConfig, target_sr: int = TARGET_SR
                    ) -> "CalibratedChannel":
        calibration = config.calibration
        noise_dir = Path(calibration.noise_bank)
        return cls(
            load_presets(calibration.presets),
            StationNoise(noise_dir, target_sr) if noise_dir.is_dir() else None,
            station_mix=calibration.station_mix,
            snr_jitter=calibration.snr_jitter_db,
            cross_station_prob=calibration.cross_station_prob,
            post_effects=config.post_effects,
            target_sr=target_sr,
        )

    # -- sampling ----------------------------------------------------------- #
    def draw_preset(self, rng: random.Random) -> Preset:
        station = (self.stations[0] if len(self.stations) == 1
                   else rng.choices(self.stations, weights=self.weights)[0])
        return rng.choice(self.by_station[station])

    def _taps_for(self, preset: Preset) -> np.ndarray:
        key = id(preset)
        if key not in self._taps:
            self._taps[key] = fir_taps(preset.band_gains_db, preset.band_edges_hz,
                                       self.target_sr)
        return self._taps[key]

    def _bed(self, preset: Preset, n: int, rng: random.Random
             ) -> tuple[np.ndarray | None, str | None]:
        """A noise bed for `preset`, and the station it actually came from."""
        if self.noise is None:
            return None, None
        cross = rng.random() < self.cross_station_prob
        station = rng.choice(self.noise.stations) if cross else preset.station
        if station not in self.noise.by_station:
            station = rng.choice(self.noise.stations)
        return self.noise.sample(n, rng, station), station

    def _through(self, x: np.ndarray, preset: Preset, snr_db: float,
                 rng: random.Random) -> tuple[np.ndarray, str | None]:
        """One pass of the fitted chain, with a real bed as its noise floor.

        The bed goes through the EQ like the synthetic noise the fit used, even
        though a harvested bed already carries a real receiver's band shape.
        Applying it twice costs nothing where it matters — the passband is flat,
        so only the already-inaudible skirts get steeper — and it guarantees the
        output's band is the fitted one, rather than whatever the bed happened to
        contain out of band.
        """
        bed, bed_station = self._bed(preset, len(x), rng)
        return apply_preset(x, self.target_sr, preset, noise=bed, snr_db=snr_db,
                            filter_noise=True,
                            taps=self._taps_for(preset)), bed_station

    # -- the backend interface ---------------------------------------------- #
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

        record = ChannelRecord(hops=hops)
        if rng.random() < PTT_PROB:
            # each end is clipped independently: a late press, an early release,
            # or both -- matching the matched profile's per-end 0.6 draw
            head = rng.uniform(*PTT_MS) if rng.random() < 0.6 else 0.0
            tail = rng.uniform(*PTT_MS) if rng.random() < 0.6 else 0.0
            x = ptt_truncation(x, self.target_sr, rng, head_ms=head, tail_ms=tail,
                               pad=pad)
            record.steps.append({"primitive": "ptt_truncation", "hop": 0,
                                 "head_ms": round(head, 1), "tail_ms": round(tail, 1)})

        # a relayed transmission passes through the relaying station's radio
        # first, at the quieter floor of a strong local signal
        for hop in range(max(hops, 1) - 1):
            relay = self.draw_preset(rng)
            snr = rng.uniform(*RELAY_SNR_DB)
            x, bed_station = self._through(x, relay, snr, rng)
            record.steps.append(self._preset_step(relay, snr, bed_station, hop))

        preset = self.draw_preset(rng)
        snr = preset.snr_est + (0.0 if self.snr_jitter is None
                                else float(self.snr_jitter.sample(rng) or 0.0))
        x, bed_station = self._through(x, preset, snr, rng)
        record.steps.append(self._preset_step(preset, snr, bed_station,
                                              max(hops, 1) - 1))
        record.snr_db = round(float(snr), 2)

        x = self._post_effects(x, rng, record, pad, interference)
        peak = float(np.abs(x).max())
        if peak > 1.0:
            x = x / peak * 0.98
        return x.astype(np.float32), record

    @staticmethod
    def _preset_step(preset: Preset, snr_db: float, bed_station: str | None,
                     hop: int) -> dict[str, Any]:
        return {"primitive": "calibrated_preset", "hop": hop,
                "clip_id": preset.clip_id, "station": preset.station,
                "noise_station": bed_station, "snr_db": round(float(snr_db), 2),
                "drive": preset.drive, "passband_hz": preset.passband_hz}

    def _post_effects(self, x: np.ndarray, rng: random.Random, record: ChannelRecord,
                      pad: int, interference: np.ndarray | None) -> np.ndarray:
        """The receiving station's own artifacts, in the order it produces them."""
        if interference is not None and rng.random() < COCHANNEL_PROB:
            level = rng.uniform(*COCHANNEL_LEVEL)
            x = cochannel_mix(x, self.target_sr, rng, level=level,
                              interference=interference)
            record.steps.append({"primitive": "cochannel_mix", "hop": 0,
                                 "level": round(level, 3)})

        if rng.random() < self.post.dropouts.prob:
            x = dropouts(x, self.target_sr, rng, dropout_prob=1.0, count_lam=1.0,
                         min_ms=10.0, max_ms=40.0)
            record.steps.append({"primitive": "dropouts", "hop": 0})

        if rng.random() < self.post.squelch.prob:
            gated = rng.random() < self.post.squelch.gated_floor_prob
            floor = rng.uniform(*(SQUELCH_FLOOR_GATED_DB if gated
                                  else SQUELCH_FLOOR_OPEN_DB))
            x = squelch_gate(x, self.target_sr, rng, floor_db=floor,
                             attack_ms=rng.uniform(*SQUELCH_ATTACK_MS),
                             release_ms=rng.uniform(*SQUELCH_RELEASE_MS),
                             threshold_db=rng.uniform(*SQUELCH_THRESHOLD_DB), pad=pad)
            x = squelch_clicks(x, self.target_sr, rng)
            record.steps.append({"primitive": "squelch_gate", "hop": 0,
                                 "floor_db": round(floor, 1), "gated_floor": gated})
            record.steps.append({"primitive": "squelch_clicks", "hop": 0})

        if rng.random() < self.post.codec.prob:
            quality = float(self.post.codec.quality.sample(rng))
            x = codec_roundtrip(x, self.target_sr, rng, compression_level=quality,
                                codec=self.post.codec.kind)
            record.steps.append({"primitive": "codec_roundtrip", "hop": 0,
                                 "codec": self.post.codec.kind,
                                 "compression_level": round(quality, 3)})
        return x
