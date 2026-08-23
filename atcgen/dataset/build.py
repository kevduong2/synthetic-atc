"""End-to-end config-driven dataset builder (02 §3, §5, §6).

    text source -> TTS -> channel backend -> post -> wavs/ + manifest.jsonl

Everything randomized is drawn here, from the `GeneratorConfig`, seeded by
`config.seed`, and written into each record's `gen` blob so a sample can be
traced back to what produced it.  The channel backend is chosen by
`config.mode` (`procedural`, `calibrated`, or a per-sample weighted draw in
`mix`) through `make_backend`.

Each sample passes the Tier 0 QC gates (05 §2) before it is written; a failing
sample is regenerated up to `config.qc.max_retries` times and then kept with
`gen.qc.ok = false`, so a run never silently shrinks.  Discard reasons and
rates, category/duration/SNR histograms and achieved quota fractions land in
`stats.json` next to the manifest, alongside the fully resolved config.

A small fraction of samples are noise-only (empty transcript, category
"noise") as Whisper hallucination control: an open-squelch bed degraded by the
same channel, see `_noise_bed`.  Pilot utterances are sometimes double-hopped
through a ground relay.  `load_manifest` loads a built set for
training (extra keys are ignored).
"""

import inspect
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..channel.chain import ProceduralChannel, UtteranceMeta
from ..channel.learned.backend import CalibratedChannel
from ..channel.primitives import TARGET_SR, NoiseBank, pink_noise
from ..config import GeneratorConfig, QCConfig, dump_resolved
from ..eval.qc import QCConfig as QCGates
from ..eval.qc import QCTally, qc_sample
from ..text.grammar import Utterance
from ..text.sources import TextSource, WeightedSampler, make_text_source
from ..tts.augment import VoiceAugment

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mode1_default.yaml"
NOISE_ONLY_SEC = (2.0, 6.0)          # duration range of the noise-only samples
NOISE_ONLY_BED_RMS = 0.03            # the bed the channel shapes when nobody speaks
DURATION_EDGES = list(range(0, 32, 2))
SNR_EDGES = list(range(-5, 45, 5))


def make_backend(config: GeneratorConfig, name: str | None = None):
    """Build the channel backend called `name` (default: the config's mode).

    The seam Mode 2 plugs into: a backend is
    `(wav, sr, rng, meta, interference=, hops=) -> (wav, ChannelRecord)`.
    """
    name = name or config.mode
    if name == "procedural":
        if config.channel is None:
            raise ValueError("mode 'procedural' requires a channel section")
        beds_dir = config.channel.noise.beds_dir
        noise_bank = NoiseBank(beds_dir) if beds_dir is not None else None
        return ProceduralChannel.from_config(
            config.channel, noise_bank=noise_bank,
            target_sr=config.output.sample_rate)
    if name == "calibrated":
        if config.calibrated is None:
            raise ValueError("mode 'calibrated' requires a calibrated section")
        return CalibratedChannel.from_config(
            config.calibrated, target_sr=config.output.sample_rate)
    raise ValueError(f"unknown channel backend: {name}")


def build_dataset(config: GeneratorConfig, out_dir: str | Path, n_samples: int,
                  text_source: TextSource | str | None = None, tts=None,
                  transcriber=None) -> Path:
    """Generate `n_samples` utterances into `out_dir`. Returns the manifest path.

    `transcriber` overrides the ASR round-trip gate's Whisper (see
    `atcgen.eval.qc`); it is only consulted when `config.qc.asr_roundtrip`.
    """
    if config.output.format != "wav":
        raise ValueError(f"unsupported output format: {config.output.format}")
    out = Path(out_dir)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    sr = config.output.sample_rate
    rng = random.Random(config.seed)
    augment_rng = random.Random(f"{config.seed}:voice-augment")

    if text_source is None or isinstance(text_source, str):
        text_source = make_text_source(text_source or "grammar")
    sampler = WeightedSampler.for_source(text_source, config.dataset.category_quotas)
    if tts is None:
        from ..tts import KokoroTTS
        tts = KokoroTTS(voices=config.tts.voices)
    voice_augment = VoiceAugment.from_config(config.voice_augment)

    names, weights, backends = _backend_pool(config)
    _, resolved_hash = dump_resolved(config, out)
    gates = _gates(config.qc)
    tally = QCTally()

    manifest_path = out / "manifest.jsonl"
    rows: list[dict] = []
    kept_with_flag = 0
    prev_wav = None                   # reused as co-channel interference material
    with open(manifest_path, "w") as manifest:
        for index in tqdm(range(n_samples), desc=f"generating ({config.mode})"):
            noise_only = rng.random() < config.dataset.noise_only_frac
            backend_name = _draw(names, weights, rng)
            utterance = None if noise_only else _next_utterance(
                sampler, text_source, rng)

            # a rejected sample is re-rendered with fresh voice/speed/channel
            # draws; the text stays put, so quota accounting stays honest
            result = None
            for attempt in range(config.qc.max_retries + 1 if config.qc.enabled else 1):
                wav, gen = _render(config, tts, voice_augment,
                                   backends[backend_name], utterance, rng,
                                   augment_rng, prev_wav)
                if not config.qc.enabled:
                    break
                result = qc_sample(wav, sr, utterance.transcript if utterance else None,
                                   gates, transcriber)
                tally.add(result)
                if result.ok:
                    break
            attempts = attempt + 1
            if result is not None and not result.ok:
                kept_with_flag += 1

            relative = f"wavs/{index:06d}.wav"
            sf.write(out / relative, wav, sr)
            if utterance is not None:
                prev_wav = wav

            gen.update({"mode": backend_name, "config_hash": resolved_hash,
                        "seed_index": index})
            if result is not None:
                gen["qc"] = {"ok": result.ok, "reason": result.reason,
                             "attempts": attempts}
            record = {
                "audio": relative,
                "text": utterance.transcript if utterance else "",
                "role": utterance.role if utterance else "none",
                "kind": utterance.kind if utterance else "noise",
                "category": utterance.category if utterance else "noise",
                "duration": round(len(wav) / sr, 3),
                "gen": gen,
            }
            manifest.write(json.dumps(record) + "\n")
            rows.append(record)

    stats = _stats(config, rows, sampler, tally, kept_with_flag, resolved_hash)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return manifest_path


def _backend_pool(config: GeneratorConfig) -> tuple[list[str], list[float], dict]:
    """Backend names, their draw weights, and one instance per distinct name."""
    if config.mode == "mix":
        names = [item.backend for item in config.backends]
        weights = [item.weight for item in config.backends]
    else:
        names, weights = [config.mode], [1.0]
    return names, weights, {name: make_backend(config, name)
                            for name in dict.fromkeys(names)}


def _draw(names: list[str], weights: list[float], rng: random.Random) -> str:
    return names[0] if len(names) == 1 else rng.choices(names, weights=weights)[0]


def _next_utterance(sampler: WeightedSampler | None, source: TextSource,
                    rng: random.Random) -> Utterance:
    return sampler.sample(rng) if sampler is not None else source.sample(rng)


def _gates(qc: QCConfig) -> QCGates:
    """The config's `qc` section as the eval module's gate thresholds."""
    return QCGates(min_duration=qc.min_duration, max_duration=qc.max_duration,
                   max_clip_frac=qc.max_clip_frac, min_rms_db=qc.min_rms_db,
                   max_rms_db=qc.max_rms_db, max_wer=qc.max_wer,
                   asr_gate=qc.asr_roundtrip)


def _render(config: GeneratorConfig, tts, voice_augment: VoiceAugment, backend,
            utterance: Utterance | None, rng: random.Random,
            augment_rng: random.Random, interference) -> tuple[np.ndarray, dict]:
    """One render: TTS -> voice augment -> channel -> post. Draws are per attempt."""
    sr = config.output.sample_rate
    if utterance is None:
        wav, record = backend(_noise_bed(sr, rng), sr, rng,
                              UtteranceMeta(kind="noise", category="noise"))
        gen: dict = {"voice": None, "speed": None}
        augment_record = {"pitch": None, "tempo": None, "eq_tilt_db": None}
    else:
        voice = rng.choice(config.tts.voices)
        speed = float(config.tts.speed.sample(rng))
        clean = _synthesize(tts, utterance.spoken, rng, voice, speed)
        clean, augment_record = voice_augment(clean, tts.sample_rate, augment_rng)
        hops = 2 if (utterance.role == "pilot"
                     and rng.random() < config.dataset.pilot_double_hop_prob) else 1
        meta = UtteranceMeta(role=utterance.role, kind=utterance.kind,
                             category=utterance.category)
        wav, record = backend(clean, tts.sample_rate, rng, meta,
                              interference=interference, hops=hops)
        gen = {"voice": voice, "speed": round(speed, 3)}
    wav = _post(wav, config.output.loudness_db.sample(rng))
    gen.update({**augment_record, "channel": record.as_dict()})
    return wav, gen


def _noise_bed(sr: int, rng: random.Random) -> np.ndarray:
    """Input for a noise-only sample: an open-squelch bed, not digital silence.

    The channel's noise is signal-relative, so feeding it zeros yields a silent
    file with only the absolute-level effects (clicks, crackle) on it — useless
    as hallucination control.  Seeding the transmit path with a quiet pink bed
    gives the chain something to band-limit, gate and code, which is what dead
    air on a receiver actually is.
    """
    n = int(sr * rng.uniform(*NOISE_ONLY_SEC))
    bed = pink_noise(n, np.random.default_rng(rng.getrandbits(64)))
    rms = float(np.sqrt(np.mean(bed.astype(np.float64) ** 2)))
    return (bed * (NOISE_ONLY_BED_RMS / rms)).astype(np.float32) if rms > 0 else bed


def _synthesize(tts, text: str, rng: random.Random, voice: str,
                speed: float) -> np.ndarray:
    """Render `text` with the voice/speed the builder drew from the config.

    `TTSEngine` is only `synthesize(text, rng)`, and Kokoro draws voice and
    speed from its own pools — but those draws belong to the builder, because
    they go in the manifest.  Engines taking explicit keywords get them;
    pool-style engines are pinned to the drawn values for the one call.
    """
    if {"voice", "speed"} <= set(inspect.signature(tts.synthesize).parameters):
        return tts.synthesize(text, rng, voice=voice, speed=speed)
    if hasattr(tts, "voices") and hasattr(tts, "speed_range"):
        pools = (tts.voices, tts.speed_range)
        tts.voices, tts.speed_range = [voice], (speed, speed)
        try:
            return tts.synthesize(text, rng)
        finally:
            tts.voices, tts.speed_range = pools
    return tts.synthesize(text, rng)


def _post(wav: np.ndarray, loudness_db: float | None) -> np.ndarray:
    """Post stage (02 §3): target loudness jitter, then peak safety."""
    x = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if loudness_db is not None and rms > 0:
        x = (x * (10.0 ** (float(loudness_db) / 20.0) / rms)).astype(np.float32)
    peak = float(np.abs(x).max())
    if peak > 0.99:
        x = (x * (0.99 / peak)).astype(np.float32)
    return x


def _histogram(values: list[float], edges: list[int]) -> dict:
    if not values:
        return {"edges": edges, "counts": [0] * (len(edges) - 1)}
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    return {"edges": edges, "counts": [int(count) for count in counts]}


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    p10, p50, p90 = (round(float(v), 3) for v in np.percentile(array, [10, 50, 90]))
    return {"n": len(values), "p10": p10, "p50": p50, "p90": p90,
            "mean": round(float(array.mean()), 3)}


def _stats(config: GeneratorConfig, rows: list[dict], sampler: WeightedSampler | None,
           tally: QCTally, kept_with_flag: int, resolved_hash: str) -> dict:
    """Run summary for `stats.json` (02 §6, 05 §2)."""
    durations = [row["duration"] for row in rows]
    snrs = [row["gen"]["channel"]["snr_db"] for row in rows
            if row["gen"]["channel"].get("snr_db") is not None]
    categories = Counter(row["category"] for row in rows)
    speech = [row for row in rows if row["category"] != "noise"]
    speech_total = len(speech) or 1
    speech_categories = Counter(row["category"] for row in speech)
    quotas = config.dataset.category_quotas
    return {
        "n_samples": len(rows),
        "mode": config.mode,
        "seed": config.seed,
        "config_hash": resolved_hash,
        "backends": dict(Counter(row["gen"]["mode"] for row in rows)),
        "categories": dict(categories),
        "category_fractions": {name: round(count / (len(rows) or 1), 4)
                               for name, count in sorted(categories.items())},
        "quotas": {
            "targets": dict(quotas),
            # achieved over speech samples: noise-only is not part of the text pool
            "achieved": {name: round(speech_categories.get(name, 0) / speech_total, 4)
                         for name in sorted(quotas)},
            "unavailable_categories": sampler.dropped_quotas if sampler else sorted(quotas),
        },
        "noise_only": {"target": config.dataset.noise_only_frac,
                       "achieved": round(categories.get("noise", 0) / (len(rows) or 1), 4)},
        "duration": {**_percentiles(durations),
                     "total_sec": round(sum(durations), 1),
                     "histogram": _histogram(durations, DURATION_EDGES)},
        "snr_db": {**_percentiles(snrs), "histogram": _histogram(snrs, SNR_EDGES)},
        "voices": dict(Counter(row["gen"]["voice"] for row in rows
                               if row["gen"].get("voice"))),
        "qc": {**tally.summary(), "enabled": config.qc.enabled,
               "asr_roundtrip": config.qc.asr_roundtrip,
               "max_retries": config.qc.max_retries,
               "kept_with_flag": kept_with_flag},
    }


def load_manifest(manifest_path: str | Path):
    """Load a built dataset as a HF Dataset with an Audio column."""
    from datasets import Audio, Dataset

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    records = [json.loads(line) for line in open(manifest_path) if line.strip()]
    for record in records:
        record["audio"] = str(root / record["audio"])
        # provenance is per-sample free-form; it is not a training column and
        # its ragged shape has no single Arrow schema
        record.pop("gen", None)
    ds = Dataset.from_list(records)
    return ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
