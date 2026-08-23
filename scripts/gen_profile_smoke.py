"""Generate a small profile smoke set and compare its Tier 1 stats to real clips.

The P2 acceptance check of 03 §6: `matched` medians must land inside the real
p10-p90 for spectral edge, SNR and the LTAS-derived stats, and `wide` must be
strictly broader than `matched` on every reported histogram.  Deliberately a
thin wrapper — TTS, channel and stats all come from the library, so what this
script tests is the profile YAMLs and nothing else.

    uv run python scripts/gen_profile_smoke.py --config configs/mode1_matched.yaml \
        --out runs/p2_smoke/matched -n 100 --ref data/real/calibration

Kokoro runs on CPU/MPS; 100 clips takes a couple of minutes.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from atcgen.channel.chain import ProceduralChannel, UtteranceMeta
from atcgen.channel.primitives import TARGET_SR, NoiseBank, resample
from atcgen.config import load_config
from atcgen.eval.channel_stats import SCALAR_KEYS, compare, compute_stats
from atcgen.text.sources import GrammarTextSource
from atcgen.tts.synthesize import KokoroTTS


def _post(wav: np.ndarray, loudness_db: float | None) -> np.ndarray:
    """Loudness jitter then peak safety — the same post stage as the builder."""
    x = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if loudness_db is not None and rms > 0:
        x = (x * (10.0 ** (float(loudness_db) / 20.0) / rms)).astype(np.float32)
    peak = float(np.abs(x).max())
    return (x * (0.99 / peak)).astype(np.float32) if peak > 0.99 else x


def _voice_layer(config, cache_dir: Path, count: int, seed: int) -> list[dict]:
    """Clean TTS renders, cached on disk.

    The cache is per profile because `tts.speed` is part of a profile — `wide`
    draws 0.95-1.55x where `matched` stops at 1.45x, and clip duration is a
    reported statistic.  Both profiles still draw the same utterance texts from
    the same seed sequence, so the channel comparison stays like-for-like.
    Caching also keeps profile tuning fast: Kokoro dominates the runtime, and a
    re-measure after a YAML edit costs seconds instead of minutes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.jsonl"
    records = ([json.loads(line) for line in index_path.read_text().splitlines() if line]
               if index_path.exists() else [])
    if len(records) >= count:
        return records[:count]

    source = GrammarTextSource()
    tts = KokoroTTS(voices=config.tts.voices,
                    speed_range=tuple(config.tts.speed.value)
                    if config.tts.speed.kind == "uniform" else (1.0, 1.0))
    for index in range(len(records), count):
        rng = random.Random(seed * 1_000_003 + index)
        utterance = source.sample(rng)
        wav = resample(tts.synthesize(utterance.spoken, rng), tts.sample_rate, TARGET_SR)
        name = f"tts_{index:05d}.wav"
        sf.write(cache_dir / name, wav, TARGET_SR)
        records.append({"name": name, "role": utterance.role, "kind": utterance.kind})
        if (index + 1) % 25 == 0:
            print(f"  tts {index + 1}/{count}")
    index_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def generate(config_path: str, out_dir: Path, count: int, seed: int,
             cache_dir: Path | None) -> Path:
    config = load_config(config_path)
    beds_dir = config.channel.noise.beds_dir
    noise_bank = NoiseBank(beds_dir) if beds_dir is not None else None
    channel = ProceduralChannel.from_config(config.channel, noise_bank=noise_bank)
    print(f"  noise beds: {beds_dir or 'none (synthetic pink/white only)'}")
    cache_dir = cache_dir or Path(f"runs/p2_smoke/tts_{config.channel.profile}")
    records = _voice_layer(config, cache_dir, count, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    previous = None
    for index, record in enumerate(records):
        rng = random.Random(seed * 1_000_003 + index)
        wav, _ = sf.read(cache_dir / record["name"], dtype="float32")
        hops = 2 if (record["role"] == "pilot"
                     and rng.random() < config.dataset.pilot_double_hop_prob) else 1
        degraded, _ = channel(wav, TARGET_SR, rng,
                              UtteranceMeta(role=record["role"], kind=record["kind"]),
                              interference=previous, hops=hops)
        degraded = _post(degraded, config.output.loudness_db.sample(rng))
        sf.write(out_dir / f"smoke_{index:05d}.wav", degraded, TARGET_SR)
        previous = degraded
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default="data/real/calibration",
                    help="real clips to compare against; skipped when absent")
    ap.add_argument("--stats-out", help="write the full stats JSON here")
    ap.add_argument("--tts-cache",
                    help="clean-TTS cache dir; defaults to one per profile, since "
                         "the profiles configure different speaking-rate ranges")
    ap.add_argument("--skip-generate", action="store_true",
                    help="only re-measure wavs already in --out")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not args.skip_generate:
        print(f"generating {args.count} clips from {args.config} -> {out}")
        generate(args.config, out, args.count, args.seed,
                 Path(args.tts_cache) if args.tts_cache else None)

    stats = compute_stats(out)
    reference = Path(args.ref)
    comparison = compare(stats, compute_stats(reference)) if reference.is_dir() else None

    print(f"\n{stats['n']} clips from {out}")
    for key in SCALAR_KEYS:
        summary = stats["summary"][key]
        line = (f"  {key:17s} p10={summary['p10']:>9.2f}  p50={summary['p50']:>9.2f}  "
                f"p90={summary['p90']:>9.2f}  spread={summary['p90'] - summary['p10']:>9.2f}")
        if comparison:
            entry = comparison["stats"][key]
            line += (f"   real p10-p90=[{entry['real_p10']:>8.2f},{entry['real_p90']:>8.2f}]"
                     f"  in_range={str(entry['median_in_range']):5s}")
        print(line)
    if comparison:
        print(f"  LTAS L1 distance: {comparison['ltas_l1_db']:.2f} dB")
        print(f"  all medians in range: {comparison['all_medians_in_range']}")
        stats["comparison"] = comparison
    if args.stats_out:
        path = Path(args.stats_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
