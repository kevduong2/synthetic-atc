"""Capped domain randomization: is a profile still inside the real envelope?

research-findings §4.3 puts the channel twin in two operating modes —
reconstruction, and *capped* domain randomization, "envelope learned from real
recordings", because unlimited distortion manufactures audio whose transcript
is no longer recoverable, i.e. mislabeled training data.  Nothing checked that
cap.  This module measures the envelope from the real corpus's own Tier 1
statistics and reports, per profile, where the randomization ranges run past
it.  It is a report, never a failure: `wide` explores past the cap on purpose
(03 §1), and the point is to know by how much.

Config space vs measured space
------------------------------
A profile declares *injected* values; the envelope is in *measured* Tier 1
statistics, and the two are not the same scale.  Injecting a 16 dB SNR reads
back as 28 dB, because the squelch gate and the harvested beds make the noise
floor non-stationary and `channel_stats`' p90/p15 estimator is sensitive to
that (see `configs/mode1_matched.yaml`'s header).  A 2750 Hz filter corner
reads back as a 2391 Hz 98%-power edge, because speech has little energy up
there to begin with.  Comparing the raw numbers would flag the profile that
was *fitted* to the real set, which is the opposite of useful.

So each rule carries a measured offset (`measured - injected`) and the check
converts the envelope into config space before comparing.  The offsets come
from one run — 200 clips of `mode1_matched` (`runs/p4_matched_stats.json`)
against that profile's own declared distributions — and `calibrate()` re-derives
them, so run it whenever a chain change moves a Tier 1 median.  `loudness_db`
is the control: the post stage applies it literally, so its offset should be
zero, and measuring it gives -0.3 dB.

Two limits worth stating.  The offset is a linear shift fitted at one point, so
it overstates how far a corner can move — the measured spectral edge is bounded
above by the speech's own content, and raising the filter corner past ~3.5 kHz
barely moves it.  And because the calibration profile is `matched`, the check
really asks "does this randomize further than the profile fitted to the real
corpus, by more than the slack", with the real p10-p90 setting the anchor.
That is the honest reading of what one measured transfer supports.

CLI
---
    python -m atcgen.channel.envelope --config configs/mode1_wide.yaml
    python -m atcgen.channel.envelope --stats runs/mode1_matched_stats.json \\
        --out configs/real_envelope.json          # regenerate the snapshot
    python -m atcgen.channel.envelope --config configs/mode1_matched.yaml \\
        --calibrate runs/p4_matched_stats.json    # re-derive the RULES offsets
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import ChannelConfig, DistSpec, GeneratorConfig

SNAPSHOT = Path(__file__).resolve().parents[2] / "configs" / "real_envelope.json"
METRICS = ("snr_db", "spectral_edge_hz", "spectral_low_hz", "rms_db")
REGENERATE = ("python -m atcgen.channel.envelope --stats runs/mode1_matched_stats.json"
              " runs/mode2_calibrated_stats.json --out configs/real_envelope.json")


@dataclass(frozen=True)
class Rule:
    """One randomization range, and the measured statistic that caps it.

    `offset` is `measured - injected` for this quantity: the envelope's real
    percentiles are shifted by `-offset` to land in config space.  `slack` is
    how far past that a profile may go before it is worth mentioning, and is
    per-metric because a dB and a hertz are not comparable.
    """

    label: str
    metric: str
    param: str
    primitive: str | None      # None: the param lives under `output`
    offset: float
    slack: float
    unit: str


RULES = (
    # +14.0 dB: an injected median of 16.2 dB reads back as 30.2 dB.  The slack
    # is wide because this offset is the least linear of the four.
    Rule("additive_noise.snr_db", "snr_db", "snr_db", "additive_noise",
         offset=14.0, slack=5.0, unit="dB"),
    # -964 Hz: a 3300 Hz corner reads back as a 2336 Hz 98%-power edge, because
    # `reapply_bandpass` runs that filter about three times and speech has
    # little energy up there to begin with.
    Rule("bandpass.high", "spectral_edge_hz", "high", "bandpass",
         offset=-964.1, slack=300.0, unit="Hz"),
    # +13 Hz, the same cascade seen from the bottom of the band.
    Rule("bandpass.low", "spectral_low_hz", "low", "bandpass",
         offset=13.1, slack=50.0, unit="Hz"),
    # -0.3 dB: the control rule -- the post stage applies this one literally, so
    # its offset should be zero, and measuring it gives -0.3.
    Rule("output.loudness_db", "rms_db", "loudness_db", None,
         offset=-0.3, slack=2.0, unit="dB"),
)


def _bounds(spec: DistSpec) -> tuple[float, float] | None:
    """The (lowest, highest) value a `DistSpec` can draw, if it is numeric."""
    if spec.kind == "uniform":
        return float(spec.value[0]), float(spec.value[1])
    if spec.kind == "beta_scaled":
        return float(spec.value[2]), float(spec.value[3])
    if spec.kind == "const":
        return (float(spec.value), float(spec.value)) if isinstance(
            spec.value, (int, float)) and not isinstance(spec.value, bool) else None
    if spec.kind == "choice":
        numbers = [float(v) for v in spec.value
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return (min(numbers), max(numbers)) if numbers else None
    return None


def measure_envelope(stats_json_paths: Iterable[str | Path]) -> dict[str, Any]:
    """The real corpus's p10-p90 envelope, distilled from `channel_stats` JSON.

    Those files carry the *real* reference percentiles alongside the synthetic
    ones (`comparison.stats[metric].real_p10/real_p90`), which is the whole
    envelope we need and the only place it is already computed.  Several files
    are merged by widening — they are usually the same reference set seen from
    different profiles, and taking the union is the conservative reading if
    they ever are not.
    """
    metrics: dict[str, dict[str, float]] = {}
    sources: list[str] = []
    n_real = 0
    for path in stats_json_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        comparison = data.get("comparison") or {}
        stats = comparison.get("stats") or {}
        found = False
        for metric in METRICS:
            entry = stats.get(metric) or {}
            if "real_p10" not in entry or "real_p90" not in entry:
                continue
            found = True
            low, high = float(entry["real_p10"]), float(entry["real_p90"])
            current = metrics.setdefault(metric, {"p10": low, "p90": high})
            current["p10"] = min(current["p10"], low)
            current["p90"] = max(current["p90"], high)
        if found:
            sources.append(str(path))
            n_real = max(n_real, int(comparison.get("n_real", 0)))
    if not metrics:
        raise ValueError("no comparison.stats real percentiles in: "
                         + ", ".join(str(p) for p in stats_json_paths))
    return {"_comment": "Measured p10-p90 envelope of the real calibration corpus, "
                        "read from channel_stats comparison output. Regenerate with: "
                        + REGENERATE,
            "n_real": n_real, "sources": sources, "metrics": metrics}


@lru_cache(maxsize=4)
def load_envelope(path: str | Path | None = None) -> dict[str, Any] | None:
    """The committed snapshot, or None when it is missing or unreadable."""
    try:
        data = json.loads(Path(path or SNAPSHOT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data.get("metrics"), dict) else None


def check_profile(config: GeneratorConfig | ChannelConfig, envelope: Mapping[str, Any],
                  slack: Mapping[str, float] | None = None) -> list[str]:
    """Human-readable findings where `config` randomizes past `envelope`.

    Accepts a whole `GeneratorConfig` or just its `ChannelConfig` (the chain
    rules are all that can be checked in the latter case).  `slack` overrides
    the per-rule defaults by rule label.  A rule whose parameter the profile
    does not declare is skipped rather than guessed at.
    """
    channel = config if isinstance(config, ChannelConfig) else getattr(config, "channel", None)
    output = None if isinstance(config, ChannelConfig) else getattr(config, "output", None)
    metrics = envelope.get("metrics") or {}
    findings = []

    for rule in RULES:
        bounds = envelope_bounds(rule, metrics)
        drawn = _profile_range(rule, channel, output)
        if bounds is None or drawn is None:
            continue
        allowed = float((slack or {}).get(rule.label, rule.slack))
        low, high = drawn
        for value, bound, side in ((low, bounds[0], "low"), (high, bounds[1], "high")):
            past = (bound - value) if side == "low" else (value - bound)
            if past > allowed:
                findings.append(
                    f"{rule.label} {side} edge {value:g} {rule.unit} is {past:.1f}"
                    f" {rule.unit} past the real p{10 if side == 'low' else 90}"
                    f" ({bound:.1f} {rule.unit} in config space; slack"
                    f" {allowed:g} {rule.unit})")
    return findings


def envelope_bounds(rule: Rule, metrics: Mapping[str, Any]) -> tuple[float, float] | None:
    """`rule`'s real p10/p90, shifted into the config's own units."""
    entry = metrics.get(rule.metric)
    if not isinstance(entry, Mapping) or "p10" not in entry or "p90" not in entry:
        return None
    return float(entry["p10"]) - rule.offset, float(entry["p90"]) - rule.offset


def _profile_range(rule: Rule, channel: ChannelConfig | None,
                   output: Any) -> tuple[float, float] | None:
    """The widest range `rule`'s parameter can draw anywhere in the profile."""
    if rule.primitive is None:
        spec = getattr(output, rule.param, None)
        return _bounds(spec) if isinstance(spec, DistSpec) else None
    if channel is None:
        return None
    spans = [span for step in channel.chain if step.primitive == rule.primitive
             for spec in [step.params.get(rule.param)]
             if isinstance(spec, DistSpec) and (span := _bounds(spec)) is not None]
    if not spans:
        return None
    return min(s[0] for s in spans), max(s[1] for s in spans)


def _injected_median(spec: DistSpec) -> float | None:
    """The middle of what `spec` draws, in the config's own units."""
    if spec.kind == "uniform":
        return 0.5 * (float(spec.value[0]) + float(spec.value[1]))
    if spec.kind == "beta_scaled":
        from scipy.stats import beta

        alpha, beta_, low, high = (float(v) for v in spec.value)
        return low + float(beta.ppf(0.5, alpha, beta_)) * (high - low)
    bounds = _bounds(spec)
    return None if bounds is None else 0.5 * (bounds[0] + bounds[1])


def calibrate(stats: Mapping[str, Any], config: GeneratorConfig) -> dict[str, float]:
    """Re-derive each rule's `measured - injected` offset from a measured run.

    `stats` is a `channel_stats` report for clips generated *by this config*;
    the offset is that run's median for the rule's metric minus the middle of
    the distribution the profile injects.  Run it whenever a chain change moves
    a Tier 1 median, and paste the numbers into `RULES`.
    """
    summary = stats.get("summary") or {}
    channel = getattr(config, "channel", None)
    output = getattr(config, "output", None)
    offsets = {}
    for rule in RULES:
        measured = (summary.get(rule.metric) or {}).get("p50")
        spec = (getattr(output, rule.param, None) if rule.primitive is None else
                next((step.params.get(rule.param) for step in (channel.chain if channel
                      else []) if step.primitive == rule.primitive), None))
        injected = _injected_median(spec) if isinstance(spec, DistSpec) else None
        if measured is not None and injected is not None:
            offsets[rule.label] = round(float(measured) - injected, 1)
    return offsets


def report(config: GeneratorConfig, envelope: Mapping[str, Any]) -> str:
    """The CLI's full picture: every rule, in or out, with its measured bound."""
    channel = getattr(config, "channel", None)
    output = getattr(config, "output", None)
    metrics = envelope.get("metrics") or {}
    findings = set(check_profile(config, envelope))
    lines = [f"real envelope: n={envelope.get('n_real', '?')} clips"
             f" from {', '.join(envelope.get('sources', []) or ['?'])}"]
    for rule in RULES:
        bounds = envelope_bounds(rule, metrics)
        drawn = _profile_range(rule, channel, output)
        if bounds is None or drawn is None:
            lines.append(f"  skip {rule.label:24s} not declared by this profile,"
                         " or not in the envelope")
            continue
        hit = [f for f in findings if f.startswith(rule.label + " ")]
        mark = "WARN" if hit else "ok  "
        lines.append(f"  {mark} {rule.label:24s} profile [{drawn[0]:g}, {drawn[1]:g}]"
                     f" vs real [{bounds[0]:.1f}, {bounds[1]:.1f}] {rule.unit}"
                     f" (p10-p90 in config space, slack {rule.slack:g})")
        lines.extend(f"       {f}" for f in hit)
    if not findings:
        lines.append("  every checked range is inside the measured envelope")
    return "\n".join(lines)


def main(argv=None) -> int:
    from ..config import load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="generator profile YAML to check")
    ap.add_argument("--envelope", default=None,
                    help=f"envelope snapshot (default: {SNAPSHOT})")
    ap.add_argument("--stats", nargs="+", default=None,
                    help="channel_stats JSON file(s) to distill a fresh envelope from")
    ap.add_argument("--out", default=None, help="write the distilled envelope here")
    ap.add_argument("--calibrate", default=None,
                    help="channel_stats JSON of a run generated by --config; prints"
                         " each rule's re-derived measured-minus-injected offset")
    args = ap.parse_args(argv)

    if args.calibrate:
        if not args.config:
            ap.error("--calibrate needs the --config that generated those clips")
        stats = json.loads(Path(args.calibrate).read_text(encoding="utf-8"))
        offsets = calibrate(stats, load_config(args.config))
        width = max(len(name) for name in offsets) if offsets else 0
        for rule in RULES:
            if rule.label in offsets:
                print(f"  {rule.label:{width}s} offset={offsets[rule.label]:>8.1f}"
                      f" {rule.unit}  (RULES says {rule.offset:g})")
        return 0

    if args.stats:
        envelope = measure_envelope(args.stats)
        if args.out:
            Path(args.out).write_text(json.dumps(envelope, indent=2) + "\n",
                                      encoding="utf-8")
            print(f"wrote {args.out}")
    else:
        envelope = load_envelope(args.envelope)
        if envelope is None:
            print(f"no envelope snapshot at {args.envelope or SNAPSHOT}; regenerate with:"
                  f"\n  {REGENERATE}", file=sys.stderr)
            return 1
    if args.config:
        print(report(load_config(args.config), envelope))
    elif not args.stats:
        print(json.dumps(envelope, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
