"""One-command regression harness for synthetic ATC generator versions.

The harness composes the individual evaluation modules without reimplementing
their metrics.  Tier 0 is copied from a generation run's ``stats.json``;
Tier 1 channel statistics are computed here; and the embedding/probe modules
are loaded only when those tiers are requested.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from .channel_stats import compare as compare_channel_stats
from .channel_stats import compute_stats

DISCARD_GATE = 0.15
PROBE_GATE = 0.65
SCHEMA_VERSION = 1
DEFAULT_OUT = Path("runs/eval/versions")


@dataclass(frozen=True)
class EvaluationArtifacts:
    """In-memory result and the versioned artifacts written for one run."""

    result: dict
    json_path: Path
    html_path: Path | None = None


@dataclass(frozen=True)
class _Inputs:
    run_dir: Path
    ref_dir: Path
    synthetic_wavs: tuple[Path, ...]
    reference_wavs: tuple[Path, ...]
    manifest_path: Path | None
    manifest_count: int | None
    stats_path: Path | None
    run_stats: dict | None
    config_hash: str | None


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("audio"), str):
            raise ValueError(f"{path}:{line_number} must contain a string 'audio' field")
        rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _resolved_config_hash(run_dir: Path, stats: Mapping | None) -> str | None:
    if stats:
        value = stats.get("config_hash")
        if isinstance(value, str) and value:
            return value
        resolved = stats.get("resolved_config")
        if isinstance(resolved, Mapping):
            value = resolved.get("config_hash")
            if isinstance(value, str) and value:
                return value
    path = run_dir / "config.resolved.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _limited(paths: list[Path], n_max: int | None) -> tuple[Path, ...]:
    return tuple(paths if n_max is None else paths[:n_max])


def _resolve_inputs(run_dir: str | Path, ref_dir: str | Path, *,
                    wavs_only: bool, n_max: int | None) -> _Inputs:
    if n_max is not None and n_max <= 0:
        raise ValueError("n_max must be positive")
    run = Path(run_dir).resolve()
    ref = Path(ref_dir).resolve()
    if not run.is_dir():
        raise ValueError(f"run directory does not exist: {run}")
    if not ref.is_dir():
        raise ValueError(f"reference directory does not exist: {ref}")

    ref_wavs = sorted(ref.glob("*.wav"))
    if not ref_wavs:
        raise ValueError(f"no .wav files found in reference directory: {ref}")

    if wavs_only:
        syn_wavs = sorted(run.glob("*.wav"))
        manifest_path = stats_path = None
        manifest_count = None
        stats = None
    else:
        manifest_path = run / "manifest.jsonl"
        stats_path = run / "stats.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing manifest.jsonl in run directory: {run}")
        if not stats_path.is_file():
            raise ValueError(f"missing stats.json in run directory: {run}")
        rows = _read_manifest(manifest_path)
        manifest_count = len(rows)
        syn_wavs = []
        for row in rows:
            wav = Path(row["audio"])
            wav = wav if wav.is_absolute() else run / wav
            if not wav.is_file():
                raise ValueError(f"manifest audio does not exist: {wav}")
            syn_wavs.append(wav.resolve())
        stats = _read_json(stats_path)

    if not syn_wavs:
        location = run if wavs_only else manifest_path
        raise ValueError(f"no synthetic .wav files found from {location}")
    return _Inputs(
        run_dir=run,
        ref_dir=ref,
        synthetic_wavs=_limited(syn_wavs, n_max),
        reference_wavs=_limited(ref_wavs, n_max),
        manifest_path=manifest_path,
        manifest_count=manifest_count,
        stats_path=stats_path,
        run_stats=stats,
        config_hash=_resolved_config_hash(run, stats),
    )


def _criterion(status: str, *, value=None, threshold: str,
               passed: bool | None, note: str) -> dict:
    return {"note": note, "pass": passed, "status": status,
            "threshold": threshold, "value": value}


def build_verdict(tier0: Mapping, channel_comparison: Mapping,
                  probe_result: Mapping | None) -> dict:
    """Evaluate the train-ready gates available to the Tier 0--2 harness."""
    qc = tier0.get("stats") if tier0.get("status") == "recorded" else None
    discard = qc.get("discard_rate") if isinstance(qc, Mapping) else None
    if isinstance(discard, (int, float)):
        tier0_gate = _criterion(
            "pass" if discard < DISCARD_GATE else "fail",
            value=discard,
            threshold=f"< {DISCARD_GATE}",
            passed=bool(discard < DISCARD_GATE),
            note="Recorded generation discard rate; no Tier 0 recomputation.",
        )
    else:
        tier0_gate = _criterion(
            "not_evaluated", threshold=f"< {DISCARD_GATE}", passed=None,
            note="Tier 0 requires a generation stats.json (unavailable in wavs-only mode).",
        )

    stats = channel_comparison.get("stats", {})
    failed = sorted(name for name, item in stats.items()
                    if not item.get("median_in_range", False))
    medians_pass = bool(stats) and not failed
    tier1_gate = _criterion(
        "pass" if medians_pass else "fail",
        value={"failed_statistics": failed,
               "statistics_evaluated": len(stats)},
        threshold="every synthetic p50 inside real p10-p90",
        passed=medians_pass,
        note="Applies to scalar channel statistics; embedding distances have no median gate.",
    )

    if probe_result is None:
        tier2_gate = _criterion(
            "not_evaluated", threshold=f"<= {PROBE_GATE}", passed=None,
            note="WavLM embeddings and the channel probe were skipped.",
        )
    else:
        accuracy = probe_result.get("balanced_accuracy")
        if not isinstance(accuracy, (int, float)):
            raise ValueError("probe result lacks numeric balanced_accuracy")
        tier2_gate = _criterion(
            "pass" if accuracy <= PROBE_GATE else "fail",
            value=accuracy,
            threshold=f"<= {PROBE_GATE}",
            passed=bool(accuracy <= PROBE_GATE),
            note="Uses k-fold balanced accuracy on frozen WavLM embeddings.",
        )

    computable = [tier0_gate, tier1_gate, tier2_gate]
    covered = all(item["pass"] is not None for item in computable)
    passed = all(item["pass"] is True for item in computable) if covered else None
    overall_status = "pass" if passed is True else "fail" if passed is False else "incomplete"
    return {
        "criteria": {
            "tier0_discard_rate": tier0_gate,
            "tier1_channel_medians": tier1_gate,
            "tier2_probe_accuracy": tier2_gate,
            "tier3_downstream_wer": _criterion(
                "not_covered", threshold="synthetic+real >= real-only baseline",
                passed=None,
                note="Tier 3 requires the separate Whisper fine-tuning protocol.",
            ),
        },
        "overall": {
            "all_tier0_to_2_criteria_covered": covered,
            "pass": passed,
            "scope": "Tier 0-2 train-ready criteria only",
            "status": overall_status,
            "tier3_covered": False,
            "train_ready": None,
            "note": "Full train-ready status is undetermined until Tier 3 passes.",
        },
    }


def _tool_versions() -> dict:
    versions = {"python": platform.python_version()}
    for distribution in ("atc-gan", "numpy", "scipy", "soundfile", "torch",
                         "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return clean or "evaluation"


def _write_json(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    identity = result["config_hash"] or result["run_name"]
    stamp = datetime.fromisoformat(result["created_utc"].replace("Z", "+00:00"))
    # Microseconds make the "one JSON per invocation" guarantee hold even for
    # two fast stats-only invocations of the same generator version.
    path = out / f"{_safe_name(identity)}_{stamp.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def _audition(inputs: _Inputs, identity: str) -> list[tuple[str, Path]]:
    synthetic = [(f"synthetic: {path.name}", path)
                 for path in inputs.synthetic_wavs[:10]]
    rng = random.Random(identity)
    count = min(5, len(inputs.reference_wavs))
    real = rng.sample(list(inputs.reference_wavs), count)
    return synthetic + [(f"real: {path.name}", path) for path in real]


def _normalize_source_labels(result: dict, inputs: _Inputs) -> dict:
    """Replace module stringifications of path lists with concise provenance."""
    synthetic_source = (inputs.run_dir if inputs.manifest_path is None
                        else inputs.run_dir / "wavs")
    if "synthetic_dir" in result:
        result["synthetic_dir"] = str(synthetic_source)
    if "real_dir" in result:
        result["real_dir"] = str(inputs.ref_dir)
    return result


def run_evaluation(run_dir: str | Path, ref_dir: str | Path, *,
                   out_dir: str | Path = DEFAULT_OUT,
                   skip_embeddings: bool = False,
                   n_max: int | None = None,
                   wavs_only: bool = False,
                   html: bool = False,
                   embedding_compare: Callable | None = None,
                   probe_runner: Callable | None = None,
                   report_builder: Callable | None = None,
                   now: datetime | None = None) -> EvaluationArtifacts:
    """Run requested tiers and write one immutable, diffable version JSON.

    The three injectable callables are test seams for expensive embedding,
    probe, and HTML work.  Production defaults are the existing eval module
    functions.
    """
    inputs = _resolve_inputs(run_dir, ref_dir, wavs_only=wavs_only, n_max=n_max)
    synthetic = compute_stats(inputs.synthetic_wavs)
    real = compute_stats(inputs.reference_wavs)
    channel_comparison = compare_channel_stats(synthetic, real)

    if skip_embeddings:
        embeddings = {"reason": "--skip-embeddings", "status": "skipped"}
        probe = {"reason": "--skip-embeddings", "status": "skipped"}
        probe_result = None
    else:
        if embedding_compare is None:
            from .embed_dist import compare_dirs
            embedding_compare = compare_dirs
        if probe_runner is None:
            from .probe import probe_dirs
            probe_runner = probe_dirs
        embeddings_result = _normalize_source_labels(
            dict(embedding_compare(inputs.synthetic_wavs, inputs.reference_wavs)),
            inputs,
        )
        probe_result = _normalize_source_labels(
            dict(probe_runner(inputs.synthetic_wavs, inputs.reference_wavs)), inputs,
        )
        embeddings = {"result": embeddings_result, "status": "evaluated"}
        probe = {"result": probe_result, "status": "evaluated"}

    tier0 = ({"source": str(inputs.stats_path), "stats": inputs.run_stats.get("qc"),
              "status": "recorded"}
             if inputs.run_stats is not None and isinstance(inputs.run_stats.get("qc"), dict)
             else {"reason": "wavs-only mode has no generation stats",
                   "status": "not_evaluated"})
    created = _utc(now)
    result = {
        "config_hash": inputs.config_hash,
        "created_utc": created.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "inputs": {
            "manifest": str(inputs.manifest_path) if inputs.manifest_path else None,
            "manifest_samples": inputs.manifest_count,
            "n_max": n_max,
            "reference_dir": str(inputs.ref_dir),
            "reference_samples_evaluated": len(inputs.reference_wavs),
            "run_dir": str(inputs.run_dir),
            "stats": str(inputs.stats_path) if inputs.stats_path else None,
            "synthetic_samples_evaluated": len(inputs.synthetic_wavs),
            "wavs_only": wavs_only,
        },
        "run_name": inputs.run_dir.name,
        "schema_version": SCHEMA_VERSION,
        "tier0": tier0,
        "tier1": {
            "channel": {"comparison": channel_comparison, "real": real,
                        "synthetic": synthetic},
            "embeddings": embeddings,
        },
        "tier2": {"probe": probe},
        "tool_versions": _tool_versions(),
    }
    result["verdict"] = build_verdict(tier0, channel_comparison, probe_result)
    json_path = _write_json(result, out_dir)

    html_path = None
    if html:
        if report_builder is None:
            from .report import build_report
            report_builder = build_report
        html_path = json_path.with_suffix(".html")
        identity = result["config_hash"] or result["run_name"]
        report_builder(
            html_path, synthetic, real=real, comparison=channel_comparison,
            audition=_audition(inputs, identity),
            title=f"Synthetic ATC evaluation — {result['run_name']}",
            qc_summary=tier0.get("stats"),
        )
    return EvaluationArtifacts(result=result, json_path=json_path, html_path=html_path)


def _metrics(result: Mapping) -> dict[str, float]:
    metrics: dict[str, float] = {}
    discard = result.get("tier0", {}).get("stats", {}).get("discard_rate")
    if isinstance(discard, (int, float)):
        metrics["tier0.discard_rate"] = float(discard)

    comparison = result.get("tier1", {}).get("channel", {}).get("comparison", {})
    ltas = comparison.get("ltas_l1_db")
    if isinstance(ltas, (int, float)):
        metrics["tier1.channel.ltas_l1_db"] = float(ltas)
    for name, entry in sorted(comparison.get("stats", {}).items()):
        for field in ("synthetic_p50", "wasserstein", "wasserstein_norm"):
            value = entry.get(field)
            if isinstance(value, (int, float)):
                metrics[f"tier1.channel.{name}.{field}"] = float(value)

    embedding = result.get("tier1", {}).get("embeddings", {}).get("result", {})
    for family, entry in sorted(embedding.get("families", {}).items()):
        for field in ("kid", "frechet"):
            value = entry.get(field)
            if isinstance(value, (int, float)):
                metrics[f"tier1.embeddings.{family}.{field}"] = float(value)

    probe = result.get("tier2", {}).get("probe", {}).get("result", {})
    for field in ("balanced_accuracy", "null_balanced_accuracy"):
        value = probe.get(field)
        if isinstance(value, (int, float)):
            metrics[f"tier2.probe.{field}"] = float(value)
    return metrics


def format_diff(old: Mapping, new: Mapping) -> str:
    """Return a compact stable table of old/new values and ``new - old``."""
    old_metrics, new_metrics = _metrics(old), _metrics(new)
    names = sorted(set(old_metrics) | set(new_metrics))
    rows = [(name, old_metrics.get(name), new_metrics.get(name)) for name in names]
    header = f"{'metric':52} {'old':>12} {'new':>12} {'delta':>12}"
    separator = "-" * len(header)

    def number(value: float | None) -> str:
        return "-" if value is None else f"{value:.6g}"

    lines = [header, separator]
    for name, old_value, new_value in rows:
        delta = (new_value - old_value
                 if old_value is not None and new_value is not None else None)
        lines.append(f"{name:52} {number(old_value):>12} {number(new_value):>12} "
                     f"{number(delta):>12}")
    return "\n".join(lines)


def format_verdict(verdict: Mapping) -> str:
    criteria = verdict["criteria"]
    labels = (
        ("Tier 0 discard rate", "tier0_discard_rate"),
        ("Tier 1 channel medians", "tier1_channel_medians"),
        ("Tier 2 probe accuracy", "tier2_probe_accuracy"),
        ("Tier 3 downstream WER", "tier3_downstream_wer"),
    )
    lines = ["Verdict"]
    for label, key in labels:
        item = criteria[key]
        value = item.get("value")
        detail = f" value={value}" if value is not None else ""
        lines.append(f"  {label:24} {item['status'].upper():13} "
                     f"({item['threshold']}){detail}")
    overall = verdict["overall"]
    lines.append(f"  {'Overall (Tier 0-2)':24} {overall['status'].upper():13}")
    lines.append("  Full train-ready status: UNDETERMINED (Tier 3 not covered)")
    return "\n".join(lines)


def main(argv=None) -> EvaluationArtifacts:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="generated run directory")
    parser.add_argument("--ref", required=True, help="real calibration wav directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="version JSON output directory")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="run only Tier 0 and Tier 1 channel statistics")
    parser.add_argument("--n-max", type=int,
                        help="deterministically limit both sets to their first N wavs")
    parser.add_argument("--wavs-only", action="store_true",
                        help="run on a bare wav directory; Tier 0 is unavailable")
    parser.add_argument("--html", action="store_true",
                        help="write the E1 HTML report next to the version JSON")
    parser.add_argument("--diff", metavar="OLD_JSON",
                        help="print per-metric deltas from an older version JSON")
    args = parser.parse_args(argv)

    artifacts = run_evaluation(
        args.run_dir, args.ref, out_dir=args.out,
        skip_embeddings=args.skip_embeddings, n_max=args.n_max,
        wavs_only=args.wavs_only, html=args.html,
    )
    print(format_verdict(artifacts.result["verdict"]))
    print(f"wrote {artifacts.json_path}")
    if artifacts.html_path:
        print(f"wrote {artifacts.html_path}")
    if args.diff:
        print("\nDelta from " + str(args.diff))
        print(format_diff(_read_json(Path(args.diff)), artifacts.result))
    return artifacts


__all__ = [
    "DEFAULT_OUT", "DISCARD_GATE", "EvaluationArtifacts", "PROBE_GATE",
    "SCHEMA_VERSION", "build_verdict", "format_diff", "format_verdict",
    "main", "run_evaluation",
]
