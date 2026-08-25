"""Reject channel artifacts derived from validation receiver clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    corpus: str | Path,
    presets: str | Path,
    noise_stats: str | Path,
    out: str | Path,
    run_inputs: str | Path | None = None,
) -> dict:
    """Collect every partition and provenance failure in one report."""
    corpus_path = Path(corpus)
    presets_path = Path(presets)
    noise_path = Path(noise_stats)
    corpus_rows = _rows(corpus_path)
    preset_rows = _rows(presets_path)
    noise_rows = _rows(noise_path)
    train_ids = {row["clip_id"] for row in corpus_rows
                 if row.get("split") == "channel_train"}

    forbidden: list[dict] = []
    for row in preset_rows:
        reasons = []
        if row.get("clip_id") not in train_ids:
            reasons.append("clip_id not in corpus channel_train")
        if row.get("split") != "channel_train":
            reasons.append("split is not channel_train")
        if reasons:
            forbidden.append({
                "artifact": "presets",
                "clip_id": row.get("clip_id"),
                "reasons": reasons,
            })
    for row in noise_rows:
        reasons = []
        if row.get("source_clip") not in train_ids:
            reasons.append("source_clip not in corpus channel_train")
        if row.get("split") != "channel_train":
            reasons.append("split is not channel_train")
        if reasons:
            forbidden.append({
                "artifact": "noise_stats",
                "clip_id": row.get("source_clip"),
                "reasons": reasons,
            })

    mismatches: list[dict] = []
    if run_inputs is not None:
        inputs_path = Path(run_inputs)
        if not inputs_path.is_file():
            mismatches.append({
                "artifact": "run_inputs",
                "reason": f"file does not exist: {inputs_path}",
            })
        else:
            inputs = json.loads(inputs_path.read_text())
            for name, path in (("presets", presets_path),
                               ("noise_stats", noise_path)):
                recorded = inputs.get(name, {}).get("sha256")
                actual = _sha256(path)
                if recorded != actual:
                    mismatches.append({
                        "artifact": name,
                        "recorded_sha256": recorded,
                        "actual_sha256": actual,
                    })

    report = {
        "ok": not forbidden and not mismatches,
        "checked": {
            "corpus": len(corpus_rows),
            "channel_train": len(train_ids),
            "presets": len(preset_rows),
            "noise_stats": len(noise_rows),
        },
        "forbidden": forbidden,
        "mismatches": mismatches,
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--noise-stats", type=Path, required=True)
    parser.add_argument("--run-inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(
        args.corpus, args.presets, args.noise_stats, args.out, args.run_inputs)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
