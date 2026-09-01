#!/usr/bin/env python
"""Rewrite absolute path prefixes inside CSV / JSON / JSONL manifests after moving the repo.

The real-audio manifests (`data/real/**/*.csv|jsonl`), the calibration corpus
(`runs/channel_data_kixd/corpus.jsonl`, `runs/calib_kixd/corpus.jsonl`) and
exported corpus `manifest.json` files all store *absolute* audio paths, by
design (the asr V2 schema). After copying the tree to another machine they
point at the old root. This rewrites the prefix in place:

    uv run python scripts/lab/relocate.py \
        --from /Users/kevin/repos/ai/atc-gan --to C:/ml/atc-gan \
        data/real runs/channel_data_kixd runs/calib_kixd --check --apply

Dry-run by default; `--apply` writes. `--check` verifies, after rewriting,
that every `audio` / `path` value points at an existing file. Backslash and
JSON-escaped (`\\\\`) spellings of the old prefix are rewritten too; the new
prefix is always written with forward slashes, which every consumer on
Windows accepts.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

EXTS = {".csv", ".jsonl", ".json"}
PATH_KEYS = ("audio", "path", "clips_dir", "wav")


def variants(prefix: str) -> list[str]:
    """Spellings of `prefix` to look for: JSON-escaped backslashes, backslashes, slashes."""
    fwd = prefix.replace("\\", "/").rstrip("/")
    back = fwd.replace("/", "\\")
    return list(dict.fromkeys([back.replace("\\", "\\\\"), back, fwd]))


def rewrite_text(text: str, src: str, dst: str) -> tuple[str, int]:
    """Replace every spelling of `src` with `dst` and normalise the rest of that path to '/'."""
    dst_fwd = dst.replace("\\", "/").rstrip("/")
    total = 0
    for v in variants(src):
        pat = re.compile(re.escape(v) + r'([^",\r\n]*)')

        def _sub(m: re.Match) -> str:
            tail = m.group(1).replace("\\\\", "/").replace("\\", "/")
            return dst_fwd + tail

        text, n = pat.subn(_sub, text)
        total += n
    return text, total


def iter_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(f for f in path.rglob("*") if f.suffix in EXTS and f.is_file()))
        elif path.is_file():
            out.append(path)
        else:
            print(f"warning: {p} does not exist, skipped", file=sys.stderr)
    return out


def referenced_paths(path: Path, text: str) -> list[str]:
    refs: list[str] = []
    if path.suffix == ".csv":
        for row in csv.DictReader(text.splitlines()):
            for k in PATH_KEYS:
                if row.get(k):
                    refs.append(row[k])
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                for k in PATH_KEYS:
                    if isinstance(row.get(k), str):
                        refs.append(row[k])
    elif path.suffix == ".json":
        try:
            obj = json.loads(text)
        except ValueError:
            return refs
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if k in PATH_KEYS and isinstance(v, str):
                        refs.append(v)
                    else:
                        stack.append(v)
            elif isinstance(cur, list):
                stack.extend(cur)
    return refs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", required=True, help="old absolute repo root")
    ap.add_argument("--to", dest="dst", required=True, help="new absolute repo root")
    ap.add_argument("paths", nargs="+", help="files or directories to rewrite")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--check", action="store_true",
                    help="after rewriting, verify referenced audio/path files exist")
    args = ap.parse_args(argv)

    files = iter_files(args.paths)
    changed = 0
    missing_total = 0
    for f in files:
        raw = f.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            print(f"skip (not utf-8): {f}", file=sys.stderr)
            continue
        new, n = rewrite_text(text, args.src, args.dst)
        missing = 0
        if args.check:
            refs = referenced_paths(f, new)
            missing = sum(1 for r in refs if Path(r).is_absolute() and not Path(r).exists())
            missing_total += missing
        if n:
            changed += 1
            if args.apply:
                f.write_bytes(new.encode("utf-8"))
        status = "rewrote" if (n and args.apply) else ("would rewrite" if n else "unchanged")
        extra = f", {missing} missing targets" if args.check else ""
        print(f"{status:<14} {n:>7} refs  {f}{extra}")
    mode = "applied" if args.apply else "dry run"
    print(f"\n{mode}: {changed}/{len(files)} files with the old prefix"
          + (f"; {missing_total} referenced files missing" if args.check else ""))
    return 1 if (args.check and missing_total) else 0


if __name__ == "__main__":
    sys.exit(main())
