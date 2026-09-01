# Agent / contributor notes

## Pre-release status: no compatibility or versioning obligations

This project is **not released and nothing is pinned**. Until that changes:

- **No backwards compatibility.** Configs, schemas (manifest rows, `RewardResult`,
  trial logs, report JSON), module APIs, and CLI flags may be changed or broken
  freely. Do not add deprecation shims, legacy code paths, or migration code.
- **No old-profile maintenance.** Superseded YAML profiles under `configs/` and
  stale run directories under `runs/` can be edited, renamed, or deleted rather
  than preserved. Don't keep a config working just because an old run used it.
- **No experiment-tracking / reproducibility versioning yet.** There is no
  requirement that today's code reproduce yesterday's runs. Existing provenance
  (resolved-config dumps, `config_hash`, `trials.jsonl`) is for *within-run*
  comparison and debugging, not a compatibility contract — don't build schema
  versioning, migration tooling, or frozen-baseline registries around it.

When the project cuts a first release/pin, this section gets replaced with the
actual compatibility policy. Until then, prefer the simplest change that moves
the work forward.

## MLBucket run tracking is optional

`mlbucket` is deliberately not a project dependency, so `uv sync` and the test
suite work on a machine with no MLBucket checkout and no server.
`atcgen/tracking.py` is the only seam: `start_run` returns a no-op run — with
one warning — when the SDK is missing or its server (`MLBUCKET_SERVER_URL`,
default http://localhost:8484) does not accept a connection within a second,
and `ATCGAN_TRACKING=off` silences tracking entirely. On a machine with a
sibling checkout, opt in with:

    uv pip install -e ../MLBucket/sdk

`uv run` keeps that install, but an explicit `uv sync` removes it (it is not in
the lockfile) — re-run the command afterwards, or use `uv sync --inexact`.
