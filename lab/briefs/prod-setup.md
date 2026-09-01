# Brief prod-setup            to: any agent with a shell on the 3080 box   from: Kevin

Goal: put the gitignored payload in place and prove the environment, so the
`lab-director` can start mission `prod-v1` without touching a broken machine.
CPU-only. You do not launch the mission, any GPU job, or any script under
`scripts/` except the checks listed here.

Inputs (Kevin provides the paths when he hands you this brief):
- `<CLIPS_ZIP>`: one archive of every station's clips, files named
  `<STATION>_YYYYMMDD_HHMMSS.wav` (may sit inside one top-level folder).
- ONE of:
  - `<SCENES_FILE>`: `synthetic_generation_deployed_airports_v2.0.1.jsonl`,
    the scene corpus (concatenated pretty-printed JSON, not line JSONL). The
    text is regenerated from it (step 4a).
  - `<DATA_ZIP>`: `atc-gan-data-payload.zip` built on the Mac; unpacks to
    `data/text/…`, `data/real/…`, `data/vocab/…` at the repo root (step 4b).

Deliverable: `lab/reports/prod-setup.md` with a PASS/FAIL line per step below
and the exact numbers. If any step fails, stop there, write what failed and
the full error, and return. If your reply is lost, the report file is the
result.

Do not: run `uv run python scripts/generate_dataset.py`, `channel_fit`,
`residual_train`, or `bench_devices.py` (the mission's phase 0 owns those);
edit anything under `atcgen/`, `configs/`, `docs/`; commit.

## Steps (PowerShell, from the repo root)

1. **Repo is current.** `git log -1 --oneline` must show commit `8fcd7c4` or
   later ("Prod-v1 launch package…"). If not: `git pull`, then re-check.
   `HUMAN.md` and `lab/missions/prod-v1.md` must exist.

2. **Toolchain.** `uv --version` works (install from https://docs.astral.sh/uv/
   if not). `nvidia-smi` runs and shows driver ≥ 560 and the RTX 3080; record
   the driver version and free VRAM. Record free disk on the repo's drive
   (`Get-PSDrive`): need the extracted archive's size plus ≥ 40 GB.

3. **Environment.**
   ```powershell
   uv python install 3.11
   uv sync
   uv run python -c "import torch, soundfile; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), soundfile.__libsndfile_version__)"
   ```
   FAIL if `cuda.is_available()` is not True or the torch version ends in
   `+cpu` (then `uv sync` did not use the CUDA index in `pyproject.toml`; report
   it, do not work around it). Record libsndfile's version (expect 1.2.2).

4. **Text and vocab** — do 4a if Kevin gave you a scenes file, else 4b.

   4a. Regenerate from the scene corpus (needs internet for the vocab step):
   ```powershell
   uv run python scripts/convert_scenes.py --scenes <SCENES_FILE> --out data/text/scenes_v2.0.1.jsonl
   uv run python scripts/expand_text_views.py --text data/text/scenes_v2.0.1.jsonl --out data/text/scenes_v2.0.1_2view.jsonl --views 2 --seed 0
   uv run python scripts/harvest_vocab.py
   ```
   4b. Unpack the Mac payload:
   ```powershell
   uv run python -m zipfile -t <DATA_ZIP>
   Expand-Archive <DATA_ZIP> -DestinationPath . -Force
   ```
   Either way, verify and record each number:
   - `data/text/scenes_v2.0.1.jsonl` has 77,888 lines; `data/text/scenes_v2.0.1_2view.jsonl` has 155,776
     (`(Get-Content <file> | Measure-Object -Line).Lines`). A different count
     means a different scene file; report it, FAIL.
   - `data/vocab/real_anchor.json` exists.
   - (4b only) `data/real/calibration/` holds 100 wavs; `data/real/kixd/kixd_dev.csv`
     and `kixd_locked_day.csv` exist. **Do not open `kixd_locked_day.csv`**.

5. **Clips.**
   ```powershell
   uv run python -m zipfile -t <CLIPS_ZIP>
   New-Item -ItemType Directory -Force reference-data-for-v1-run | Out-Null
   Expand-Archive <CLIPS_ZIP> -DestinationPath reference-data-for-v1-run/airport_clips_v2
   # only if the station table below lacks KSDL_TOWER and data/real/calibration exists:
   # Copy-Item data/real/calibration/*.wav reference-data-for-v1-run/airport_clips_v2/<same folder as the other wavs>/
   Get-ChildItem reference-data-for-v1-run/airport_clips_v2 -Recurse -Filter *.wav | ForEach-Object { $_.BaseName -replace '_\d{8}_\d{6}$','' } | Group-Object | Sort-Object Name | Select-Object Count,Name
   ```
   FAIL on the integrity test. Put the full station table in the report.
   PASS requires every deployed airport present: **KEUG, KOJC, S50, KSLE,
   KIXD, KSDL** (facility suffixes like `_TOWER` are fine; report the exact
   spellings). Also report the number of wavs whose name does NOT match
   `^.+_\d{8}_\d{6}\.wav$`; those would land in station `unknown`:
   `Get-ChildItem ... -Filter *.wav | Where-Object { $_.Name -notmatch '^.+_\d{8}_\d{6}\.wav$' } | Measure-Object`
   If the archive extracted into a nested folder, say so; it is fine (the
   ingest walks subdirectories). The calibration-wav merge is only needed when
   the archive has no KSDL_TOWER station; if `data/real/calibration` is absent
   and KSDL_TOWER is present, skip it and say so — not a failure.

6. **Tests.** `uv run pytest -q` — record the pass/skip/fail counts. Expect
   ~780 passed. Any failure: paste it, FAIL.

7. **Lab state.** `uv run python scripts/lab/jobs.py lock status` must report
   `held: false`. `Get-Content lab/STATE.md` shows "Mission: none started".

8. **HF cache (optional).** If the system drive has < 20 GB free, tell Kevin
   to set `$env:HF_HOME` to a data drive before the mission; do not set it
   yourself.

## Report format (`lab/reports/prod-setup.md`)

```
# prod-setup            <UTC time>
commit: <hash>   uv: <ver>   python: <ver>   driver: <ver>   VRAM free: <MB>   disk free: <GB>
torch: <ver>   cuda: True/False   gpu: <name>   libsndfile: <ver>
1 repo        PASS/FAIL
2 toolchain   PASS/FAIL
3 env         PASS/FAIL
4 data        PASS/FAIL   via 4a|4b   text 77888 / 155776   vocab present
5 clips       PASS/FAIL   <station table>   unmatched names: <n>   nested folder: yes/no
6 tests       PASS/FAIL   <n> passed, <n> skipped, <n> failed
7 lab         PASS/FAIL
READY FOR prod-v1: YES / NO   (<one line: what blocks, if anything>)
```

When every line is PASS, Kevin prompts the `lab-director` with:
`Run lab/missions/prod-v1.md as mission prod-v1.`
