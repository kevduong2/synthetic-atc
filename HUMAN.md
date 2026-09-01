# HUMAN.md — what Kevin does before launching the V1 production run

Everything an agent can do is already in `docs/runbook-v1-3080.md` and
`lab/missions/prod-v1.md`. This is the rest: the steps that need your hands,
your data, or your decision. Work top to bottom; the run cannot start until
the first four are done.

## 1. Get the repo onto the 3080 box

- [ ] **Commit the current working tree.** The review changes (per-station
      calibration, PowerShell runbook, mission file, shard/resample scripts)
      are staged but not committed. `git log -1` on the box must show them.
- [ ] **Pick a transfer path.** This repo has no git remote. Either add one
      (`gh repo create --private --source . --push` or an existing host) and
      clone on the box, or copy the tree by hand. The gitignored payload
      travels separately either way (item 3).

## 2. The full clip archive (the actual gate on this run)

- [ ] Obtain the complete, un-truncated archive. The last one stopped at
      1.2 GB; `python -m zipfile -t <zip>` must pass before anything else.
- [ ] It must contain every deployed airport: **KEUG, KOJC, S50, KSLE, KIXD,
      KSDL**, as `<STATION>_YYYYMMDD_HHMMSS.wav` with one consistent
      `ICAO_FACILITY` spelling per receiver. Details and the count command
      are in `docs/data-handoff.md`. A station that is absent stops the
      mission at phase 0 by design; deliver even ten minutes for a missing
      airport rather than nothing.
- [ ] A `README.txt` in the archive naming each station's receiver and audio
      origin (own SDR vs someone's feed). That line is the licensing gate
      (`docs/data-licensing.md`); LiveATC-derived audio cannot go down the
      commercial path.

## 3. Copy the gitignored payload (runbook §0.1)

| What | Where it goes on the box |
|---|---|
| `data/` (110 MB: text corpora + real-audio manifests) | `<repo>/data/` |
| the clip archive | anywhere; extract to `<repo>/reference-data-for-v1-run/airport_clips_v2/` |
| `data/real/calibration/` (own-SDR KSDL/KSLE/SEATTLE_CENTER wavs; inside `data/`) | merged into the clips dir per runbook §0.3 |
| optional: `runs/calib_kixd/`, `runs/channel_data_kixd/` (2 GB) | only to compare new KIXD presets with the overnight ones |
| optional: `reference-data-for-v1-run/asr/` (~4.5 GB) | only if you take the 24-hour plan below (asr training step) |

- [ ] **Disk:** the render writes ~160k wavs, roughly 25–30 GB, plus ~3 GB
      of HF model downloads and the clip set. Have ~60 GB free. Set
      `$env:HF_HOME` if the system drive is small.
- [ ] NVIDIA driver ≥ 560 (the cu126 torch wheel needs it).

## 3b. Or hand items 3–4 to an agent

`lab/briefs/prod-setup.md` is a self-contained brief for any agent with a
shell on the box: it unpacks both zips, checks the counts, runs the
environment gate and the tests, and writes `lab/reports/prod-setup.md` with a
READY YES/NO line. Give it the two zip paths. The data zip is built on the Mac
with the payload command in that brief's header (or ask Claude for
`atc-gan-data-payload.zip`, ~19 MB).

## 4. Environment gate on the box (10 minutes, runbook §0.2)

```powershell
uv python install 3.11
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q
```

CUDA must print `True`. If torch ends in `+cpu`, `uv sync` did not use the
CUDA index; check `pyproject.toml`'s `[tool.uv.sources]` block survived.

## 5. Decisions to make now, not at 3 a.m. (pre-registered branches)

The mission's three stop points are now pre-registered branches in
`lab/missions/prod-v1.md` (ticked 2026-09-01). Two run without you; the third
stops with the decision packet ready.

| Stop | Ticked branch | What you rejected |
|---|---|---|
| [x] A deployed station ends with < 30 kept presets (D1) | **B1:** proceed, keep it in `station_mix`, flag "thin calibration" in the report | dropping it from `station_mix` (its airport would get other towers' channels); stopping for more audio |
| [x] Residual selection is `no_eligible_candidate` (D2) | **B2:** auditor diagnoses (gates failed vs eval never scored), one retry with `--seed 1`; second failure stops | rendering with `residual.enabled: false` (valid but not the frozen recipe) |
| [x] In-band LTAS gap > 2 dB or 4 kHz still > +8 dB (D3) | **B3:** build the 4-row filter table (`scripts/analysis/filter_variants.py`: off / on / +LP 3.8 kHz / +LP+HP 150 Hz, KID ± SE + LTAS), then **stop for you** | pre-authorizing the band edge (a frozen-config change; you can still do this later by editing B3) |

So the only expected touchpoint is B3, and only if fidelity misses. To make
it fully unattended, change B3 to "apply the LP edge when the table shows KID
better by > 1 SE with LTAS moving toward real" — your call, it edits a frozen
config.

- [ ] `cross_station_prob` stays 0.1 (frozen; becomes live with several
      stations). Keep unless the fidelity check implicates it.
- [ ] Any shard that fails twice stops the mission (already written).

## 6. Launch

- [ ] Open the repo in VS Code on the box, pick the `lab-director` agent, and
      prompt: `Run lab/missions/prod-v1.md as mission prod-v1.`
- [ ] If a model name in `.github/agents/*.agent.md` is not in the Copilot
      picker on that machine, edit the `model:` list; the first available
      entry wins.
- [ ] Expect at most one human touchpoint (B3, fidelity miss). Everything
      else runs from files; check `lab/STATE.md`, not the chat.

## 7. Rough clock (bench first; these are MPS-extrapolated)

| Phase | Est. on a 3080 |
|---|---:|
| §0 setup + bench | 0.5 h |
| §1 calibration (fit ~900 clips on CUDA) | 0.5–1 h |
| §2 residual, 5,000 steps | 0.5–1 h |
| §5 fidelity check | 0.3 h |
| §3 render, 4 shards + noise | 4–6 h |
| §4 gate (~160k clips through two teachers) + export | 2–4 h |
| **Production total** | **~9–13 h** |

## 8. If you have 24 hours: what to spend the slack on

Not on RL and not on a bigger generator. The evidence says why: the
fine-tune reward is blind to channel quality at any feasible budget (a channel
wrecked to 0–6 dB SNR was invisible, t=0.10), and the talker knobs are frozen
on 4–10 paired seeds. A longer search fits noise; a bigger residual changes
a frozen recipe with no measurement behind it. Spend the hours on
measurements that change decisions, in this order:

1. **The asr feedback loop (≈4–8 h, the only step that measures the product
   metric).** Train the asr repo's model on real + `V1.0.0` synthetic vs the
   same run without synthetic (same seed, same real data), compare
   `real_val_awer`. whisper-medium at batch 1 with gradient checkpointing is
   unmeasured on 10 GB; treat epoch 1 as a smoke test with a pre-authorized
   kill on OOM and `whisper-small` as the fallback. Recipe:
   `.github/skills/asr-feedback-loop/SKILL.md` §1–§3. Needs the asr repo on
   the box (item 3).
2. **The experiment window's Phase 1 gate (≈0.5 h).** Does 2× data + 500
   fine-tune steps let the reward see the channel? Six cells, decision rule
   D1 already written in `agents-experiment-handoff.md`. It decides how every
   future GPU window is spent.
3. **Phase 2B, the whisper-small transfer check (≈1.5 h).** This is the
   "bigger model on a 3080" experiment, already designed: do the two frozen
   decisions (`aug_off` harmful, `pitch_off` not free) hold on `small.en`?
   Needs the `--model` flag on `rl_power_check.py` (a small additive change
   the engineer makes with a test).

Run 2 and 3 as mission `win2` after `prod-v1` finishes, never interleaved:
one GPU stream, and the two missions have different frozen-config rules.

## 9. Not blocking, still yours

- [ ] Licensing (runbook §6): KIXD provenance vs LiveATC terms; CMU AirLab
      email re TartanAviation.
- [ ] After the final model exists: the single read of
      `data/real/kixd/kixd_locked_day.csv`. Once, ever.
