# Experiment handoff — 6-hour RTX 3080 window

**Scope:** this is the six-hour *experiment* window (evidence for V1.1). The
*production* run — recalibrate on the full clip set, train the residual,
render, gate, export — is a separate mission: `docs/runbook-v1-3080.md` with
`lab/missions/prod-v1.md`. Do not mix the two in one GPU session.

**Audience:** the GitHub Copilot agent team in `.github/agents/` (Kevin prompts
`lab-director`; it delegates to `senior-researcher`, `experiment-engineer`,
`lab-assistant`, `results-auditor`), or a junior scientist running solo.
**Hardware:** Windows 11, one RTX 3080 (10 GB), CUDA, PowerShell. One GPU
stream at a time — never two GPU-heavy jobs concurrently (we measured 40%
slowdown on both when it happened); `scripts/lab/jobs.py launch --gpu` enforces
this with a lock.
**Hard rule:** the V1 production config is FROZEN (`docs/runbook-v1-3080.md`).
Nothing in this window changes it unless an experiment below meets its
pre-registered decision rule. You are generating evidence for V1.1, not
re-litigating V1.

Read first (30 min, before touching the GPU):
- `.github/skills/lab-protocol/SKILL.md` — how the team coordinates (files, not
  messages; brief/spec/report templates; the discipline rules in Part 1).
- `docs/results.md` — the 2026-09-01 addendum is the evidence base you're extending.
- `docs/runbook-v1-3080.md` §5 — what is frozen and why.
- This file, fully, before starting Part 0.

Who does what (the director briefs; nobody else touches the GPU):

| Phase | Agent | Skill(s) |
|---|---|---|
| Part 0 setup gate, Phase 0 bench | experiment-engineer | gpu-jobs |
| every running job | lab-assistant (watch every 5 min, returns on events) | monitor-run |
| Phase 1 gate, 2A/2B arms, 3 residual, 4 sweep | experiment-engineer, with specs from senior-researcher | generator-config, gpu-jobs, paired-analysis |
| D1–D4 numbers before the director acts on them | results-auditor | paired-analysis |
| Phase 5 writeup | engineer supplies numbers, lab-assistant transcribes, director signs off | lab-protocol |

---

## Part 0 — Windows / 3080 setup gate (do this once, before Phase 0)

Nothing below touches the GPU; it is the difference between a 6-hour window
and a 6-hour debugging session. Every item is a pass/fail check; write the
results into `lab/reports/win2-p0-setup.md`.

**0.1 Copy the gitignored payload from the Mac** (same relative layout under
the repo root; `git clone` brings none of it):

| Path | Size | Needed for |
|---|---:|---|
| `data/` | 110 MB | dev/locked/EU manifests, text sources |
| `reference-data-for-v1-run/updated_kixd_clips/` | 3.7 GB | the real KIXD audio every `data/real/kixd/*.csv` row points at |
| `runs/calib_kixd/` | 1.7 GB | ingested 16 kHz clips; `runs/channel_data_kixd/corpus.jsonl` points here |
| `runs/channel_data_kixd/` | 347 MB | folds, 400 presets, 15,302-segment noise bank (Phase 3, mode 2 renders) |
| `runs/gan_a_base_kixd/`, `runs/gan_val_base_kixd/` | 38 MB | probe TTS for residual training |
| `runs/e1_artifacts/`, `runs/e1_mode2_kixd/` | 24 MB | real LTAS curve JSON; mode 2 render for KID |
| `runs/dryrun_v1_main/` | 34 MB | Phase 4 gate sweep input |
| `runs/power_check_kixd/` | 1.2 GB | optional: last night's 22 cells, for auditing against |
| `reference-data-for-v1-run/asr/` | ~4.5 GB with zip | optional this window; the asr trainer (`asr-feedback-loop` skill) |

**0.2 Environment** (PowerShell):

```powershell
uv python install 3.11                 # 3.11 or 3.12; the CUDA torch wheels exist for both
uv sync                                # torch comes from the cu126 index on Windows (pyproject.toml); driver >= 560 required
uv run python -c "import torch, soundfile; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), soundfile.__libsndfile_version__)"
uv run pytest -q                       # expect ~780 passed (775 + 2 skipped on the Mac); the suite is Windows-clean
```

`cuda.is_available()` must print `True` (a `+cpu` torch means the CUDA index was
not used); libsndfile should be 1.2.2 (the MP3 compression table was measured
there; note the number if it differs). MLBucket is optional and absent on this
machine: `atcgen/tracking.py` no-ops with one warning; do nothing.

**0.3 Rewrite the absolute Mac paths** stored in every real-audio manifest
and the calibration corpus (dry run first, then apply; expect 0 missing):

```powershell
uv run python scripts/lab/relocate.py --from /Users/kevin/repos/ai/atc-gan --to <ABS-PATH-OF-THIS-REPO> data/real runs/channel_data_kixd runs/calib_kixd --check
uv run python scripts/lab/relocate.py --from /Users/kevin/repos/ai/atc-gan --to <ABS-PATH-OF-THIS-REPO> data/real runs/channel_data_kixd runs/calib_kixd --check --apply
```

**0.4 Smoke the whole chain on 3 clips** (TTS → channel → manifest; downloads
Kokoro and the spaCy model on first use):

```powershell
uv run python scripts/generate_dataset.py --config configs/mode1_matched_kixd.yaml --n-samples 3 --out runs/win2_smoke --seed 0
uv run python scripts/lab/jobs.py lock status      # must be free
```

**0.5 Copilot:** open the repo in VS Code, pick the `lab-director` agent, and
prompt: `Run agents-experiment-handoff.md as mission win2.` Model names in the
agent files follow the Copilot model picker (`Claude Opus 5`, `GPT-5.6 Sol`,
`GPT-5.6 Luna`); if a name is not offered on this machine, edit the `model:`
list, the first available entry wins.

---

## Part 1 — Lab discipline (what made last night work)

These are not suggestions. Each one is a lesson bought with wasted cells.

1. **Gate before you search.** Never launch an optimization without first
   proving the reward can see the thing you're optimizing. Run a power check:
   known-good arm, known-bad arm, base, ≥2 paired seeds. If known-bad doesn't
   separate from base by ≥2× the pooled seed-to-seed spread on the PAIRED
   statistic, the search will fit noise — don't run it. Last night a channel
   wrecked to 0–6 dB SNR with 40% dropouts was invisible to the reward
   (t=0.10); a 25-trial CEM would have "found" a winner anyway.

2. **Paired seeds or nothing.** A seed fixes both the generation draw and the
   fine-tune order (common random numbers), so seed-to-seed nuisance cancels in
   paired differences. Always report: per-seed paired diffs, direction count
   (n/n), paired t. Never quote unpaired separation. Never trust 2 seeds:
   aug_off's separation read 3.5× at 2 seeds, 2.6× at 3, 1.5× at 4. Minimum 4
   paired seeds for any claim; 8–10 to resolve ~1 WER point.

3. **Bounded WER is the decision metric.** Per-row errors are capped at
   reference length inside `TrueRewardHarness` (whisper-tiny loops — one row
   produced 96 errors on 17 words and outweighed an entire channel
   manipulation). Raw counts still land in each cell's `dev_rows.jsonl` with a
   `capped` flag. If you change any metric mid-run, use a fresh `--out`:
   `runs/power_check_kixd/summary.json` is permanently poisoned by exactly
   this mistake — recompute from dev_rows via `scripts/analysis/paired_report.py`.

4. **Fixed interpretable arms beat CEM at these budgets.** ≤30 trials cannot
   estimate a >4-dim response surface. Design arms that isolate one question
   each, share the base arm as the control, and check additivity (last night
   speed_fixed + voiceaug_off reconstructed aug_off to +0.001 — that additivity
   check is what made the numbers trustworthy).

5. **Locked data is read once, ever.** `data/real/kixd/kixd_locked_day.csv`
   (day 2025-08-08, 337 rows) has never been touched by any selection. It is
   NOT for this window. Dev slice is `data/real/kixd/kixd_dev.csv` (200 rows,
   day 2025-08-07). The EU rows (`data/real/eu_heldout/`) are monitor-only —
   whisper-tiny loops on them; never put them in the number an optimizer sees.

6. **KID only on matched audio.** Raw KID against the KIXD reference is ~35–40%
   padding/level contamination (reference has 18.8% digital-zero samples and
   sits 9 dB colder than synthetic). Protocol: trim both sides to the
   energy-active region, RMS-normalize to a common level, then
   `python -m atcgen.eval.embed_dist`. Tool: `scripts/analysis/make_matched_sets.py`.
   Fixed 1,000-clip reference subset reproduces the full-set number.

7. **Timebox and pre-register.** Every experiment below has a budget, a
   decision rule, and a kill criterion written BEFORE you run it. When the
   clock crosses a checkpoint, follow the written branch — do not extend a
   running experiment because it "feels close." Write results into your report
   file as each stage lands, not at the end.

8. **Known traps** (each cost us time last night):
   - `--dev-indices` default is `0:200`; on a mixed dev file that silently
     selects one source. Every run prints its dev composition — read that line.
   - `export_corpus_csv.py --version` is strictly `V<int>.<int>.<int>`.
   - `scripts/build_paired_views.py` writes 24 kHz Kokoro-native audio;
     `channel_fit` requires 16 kHz probes (resample snippet in runbook §1a).
   - Preset `passband_hz` fields look degenerate; the real EQ is
     `band_edges_hz`/`band_gains_db`. Rendered audio is full-band.
   - `sequential:` text sources refuse `--n-samples` beyond file length.
   - Voice/speed per clip live under the manifest row's nested `gen` key.
   - Trial-to-trial baseline caches key off corpus; if you add a model flag
     (E-B below), verify the cache keys off the model too, else fresh `--out`.

9. **If you delegate to subagents:** put specs and results in FILES, not
   messages (we lost 5 of 5 approvals to a one-way message failure; agents
   that wrote reports to agreed paths never lost work). Give every agent
   pre-authorized decision rules so a lost message can't stall the lab. Verify
   agents' factual claims that gate decisions (an agent's "VAD-trimmed
   directory" turned out to be byte-identical to the raw one). The templates
   and the `lab/` layout are in `.github/skills/lab-protocol/SKILL.md`; the
   results-auditor exists for the verification step.

10. **Windows shell:** PowerShell has no `\` line continuation and no
    heredocs. Every command below is one line (or backtick-continued); put
    ad-hoc Python into `scripts/analysis/` instead of pasting it. Long jobs
    go through `scripts/lab/jobs.py launch --gpu --id <id> -- <command>` so
    they survive the tool call and the lab-assistant can watch them
    (`.github/skills/gpu-jobs/SKILL.md`).

---

## Part 2 — The window, phase by phase

Clock notation: T+H:MM from when you start Phase 0. Checkpoints are hard.

### Phase 0 — Bench + sanity (T+0:00 → T+0:25)

```powershell
uv run pytest -q                                   # expect ~780 passed
uv run python scripts/lab/jobs.py launch --gpu --id win2-p0-bench -- uv run python scripts/bench_devices.py --device cuda --out runs/win2_bench.json
```

Record from the bench: TTS s/render, FastCUT s/step, whisper-tiny SFT s/step.
Then measure the production-budget trial cost directly — run ONE cell (the
gate runs on the procedural KIXD profile, see Phase 1 for why):

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id win2-p0-cell -- uv run python scripts/rl_power_check.py --out runs/win2_gate --base-config configs/mode1_matched_kixd.yaml --arms base --seeds 0 --dev-corpus data/real/kixd/kixd_dev.csv --n-synth 400 --ft-steps 500 --device cuda
```

`--n-synth`, `--ft-steps`, `--device`, `--dev-corpus` all exist on
`rl_power_check.py` (`--help`). There is no `--model` flag yet; that is Phase
2B's job.

Let **C** = wall-clock of that cell in minutes. Every budget below assumes
C ≈ 4–6. If C > 8, halve seeds everywhere (minimum 4 for claims, 2 for gates)
and drop E-D. Write C at the top of your report file.

### Phase 1 — E-G: the gate — can a production-budget reward see the channel? (T+0:25 → T+1:15)

**Question.** Last night's reward (200 clips / 300 steps / whisper-tiny) was
blind to channel quality. Does 2× data + longer fine-tune restore sight?
Everything channel-side hinges on this.

**Run.** In the same `runs/win2_gate` dir (same budget flags as Phase 0; the
Phase 0 cell is reused, so this adds 5 cells ≈ 5C ≈ 25–35 min):

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id win2-p1-gate -- uv run python scripts/rl_power_check.py --out runs/win2_gate --base-config configs/mode1_matched_kixd.yaml --arms base,degraded --seeds 0,1,2 --dev-corpus data/real/kixd/kixd_dev.csv --n-synth 400 --ft-steps 500 --device cuda
```

Why mode 1: `degraded` edits `channel.chain` steps, which only a procedural
profile has; `rl_power_check.py` refuses it on a mode 2 config by design. The
gate question is about the reward, not the backend.

**Decision D1 (pre-registered).** Compute paired degraded-vs-base with
`uv run python scripts/analysis/paired_report.py runs/win2_gate` (the run
directory is a real argument now; it used to be hardcoded) and have the
results-auditor recompute it before deciding.
- If |paired mean| ≥ 2× paired SE and 3/3 seeds agree in direction → **channel
  is visible** → Phase 2A.
- Otherwise → **still blind** → Phase 2B. Do not run any WER-based channel
  search this window, and say so explicitly in the report — that is a result,
  not a failure.

### Phase 2A — E-C: calibrated-channel search (only if D1 passed) (T+1:15 → T+3:45)

Fixed arms, not CEM, on `configs/mode2_fastcut_kixd.yaml`, 4 paired seeds each.
Five arms (read the base config's actual values first; arms are single-knob
moves within `mode2_safe`'s bounds — see `atcgen/rl/space.py:mode2_safe_space`):

1. `base`
2. `snr_lo_down` — snr_jitter low edge −6 dB (harder noise floor)
3. `snr_hi_down` — snr_jitter high edge −6 dB (compresses toward harder)
4. `post_fx_up` — the post-effects probs (squelch/crackle/dropouts) +0.15 each
5. `codec_low` — codec quality_lo reduced one notch

Add arms to `rl_power_check.py`'s arm table (additive; extend its parametrized
arm test; the knob constructors and an example are in
`.github/skills/generator-config/SKILL.md` §5). 4 new arms × 4 seeds + base
already has 3 → ~17C ≈ 70–100 min. Note these arms run on the mode 2 profile
while the gate ran on mode 1: give them a fresh `--out runs/win2_arms` and
include `base` there too.
Budget stop: if the clock passes T+3:45 mid-arm, stop; the run resumes
per-cell, report what completed.

**Decision D2.** An arm wins only if: paired |t| ≥ 2.0, ≥3/4 seeds one
direction, AND matched KID of its render is not worse than base by more than
1 SE (render 150 clips of the winning arm config for the KID check — reuse the
E1 tooling). A WER win that costs fidelity is a note, not a change.

### Phase 2B — E-B: proxy-transfer check on whisper-small (if D1 failed — else run this in Phase 3's slack) (T+1:15 → T+3:00)

**Question.** All of last night ranks configs by whisper-tiny. Cross-family
correlation is weak (r≈0.41 literature); within-family transfer is the bet.
Verify the two decisions that matter transfer to `openai/whisper-small.en`.

**Implement.** The model id is the `base_model` keyword of
`TrueRewardHarness.__init__` in `atcgen/rl/reward.py` (`openai/whisper-tiny.en`);
`rl_power_check.py` constructs the harness without passing it. Add a `--model`
flag threaded to `base_model` (default tiny, additive, with a test), and check
that the baseline cache slug includes the model id before sharing any out dir. VRAM: small.en fine-tunes fine in 10 GB at batch 8;
if OOM, batch 4 + grad accumulation ×2. **Fresh out dir** (`runs/win2_small`)
— do not share caches with tiny runs unless you verified the baseline cache
keys off the model id.

**Run.** Arms `base,aug_off,pitch_off` × seeds 0,1,2,3 at the production
budget. Cells are slower than C (bigger model) — measure the first one; if a
cell exceeds 12 min, drop `pitch_off` and run `base,aug_off` × 4 seeds.

**Decision D3.**
- `aug_off` harmful on small too (≥3/4 seeds) → the freeze's core claim
  transfers; say so with numbers.
- `pitch_off` NOT harmful on small (≤2/4 seeds, |t| < 1) → reopen the
  pitch-vs-KID trade as a V1.1 candidate; do NOT change V1.
- Anything contradicting the freeze → flag loudly; evidence, not action.

### Phase 3 — E-E: does the trained residual close the spectral leakage? (T+3:00/3:45 → T+5:00)

**Question.** Both modes leak +13–24 dB at 4 kHz and mode 2 runs +13.5 dB hot
at 100 Hz vs real KIXD (real has ~nothing at 4 kHz; whisper's mel front-end
sees it — a domain giveaway). The FastCUT residual is trained adversarially
against real audio and historically closes spectral gaps. Nothing has measured
whether it closes THIS one.

**Run.**
1. Train the residual for real on the KIXD artifacts already on disk (~30–40
   min on CUDA; this doubles as the CUDA validation of runbook §2):

```powershell
uv run python scripts/lab/jobs.py launch --gpu --id win2-p3-resid -- uv run python -m atcgen.channel.learned.residual_train `
  --corpus runs/channel_data_kixd/corpus.jsonl --split channel_train --val-split channel_val `
  --tts-dir runs/gan_a_base_kixd/clean --val-tts-dir runs/gan_val_base_kixd/clean `
  --presets runs/channel_data_kixd/train/presets.jsonl --noise-bank runs/channel_data_kixd/train/noise `
  --out runs/fastcut_kixd_cuda --device cuda `
  --steps 5000 --batch-size 12 --crop-frames 128 --lr 2e-4 --base 48 --n-res 6 --scales 1 2 4 --num-patches 256 `
  --nce-mode source+identity --lambda-nce 10.0 --lambda-gan 1.0 --r1-gamma 1.0 --r1-every 16 --ema-decay 0.9995 `
  --residual-scale-max 0.20 --a-renders 4 --eval-every 500 --eval-clips 64 --save-every 500 --seed 0
```

(`--resume <ckpt>` exists if the window is interrupted; `--save-every 500`
leaves restart points.)

2. Copy `configs/mode2_fastcut_kixd.yaml` → `configs/mode2_kixd_resid.yaml`;
   set `calibrated.residual.enabled: true` and point its checkpoint at
   `runs/fastcut_kixd_cuda/G_selected.pt`.
3. Render 150 clips with it (`scripts/generate_dataset.py`, scene text,
   seed 0) and measure, with residual OFF as the control:
   - LTAS at 100 Hz / 200 / 400 / 1k / 2k / 3k / 4k vs the real reference
     (`scripts/analysis/ltas_check.py`; real curve values in `docs/results.md`).
   - Matched KID vs the fixed 1,000-clip reference subset, both renders:
     `uv run python scripts/analysis/make_matched_sets.py --out runs/win2_kid --syn off=runs/win2_resid_off/wavs --syn on=runs/win2_resid_on/wavs`
     then the `embed_dist` commands it prints (`--device cuda`; the embedder
     now defaults to CUDA when present).
4. If the 4 kHz excess is still > +8 dB with the residual on: test the cheap
   fix offline — steep low-pass (8th-order Butterworth at 3.8 kHz) applied to
   the rendered wavs in an analysis script (scipy, no pipeline change),
   re-measure matched KID. Same for a 150 Hz high-pass if 100 Hz is still hot.

**Decision D4.** Report a 4-row table (residual off / on / on+LP / on+LP+HP ×
LTAS-gap and matched KID). If a filter variant improves matched KID by > 1 SE
with LTAS moving toward real: recommend adding the final band edge to the
channel config as a V1.1 change (implementation = one chain-step config
change; spec it in the report, don't ship it).

### Phase 4 — E-F (stretch, only if ≥45 min left): gate-teacher ceiling sweep (T+5:00 → T+5:45)

The frozen config gates at ~26% teacher-rejected / 18% gold. Re-run
`scripts/gate_dataset.py` over the SAME existing render (`runs/dryrun_v1_main`
or Phase 3's 150-clip sets — no new rendering) at 2–3 teacher-WER-ceiling
values (find the ceiling in the gate config; document its current value).
Report tier yields per ceiling. Output: a yield-vs-ceiling table for the owner
decision named in runbook §4 — no threshold change, evidence only.

### Phase 5 — Writeup (T+5:15 → T+6:00, protected — start on time no matter what)

Deliverables, in the repo:
1. Append a dated addendum to `docs/results.md`: every phase, full paired
   tables (per-seed diffs, direction counts, t), decision-rule outcomes
   (D1–D4) stated as pass/fail against their pre-registered criteria, and an
   explicit "changes recommended for V1.1" list (possibly empty — a clean
   null on every phase is a fine outcome and worth exactly as much as a win).
   The per-phase reports under `lab/reports/` are the source; the addendum is
   the digest.
2. Any new analysis scripts → `scripts/analysis/` (never leave tools in tmp
   dirs; last night's had to be rescued).
3. `runs/win2_*` left intact. Locked day still untouched — state that
   explicitly in the addendum.
4. Tests still green (`uv run pytest -q`); any script you extended keeps its
   parametrized arm/flag tests passing.

Priority order if time collapses: Phase 1 > Phase 2B > Phase 3 > Phase 2A > 4.
The gate (Phase 1) and the transfer check (2B) are the two results that change
how every future window is spent — protect them.
