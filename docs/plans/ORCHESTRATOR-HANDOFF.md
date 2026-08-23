# Orchestrator Handoff Prompt

Copy everything below the line into the orchestrator agent.

---

You are orchestrating the implementation of a two-mode synthetic ATC radio audio generator in the repo at `/Users/kevin/repos/ai/atc-gan`. All design work is done — your job is to execute it with subagents, not redesign it.

## Required reading (in this order, before spawning anything)

1. `docs/plans/README.md` — doc index + master roadmap with task ordering
2. `docs/plans/02-architecture.md` — shared interfaces, config system, manifest format
3. `docs/plans/03-mode1-procedural-plan.md`, `docs/plans/04-mode2-calibrated-plan.md`, `docs/plans/05-evaluation-plan.md` — per-task specs and acceptance criteria
4. `docs/plans/01-codebase-analysis.md` — current code map and known miscalibrations
5. `docs/plans/00-research-findings.md` — the research decisions; consult when a subagent proposes deviating

Give each subagent the specific doc sections for its task, not the whole set.

## Ground rules (apply to every subagent)

- Python 3.11, `uv` project. Run `uv run pytest` after every task; the suite must be green. New code gets unit tests (synthetic fixtures, no network, no GPU in tests).
- Nothing is in production: no backwards compatibility, no compat wrappers, no deprecation paths. Replace `dsp.py` outright in S2 (port its effect implementations into `primitives.py`, port/rewrite its tests per-primitive, delete it); the CLI and manifest move directly to the new config-driven forms. Only discipline: the S2 port initially uses parameter values matching current behavior — distribution retuning (P2 profiles) is a separate commit so audible regressions are bisectable.
- Do not touch `atcgen/text/grammar.py`/`lexicon.py` content (other team's scope). Only the `TextSource` record contract may be extended per 02 §5.
- Kokoro TTS is fixed; don't swap engines. Don't add heavy deps without need — YAML lib and the `[eval]` extra (CLAP/WavLM) are pre-approved per 02 §7.
- This machine is CPU/MPS. Tasks marked GPU (M2.4, M2.6, P4 fine-tune) — build the code and a smoke test, but do NOT launch long trainings; leave a documented run command for the 5080 box.
- Real audio for calibration lives in `data/real/calibration/` (100 clips, unlabeled). Never commit large generated datasets; `data/` output dirs stay gitignored (check `.gitignore` covers new paths).
- Commit per task with the task ID in the message (e.g. `S2: extract channel primitives`). One task = one reviewable unit.
- Every task's plan doc lists acceptance criteria ("*Accept:*" lines). A task is not done until they're met and demonstrated (test output, generated stats, or audition files noted in the commit).

## Execution waves

Run tasks within a wave in parallel where files don't overlap; waves are sequential.

**Wave 1 — foundations (parallel-safe):**
- S1: `atcgen/config.py` — typed config + distribution specs + YAML profiles + resolved-config dump (02 §4)
- E1: `atcgen/eval/qc.py`, `channel_stats.py`, `report.py` — QC gates + Tier 1 stats vs `data/real/calibration/` (05 §2, §4)

**Wave 2 — refactor (single agent; restructures the channel module):**
- S2: port `dsp.py` effects into `atcgen/channel/primitives.py` + `chain.py`, delete `dsp.py`, port tests to per-primitive tests; full suite green (03 §2)

**Wave 3 — parallel tracks:**
- Track A (Mode 1): P1 new primitives (squelch_gate, ptt_truncation, mic_coloration, fading, agc_attack, extended codec) → P2 `wide`/`matched` profiles validated with E1 stats (03 §3, §6)
- Track B (Mode 2 data): M2.1 `dataset/local_corpus.py` + `noise_harvest.py` (04 §2.2, roadmap)
- Track C: S3 builder integration — config-driven `build.py`, provenance manifest records, category quotas, Tier 0 gates wired in (02 §5–6)

**Wave 4:**
- M2.2 per-clip channel fitting (`channel_fit.py`, statistics-matching variant first) → M2.3 CalibratedChannel backend (04 §2.1, §2.3)
- E2 embedding distances + channel probe (05 §2 Tier 2)
- P3 voice-augment layer (03 §4)

**Wave 5:**
- M2.4 residual CUT training code + smoke test (GPU-deferred run), M2.5 expansion workflow `dataset/expand.py`, E3 Tier 3 protocol extensions to `training/`, E4 `scripts/eval_synthetic.py` harness

Stop after Wave 5 and report; M2.6/P4 (fine-tune comparisons) and M2.7 (diffusion spike) are decision points for the user, not yours.

## Verification & reporting

After each wave: run the full test suite, generate a 25-sample smoke set exercising the new code path, run E1 stats against the calibration clips where applicable, and summarize per-task status (done/accept-criteria evidence/deviations) before starting the next wave. If a subagent hits a genuine spec ambiguity, resolve it from the plan docs' cited rationale; if the docs truly don't answer it, make the smallest reasonable choice and log it in the wave report — don't stall.
