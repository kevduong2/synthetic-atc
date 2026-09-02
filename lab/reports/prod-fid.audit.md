# Audit prod-fid          (report: lab/reports/prod-fid.md; brief: lab/briefs/prod-fid.md; mission: lab/missions/prod-v1.md)

AUDIT: **PASS-WITH-NOTES** — every number in the report reproduces exactly from the raw artifacts and the D3 FAIL is correctly derived and robust, but the B3 table Kevin is asked to decide from is built on 73/77-clip cohorts (below the lab's ~150-per-side reportability floor) with `off` and `on` drawn from disjoint clips, and the Director summary states an unpaired sub-2×SE KID difference as a residual improvement.

## Verdict table

| # | Check | Verdict |
|---|---|---|
| 1 | Pre-registration: D3 rule, B3 branch, budget, kill criterion match what was written before the run | CONFIRMED |
| 2 | Matched KID 0.003443 ± 0.000731 recomputed from the embed_dist artifact, not `summary.json` | CONFIRMED |
| 3 | LTAS gap table at 100/200/400/1k/2k/3k/4k independently recomputed | CONFIRMED |
| 4 | In-band max 2.8 dB → D3 FAIL derived per the mission rule (in-band ≤ 2 dB) | CONFIRMED |
| 5 | B3 trigger (in-band > 2 dB, or residual-on 4 kHz > +8 dB) correctly fired | CONFIRMED |
| 6 | 4-row filter table: KID ± SE and all seven LTAS bands present, values reproduce | CONFIRMED |
| 7 | Matched-KID hygiene: energy-trim + RMS-normalize both sides, fixed 1,000-clip reference | CONFIRMED |
| 8 | Matched-KID hygiene: ≥ ~150 clips per side, same count on both synthetic sides, paired comparison | **CONTRADICTED** |
| 9 | D2 gate: `selection.status == "selected"`, `kid_mean` ≈ 0.0058, and the render used that checkpoint | CONFIRMED |
| 10 | Gating factual claims: render size, cohort split, distinct variant dirs, frozen values in `config.resolved.yaml`, test count | CONFIRMED |
| 11 | Data discipline: no artifact or command reads `kixd_locked_day` | CONFIRMED |
| 12 | Band decomposition (on → on+LP → on+LP+HP) is additive | CONFIRMED |
| 13 | Reference curve is representative of the real target set | **UNVERIFIABLE / material caveat** |

## Evidence

**1. Pre-registration.** `lab/specs/prod-fid.md` does not exist (`lab/specs/` holds only `.gitkeep`), but D3, branch B3 and the kill rule were pre-registered in `lab/missions/prod-v1.md` (branches "ticked by Kevin 2026-09-01") and restated in `lab/briefs/prod-fid.md` (written 2026-09-02, before the run). The report's decision line and B3 scope match both, verbatim, with no post-hoc change. The commands in the report are character-for-character the runbook §5 block (`docs/runbook-v1-3080.md` lines 260–268).

**2. Primary matched KID.** Read from the raw evaluation output, not a summary:
`runs/prod_fid/kid/kid_v1_matched.json` → `wavlm.kid = 0.003443`, `wavlm.kid_std = 0.000731`, `n_synthetic = 150`, `n_real = 997`; `clap.kid = 0.001003`, `kid_std = 0.000118`. Matches the report at printed precision. The four B3 JSONs likewise match every digit:

| Variant | file | wavlm kid / std | clap kid / std | n_syn |
|---|---|---|---|---:|
| off | `kid_b3/kid_off_matched.json` | 0.003928 / 0.000737 | 0.001079 / 0.000113 | 77 |
| on | `kid_b3/kid_on_matched.json` | 0.003043 / 0.000525 | 0.000977 / 0.000097 | 73 |
| on+LP | `kid_b3/kid_on_lp_matched.json` | 0.003364 / 0.000630 | 0.000922 / 0.000099 | 73 |
| on+LP+HP | `kid_b3/kid_on_lp_hp_matched.json` | 0.003599 / 0.000715 | 0.000875 / 0.000092 | 73 |

A full re-derivation of KID from the embeddings needs the GPU and was not run (auditor is CPU-only); everything checkable without it is checked below.

**3–5. LTAS and D3.** Recomputed independently, from the wav directories, with fresh `--out` files:

```powershell
uv run python scripts/analysis/ltas_check.py runs/calib_v2/clips runs/prod_fid/wavs --label real --label v1 --limit 1000 --json runs/prod_fid/audit_ltas.json
uv run python scripts/analysis/ltas_check.py runs/prod_fid/variants/off runs/prod_fid/variants/on runs/prod_fid/variants/on_lp runs/prod_fid/variants/on_lp_hp --label off --label on --label on+LP --label on+LP+HP --json runs/prod_fid/audit_ltas_b3.json
```

Reproduced exactly, all seven bands, all five rows: V1 `+16.29 / +1.85 / −0.99 / +2.77 / −1.69 / +0.08 / +16.40`, `max_abs_gap_1k_3k = 2.77`; off `+17.30 / +3.05 / −1.63 / +2.30 / −0.85 / +1.31 / +18.63`; on `+13.85 / −0.90 / −1.24 / +2.41 / −3.70 / −2.73 / +9.60`; on+LP identical to `on` except 3 k `−2.78` and 4 k `−3.43`; on+LP+HP `−6.61 / −1.64 / −1.14 / +2.38 / −3.70 / −2.65 / −3.33`. Every rounded value in both report tables is correct.

D3 as pre-registered is "in-band LTAS gap ≤ 2 dB AND matched KID reported with SE". Observed in-band max 2.77 dB > 2 → FAIL. Correct. B3's trigger is "in-band gap > 2 dB, **or** the 4 kHz excess still > +8 dB with the residual on": both limbs fire (2.77 dB; residual-on 4 kHz +9.60 dB). B3 was correctly mandatory, and the mission's stop-for-Kevin outcome is the correct terminal state.

**7. Matched-set hygiene (the part that holds).** Aggregate statistics over the sets themselves confirm identical treatment on both sides — trim to the active region and normalize to −26 dB:

```
set                           n   dur_med   zerofrac   rms_med
kid/ref_raw                1000      5.25      0.163    -27.98
kid/ref_matched             997      4.27      0.022    -26.00
kid/v1_raw                  150      4.83      0.021    -23.46
kid/v1_matched              150      4.59      0.022    -26.00
kid_b3/ref_matched          997      4.27      0.022    -26.00
kid_b3/off_matched           77      4.54      0.029    -26.00
kid_b3/on_matched            73      4.78      0.015    -26.00
kid_b3/on_lp_matched         73      4.78      0.016    -26.00
kid_b3/on_lp_hp_matched      73      4.78      0.021    -26.00
```

The reference is the fixed 1,000-clip seed-0 draw (997 survive the trim), and `kid/ref_matched` and `kid_b3/ref_matched` are content-identical — same aggregate SHA-256 over sorted per-file hashes (`b50c7b33046cc271da30…`). Both KID batches were scored against the same reference, so the five rows are mutually comparable. Only `*_matched` vs `ref_matched` numbers appear in the report; no raw KID is quoted.

**8. CONTRADICTED — clip counts and pairing.** `.github/skills/paired-analysis/SKILL.md` §7: "KID on un-matched audio, **on fewer than ~150 clips per side**, or against a reference that overlaps calibration clips, is not reportable"; §4: "Same clip count on both synthetic sides." Both are breached by the B3 table:

- Every B3 row has 73–77 clips on the synthetic side, roughly half the floor. This is structural, not an oversight: `filter_variants.py --manifest` splits the single 150-clip render by whether `residual_translate` appears in the manifest row's chain, and `residual.apply_prob = 0.5`, so 77 off + 73 on = 150. A 150-clip render cannot yield 150 clips per cohort.
- `off` and `on` are therefore **different clips**, not the same clips with and without residual. The off→on KID difference is unpaired and cohort-confounded, and is 0.000885 against a combined subset spread of 0.000905 — **0.98×**, below the lab's 2×SE visibility bar.
- Despite that, the Director summary states flatly: "The residual improves matched KID and reduces the upper-band excess." Under §9 claim discipline that is a confirmatory claim from a sub-threshold, unpaired development difference. The Interpretation section does note the cohort difference for LTAS, so the report contradicts its own summary.
- The `on` → `on+LP` → `on+LP+HP` comparisons are the same 73 clips and *are* paired; the report's hedging there ("+0.000321 remains within the reported subset spread") is correct and stands.

Fix, in order of cost: (a) reword the summary to "residual-on and residual-off cohorts differ by 0.9× the combined subset spread — not separated at this n"; (b) to make the row reportable, re-render at 300+ clips, or render the same 150 texts twice with `--set calibrated.residual.apply_prob=0` and `=1` so off/on become paired at n=150:

```powershell
uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid_off --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0 --set calibrated.residual.apply_prob=0
uv run python scripts/generate_dataset.py --config configs/mode2_v1.yaml --n-samples 150 --out runs/prod_fid_on --seed 0 --text sequential:data/text/scenes_v2.0.1.jsonl --set dataset.noise_only_frac=0 --set calibrated.residual.apply_prob=1
```

**9. D2 re-verified.** `runs/fastcut_v1/validation_report.json` → `selection = {step: 3500, kid_mean: 0.0057963989421491655, kid_se: 0.0003688342183144903, rule: "lexicographic_v1.1_fold_paired_tiebreak", status: "selected", sha256: "1f1d95f3…f972107dd"}`, 10 evaluations. `kid_mean` rounds to 0.0058 as reported. The selection `sha256` equals the SHA-256 of `runs/fastcut_v1/G_selected.pt` (`1f1d95f32adcfda772e089799e41f832beab7b039523378589c9cf4f972107dd`), and `runs/prod_fid/config.resolved.yaml` sets `calibrated.residual.checkpoint: runs/fastcut_v1/G_selected.pt` with `enabled: true` — the fidelity render used the selected checkpoint, not `G_ema.pt`.

**10. Factual claims verified by inspection.** Counts and aggregate content hashes (no per-file enumeration surfaced):

```
runs/prod_fid/wavs                  n=150   runs/prod_fid/kid/v1_matched      n=150
runs/prod_fid/variants/off          n= 77   runs/prod_fid/variants/on         n= 73
runs/prod_fid/variants/on_lp        n= 73   runs/prod_fid/variants/on_lp_hp   n= 73
```

77 + 73 = 150, consistent with `apply_prob: 0.5`. All four variant directories have distinct aggregate hashes (`6d5c04fe…`, `4d7ea4eb…`, `267dbe9d…`, `f2210994…`), so "different" directories really are different audio; likewise the four B3 matched sets. `runs/prod_fid/stats.json` shows `n_samples: 150`, `mode: calibrated`, `seed: 0`, `backends.calibrated: 150`. `config.resolved.yaml` carries the frozen §5 values unchanged (`tts.speed [1.0, 1.4]`, `pitch_semitones.prob 0.5`, `tempo.prob 0.3`, `eq_tilt_db.prob 0.4`), the exact six-station `station_mix`, and `dataset.noise_only_frac: 0.0`. `uv run pytest -q` reproduces **781 passed, 3 skipped** (98 s), matching the report.

**11. Data discipline.** A search of `runs/prod_fid/config.resolved.yaml`, `manifest.jsonl`, `stats.json`, the report and the brief for `kixd_locked_day` returns exactly one hit — the brief's "Do not" line. No run artifact, config or command touches the locked day. No EU-row or `channel_val` selection is involved in this phase.

**12. Additivity.** LP alone moves only 3 k (−2.73 → −2.78), 3.4 k (+0.82 → +0.17) and 4 k (+9.60 → −3.43); adding HP moves only 100 Hz (+13.85 → −6.61) and 200 Hz (−0.90 → −1.64) and leaves 4 kHz at −3.33 vs −3.43. The two band edges act on disjoint bands and compose as expected. No power gate applies — this is a fidelity phase with no WER arms and no null claimed.

**13. Material caveat on the reference curve.** `ltas_check.py` scores every row against a hardcoded `real_reference_db` (the overnight KIXD curve). The same run measures the actual target set, `runs/calib_v2/clips`, against that curve — and the report omits that row. It is large:

```
real (runs/calib_v2/clips, n=1000)  gap: +5.5 / -0.8 / -1.1 / +7.4 / -0.1 / +0.7 / +6.1   max|gap| 1-3k = 7.4 dB
```

The real multi-station set misses its own reference by **7.4 dB in band** and 6.1 dB at 4 kHz, so the reference is not descriptive of the V1 target. Measured against the real cohort actually rendered against, V1's gaps are `+10.8 / +2.7 / +0.1 / −4.6 / −1.6 / −0.6 / +10.3`, in-band max **4.6 dB**.

This does not change the verdict — D3 FAILs under either reference (2.8 dB and 4.6 dB both exceed 2 dB) and B3 triggers under either (4 kHz excess +16.4 or +10.3, both > +8) — so the packet's terminal state is robust. But it is material to the decision the packet gates: the band-edge magnitudes differ by about 6 dB at both edges (100 Hz +16.3 vs +10.8; 4 kHz +16.4 vs +10.3), and the 100 Hz sign of the residual-on cohort's correction changes character. A band edge sized off the +16.4 figure would over-correct against the real set. Note also that the runbook states the in-band rule as "≤ ~2 dB (**measurement floor**)", which makes a 2.8 dB observed miss a marginal FAIL, and that `make_matched_sets.py` draws its reference from all of `runs/calib_v2/clips`, which includes the train split whose presets were fit in P1 — the absolute KID level is therefore optimistic, though identically so for all five rows.

Fix: add the `real` row to the fidelity table (it is already in `runs/prod_fid/ltas.json`), and quote the V1-vs-real-cohort gaps alongside the V1-vs-reference gaps, before Kevin sizes an edge.

## Recompute artifacts

`runs/prod_fid/audit_ltas.json`, `runs/prod_fid/audit_ltas_b3.json` (this audit's independent LTAS recomputation; fresh `--out`, the report's files untouched). No report, config, code or `docs/results.md` file was modified by this audit.
