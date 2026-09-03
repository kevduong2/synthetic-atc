# Audit prod-close                (mission: lab/missions/prod-v1.md, prod-close row)

AUDIT: **PASS-WITH-NOTES** — every corpus, gate, provenance and fidelity number in the close-out addendum reproduces exactly from raw artifacts and the frozen recipe is the one that rendered all 160,576 rows, but the addendum's fidelity-history table implies the low-pass landed *after* D3' when it landed before it, six `lab/` deliverables (including the audit that cleared the render) are still uncommitted, and the exported corpus carries no gate tier, so the 62,341 rejected rows are indistinguishable to a downstream consumer.

## Verdicts

| # | Check | Verdict |
|---|---|---|
| 1 | D4 artifacts: `corpus_train.csv` + `corpus_test.csv` + `manifest.json`, exactly three files, 157,462 + 3,114 = 160,576 | CONFIRMED |
| 2 | `manifest.json` internally consistent: digests match the files, `rows_in_manifest` = train + test = 4 x 38,944 + 4,800 | CONFIRMED |
| 3 | Gate tiers reconcile: per-shard `gate_stats.json` sum to 19,694 / 29,976 / 43,765 / 62,341; each shard sums to 38,944 | CONFIRMED |
| 4 | Provenance: every rendered run's `config.resolved.yaml` equals the final `configs/mode2_v1.yaml` (LP@3.8 kHz + peaking_eq, residual `G_selected.pt`) up to seed / `noise_only_frac` | CONFIRMED |
| 5 | The rendering config is the D3''-cleared one (identical chain and residual to `runs/prod_fid_d3pp_a1`) | CONFIRMED |
| 6 | Addendum (docs/results.md L317-379): station clips, presets, selection block, fidelity values, gate yields, corpus counts all trace to a report or artifact | CONFIRMED |
| 7 | Addendum fidelity-history row order: "LP step ... applied before the recipe correction" placed *below* the D3' row | **CONTRADICTED** |
| 8 | Locked data: `kixd_locked_day.csv` untouched; no run config, command or artifact references it | CONFIRMED (vacuous — see below) |
| 9 | Commit trail: p1 / p2 / fid / fid-rerun / fix / p3 / p4 phase commits exist and `configs/mode2_v1.yaml` is committed clean | CONFIRMED |
| 10 | All `lab/` deliverables committed | **CONTRADICTED** |
| 11 | Pre-registration: the D4 rule executed is the mission's D4 (160,576 pre-gate rows + three files), unchanged | CONFIRMED |
| 12 | Split hygiene: no audio path on both sides; noise-only rows train-only; test frac 0.01999 | CONFIRMED |
| 13 | Gate tier reachable from the exported corpus | **CONTRADICTED** |

## Recomputation detail

**1-2. Corpus.** `data/corpus/V1.0.0` holds exactly three files. One aggregate
`csv.reader` pass per file: `corpus_train.csv` 157,462 data rows,
`corpus_test.csv` 3,114, header `audio,text,suspect` both. 157,462 + 3,114 =
160,576 = 4 x 38,944 + 4,800. Recomputed SHA-256 —
train `9f5e2de31cc18514912665deec35c70f5eafed9454bdb1dc03823d14aa87f560`,
test `3bfbd9a24e928107e453d69448023bfd668aa75d1f096bb8a65b4861012b9a6a` —
match `manifest.json.sha256` digit for digit. `rows_in_manifest` 160,576,
`train_rows` 157,462, `test_rows` 3,114, `noise_only_in_train` 4,800,
`noise_only_dropped` 0, per-dataset counts 38,944 x 4 + 4,800: consistent.
Rows recounted by source run directly from the CSV audio paths give
s1 38,132 + 812, s2 38,192 + 752, s3 38,171 + 773, s4 38,167 + 777, noise
4,800 + 0 — each speech run exactly 38,944, noise exactly 4,800.

**3. Gate.** From `runs/train_v1_s{1..4}/gate_stats.json`, never `summary`
text: s1 4,830 / 7,569 / 11,106 / 15,439; s2 4,935 / 7,422 / 10,934 / 15,653;
s3 4,964 / 7,443 / 10,929 / 15,608; s4 4,965 / 7,542 / 10,796 / 15,641. Each
row sums to 38,944; column totals 19,694 / 29,976 / 43,765 / 62,341, grand
total 155,776. Matches `lab/reports/prod-p4.md` and the addendum exactly.

**4-5. Provenance.** A flattened key-by-key diff of each render's
`config.resolved.yaml` against `_plain(load_config('configs/mode2_v1.yaml'))`
returns exactly two differing leaves per speech shard — `seed` (7 / 17 / 27 /
37) and `dataset.noise_only_frac` (0.03 -> 0.0) — and for the noise run `seed`
8 and `noise_only_frac` 1.0. Nothing else differs. All five carry
`chain[0] lowpass cutoff 3800 / order 8 / zero_phase true`,
`chain[1] peaking_eq f0 1100 / gain 7.0 / q 1.7 / prob 1.0`, and
`residual.enabled true` on `runs/fastcut_v1/G_selected.pt`
(`apply_prob 0.5`, `alpha 1.0`, `residual_scale_max 0.35`).
`runs/prod_fid_d3pp_a1/config.resolved.yaml` — the render that cleared D3'' —
differs from the same base only in `noise_only_frac`, so the shipped audio was
made by the cleared recipe. The checkpoint on disk is `G_selected.pt`,
17,625,973 bytes, SHA-256 `1f1d95f3…72107dd`, equal to
`validation_report.json.selection.sha256`, and *not* byte-identical to
`G_ema.pt` (`a0e81a75…`, 17,623,523 bytes); `selection.status == "selected"`,
step 3500, 10/10 evaluations `gates_ok`, no null `kid_mean`.
`git log -p -- configs/mode2_v1.yaml` shows only four touches, the LP added in
`2e2dee2` and the bell in `72ba48a`; the file is committed and clean.

**6. Addendum, number by number.** Station clips 126 / 86,104 / 10,305 / 8,157
/ 8,000 / 94,537 = `lab/reports/prod-p0-setup.md`. Presets 85 / 143 / 144 /
145 / 150 / 142 = `lab/reports/prod-p1-calib.md`. Selection "selected, 3,500,
0.0058 +/- 0.0004" rounds `validation_report.json`'s 0.0057963989 /
0.0003688342 correctly. D3 2.8 dB = report table (auditor-exact 2.77 in
`prod-fid.audit.md`); D3' 4.69 dB reproduced from
`runs/prod_fid_d3prime/ltas.json` (`max_abs_gap_1k_3k` 4.69, gap vector
+7.94 / +1.77 / +1.90 / -0.45 / -4.69 / -1.09 / +0.02 / +2.90 / -6.55); D3''
1.43 dB reproduced from `runs/prod_fid_d3pp_a1/ltas.json` (-0.53 / -0.34 /
-1.43, `max_abs_gap_1k_3k` 1.43); KID 0.004331 +/- 0.000938 reproduced from
`kidsets/kid_v1_matched.json` (`wavlm.kid` 0.004331, `kid_std` 0.000938,
150 synthetic / 997 reference), and the guardrail arithmetic checks:
0.004134 + 2 x 0.000797 = 0.005728. The struck off-vs-on claim and the
"77/73-clip, disjoint, unpaired" wording match `prod-fid.audit.md` item 8.
Gate and corpus tables match §3 and §1 above. No transcription error found in
any figure.

**8. Locked data.** `data/real/kixd/kixd_locked_day.csv` does not exist on this
box — `data/real/` itself is absent (as `prod-p0-setup.md` already recorded),
and git does not track the path. A search of `runs/**/config.resolved.yaml`,
`lab/jobs/**/cmd.json` and the tracked tree returns hits only in "do not touch"
lines of briefs, specs and prior audits. The addendum's claim is therefore true
but vacuous: the locked day could not have been read because it is not present.
Nothing here supports a *confirmatory* model claim later; that still needs the
locked day on the machine that trains the model.

**12. Split hygiene.** Zero audio-path overlap between the two CSVs (157,462
and 3,114 unique paths); all 4,800 empty-text noise rows in train, none in
test; `suspect` is `False` on every row; test fraction of speech 0.01999
against the 0.02 target.

## What would fix each non-CONFIRMED line

**7. Fidelity-history ordering (docs/results.md L344-348).**
`runs/prod_fid/config.resolved.yaml` has an *empty* chain,
`runs/prod_fid_d3prime` has the low-pass only, `runs/prod_fid_d3pp_a1` has
low-pass + peaking_eq, and the LP commit `2e2dee2` predates the D3' render.
So D3' was measured *with* the LP, and the table's row order (D3, D3', "LP
step", "Recipe correction", D3'') reads as if the LP were a response to D3'.
Fix, wording only: move the `LP step` row above the D3' row and change its
note to "Landed before the D3' re-measurement (commit `2e2dee2`); blameless
in band, fixed the 4 kHz excess". No number changes.

**10. Uncommitted `lab/` deliverables.** `git status` shows untracked
`lab/briefs/prod-p4.md`, `lab/briefs/prod-p4-gate-s{1,2,3,4}-watch.md` and
`lab/reports/prod-fix-1khz.audit.md` — the last being the audit that cleared
the render and is cited by both `lab/STATE.md` and the addendum — plus
modified `lab/STATE.md`, `docs/results.md` and `lab/briefs/prod-fid-watch.md`,
and a modified shared skill `.github/skills/monitor-run/SKILL.md` (the
15-minute watch-cadence rule). Fix, before sign-off:

```powershell
git add lab/briefs/prod-p4.md lab/briefs/prod-p4-gate-s1-watch.md lab/briefs/prod-p4-gate-s2-watch.md lab/briefs/prod-p4-gate-s3-watch.md lab/briefs/prod-p4-gate-s4-watch.md lab/reports/prod-fix-1khz.audit.md lab/reports/prod-close.audit.md lab/briefs/prod-fid-watch.md .github/skills/monitor-run/SKILL.md lab/STATE.md docs/results.md
git commit -m "prod-v1: close PASS-WITH-NOTES"
```

**13. Gate tier not reachable from the corpus.** `scripts/gate_dataset.py`
writes `manifest_gated.jsonl` and deletes nothing, but
`scripts/export_corpus_csv.py::read_manifest` reads `manifest.jsonl` — the
pre-gate file — and the CSV schema is `audio,text,suspect` with no tier
column. The exported corpus therefore contains all 160,576 rows, including the
62,341 the gate rejected, with nothing to distinguish them; the script's own
docstring ("rows that failed QC or the gate were already filtered upstream")
is false for this export. This does **not** violate the pre-registered D4,
which asks for exactly 160,576 rows, and it is the runbook's literal §4
command — so P4 is compliant — but the runbook's own advice ("training should
use the gated tier mix per the recipe, not gold-only") cannot be followed from
`data/corpus/V1.0.0/` alone. Fix, owner's choice, one of: (a) note in the
addendum and the asr handoff that tiers live in
`runs/train_v1_s{1..4}/manifest_gated.jsonl` and must be joined on the audio
path; (b) re-export from the gated manifests with a `tier` column. Either is a
CPU-only change; no re-render is implied.

## Notes (not verdicts)

- `manifest.json.source.config_hash` `647b090a…` is `rows[0]`'s lineage, i.e.
  shard 1 only; s2 / s3 / s4 / noise hash to `d7932584…`, `0ffc2799…`,
  `bd6d6816…`, `ed59a9dd…`. Those differ only through `seed` and
  `noise_only_frac` (verified above), so the single hash under-describes rather
  than misdescribes the corpus. `git_revision` is recorded as `72ba48a-dirty`.
- The split key is `(kind, text)`, so 5 transcript strings (126 train rows,
  28 test rows) appear on both sides via different airports. That is the
  documented behaviour of the stratified group split, not leakage of a group,
  but it is worth knowing before any per-utterance memorization claim.
- Recomputation was CPU-only and used aggregate commands throughout; no file
  inside a dataset or corpus directory was listed or read individually. Nothing
  in this audit required the GPU, so no line is UNVERIFIABLE.
