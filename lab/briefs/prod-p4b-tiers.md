# Brief prod-p4b-tiers          to: experiment-engineer   from: lab-director   written 2026-09-03
Goal: add a per-row gate tier column to the V1.0.0 corpus export, as requested by Kevin (post-close follow-up). Pre-release repo: schema may change freely, no compat shims (AGENTS.md).
Context: lab/reports/prod-close.audit.md item 3 — the export reads pre-gate manifest.jsonl, so all 62,341 rejected rows ship indistinguishably; per-clip tier assignments exist in the four shards' gate output JSONs (see lab/reports/prod-p4.md for paths). Noise rows were never gated.
Deliverable: lab/reports/prod-p4b-tiers.md. Director summary: the column name/values as shipped, per-tier row counts in the re-exported corpus_train.csv/corpus_test.csv (must reconcile with 19,694/29,976/43,765/62,341 + 4,800 noise), and confirmation manifest.json was regenerated with new hashes.
Steps:
1. Extend scripts/export_corpus_csv.py to join per-clip tiers from the gate outputs into a `gate_tier` column (values: gold/silver/adversarial/rejected for speech, `noise` for the noise set — or the repo's existing naming if docs/gate.md defines one). Unit test for the join per repo conventions; full suite green.
2. Re-run the export over the existing five run dirs into data/corpus/V1.0.0/ (overwrite; CPU, inline). Do NOT re-render or re-gate anything.
3. Verify: total rows still 157,462 + 3,114; per-tier counts reconcile exactly with the gate reports; every row has a non-null gate_tier.
4. Update the corpus schema documentation where it lives (docs/data-handoff.md or equivalent — one place, brief).
5. Commit `prod-v1: p4b gate_tier column` (code, test, docs, manifest; CSVs only if tracked).
Budget: 60 min. Kill: join produces any unmatched clip id → stop and report (do not fabricate tiers).
Dataset-read guardrail (standing): aggregate commands only; never list/read individual files inside corpus/run dirs; 5-min cap on ad-hoc reads only.
Do not: touch data/real/kixd/kixd_locked_day.csv; re-run gates; alter tier assignments.
If your reply is lost: the report file is the result.
