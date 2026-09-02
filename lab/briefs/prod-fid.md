# Brief prod-fid          to: experiment-engineer   from: lab-director   written 2026-09-02
Goal: run the runbook §5 fidelity check on the frozen V1 config with the selected residual: fidelity render, matched KID (with SE), and LTAS table at 100/200/400/1k/2k/3k/4k Hz.
Inputs: docs/runbook-v1-3080.md §5 fidelity procedure (follow exactly); lab/missions/prod-v1.md (D3, branch B3); configs/mode2_v1.yaml (D1+D2 complete, residual enabled, strict-load verified); .github/skills/paired-analysis/SKILL.md (matched KID via make_matched_sets.py + atcgen.eval.embed_dist, LTAS via ltas_check.py); .github/skills/gpu-jobs/SKILL.md.
Deliverable: lab/reports/prod-fid.md. Director summary must state: matched KID ± SE, the LTAS gap at each of 100/200/400/1k/2k/3k/4k, the in-band max gap, the 4 kHz excess, and D3 PASS/FAIL per: in-band LTAS gap ≤ 2 dB AND matched KID reported with SE. The 4 kHz / 100 Hz gaps are reported, not gated.
Steps: runbook §5 exactly. GPU steps via scripts/lab/jobs.py launch --gpu, job ids prod-fid (render) and prod-fid-* for evals; one stream. For any job expected > 10 min, launch, verify started, note it in the report, and return "launched, watch pending" — the director runs the watcher. When re-invoked, continue from the last completed step.
Budget: 60 min of your own activity per invocation (GPU job time not counted).
Pre-authorized branch B3 (fidelity miss: in-band gap > 2 dB, or 4 kHz excess still > +8 dB with residual on): build the filter table — scripts/analysis/filter_variants.py on the fidelity render, then matched KID + LTAS for off / on / on+LP / on+LP+HP — write the 4-row decision packet (KID ± SE and LTAS at all seven bands) into the report, then STOP. A band edge is a frozen-config change and stays Kevin's.
Kill criteria: crash twice on the same step → stop and report.
Dataset-read guardrail (standing rule): never list/glob/read individual files inside dataset directories; aggregate shell commands only; 5-min cap on ad-hoc reads ONLY, never on launched compute jobs.
Commit rule (standing): at phase close, commit deliverables with message `prod-v1: fid <verdict>`.
Do not: start §3 renders; touch data/real/kixd/kixd_locked_day.csv; change frozen values (B3 produces a packet, not a diff).
If your reply is lost: the report file is the result.
