# Documentation

The docs describe the current pre-release PoC. The code remains authoritative
for behavior and thresholds.

- [research-findings.md](research-findings.md) — the Aug 2026 engineering proposal.
- [plans/research-integration.md](plans/research-integration.md) — PoC mapping and deviations.
- [plans/fastcut-asr-research-plan.md](plans/fastcut-asr-research-plan.md) — current
  FastCUT/channel-to-ASR research conclusion, experiment design, and implementation roadmap.
- [architecture.md](architecture.md) — system circuit, invariants, and module map.
- [systems-manual.html](systems-manual.html) — illustrated architecture; open in a browser.
- [generation.md](generation.md) — scenario, speech, channel, and manifest generation.
- [gate.md](gate.md) — teacher verification, entity checks, and dataset tiers.
- [training-and-eval.md](training-and-eval.md) — student recipe, splits, and evaluation.
- [rl-loops.md](rl-loops.md) — L1, L2, and L3 boundaries and operations.
- [cli-reference.md](cli-reference.md) — current command-line entry points and options.
- [results.md](results.md) — evidence snapshot for 2026-08-25.
- [known-issues.md](known-issues.md) — operational, measurement, and corpus caveats.
- [data-licensing.md](data-licensing.md) — data provenance and license status.
- [data-handoff.md](data-handoff.md) — how to deliver the remaining airport audio (format, balance, transcripts).
- [runbook-v1-3080.md](runbook-v1-3080.md) — the V1 production run on the RTX 3080 (PowerShell): recalibrate, residual, sharded render, gate, export; lab mission `lab/missions/prod-v1.md`.
- [../agents-experiment-handoff.md](../agents-experiment-handoff.md) — the RTX 3080 window: Windows setup gate, phases, decision rules; run by the Copilot agent team in `.github/agents/`.
