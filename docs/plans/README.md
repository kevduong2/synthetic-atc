# atc-gan Design & Planning Docs

Deliverables of the research/design pass (2026-08-23). Scope: audio generation only — text/scenario generation belongs to the other team and plugs in via `TextSource` (JSONL contract in [02](02-architecture.md) §5). Kokoro TTS is a fixed dependency.

| Doc | Contents |
|---|---|
| [00-research-findings.md](00-research-findings.md) | Phase 1 synthesis: 2026 SOTA conclusions + locked design decisions. Full sourced surveys in `../research/`. |
| [01-codebase-analysis.md](01-codebase-analysis.md) | Phase 2: what exists, measured properties of the local calibration clips, gap analysis, target layout. |
| [02-architecture.md](02-architecture.md) | Shared infrastructure: pipeline stages, `ChannelBackend` interface, config system, text-source contract, output format. |
| [03-mode1-procedural-plan.md](03-mode1-procedural-plan.md) | Mode 1: randomized procedural radio channel — primitives, parameter table, `wide`/`matched` profiles. |
| [04-mode2-calibrated-plan.md](04-mode2-calibrated-plan.md) | Mode 2: calibrated channel learned from real clips — fitted presets + noise bank + gated residual CUT; 1k→10k expansion workflow. |
| [05-evaluation-plan.md](05-evaluation-plan.md) | Metric tiers (QC gates → distribution match → channel probe → downstream WER), acceptance criteria, module layout. |

## Master roadmap (task IDs from the per-doc plans)

Ordering minimizes idle dependencies; S/E tracks are CPU-friendly, M2.4+ and Tier 3 need the 5080.

1. **S1–S3** (02): config system → primitives/chain refactor (`dsp.py` retired, tests ported) → builder integration (provenance, quotas).
2. **E1** (05): QC gates + channel statistics + report — needed by every later acceptance check.
3. **P1–P2** (03): new primitives (squelch gate, PTT truncation, mic coloration, fading) → `wide`/`matched` profiles validated against the calibration clips. ⇒ **Mode 1 shippable.**
4. **M2.1–M2.3** (04) with **E2** (05): local corpus + noise bank → per-clip channel fitting → CalibratedChannel backend, gated by the channel probe. ⇒ **Mode 2 v1 (DSP-hybrid) shippable.**
5. **P3** (03) voice-augment layer; **M2.4** residual CUT (gated); **M2.5** expansion workflow.
6. **E3–E4 + P4 + M2.6** : fixed Tier 3 WER protocol; Mode 1 vs Mode 2 vs mix decision; regression harness.
7. Optional/later: **M2.7** diffusion spike (only if CUT fails its gates); VC/accent-conversion stage (research says biggest remaining upside).
