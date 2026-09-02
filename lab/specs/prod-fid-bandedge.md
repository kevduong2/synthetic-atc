# Spec prod-fid-bandedge

Question: Does the single B3-supported upper-edge correction leave the frozen V1 recipe inside the unchanged D3 fidelity rule when LTAS is measured against the real multi-station target cohort?

Hypothesis: if the remaining D3 miss is no larger than the LTAS measurement floor on a fresh render, then V1 with one 3.8 kHz low-pass step has maximum absolute real-cohort LTAS gap at 1/2/3 kHz <= 2.0 dB while matched WavLM KID remains reportable with its SE.

Decision this changes: `on+LP` is the sole D3' candidate and ships only on PASS; PASS releases Section 3, while FAIL stops the mission without another filter arm or a changed gate.

## Frozen config decision

Retain the enabled FastCUT residual and every existing frozen value, including its application behavior. Append exactly one deterministic final Mode 2 channel step, after the residual and existing post-effects, to `configs/mode2_v1.yaml`:

```yaml
calibrated:
  post_effects:
    chain:
      - primitive: lowpass
        cutoff_hz: 3800
        order: 8
        zero_phase: true
```

This step is the 8th-order Butterworth 3.8 kHz LP evaluated by `scripts/analysis/filter_variants.py`; it applies to every rendered waveform. Do not add a high-pass edge. The paired `on`/`on+LP` development rows support the LP as an upper-edge correction without a separable matched-KID loss at this sample size. The 150 Hz HP overcorrected 100 Hz from +13.8 to -6.6 dB against the old curve, does not address the in-band miss, and is not supported for shipping. The disjoint `off`/`on` cohorts do not support a residual-effect claim.

## Measurement reference fix

Gate D3' against the empirical LTAS of the fixed real target cohort in `runs/calib_v2/clips`, processed with the same per-clip power normalization, PSD, and peak normalization as the synthetic render. At each frequency, define the gap as synthetic-cohort dB minus real-cohort dB. The hardcoded KIXD curve in `scripts/analysis/ltas_check.py` is 7.4 dB away from this cohort in band and is not representative of the production multi-station target; retain its gaps only as diagnostics.

This re-reference is a measurement fix, not a gate relaxation: the in-band threshold remains exactly 2.0 dB. Existing development measurements predict that D3' may still fail because the observed real-cohort in-band miss is 4.6 dB and an edge filter does not repair it; that prediction must not alter the rule.

## D3' preregistration

Candidate: fresh 150-clip seed-0 fidelity render under `runs/prod_fid_d3prime`, with the frozen Section 5 command and only the one config step above changed. No arms or search.

**D3' PASS if and only if** (a) `max(abs(LTAS_syn(f) - LTAS_real_cohort(f))) <= 2.0 dB` for `f in {1000, 2000, 3000} Hz`, and (b) primary WavLM KID is computed on energy-trimmed, -26 dB RMS-matched audio against the fixed seed-0 real reference subset and reported as KID +/- SE. There is no post-hoc KID threshold; availability of the valid matched estimate with SE is the unchanged D3 limb.

Report but do not gate: LTAS gaps at 100/200/400/4000 Hz, all hardcoded-reference LTAS gaps, matched CLAP KID +/- SE, clip counts, and the real and synthetic peak frequencies. Raw KID and the B3 `off`/`on` KID difference are not reportable decision evidence.

Budget and checkpoint: one production fidelity cell (150 clips), one matched-set build, one primary matched-KID evaluation, and one LTAS evaluation; expected wall-clock <= 60 minutes, and stop after the audited D3' report. Artifacts that prove it ran under `runs/prod_fid_d3prime`: fresh `manifest.jsonl`, `config.resolved.yaml`, LTAS JSON containing both cohort curves and direct gaps, matched-set summary, and matched-KID JSON with counts and SE.

Kill criterion: if either D3' limb fails, or the matched estimate/reference provenance is invalid, record D3' FAIL and STOP `prod-v1`; do not start Section 3, try HP or another edge, change the reference again, or relax 2.0 dB. Any further recipe correction is a new preregistered mission decision, not a continuation of this rerun.

## Ideas (unscheduled)

- Mid-band correction: parked because B3 did not evaluate one and adding it would answer a second question after seeing D3.
- 150 Hz HP: parked because it overcorrected the low edge and cannot improve the gated band.
- Paired full-size residual ablation: parked because it can repair the B3 evidence base but cannot change this frozen residual decision inside `prod-v1`.