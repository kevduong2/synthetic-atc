# Report prod-p1-calib                (brief: lab/briefs/prod-p1-calib.md)
## Director summary (<=150 words)
PASS. The completed fit retained 1,232 of 1,302 presets. Deployed counts are KEUG 85 (`KEUG_CASCADE_APR_DEP`), KOJC 143 (`KOJC_KC_CENTER`), S50 144 (`S50_SEATTLE_CENTER`), KSLE 145 (`KSLE_TOWER`), KIXD 150 (`KIXD_TOWER`), and KSDL 142 (`KSDL_TOWER`). Additional unselected pools are `KSDL_PHOENIX_AP_DP` 138, `KSLE_GROUND` 131, `S12_CTAF` 4, and `unknown` 150. `configs/mode2_v1.yaml` loads and resolves with exactly the six deployed facilities in its uniform `station_mix`; all have n >= 30, so D1 is PASS and B1 does not apply. The resolved config hash is `6bc8b4b1f3d54a2a1238acfc73ab43a199c84a81d17052620f112fa4b59a825a`. No section 2 or later work ran. Recommended next action: brief and launch `prod-p2-resid` under D2/B2.

## Results

The fit processed 1,302 candidates from `channel_train`, retained 1,232, and
dropped 70 (5.38%). Its recorded runtime was 8,398.1 seconds (139 minutes
58.1 seconds), using 300 optimization steps and two probes per candidate.
The reward-cell unit C is not applicable to this calibration phase.

| Facility in `presets_stats.json` | Kept presets | In production `station_mix` |
|---|---:|---|
| `KEUG_CASCADE_APR_DEP` | 85 | yes (KEUG) |
| `KOJC_KC_CENTER` | 143 | yes (KOJC) |
| `S50_SEATTLE_CENTER` | 144 | yes (S50) |
| `KSLE_TOWER` | 145 | yes (KSLE) |
| `KIXD_TOWER` | 150 | yes (KIXD) |
| `KSDL_TOWER` | 142 | yes (KSDL) |
| `KSDL_PHOENIX_AP_DP` | 138 | no |
| `KSLE_GROUND` | 131 | no |
| `S12_CTAF` | 4 | no |
| `unknown` | 150 | no |

The source manifest contained 10,214 `channel_train` rows and 1,888
`channel_val` rows. The `--per-station 150` selection produced the 1,302 fit
candidates before global fit-loss QC.

### Config smoke

`load_config` accepted `configs/mode2_v1.yaml`. `dump_resolved` wrote
`runs/prod-p1-config-smoke/config.resolved.yaml` with hash
`6bc8b4b1f3d54a2a1238acfc73ab43a199c84a81d17052620f112fa4b59a825a`.
The smoke asserted that the configured station names exactly equal the six
deployed names above and that each corresponding retained count is at least
30. A structured comparison with `configs/mode2_fastcut_kixd.yaml` found only
the five authorized changed leaves: calibration `corpus_dir`, `presets`,
`noise_bank`, and `station_mix`, plus `residual.checkpoint`.

## Decision rules

D1: each deployed facility has n >= 30 and `station_mix` lists exactly those
six -> observed minimum n = 85 and exact six-name equality -> **PASS**.

B1: any deployed facility has n < 30 -> no deployed facility does -> **NOT
TRIGGERED**.

## Interpretation

At this calibration budget, the retained preset pool supports balanced
uniform draws across all six deployed facilities. The smallest deployed pool,
KEUG, has 85 retained presets, leaving a 55-preset margin over the D1 floor.
The additional facilities remain represented in the fit artifacts but cannot
be drawn as the primary channel because they are absent from `station_mix`.

## Artifacts and exact commands

- Presets: `runs/channel_data_v2/train/presets.jsonl`
- Fit statistics: `runs/channel_data_v2/train/presets_stats.json`
- Fit log and status: `lab/jobs/prod-p1-fit/log.txt`, `lab/jobs/prod-p1-fit/status.json`
- Production config: `configs/mode2_v1.yaml`
- Resolved smoke artifact: `runs/prod-p1-config-smoke/config.resolved.yaml`
- Fit: `uv run python -m atcgen.channel.learned.channel_fit runs/channel_data_v2/corpus.jsonl runs/channel_data_v2/train/presets.jsonl --probe-dir runs/gan_a_base_v1/clean --split channel_train --per-station 150 --device cuda`
- Aggregate counts: `uv run python -c "import json; s=json.load(open('runs/channel_data_v2/train/presets_stats.json')); print(json.dumps({'kept':s['kept'],'dropped':s['dropped'],'stations':{k:v['n'] for k,v in s['stations'].items()}}, sort_keys=True))"`
- Config smoke: `uv run python -c "from atcgen.config import load_config, dump_resolved; import json; c=load_config('configs/mode2_v1.yaml'); s=json.load(open('runs/channel_data_v2/train/presets_stats.json')); expected={'KEUG_CASCADE_APR_DEP','KOJC_KC_CENTER','S50_SEATTLE_CENTER','KSLE_TOWER','KIXD_TOWER','KSDL_TOWER'}; actual=set(c.calibrated.calibration.station_mix); assert actual == expected, (actual, expected); assert all(s['stations'][name]['n'] >= 30 for name in expected); p,h=dump_resolved(c,'runs/prod-p1-config-smoke'); print(c.calibrated.calibration); print('resolved', p, h)"`
- Validation: `uv run pytest -q` -> 780 passed, 3 skipped, 0 failed in 87.85 seconds.

## Not done / deviations

- Resume note 2 prohibited relaunching the completed fit, so this execution
	began with its emitted statistics and completed only the remaining work.
- The first fit attempt was killed at 90 minutes under the earlier guardrail
	misreading. The exact command was relaunched before Resume note 2 and ran
	139 minutes 58.1 seconds to exit 0, consistent with the clarified rule that
	watched runbook compute jobs continue to their expected duration.
- Section 1a and the pre-fit section 1b artifacts were reused under the
	brief's resumability instruction; they were not regenerated.
- The residual is intentionally disabled pending P2 selection. No section 2
	or later command ran, and the locked-day corpus was not read.
