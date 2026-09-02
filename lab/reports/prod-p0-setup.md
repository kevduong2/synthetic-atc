# Report prod-p0-setup                (brief: lab/briefs/prod-p0-setup.md)
## Director summary (<=150 words)
PASS. The synchronized environment has PyTorch 2.14.0+cu126, `cuda.is_available() == True`, an NVIDIA GeForce RTX 3080, and libsndfile 1.2.2. Pytest completed with 780 passed, 3 skipped, and 0 failed. CUDA bench medians were TTS 0.09708 s/render, GAN ordinary 0.09846 s/step (R1 0.03066 s/step), and Whisper-tiny SFT 0.13782 s/step. The extracted archive contains 209,259 WAVs: KEUG 126, KOJC 86,104, S50 10,305, KSLE 8,157, KIXD 8,000, KSDL 94,537, S12 14, and unknown 2,016. All six deployed station prefixes are present: PASS. The optional `data/real/calibration` and `data/real` manifest directories were absent, so no clips were copied and no Mac paths were relocated. Recommended next action: start `prod-p1-calib` while retaining the unknown-filename count as an ingestion diagnostic.

## Results

### Environment and tests

| Check | Result |
|---|---|
| PyTorch | 2.14.0+cu126 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 3080 |
| libsndfile | 1.2.2 |
| pytest | 780 passed, 3 skipped, 0 failed, 137 warnings in 89.75 s |
| GPU lock before bench | free |

### CUDA benchmark

All values below are medians from `runs/bench/cuda.json`.

| Section | Work unit | Median | Status |
|---|---:|---:|---|
| TTS | 20 renders | 0.097083 s/render | ok |
| GAN ordinary | 300 steps | 0.098457 s/step | ok |
| GAN R1 | 10 steps | 0.030655 s/step | ok |
| Whisper-tiny SFT | 30 optimizer steps | 0.137818 s/step | ok |
| WavLM embedding + KID | 64 clips | 1.732603 s/run | ok |
| DSP | - | - | skipped: `--presets` and `--noise-bank` were not supplied by runbook section 0 |

Peak allocated memory was 1,357,712,384 bytes for GAN, 2,102,380,544 bytes for SFT, and 468,879,872 bytes for WavLM.

### Extracted clips

The archive was already extracted under `reference-data-for-v1-run/airport_clips_v2/clips`; the parent directory contained no top-level WAVs. Counts were produced by one aggregate PowerShell pipeline with no individual filenames emitted.

| Airport prefix | WAV count | Facility labels observed |
|---|---:|---|
| KEUG | 126 | KEUG_CASCADE_APR_DEP |
| KOJC | 86,104 | KOJC_KC_CENTER |
| S50 | 10,305 | S50_SEATTLE_CENTER |
| KSLE | 8,157 | KSLE_GROUND (3,223); KSLE_TOWER (4,934) |
| KIXD | 8,000 | KIXD_TOWER |
| KSDL | 94,537 | KSDL_PHOENIX_AP_DP (78,546); KSDL_TOWER (15,991) |
| S12 | 14 | S12_CTAF |
| unknown | 2,016 | filenames not matching `<STATION>_YYYYMMDD_HHMMSS.wav` |
| **Total** | **209,259** | |

## Decision rules

P0: every deployed station prefix KEUG, KOJC, S50, KSLE, KIXD, and KSDL has at least one WAV -> observed counts 126, 86,104, 10,305, 8,157, 8,000, and 94,537 respectively -> **PASS**.

## Interpretation

The environment and required CUDA sections are operational at this budget. The clip archive contains all six deployed airport prefixes, so P1 calibration may proceed. The 2,016-file unknown bucket should remain visible in downstream ingestion statistics; this setup result does not establish whether those files are usable.

## Artifacts and exact commands

- Benchmark: `runs/bench/cuda.json`
- Environment synchronization: `uv sync`
- CUDA sanity: `uv run python -c "import torch, soundfile; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), soundfile.__libsndfile_version__)"`
- Tests: `uv run pytest -q`
- Lock check: `uv run python scripts/lab/jobs.py lock status`
- Bench: `uv run python scripts/bench_devices.py --device cuda --gan --wavlm --tts --dsp --sft --out runs/bench/cuda.json`
- Dataset scale: `$dataset = 'reference-data-for-v1-run/airport_clips_v2'; $clips = Join-Path $dataset 'clips'; [pscustomobject]@{ DatasetExists = (Test-Path $dataset -PathType Container); RootWavs = if (Test-Path $dataset -PathType Container) { (Get-ChildItem $dataset -Filter *.wav -File | Measure-Object).Count } else { 0 }; ClipsSubdirExists = (Test-Path $clips -PathType Container); ClipsSubdirWavs = if (Test-Path $clips -PathType Container) { (Get-ChildItem $clips -Filter *.wav -File | Measure-Object).Count } else { 0 } } | ConvertTo-Json -Compress`
- Station counts: `$clips = 'reference-data-for-v1-run/airport_clips_v2/clips'; Get-ChildItem $clips -Filter *.wav -File | ForEach-Object { if ($_.BaseName -match '^(?<station>.+)_\d{8}_\d{6}$') { $Matches.station } else { 'unknown' } } | Group-Object | Sort-Object Name | Select-Object Count, Name | ConvertTo-Json -Compress`

## Not done / deviations

- Archive testing and extraction were not run because the complete 209,259-WAV extraction was already present.
- `data/real/calibration` was absent, so there were no optional own-SDR WAVs to copy.
- `data/real` was absent, so there were no Mac-origin manifests to relocate.
- The exact bench flag set requested DSP, but section 0 supplies neither `--presets` nor `--noise-bank`; the benchmark recorded DSP as skipped while TTS, GAN, WavLM, and SFT completed.
- No dataset-reading command exceeded five minutes, and no individual dataset filename was emitted.