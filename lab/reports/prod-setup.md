# prod-setup            2026-09-01T23:16:38Z
commit: a7023ab0235ca4d4944011fea5f2a27f48f9808b   uv: 0.11.19   python: 3.12.10   driver: 616.56   VRAM free: 8594 MB   disk free: 1758.47 GB
torch: 2.14.0+cu126   cuda: True   gpu: NVIDIA GeForce RTX 3080   libsndfile: 1.2.2

1 repo        PASS   required commit 8fcd7c4 is an ancestor of HEAD; HUMAN.md and lab/missions/prod-v1.md present
2 toolchain   PASS   RTX 3080 present; driver 616.56; archive 22.34 GB compressed / 27.51 GB extracted; required free space 67.51 GB; F: free 1758.47 GB
3 env         PASS   CUDA available; torch is a CUDA build; uv sync completed
4 data        PASS   via 4a (existing generated artifacts verified)   text 77888 / 155776   vocab present
5 clips       PASS   ZIP integrity passed; mapped extraction retains 209259 WAVs / 29539755156 bytes; approved facility proxies supply KEUG, KOJC, S50, and KSDL

Mapped station table:

| Count | Name |
|---:|---|
| 126 | KEUG_CASCADE_APR_DEP |
| 8000 | KIXD_TOWER |
| 86104 | KOJC_KC_CENTER |
| 78546 | KSDL_PHOENIX_AP_DP |
| 15991 | KSDL_TOWER |
| 3223 | KSLE_GROUND |
| 4934 | KSLE_TOWER |
| 14 | S12_CTAF |
| 10305 | S50_SEATTLE_CENTER |

Unmatched names: 2016. These would land in station `unknown`.

| Count | Malformed pattern |
|---:|---|
| 1 | CASSCADE_APR_DEP_&lt;N&gt;_&lt;N&gt; - Copy.wav |
| 12 | KIXD_&lt;M-D-YYYY&gt;_clip&lt;N&gt;.wav |
| 2000 | KIXD_TOWER_&lt;M-D-YYYY&gt;_clip&lt;N&gt;.wav |
| 1 | KSLE_GROUND_&lt;N&gt;_&lt;N&gt; - Copy.wav |
| 2 | SEATTLE_CENTER_&lt;N&gt;_&lt;N&gt; - Copy.wav |

Nested folder: yes (`airport_clips_v2/clips`). Calibration merge skipped because KSDL_TOWER is already present; `data/real/calibration` contains no WAVs. Timestamped facility aliases applied by owner direction: `CASSCADE_APR_DEP -> KEUG_CASCADE_APR_DEP`, `PHOENIX_AP_DP -> KSDL_PHOENIX_AP_DP`, `SEATTLE_CENTER -> S50_SEATTLE_CENTER`, and `KC_CENTER -> KOJC_KC_CENTER`. Audio bytes are unchanged.

6 tests       PASS   780 passed, 3 skipped, 0 failed, 137 warnings in 88.28s
7 lab         PASS   GPU lock held: false; Mission: none started
8 HF cache    NOT NEEDED   C: free 98.90 GB, above the 20 GB warning threshold

READY FOR prod-v1: YES   all setup gates passed