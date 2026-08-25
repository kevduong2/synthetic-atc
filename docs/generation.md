# Generation

The generation path is implemented in `atcgen.dataset.build`. It samples a
text source, renders the spoken form, applies voice variation, passes the
waveform through a procedural or calibrated channel, normalizes its output
level, and writes a WAV plus one JSON manifest row. The surrounding component
boundaries are described in [the architecture notes](architecture.md); the
command-line entry point is documented in [the CLI reference](cli-reference.md).
The [systems manual](systems-manual.html) and [research findings](research-findings.md)
give the larger design context.

## Text and scenario layer

`atcgen.text.grammar` builds controller and pilot exchanges from templates.
`generate_exchange` creates a `Scenario`, composes `Line` objects, and returns
only an exchange that passes validation. `generate_utterance` chooses one line
from a validated exchange. The supported regions are `us`, `eu`, and
`mixed`; a `mixed` draw chooses European phraseology with probability `0.6`
and US phraseology otherwise.

`ScenarioConfig` is the grammar configuration:

- `region` defaults to `"mixed"`.
- `readback_error_prob` defaults to `0.05`. At most one legal value is
  corrupted in an exchange; the grammar queues a controller correction, and
  the wrong readback is labelled with what was actually spoken.
- `confusable_callsign_prob` defaults to `0.05`. When it fires, the exchange
  contains a same-airline callsign that differs at one digit.
- `phonetic_respelling_prob` defaults to `0.5`. It controls ATC digit variants
  and also makes grouped readings possible.
- `vocab_path` defaults to `data/vocab/real_anchor.json`.
- `max_retries` defaults to `25`.

The three probability fields must be in `[0, 1]`. An invalid region or
probability raises `ValueError`. A failed exchange is redrawn up to
`max_retries`; if no valid exchange is found, `generate_exchange` raises
`ValueError` rather than returning an invalid label.

Each `Utterance` contains `spoken`, `transcript`, `role`, `kind`, `meta`,
`weight`, `category`, `entities`, and `display`. `spoken` retains the
punctuation used for prosody. `_transcript` removes `, . ! ? ; :` and
collapses whitespace. `display` is a controller-facing rendering; an empty
display means the transcript is used. `weight` defaults to `1.0` and
`category` to `"routine"`.

The validator is deterministic for a completed draw. `validate_utterance`
checks non-empty spoken text, single-spacing in the transcript, the allowed
role values, legal entities, and the invariant that an entity's `spoken`
substring occurs in the transcript. It then calls
`atcgen.entities.extract_entities` on the transcript:

- every born-labeled entity must be recovered with the same `(type, value)`;
- a recoverable, unlabelled critical entity is an error; and
- `validate_exchange` checks that all callsigns are compatible with the
  exchange's expected callsigns, including safe abbreviated readbacks.

This round trip is a consistency check, not the way the grammar creates
labels. `Line.build` collects each `Slot.entity` while composing the line, so
the grammar's output is born-labeled and is never parsed to manufacture its
own ground truth.

## Vocabulary and text sources

`atcgen.text.lexicon` owns the built-in phraseology pools, spoken renderers,
and canonical value pickers. `Style.draw` makes one pronunciation draw per
transmission:

- `atc_variants` selects `tree`, `fife`, `niner`, and `fower` in place of
  `three`, `five`, `nine`, and `four`;
- `group_numbers` can render forms such as `one twenty seven`; and
- `decimal_word` is `decimal` for `eu` and `point` for `us`.

`Vocab.load` starts with `DEFAULT_AIRLINES`, then merges the optional anchor
file. Harvested airline phrases are weighted by their counts; harvested
European stations and waypoints are added to their respective pools. Missing
or unreadable vocabulary falls back to the built-in lists. `make_slot` turns a
canonical value into a `Slot(spoken, display, entity)`. The entity is attached
at that point, while unlabelled phrases such as greetings, wind, and taxi
routes use `plain`.

`atcgen.text.sources` exposes a `TextSource` protocol with
`sample(rng) -> Utterance`. `GrammarTextSource` wraps `ScenarioConfig` and
accepts its fields as keyword overrides. `JsonlTextSource` reads external
records and validates any supplied entities with `check_entity`.

`make_text_source` accepts these forms:

```text
grammar
grammar:region=eu,readback_error_prob=0.1
```

It also accepts a dictionary such as `{"kind": "grammar", "region":
"eu"}` or `{"kind": "jsonl", "path": "..."}`. Anything else is treated as
a JSONL path. Probability and retry values in an inline grammar spec are
coerced to `float` and `int` respectively. A finite source can be wrapped in
`WeightedSampler`; it draws a category first, then an utterance by positive
`weight`, and applies `dataset.category_quotas` when categories are present.

### Real vocabulary anchor

`scripts/harvest_vocab.py` reads only the first 8,000 rows of the
`jacktol/atc-dataset` train split: `train[:8000]`. It reads the `text` column
without decoding audio and counts candidate airlines, facility phrases, and
waypoints. It prunes likely fragments, keeps counts at or above the default
`--min-count 3`, assigns collision-free ICAO-like codes, and writes
`data/vocab/real_anchor.json`.

The file records `corpus`, `split`, `utterances`, `min_count`, and the three
vocabulary sections. The train prefix is intentional: reward validation,
model selection, and locked-test rows are not used to shape the grammar. The
grammar therefore sees real carrier, station, and waypoint forms without
reading the evaluation portions. If the artifact is absent, generation still
uses the built-in lexicon.

## Entity schema and parsing

`atcgen.entities.Entity` is a frozen record with `type`, canonical `value`,
the spoken rendering in `spoken`, and `critical`. If `critical` is omitted,
it is derived from `CRITICAL_TYPES`.

The complete `ENTITY_TYPES` tuple is:

```text
callsign, runway, heading, altitude, flight_level, frequency,
speed, squawk, altimeter, waypoint, atis
```

The critical types are `callsign`, `runway`, `heading`, `altitude`,
`flight_level`, `frequency`, `speed`, `squawk`, and `altimeter`. `waypoint` and
`atis` are non-critical by default. Canonical forms include `CSA123`, `24L`,
`270`, `3500ft`, `FL350`, `127.825`, `250`, `4521`, `Q1013` or `A2992`,
`PADKA`, and `C`.

`check_value` validates both format and domain. It enforces runway `01`--`36`,
heading `001`--`360`, flight level `FL010`--`FL450`, altitude `100`--`45000`
feet in round hundreds, frequency `118.000`--`136.975` on a 5 kHz step,
speed `60`--`350` knots, octal four-digit squawks, QNH `900`--`1100`, and
altimeter `A` values from `2800`--`3150`. `check_entity` adds the invariant
that `critical` matches the entity type.

`extract_entities` is a precision-first parser for spoken-form text. Its
number scanner accepts literal digits and words, including `niner`, `tree`,
`fife`, and `fower`; it handles `decimal` and `point`, arithmetic group forms
with `hundred` or `thousand`, and runway ordinal suffixes such as `26th`.
Runway side words map to `L`, `R`, or `C`. Unit suffixes anchor `feet` and
`knots`, vertical-clearance words can anchor a bare altitude, and callsign
matching uses airline telephony phrases.

Telephony lookup uses `joined_key`, so word boundaries do not change a
carrier's code: `aero mexico` and `aeromexico`, or `wizz air` and `wizzair`,
are equivalent. The same airline table is used for grammar, gate, and
evaluation when no explicit table is supplied. This keeps canonical values
comparable even when an ASR hypothesis joins or separates telephony words.

`score_entities` returns `EntityScore`. It matches a multiset of `(type,
value)` pairs, records per-type true positives, false positives, false
negatives, and same-type substitutions, and derives precision, recall, F1, and
callsign accuracy. `critical_substitution_rate` is
`critical_sub / critical_ref`: a critical value present on both sides with a
different value counts as a substitution, not merely as a miss.

The grammar attaches entities structurally while composing slots. The parser
exists for teacher or student hypotheses and for real transcripts, where
labels are not available structurally. This distinction is also used by the
[verification gate](gate.md).

## TTS and voice variation

`atcgen.tts.synthesize` defines the `TTSEngine` protocol: an engine has
`sample_rate` and `synthesize(text, rng) -> np.ndarray`. `KokoroTTS` uses
Kokoro's native `SAMPLE_RATE` of `24000`, concatenates its generated chunks,
and peak-normalizes a nonempty result to `0.9`. Its default voice pool is
`af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`,
`am_michael`, `am_eric`, `am_onyx`, `bf_emma`, `bf_isabella`,
`bm_george`, and `bm_lewis` as declared by `KOKORO_VOICES` and the config
defaults. The builder draws a configured voice and speed and pins a
pool-style engine to those values for the render.

`VoiceAugment` in `atcgen.tts.augment` applies independently sampled pitch,
tempo, and EQ-tilt effects to clean mono TTS. The default config distributions
are pitch `prob: 0.5, uniform: [-2, 2]`, tempo `prob: 0.3, uniform: [0.9,
1.1]`, and EQ tilt `prob: 0.4, uniform: [-3, 3]`. A `VoiceConverter` can run
after these effects. The result and the sampled values are returned so the
builder can record them in `gen`.

The [known-issues](known-issues.md) document records the current Kokoro-on-MPS
nondeterminism caveat. A fixed random seed does not make MPS waveforms
identical; manifest text and drawn parameters are the safer provenance fields.

## Channel twin

### Procedural primitives

`atcgen.channel.primitives` provides stateless functions with the common
`effect(x, sr, rng, **params) -> np.ndarray` shape. The `PRIMITIVES` registry
contains:

```text
mic_coloration, ptt_truncation, narrowband_roundtrip, resample_chain,
bandpass, agc_wander, agc_attack, am_distortion, soft_clip, dropouts,
fading, additive_noise, hum, crackle, heterodyne, squelch_gate,
squelch_clicks, cochannel_mix, codec_roundtrip
```

`NoiseBank` can supply cross-faded random crops from real WAV noise beds;
otherwise `additive_noise` draws pink or white noise. The channel target rate
is `TARGET_SR = 16000`.

### Mode 1 chain stages

`ProceduralChannel` divides configured steps by physical location:

- `SOURCE_ONCE` is `mic_coloration` and `ptt_truncation`. These model the
  talker's source and run once even when the transmission is relayed.
- The per-hop stage is every configured step outside `SOURCE_ONCE` and
  `RECEIVER_END`. It runs once for each hop.
- `RECEIVER_END` is `cochannel_mix`, `agc_attack`, `squelch_gate`,
  `squelch_clicks`, and `codec_roundtrip`. It runs after the final hop.

The builder gives pilot utterances a second hop with probability
`dataset.pilot_double_hop_prob`, default `0.5`. At a relay boundary the audio
is re-filtered before the next hop; per-hop additive noise uses the fixed
`HOP2_SNR_DB = (10.0, 25.0)` range on hops after the first. `ChannelRecord`
stores `hops`, `clean_arm`, `snr_db`, and the applied `steps`, including drawn
parameters and hop numbers. Its `applied()` method lists the primitive names.

The clean arm is selected by `clean_arm_prob`. It keeps only `bandpass`, drops
the receiver-end stage, and forces `hops` to `1`; the input is still resampled
to the target rate and framed with the chain's `PAD_SEC` padding.

When `reapply_bandpass` is true, the channel remembers the declared filter's
drawn low and high cutoffs. It re-applies that same passband at three kinds of
boundary:

- a relay boundary;
- before the first receiver-side step in `AFTER_RECEIVER_FILTER` after a
  splattering primitive; and
- at `chain_end` after the final stage.

The pending set includes nonlinear distortion and clipping, broadband noise,
hum, crackle, dropouts, co-channel audio, and squelch artifacts. These stages
can add energy outside the radio passband. The re-application models the
receiver filter after that energy is added, while using the same draw rather
than inventing a new passband. It is enabled by default and can be disabled
for the ablation `channel.reapply_bandpass: false`.

### Level and envelope control

`atcgen.channel.loudness` implements EBU R128 through `integrated_lufs` and
`normalize_lufs`. The target constant is `-23.0` LUFS, integration requires a
`0.400` second block, and a short or silent clip falls back to RMS or passes
through unchanged. The profiles currently use `output.loudness_mode: rms`:
the builder samples `output.loudness_db` per clip and then applies a `0.99`
peak ceiling. LUFS is an available post-stage mode, not the current profile
default.

`atcgen.channel.envelope` checks whether declared numeric ranges exceed a
measured real p10--p90 envelope. `configs/real_envelope.json` records `100`
real clips and the measured ranges for `snr_db`, `spectral_edge_hz`,
`spectral_low_hz`, and `rms_db`: respectively `[14.076, 36.613]`,
`[1828.1, 2735.96]`, `[156.2, 265.6]`, and `[-25.09, -18.794]`. The checker
converts those measured ranges back to configuration units using the rule
offsets and reports findings; it warns rather than failing a build.
`mode1_matched` is within the checked envelope, while `mode1_wide`
intentionally explores beyond it.

### Mode 2 calibration and learned stages

`atcgen.channel.learned.channel_fit` fits one `Preset` per real calibration
clip. The weakly supervised target is the clip's long-term spectrum,
per-bin variability, frame-energy distribution, and modulation spectrum; a
clean probe is passed through a differentiable fitted chain. The default fit
uses `300` steps and `2` probes on CPU. `preset.py` defines the JSONL schema
and its NumPy evaluator, including tanh drive, polynomial coloration, AGC,
log-frequency EQ gains, normalization, and fitted noise. `backend.py`
samples a preset per utterance, pairs it with station-indexed harvested noise,
adds SNR jitter, and applies shared post-effects such as squelch, dropouts,
co-channel audio, and codec artifacts. This is per-clip calibration, not one
global channel parameter set.

`learned/residual.py` contains the optional CUT residual. It changes
log-magnitude STFT values by a bounded additive residual, reuses the input
phase, and clamps the residual to `residual_scale_max` (default `0.35`).
`residual_train.py` trains it from fitted-DSP outputs with post-effects off
against the real calibration corpus, using the FastCUT contrastive objective.
At generation it sits between the fitted chain and post-effects and is sampled
with `residual.apply_prob`. `load_translator` returns no translator when the
checkpoint is absent; the backend warns and uses fitted DSP only. The checked-in
`mode2_default.yaml` sets `residual.enabled: false`.

`atcgen.channel.gan` is separate from this live Mode 2 path. It contains a
log-magnitude CycleGAN `model.py`, a trainer for clean-TTS and real-ATC folders,
and `GanChannel` inference. `build.make_backend` and config validation expose
`procedural`, `calibrated`, and `mix`, not a GAN backend, so this directory is
an available comparison/training path rather than part of ordinary dataset
generation.

## Dataset build and real-data preparation

`build_dataset` writes `wavs/` and `manifest.jsonl`. A generated row contains:

```text
audio, text, text_display, role, kind, category, duration, entities,
gen, lineage
```

`audio` is a relative WAV path. `text` is the transcript label;
`text_display` is the display rendering or transcript fallback. `entities` is
serialized directly from the `Utterance`, never reconstructed from audio.
`gen` records the selected voice and speed, voice-augmentation draws, channel
record, backend mode, config hash, seed index, and Tier 0 QC result when QC is
enabled. `lineage` is copied to every row and records `config_hash`, `profile`,
`mode`, `seed`, `text_source`, `atcgen_version`, `git_revision`, and
`built_at`.

Tier 0 QC is enabled by default with three retries. A failed render is retried
with new voice, speed, and channel draws while keeping its text; after the
retry budget it is written with `gen.qc.ok: false` instead of silently
shrinking the requested count. Noise-only rows use an empty text and entity
list, and are drawn at `dataset.noise_only_frac` (default `0.03`).

`atcgen.dataset.expand` combines a station-stratified real train subset with
enough synthetic rows to reach `target_total`, writes a separate
`holdout_manifest.jsonl`, and marks combined rows with `origin: real` or
`origin: synthetic`. Synthetic text comes from real-train transcripts and can
be extended with `expansion.external_texts`.

`atcgen.dataset.local_corpus` prepares local receiver WAVs by converting to
mono 16 kHz, dropping unreadable, duplicate, and silence-only files, parsing
station prefixes from filenames, and assigning station-stratified train and
holdout splits. `atcgen.dataset.real_atc` loads `jacktol/atc-dataset`, renames
`transcription` to `text`, and provides exports for real ATC audio and a
CycleGAN domain. `atcgen.dataset.noise_harvest` uses the local-corpus energy
VAD to write non-speech segments of at least `200` ms by default, with station,
RMS, spectral centroid, and `squelch_gated` metadata.

## Profiles

The YAML profiles currently checked in have these purposes:

- `mode1_default.yaml`: the simplest procedural chain, retained as the plain
  starting point; its ranges predate the real-corpus fit.
- `mode1_matched.yaml`: procedural Mode 1 narrowed and fitted to the local
  calibration statistics, including harvested receiver noise.
- `mode1_wide.yaml`: deliberate domain randomization over the wider union of
  defensible ranges, including values beyond the measured envelope.
- `mode2_default.yaml`: calibrated Mode 2 with per-clip presets and station
  noise, plus shared event effects; the CUT residual is disabled.
- `ablation_pitch_off.yaml`: the Mode 1 wide arm with only
  `voice_augment.pitch_semitones` disabled.

The generation result is the input to the [training and evaluation
flow](training-and-eval.md), and the verification step is documented in
[gate.md](gate.md). Tier selection and later optimization are covered in the
[RL loop notes](rl-loops.md) and [results](results.md).
