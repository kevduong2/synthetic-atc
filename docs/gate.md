# Verification gate

The gate answers a narrow question: does the post-channel waveform support the
label that will be used for training? It runs after dataset construction, on
the waveform in `wavs/`, not on the clean TTS signal. Its rule is reject,
never relabel. Relabeling a failed render with whatever a teacher happened to
hear would launder a generator error into ground truth and teach the student
the generator's mistake.

Every input row remains in the gated manifest. The gate adds a `tier` and a
machine-readable `gate` blob; downstream code chooses which tiers to train on.
The generation path and manifest fields are described in
[generation.md](generation.md). The downstream training and evaluation flow
is in [training-and-eval.md](training-and-eval.md), and the observed matrix
results are summarized in [results.md](results.md). The
[CLI reference](cli-reference.md) covers the repository's command conventions.

## Frozen teacher pool

`atcgen.gate.teachers` defines exactly two default teachers:

- `openai/whisper-base.en` is an encoder-decoder model with a language-modelled
  seq2seq decoder. It can recover fluent English from poor audio, but that
  decoder can also invent plausible speech and hallucinate on silence.
- `facebook/wav2vec2-base-960h` is a frame-synchronous CTC model without a
  decoder language model. Its failure mode is more often silence or letter
  soup, which supplies a different check on the seq2seq teacher.

The architectural difference matters more here than two checkpoints from one
family: agreement is evidence only when the teachers' failure modes are not
identical. Both teachers are frozen. `default_teachers` constructs one
`WhisperTeacher` and one `CTCTeacher`; neither is fine-tuned on generated
audio.

Inference is batched. The module docstring reports, on 24 M-series smoke clips
with batch size 8, `whisper-base.en` at `16.4` clips/s on MPS and `5.6` on
CPU, and `wav2vec2-base-960h` at `29.8` on MPS and `6.3` on CPU. Batch 8 was
faster than batch 16 for both teachers because short ATC clips otherwise pay
for more padding. The reported aggregate for the pair is about `10.6`
clips/s, or about six minutes for 4,000 clips. Actual runs should use the
`throughput` object written by the gate rather than assuming those smoke-set
figures.

## Tiers and thresholds

`GateConfig` applies the audio checks, teacher WER, and critical-entity rules.
The defaults are:

| field | default | meaning |
| --- | ---: | --- |
| `gold_wer` | `0.25` | upper best-teacher WER for gold |
| `silver_wer` | `0.50` | upper best-teacher WER for silver |
| `adversarial_wer` | `0.90` | ceiling above which a speech row is rejected |
| `gold_critical_recall` | `0.5` | critical-slot recall required for gold |
| `adversarial_critical_recall` | `0.5` | critical-slot recall required |
| `adversarial_cap` | `0.05` | maximum adversarial share selected for a mix |

The remaining defaults are `noise_max_words=2`,
`noise_requires_consensus=True`, `repeat_threshold=0.80`,
`repeat_min_lag_sec=0.25`, and `repeat_frame_sec=0.02`. Audio QC mirrors the
evaluation QC thresholds: duration `0.5`--`30.0` seconds, maximum clipped
fraction `0.01`, RMS from `-40.0` to `-8.0` dB.

For a non-empty reference, the tier rules are:

- `gold`: no rejection reason, best teacher WER at most `0.25`, and critical
  recall at least `0.5`;
- `silver`: no rejection reason and best teacher WER at most `0.50`;
- `adversarial`: no rejection reason, best teacher WER above `0.50` and at
  most `0.90`, and critical recall at least `0.5`; and
- `rejected`: any audio failure, repeated segment, critical substitution,
  WER above `0.90`, or an unverified hard clip.

The `silver` rule intentionally does not add a critical-recall condition after
the gold test. A low-WER row with missed critical slots can therefore be
silver, while a hard row must meet the adversarial recall bar. Any critical
substitution rejects before these tier comparisons.

Noise-only rows have an empty reference and no WER. A teacher is considered to
have heard speech only when it emits more than `noise_max_words` normalized
words. With the default `noise_requires_consensus=True`, all teachers must
cross that threshold before `noise_row_has_speech` is added; this prevents one
seq2seq hallucination from discarding a useful anti-hallucination row. A clean
noise-only row is gold unless an audio check fails.

`TRAINABLE_TIERS` is exactly `("gold", "silver", "adversarial")`. Rejected
rows are retained for audit and statistics, never for a selected training
mix.

## Critical-slot verification

`verify_entities` compares the reference `Entity` list with each teacher's
parsed hypotheses. The rule is deliberately asymmetric:

- A slot is `verified` when any teacher recovers its exact `(type, value)`.
- A slot is `missed` when no teacher recovers it and there is no contradictory
  slot-type evidence.
- A slot is `substituted` when every teacher that produced that slot type
  produced a value that is not present in the reference for that type.

The first case is positive evidence that the waveform carries the value. The
third case is semantic evidence against the label. `evaluate_row` adds
`critical_entity_substitution` for any substituted critical slot and rejects
the row regardless of WER. A teacher that never engaged with a slot type does
not supply negative evidence, so absence alone is a miss and only lowers
critical recall.

`hypothesis_entities` parses both the raw teacher string and its
`normalize_atc` rendering, then unions the recovered entities for positive
evidence. It also returns the raw rendering's entities separately as
`asserted`. Negative evidence uses only `asserted`: normalizing an ASR string
can re-spell an ambiguous number and create a false contradiction. Thus a
normalized rendering can help verify a slot, but it cannot by itself turn a
slot into a substitution.

The per-row entity report contains each slot's `type`, canonical `value`,
`critical` flag, verdict, `verified_by` teachers, and contradictory `heard`
values. It also contains `critical_total`, `critical_verified`,
`critical_substituted`, `critical_missed`, `critical_recall`,
`all_critical_verified`, and `any_critical_substitution`.

## Per-row evaluation

`evaluate_row(row, hypotheses, audio, config)` returns `(tier, gate_blob)`.
The row needs `text` and, for speech, serialized ground-truth `entities`.
The hypothesis mapping is teacher name to raw text; `audio` is the result from
`audio_checks`.

For each teacher, the gate stores:

- `hyp`, the raw un-normalized text;
- `hyp_normalized`, the text passed to WER;
- normalized word count;
- `wer` against the normalized row text, or `None` for a noise-only row;
- parsed `entities` from raw and normalized renderings; and
- `asserted`, the raw-rendering entities used for negative evidence.

`audio_checks` runs the audio QC thresholds and `repeat_score`. The latter
autocorrelates the 20 ms loudness envelope only at lags of at least 0.25 s;
the default score threshold is `0.80`. A failing QC result contributes an
`audio_<reason>` reason, and a repeated clip contributes
`audio_repeated_segment`.

For speech rows, `best_teacher` is the teacher with the lowest WER and
`best_wer` is its value. The gate then verifies entities, applies the
substitution and hard-clip rules, and writes `reasons`. A hard row in the
`0.50`--`0.90` WER band receives `hard_clip_unverified_entities` when its
critical recall is below `adversarial_critical_recall`.

## Dataset pass and stored output

`gate_dataset` accepts a dataset directory containing `manifest.jsonl` or a
manifest path. It reads audio in batches, calls every supplied teacher once
per batch, evaluates each row, and writes:

```text
manifest_gated.jsonl
gate_stats.json
```

The default `batch_size` is `8`. Every output row is the original row plus
`tier` and `gate`; `audio`, `text`, `text_display`, `entities`, `gen`, and
`lineage` remain present. The gate never deletes a rejected audio file or
rewrites its text. `gate_stats.json` records the resolved config, teacher
names, tier counts and fractions, rejection reasons and rates, WER
distributions, best-teacher counts, entity rates, and aggregate throughput.

The stored `lineage` is the builder's provenance blob. It keeps the config
hash, profile, mode, seed, text source, code metadata, and build time attached
to a gated row, while `gate` records what the teachers and audio checks did to
that row. This allows a rejected or selected row to be traced without relying
on a separate run directory.

`load_gated` reads `manifest_gated.jsonl` from a path or directory. It does not
run inference. `retier` re-applies new `GateConfig` thresholds to already
stored raw hypotheses and audio/entity blobs, so threshold sweeps do not call
the teachers again. It returns rows with updated `tier` and `gate` blobs.

`select_tiers` accepts rows or a gated manifest and defaults to `("gold",)`.
It validates tier names, preserves manifest order, and filters to the requested
tiers. If `adversarial` is requested, it keeps only the first rows needed to
make the adversarial fraction no greater than `adversarial_cap` relative to
the other selected tiers. With the default cap, adversarial rows can occupy at
most five percent of the selected mix.

## Command-line use

Run the gate on a built dataset with:

```bash
uv run python scripts/gate_dataset.py --dataset runs/demo
```

The script writes the gated manifest and statistics next to the dataset unless
`--out` is supplied. `--batch` defaults to `8`; `--max-samples` limits a
smoke pass; `--device` accepts `mps`, `cpu`, or `cuda` and otherwise uses the
teacher device picker; and `--quiet` disables the progress bar. Every field in
`GateConfig` is also exposed as a flag, including `--gold-wer`,
`--silver-wer`, `--adversarial-wer`, the critical-recall thresholds,
`--adversarial-cap`, and the audio/repeat settings.

For example:

```bash
uv run python scripts/gate_dataset.py --dataset data/train_v1 \
    --out runs/gate_strict --gold-wer 0.15 --device cpu
```

The CLI always constructs the two default frozen teachers. Programmatic calls
to `gate_dataset` may supply another `Teacher` list, which is how the tests
use deterministic fake teachers.

## Observed matrix result

The `matrix_v1` 8,000-row pool was split as follows by the default gate:

| tier | rows | fraction |
| --- | ---: | ---: |
| gold | 1,950 | 24.37% |
| silver | 2,290 | 28.63% |
| adversarial | 383 | 4.79% |
| rejected | 3,377 | 42.21% |

The rounded operational summary is 24% gold, 29% silver, 4.8%
adversarial, and 42% rejected. The downstream locked-test comparison gives
the gate a measurable cost-benefit: A2, trained on the gated synthetic pool,
has `0.5728` WER versus `0.6268` for A2u, the same pool ungated, a `-5.40`
absolute-WER difference. Critical substitution rate is `19.5%` for A2 versus
`25.4%` for A2u. The matrix's paired comparison and the rest of the result
context are recorded in [results.md](results.md).
