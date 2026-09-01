"""The true (non-proxy) reward: generate -> fine-tune whisper-tiny -> dev WER.

`TrueRewardHarness` implements `atcgen.rl.types.RewardFn`. Each call renders
a fresh synthetic batch from a candidate config, fine-tunes a clean copy of
`base_model` on it for a short fixed number of steps, and scores the delta in
ATC-normalized WER against a zero-shot baseline on a fixed real ATC dev set.

Two things keep candidates comparable to each other rather than to text
luck or dev-set luck:

- **Common random numbers for text.** All candidates draw utterances from the
  same fixed pool (`text_pool.jsonl`, built once from the grammar source with
  a seeded RNG), via `JsonlTextSource`. Differences in reward come from the
  channel/TTS knobs in the config, not from which sentences got generated.
- **A fixed dev set and a cached baseline.** The dev slice and the zero-shot
  WER of `base_model` on it are computed once per (model, corpus, split,
  indices) and reused across every trial in a run (and across process
  restarts, via a JSON cache file).

The generator's own `config.seed` is overridden to a fixed harness value on
every call for the same reason: it controls voice/channel draws inside
`build_dataset`, and letting it vary with the candidate would confound config
effects with draw luck.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from training.evaluate import build_report, pick_device
from training.normalize import normalize_atc

from ..config import load_config
from ..dataset.build import build_dataset, load_manifest
from ..dataset.real_atc import SOURCE_KEY, load_real_atc
from ..text.sources import JsonlTextSource, make_text_source
from .finetune_lite import finetune, prepare_features, transcribe
from .stats import wer_counts
from .types import RewardResult

GEN_SEED = 20260824  # fixed generator seed forced onto every candidate config

#: Dev slice the reward is measured on when a caller names none. `dev_corpus`
#: also accepts a local `audio,text` CSV/JSONL manifest (see
#: `atcgen.dataset.real_atc.load_local_corpus`), which is how a run targets our
#: own transcribed receiver rather than a public corpus from elsewhere.
DEFAULT_DEV_CORPUS = "jacktol/atc-dataset"
DEFAULT_DEV_SPLIT = "train"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def write_text_pool(path: str | Path, size: int, seed: int) -> Path:
    """Write (or reuse) a fixed pool of `size` grammar utterances at `path`.

    One record per line: spoken/transcript/role/kind/category/weight, the
    schema `JsonlTextSource` expects. Callers that want the same text across
    several config comparisons (a search harness's trials, or the base vs.
    best arms of `scripts/rl_verify.py`) point at the same `path` and reuse
    it; a fresh comparison uses a fresh `path` and/or `seed`.
    """
    path = Path(path)
    if path.exists():
        return path
    source = make_text_source("grammar")
    rng = random.Random(seed)
    with open(path, "w") as handle:
        for _ in range(size):
            utterance = source.sample(rng)
            handle.write(json.dumps({
                "spoken": utterance.spoken, "transcript": utterance.transcript,
                "role": utterance.role, "kind": utterance.kind,
                "category": utterance.category, "weight": utterance.weight,
            }) + "\n")
    return path


def render_and_finetune(config: Mapping[str, Any], trial_dir: str | Path, *,
                        base_model: str, pool_path: str | Path, n_synth: int,
                        ft_steps: int, ft_batch: int, ft_lr: float, ft_seed: int,
                        gen_seed: int, device, processor) -> Any:
    """Render `n_synth` clips from `config` against the shared text pool at
    `pool_path` and fine-tune a fresh `base_model` on them. Returns the
    trained model. `trial_dir/config.yaml` is the resolved config actually
    used (`config.seed` forced to `gen_seed`), audit trail included.

    This is the shared core of `TrueRewardHarness.__call__` and the A/B arms
    in `scripts/rl_verify.py`: both need "one config in, one fine-tuned model
    out" against the same fixed text pool and generator seed.
    """
    trial = Path(trial_dir)
    trial.mkdir(parents=True, exist_ok=True)

    payload = dict(config)
    payload["seed"] = gen_seed
    config_path = trial / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    cfg = load_config(config_path)

    synth_dir = trial / "synth"
    build_dataset(cfg, synth_dir, n_synth, JsonlTextSource(str(pool_path)))
    manifest = load_manifest(synth_dir / "manifest.jsonl")
    features = prepare_features(manifest, processor)

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.to(device)

    return finetune(model, features, steps=ft_steps, batch_size=ft_batch,
                    lr=ft_lr, seed=ft_seed, device=device)


class TrueRewardHarness:
    """`RewardFn`: config -> (fresh synth batch -> short fine-tune -> dev WER)."""

    def __init__(self, work_dir: str | Path, *,
                base_model: str = "openai/whisper-tiny.en",
                dev_corpus: str = DEFAULT_DEV_CORPUS,
                dev_split: str = DEFAULT_DEV_SPLIT,
                dev_indices: tuple[int, int] = (0, 200),
                text_pool_size: int = 400, text_seed: int = 1234,
                n_synth: int = 200, ft_steps: int = 300, ft_batch: int = 8,
                ft_lr: float = 1e-5, ft_seed: int = 0,
                device: str | None = None, keep_audio: bool = True,
                gen_seed: int = GEN_SEED) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.base_model = base_model
        self.dev_corpus = dev_corpus
        self.dev_split = dev_split
        self.dev_indices = dev_indices
        self.text_pool_size = text_pool_size
        self.text_seed = text_seed
        self.n_synth = n_synth
        self.ft_steps = ft_steps
        self.ft_batch = ft_batch
        self.ft_lr = ft_lr
        self.ft_seed = ft_seed
        self.keep_audio = keep_audio
        self.gen_seed = gen_seed
        self.device = pick_device(device)

        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.pool_path = self._ensure_text_pool()

        self._dev_refs: list[str] | None = None
        self._dev_categories: list[str | None] | None = None
        self._dev_sources: list[str | None] | None = None
        self._dev_paths: list[str] | None = None
        self._dev_features: list[dict] | None = None
        self._baseline_report: dict | None = None

    # -- setup, cached across calls -----------------------------------------

    def _ensure_text_pool(self) -> Path:
        return write_text_pool(self.work_dir / "text_pool.jsonl",
                               self.text_pool_size, self.text_seed)

    def _ensure_dev(self) -> None:
        if self._dev_features is not None:
            return
        dataset = load_real_atc(self.dev_split, self.dev_corpus)
        lo, hi = self.dev_indices
        dataset = dataset.select(range(lo, min(hi, len(dataset))))
        self._dev_refs = list(dataset["text"])
        self._dev_categories = (
            list(dataset["category"]) if "category" in dataset.column_names
            else [None] * len(dataset)
        )
        self._dev_sources = (
            [str(value) for value in dataset[SOURCE_KEY]]
            if SOURCE_KEY in dataset.column_names else [None] * len(dataset)
        )
        # the clip each row came from, so a per-utterance dump can be joined
        # back to capture time (KIXD filenames carry it) for clustered CIs
        self._dev_paths = [
            (row.get("path") if isinstance(row, Mapping) else None) or ""
            for row in dataset["audio"]
        ]
        self._dev_features = prepare_features(dataset, self.processor)

        # A mixed dev manifest is usually written one source after another, so
        # a contiguous `--dev-indices` slice shorter than the file selects one
        # source and silently answers a narrower question than the run intended.
        # Print the composition rather than guess at a fix.
        if any(self._dev_sources):
            counts = Counter(source or "unlabeled" for source in self._dev_sources)
            print(f"[dev] {len(self._dev_refs)} rows from {self.dev_corpus}: "
                  + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())),
                  flush=True)

    def _baseline_cache_path(self) -> Path:
        lo, hi = self.dev_indices
        slug = _slug(f"{self.base_model}__{self.dev_corpus}__{self.dev_split}__{lo}-{hi}")
        return self.work_dir / "baseline" / f"{slug}.json"

    def _ensure_baseline(self) -> None:
        if self._baseline_report is not None:
            return
        self._ensure_dev()
        cache_path = self._baseline_cache_path()
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            # A baseline cached before `wer_bounded` existed cannot be compared
            # against a bounded post-fine-tune number -- the difference would
            # be two different metrics subtracted, and it would read as a large
            # spurious gain. Treat it as a miss and recompute.
            if "wer_bounded" in cached:
                self._baseline_report = cached
                return
            print(f"[dev] baseline cache {cache_path.name} predates bounded WER; "
                  "recomputing", flush=True)

        model = WhisperForConditionalGeneration.from_pretrained(self.base_model)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.to(self.device).eval()
        hypotheses = self.transcribe_dev(model)
        report = self.dev_report(hypotheses, self.base_model)
        # the zero-shot rows sit beside the cached report: a paired comparison
        # of a trial against the baseline needs both sides per utterance
        self.write_dev_rows(cache_path.with_name(f"{cache_path.stem}_rows.jsonl"),
                            hypotheses)
        del model
        self._release_device_memory()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        self._baseline_report = report

    @property
    def baseline_report(self) -> dict:
        self._ensure_baseline()
        return self._baseline_report

    def transcribe_dev(self, model) -> list[str]:
        """Greedy hypotheses for `model` over the fixed dev set."""
        self._ensure_dev()
        return transcribe(model, self.processor, self._dev_features, self.device)

    def dev_report(self, hypotheses: list[str], model_name: str | None = None) -> dict:
        """Tier 3 report over already-decoded `hypotheses`.

        Split out from `evaluate_model` so a caller that also wants the
        per-utterance rows decodes once and scores once.  `by_source` is added
        when the dev manifest declares a `source` column: a mixed dev set
        (local rows beside public ones) can move in aggregate because one half
        moved and the other did not, and the aggregate alone cannot say which.
        """
        self._ensure_dev()
        dataset_name = (f"{self.dev_corpus}:{self.dev_split}"
                        f"[{self.dev_indices[0]}:{self.dev_indices[1]}]")
        report = build_report(self._dev_refs, hypotheses, self._dev_categories,
                              model=model_name or self.base_model,
                              dataset=dataset_name)
        # `wer` stays the standard unbounded number; `wer_bounded` is what the
        # reward is scored on, so both are on the record for every trial.
        report["wer_bounded"] = self._bounded_wer(hypotheses)
        by_source = self._by_source(hypotheses)
        if by_source:
            report["by_source"] = by_source
        return report

    def evaluate_model(self, model) -> dict:
        """Tier 3 report for `model` on the fixed dev set."""
        return self.dev_report(self.transcribe_dev(model),
                               getattr(model, "name_or_path", self.base_model))

    def _row_counts(self, hypotheses: list[str]) -> list[tuple[int, int]]:
        """Per-utterance (errors, reference words), ATC-normalized.

        `wer_counts` applies `normalize_atc` to reference and hypothesis
        through the same call, which is the same normalization
        `training.evaluate.build_report` uses for the aggregate -- so these
        rows sum to that aggregate rather than to a second opinion of it.

        Noise-only rows (empty reference) count as (0, 0) for the same reason
        `_measures` drops them: every word of the hypothesis would be an
        insertion against nothing, so including them would inflate the
        numerator over a denominator they contribute nothing to. What the
        model said there is scored separately, as the hallucination rate.
        """
        return [wer_counts(reference, hypothesis) if reference.strip() else (0, 0)
                for reference, hypothesis in zip(self._dev_refs, hypotheses)]

    def _bounded_wer(self, hypotheses: list[str]) -> dict:
        """Corpus WER with each row's errors capped at its reference length.

        An unbounded per-row WER has no upper limit: insertions are counted
        against a denominator the row does not grow, so a single utterance can
        contribute arbitrarily many errors.  whisper-tiny's repetition failure
        mode does exactly that -- it emits the same phrase until it hits
        `max_new_tokens`, and one 17-word row in the E0 run produced 96 errors,
        outweighing the entire degraded-channel manipulation it was supposed to
        be measuring.

        Capping at 1.0 WER per row makes a looping row count as "this row was
        completely wrong", which is all it can honestly mean, and no more.  The
        alternative -- dropping the row -- discards a real failure and changes
        the denominator between arms; this keeps every row and bounds its
        influence.

        This is the reward's aggregate only.  `write_dev_rows` keeps the raw
        uncapped counts so a bootstrap can still see the loops, and
        `report["wer"]` remains the standard unbounded number that every other
        evaluation in the repo reports.
        """
        counts = self._row_counts(hypotheses)
        capped = [(min(errors, words), words) for errors, words in counts]
        errors = sum(row_errors for row_errors, _ in capped)
        words = sum(row_words for _, row_words in capped)
        return {
            "atc_normalized": (errors / words) if words else None,
            "errors": errors,
            "reference_words": words,
            "n_capped_rows": sum(1 for e, w in counts if e > w),
            "discarded_errors": sum(e - w for e, w in counts if e > w),
        }

    def _by_source(self, hypotheses: list[str]) -> dict[str, dict]:
        if not any(self._dev_sources):
            return {}
        counts = self._row_counts(hypotheses)
        totals: dict[str, list[int]] = {}
        for source, (errors, words) in zip(self._dev_sources, counts):
            bucket = totals.setdefault(source or "unlabeled", [0, 0, 0, 0])
            bucket[0] += errors
            bucket[1] += words
            bucket[2] += 1
            bucket[3] += min(errors, words)
        # both numbers per source: `wer` reconciles with `report["wer"]`,
        # `wer_bounded` with the aggregate the reward is actually scored on
        return {
            name: {"samples": rows, "ref_words": words,
                   "wer": (errors / words) if words else None,
                   "wer_bounded": (bounded / words) if words else None}
            for name, (errors, words, rows, bounded) in sorted(totals.items())
        }

    def write_dev_rows(self, path: str | Path, hypotheses: list[str]) -> Path:
        """Dump one JSON object per dev utterance for offline analysis.

        Carries the error and reference-word *counts*, not just the ratio: a
        paired bootstrap over utterances (or clustered by capture day, which
        the audio path encodes) has to resample the counts and re-divide, and
        a per-row WER alone cannot be re-aggregated correctly.
        """
        self._ensure_dev()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "index": index,
                "audio": self._dev_paths[index],
                "source": self._dev_sources[index],
                "reference": reference,
                "hypothesis": hypothesis,
                "errors": errors,
                "ref_words": words,
                "wer": (errors / words) if words else None,
                # raw and uncapped on purpose (the reward's aggregate bounds
                # them, a bootstrap over these rows should not): `capped` marks
                # the rows the reward clipped, which is how you find the loops
                "capped": errors > words,
                # empty reference: excluded from WER, scored here instead
                "hallucinated": (None if reference.strip()
                                 else bool(normalize_atc(hypothesis).strip())),
            }
            for index, (reference, hypothesis, (errors, words)) in enumerate(
                zip(self._dev_refs, hypotheses, self._row_counts(hypotheses)))
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8")
        return path

    def _release_device_memory(self) -> None:
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- the reward call ------------------------------------------------------

    def __call__(self, config: Mapping[str, Any], trial_dir: str) -> RewardResult:
        self._ensure_baseline()
        baseline_wer = self._baseline_report["wer_bounded"]["atc_normalized"]

        trial = Path(trial_dir)
        start = time.monotonic()
        model = render_and_finetune(
            config, trial, base_model=self.base_model, pool_path=self.pool_path,
            n_synth=self.n_synth, ft_steps=self.ft_steps, ft_batch=self.ft_batch,
            ft_lr=self.ft_lr, ft_seed=self.ft_seed, gen_seed=self.gen_seed,
            device=self.device, processor=self.processor)
        ft_seconds = time.monotonic() - start
        synth_dir = trial / "synth"

        start = time.monotonic()
        hypotheses = self.transcribe_dev(model)
        report = self.dev_report(hypotheses,
                                 getattr(model, "name_or_path", self.base_model))
        self.write_dev_rows(trial / "dev_rows.jsonl", hypotheses)
        eval_seconds = time.monotonic() - start

        post_wer = report["wer_bounded"]["atc_normalized"]
        loss_curve_tail = list(getattr(model, "_ft_losses", []))[-10:]

        del model
        self._release_device_memory()

        if not self.keep_audio:
            shutil.rmtree(synth_dir / "wavs", ignore_errors=True)

        return RewardResult(
            reward=baseline_wer - post_wer,
            wer_after=post_wer,
            wer_baseline=baseline_wer,
            hallucination_rate=report["hallucination"]["rate"],
            proxy=False,
            metrics={
                "raw_wer_after": report["wer"]["raw"],
                # the unbounded aggregate the reward is *not* scored on, kept
                # alongside so a loop-driven divergence is visible per trial
                "unbounded_wer_after": report["wer"]["atc_normalized"],
                "n_capped_rows": report["wer_bounded"]["n_capped_rows"],
                "by_source": report.get("by_source"),
                "n_synth": self.n_synth,
                "n_dev": len(self._dev_refs),
                "ft_seconds": round(ft_seconds, 3),
                "eval_seconds": round(eval_seconds, 3),
                "loss_curve_tail": loss_curve_tail,
                "report": report,
            },
        )
