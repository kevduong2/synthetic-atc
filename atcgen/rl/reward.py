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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from training.evaluate import build_report, pick_device

from ..config import load_config
from ..dataset.build import build_dataset, load_manifest
from ..dataset.real_atc import load_real_atc
from ..text.sources import JsonlTextSource, make_text_source
from .finetune_lite import finetune, prepare_features, transcribe
from .types import RewardResult

GEN_SEED = 20260824  # fixed generator seed forced onto every candidate config


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
                dev_corpus: str = "jacktol/atc-dataset", dev_split: str = "train",
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
        self._dev_features = prepare_features(dataset, self.processor)

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
            self._baseline_report = json.loads(cache_path.read_text())
            return

        model = WhisperForConditionalGeneration.from_pretrained(self.base_model)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.to(self.device).eval()
        report = self.evaluate_model(model)
        del model
        self._release_device_memory()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        self._baseline_report = report

    @property
    def baseline_report(self) -> dict:
        self._ensure_baseline()
        return self._baseline_report

    def evaluate_model(self, model) -> dict:
        """Tier 3 report for `model` on the fixed dev set."""
        self._ensure_dev()
        hypotheses = transcribe(model, self.processor, self._dev_features, self.device)
        dataset_name = f"{self.dev_corpus}:{self.dev_split}[{self.dev_indices[0]}:{self.dev_indices[1]}]"
        return build_report(self._dev_refs, hypotheses, self._dev_categories,
                            model=getattr(model, "name_or_path", self.base_model),
                            dataset=dataset_name)

    def _release_device_memory(self) -> None:
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- the reward call ------------------------------------------------------

    def __call__(self, config: Mapping[str, Any], trial_dir: str) -> RewardResult:
        self._ensure_baseline()
        baseline_wer = self._baseline_report["wer"]["atc_normalized"]

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
        report = self.evaluate_model(model)
        eval_seconds = time.monotonic() - start

        post_wer = report["wer"]["atc_normalized"]
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
                "n_synth": self.n_synth,
                "n_dev": len(self._dev_refs),
                "ft_seconds": round(ft_seconds, 3),
                "eval_seconds": round(eval_seconds, 3),
                "loss_curve_tail": loss_curve_tail,
                "report": report,
            },
        )
