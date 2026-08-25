#!/usr/bin/env python
"""GRPO post-training for a Whisper ASR student (research-findings §4.6 / D9).

Stage 3 of the L2 recipe: SFT on real -> continued training on a synthetic mix
-> **GRPO on the ASR**. For each training clip the policy samples a *group* of
G hypotheses at temperature; each hypothesis is scored by a reward built from
ATC-normalized WER plus the three anti-degeneracy penalties the MLC-SLM 2026
system reports as mandatory from day one (hallucination, repetition, length
deviation) -- ASR GRPO without them drifts to degenerate outputs. Advantages
are group-relative (the "G" in GRPO: no value network, the group mean is the
baseline), and a KL term to the frozen SFT checkpoint keeps the policy from
walking away from the supervised solution.

Design notes worth knowing before editing:

- **Full fine-tuning only.** HF PEFT LoRA is structurally incompatible with
  Whisper's log-mel encoder (§4.6), so every parameter is trained. whisper-tiny
  is 39M params, so the frozen reference is just a second copy in memory.
- **Reward components are logged separately** to `metrics.jsonl` every step.
  A single scalar reward hides reward hacking; per-component curves make a
  policy that buys WER with length blowup immediately visible (D3 spirit).
- **The encoder runs once per clip, not once per rollout.** Whisper's encoder
  is 1500 positions of fixed cost and dominates a forward pass; sampling G
  hypotheses for the same audio means one encoder pass whose hidden states are
  `repeat_interleave`d across the group (gradients accumulate correctly
  through the expansion). Measured ~6x on the policy update at G=6.
- **`model.eval()` during both sampling and scoring.** Rollouts must be scored
  under the distribution that produced them; eval mode does not disable
  gradients, only dropout, so on-policy correctness holds regardless of the
  checkpoint's dropout setting.
- Checkpoints are written with `save_pretrained` + `WhisperProcessor`, i.e.
  loadable by `training/evaluate.py --model <dir>`.

CLI:
  uv run python training/grpo.py --init runs/sft/ckpt \
      --manifest data/train_v1 --real-split train --real-indices 3000:4000 \
      --dev-split train --dev-indices 9000:9400 \
      --steps 300 --batch 4 --group 6 --out runs/grpo_x
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jiwer
import numpy as np
import torch
from torch.optim import AdamW
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atcgen.dataset.build import load_manifest          # noqa: E402
from atcgen.dataset.real_atc import load_real_atc        # noqa: E402
from training.evaluate import pick_device                # noqa: E402
from training.normalize import normalize_atc             # noqa: E402

SR = 16000


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    """Reward shaping constants. All exposed as CLI flags."""

    w_rep: float = 0.5
    w_len: float = 0.3
    w_hal: float = 1.0
    wer_clip: float = 2.0
    len_tol: float = 0.3
    len_clip: float = 2.0
    rep_ngram: int = 4


@dataclass(frozen=True)
class Reward:
    """One hypothesis's reward and the raw (unweighted) component values."""

    total: float
    wer: float
    repetition: float
    length: float
    hallucination: float


def normalized_wer(reference: str, hypothesis: str) -> float:
    """ATC-normalized WER of one pair. Empty reference -> 0.0 (undefined)."""
    ref = normalize_atc(reference).strip()
    hyp = normalize_atc(hypothesis).strip()
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))


def repetition_ratio(text: str, n: int = 4) -> float:
    """Degeneracy score in [0, 1]: duplicated n-grams or a single-token loop.

    Two failure modes, one number. The n-gram term is the fraction of `n`-grams
    that are not the first occurrence of themselves, which catches phrase-level
    looping ("cleared to land cleared to land ..."). The loop term catches the
    short catastrophic case the n-gram term structurally cannot -- "the the the
    the" has exactly one 4-gram -- by measuring how far the most frequent token
    is past half the transcript. The larger of the two wins.
    """
    words = normalize_atc(text).split()
    if not words:
        return 0.0
    top = Counter(words).most_common(1)[0][1] / len(words)
    loop = max(0.0, (top - 0.5) / 0.5)
    if len(words) < n:
        return loop
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    duplicated = 1.0 - len(set(grams)) / len(grams)
    return max(duplicated, loop)


def length_deviation(reference: str, hypothesis: str, tol: float = 0.3,
                     clip: float = 2.0) -> float:
    """Relative word-count deviation beyond a `tol` tolerance band, clipped."""
    ref_words = len(normalize_atc(reference).split())
    hyp_words = len(normalize_atc(hypothesis).split())
    ratio = abs(hyp_words - ref_words) / max(ref_words, 1)
    return min(max(0.0, ratio - tol), clip)


def score_hypothesis(reference: str, hypothesis: str,
                     weights: RewardWeights = RewardWeights()) -> Reward:
    """Reward for one sampled hypothesis against its reference transcript.

    Noise-only rows (empty reference) exist precisely to train the
    hallucination penalty, and WER/length/repetition are undefined or
    meaningless against an empty target, so those rows score on the
    hallucination term alone.
    """
    ref_empty = not normalize_atc(reference).strip()
    hyp_empty = not normalize_atc(hypothesis).strip()

    if ref_empty:
        hallucination = 1.0 if not hyp_empty else 0.0
        return Reward(total=-weights.w_hal * hallucination, wer=0.0,
                      repetition=0.0, length=0.0, hallucination=hallucination)

    wer = min(normalized_wer(reference, hypothesis), weights.wer_clip)
    repetition = repetition_ratio(hypothesis, weights.rep_ngram)
    length = length_deviation(reference, hypothesis, weights.len_tol, weights.len_clip)
    total = -(wer + weights.w_rep * repetition + weights.w_len * length)
    return Reward(total=total, wer=wer, repetition=repetition, length=length,
                  hallucination=0.0)


def group_advantages(rewards: np.ndarray, min_std: float = 1e-6
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Group-relative advantages for a (n_groups, group_size) reward matrix.

    Returns `(advantages, keep)`. A group whose rollouts all scored the same
    carries no preference signal at all -- normalizing it would amplify
    floating-point noise into a full-magnitude gradient -- so it is zeroed and
    masked out of the update instead.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    keep = np.repeat(std >= min_std, rewards.shape[1], axis=1)
    advantages = np.where(keep, (rewards - mean) / np.maximum(std, min_std), 0.0)
    return advantages, keep


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class DataSpec:
    """Where a pool of (audio, reference) rows comes from.

    Synthetic manifests and a real-corpus index range compose freely: a GRPO
    pool is normally "the synthetic mix plus a real anchor slice", and the
    noise-only rows carried in synthetic manifests are what give the
    hallucination penalty anything to bite on.
    """

    manifests: list[str] = field(default_factory=list)
    real_corpus: str = "jacktol/atc-dataset"
    real_split: str | None = None
    real_indices: tuple[int, int] | None = None

    def describe(self) -> str:
        parts = list(self.manifests)
        if self.real_split and self.real_indices:
            lo, hi = self.real_indices
            parts.append(f"{self.real_corpus}:{self.real_split}[{lo}:{hi}]")
        return " + ".join(parts) or "<empty>"


@dataclass
class Utterance:
    audio: np.ndarray
    text: str


class UtterancePool:
    """Lazy (audio, text) view over one or more HF datasets.

    Rows are held as `(dataset, index)` pairs and decoded on access: a Whisper
    log-mel feature is ~1 MB, so materializing features for a few thousand
    clips would cost more RAM than the model. Indexing an HF `Audio` column
    decodes the single clip on demand, which keeps the pool essentially free.
    """

    def __init__(self, datasets: list, rows: list[tuple[int, int]] | None = None) -> None:
        self._datasets = list(datasets)
        self._rows = list(rows) if rows is not None else [
            (d, i) for d, ds in enumerate(self._datasets) for i in range(len(ds))
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Utterance:
        dataset_index, row = self._rows[int(index)]
        example = self._datasets[dataset_index][row]
        return Utterance(
            audio=np.asarray(example["audio"]["array"], dtype=np.float32),
            text=example.get("text") or "",
        )

    def select(self, indices) -> "UtterancePool":
        return UtterancePool(self._datasets, [self._rows[int(i)] for i in indices])

    @property
    def texts(self) -> list[str]:
        return [self[i].text for i in range(len(self))]


def concat_pools(pools: list[UtterancePool]) -> UtterancePool:
    """One pool over the rows of several, preserving their order."""
    datasets: list = []
    rows: list[tuple[int, int]] = []
    for pool in pools:
        offset = len(datasets)
        datasets.extend(pool._datasets)
        rows.extend((offset + dataset_index, row) for dataset_index, row in pool._rows)
    return UtterancePool(datasets, rows)


def manifest_path(path: str | Path) -> Path:
    """Accept either a built dataset directory or the manifest file itself."""
    path = Path(path)
    return path / "manifest.jsonl" if path.is_dir() else path


def load_pool(spec: DataSpec) -> UtterancePool:
    datasets = [load_manifest(manifest_path(path)) for path in spec.manifests]
    if spec.real_split and spec.real_indices:
        real = load_real_atc(split=spec.real_split, corpus=spec.real_corpus)
        lo, hi = spec.real_indices
        datasets.append(real.select(range(lo, min(hi, len(real)))))
    pool = UtterancePool(datasets)
    if not len(pool):
        raise ValueError(f"empty utterance pool for {spec.describe()}")
    return pool


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------

def load_policy(init: str, device: torch.device) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(init)
    model.config.use_cache = False
    return model.to(device)


def decoder_prompt_ids(model) -> list[int]:
    """The decoder prompt `generate` runs before the first sampled token."""
    generation = model.generation_config
    start = generation.decoder_start_token_id
    if start is None:
        start = model.config.decoder_start_token_id
    forced = generation.forced_decoder_ids or []
    return [start] + [token for _, token in sorted(forced, key=lambda pair: pair[0])]


def ensure_decoder_prompt(sequences: torch.Tensor, prompt: list[int]) -> torch.Tensor:
    """Put Whisper's forced decoder prompt back on generated sequences.

    Whisper's short-form `generate` returns *only the newly generated tokens* --
    the `<|startoftranscript|><|notimestamps|>` prompt it decoded from is
    stripped off the front. Teacher-forcing those returned ids directly feeds
    the decoder a transcript with no start-of-transcript token and every
    position shifted by one, which silently scores each sampled token under the
    distribution for its predecessor: log-probabilities collapse to roughly
    uniform-random (~-9 per token against a 51.9k vocab) and the policy
    gradient explodes. Restoring the prompt is what makes the scored
    distribution the one that actually did the sampling.
    """
    if sequences.numel() and int(sequences[0, 0]) == prompt[0]:
        return sequences
    prefix = torch.tensor(prompt, dtype=sequences.dtype, device=sequences.device)
    return torch.cat([prefix.expand(sequences.size(0), len(prompt)), sequences], dim=1)


def prefix_length(sequences: torch.Tensor, special_ids: set[int], eos_id: int) -> int:
    """Number of leading forced/special tokens that are not policy decisions.

    Whisper's decoder always opens with `<|startoftranscript|>` plus whatever
    task/language/timestamp tokens the generation config forces. Those carry no
    gradient signal (they were never sampled), and their layout differs between
    the English-only and multilingual checkpoints, so it is read off the
    batch -- leading special tokens shared by every row -- rather than
    reconstructed from generation-config internals. Run it on sequences that
    have already been through `ensure_decoder_prompt`.
    """
    row = sequences[0].tolist()
    count = 0
    for position, token in enumerate(row):
        if token == eos_id or token not in special_ids:
            break
        if not bool((sequences[:, position] == token).all()):
            break
        count += 1
    return max(count, 1)


def token_mask(targets: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    """1.0 for real target tokens up to and including the first EOS.

    Whisper pads with the EOS id, so "everything up to the first EOS" and
    "everything that is not padding" are the same statement; the EOS itself is
    kept because learning *when to stop* is part of the policy.
    """
    is_eos = (targets == eos_id).float()
    after_first_eos = is_eos.cumsum(dim=1) - is_eos
    mask = (after_first_eos == 0).float()
    if pad_id != eos_id:
        mask = mask * (targets != pad_id).float()
    return mask


def score_sequences(model, input_features: torch.Tensor, sequences: torch.Tensor,
                    group: int, prefix: int, eos_id: int, pad_id: int,
                    temperature: float = 1.0):
    """Teacher-forced scoring of sampled sequences under `model`.

    Returns `(seq_logprob, mask, logprobs)` where `seq_logprob` is the
    length-normalized sequence log-probability (normalizing keeps a long
    hypothesis from dominating the gradient purely by having more tokens) and
    `logprobs` is the full-vocabulary log-softmax at the scored positions, kept
    for the KL term.

    Logits are divided by the sampling `temperature` so the scored distribution
    is the one the rollouts were actually drawn from -- the update is only
    on-policy if the sampler and the scorer agree. One gap remains by design:
    `generate` also applies Whisper's suppressed-token processors, which strip
    a little probability mass off ~90 special tokens the policy would rarely
    emit anyway. Renormalizing for that is not worth a second logits pass.
    """
    encoder_states = model.model.encoder(input_features).last_hidden_state
    outputs = model(
        encoder_outputs=BaseModelOutput(
            last_hidden_state=encoder_states.repeat_interleave(group, dim=0)),
        decoder_input_ids=sequences[:, :-1],
    )
    logits = outputs.logits[:, prefix - 1:].float() / temperature
    targets = sequences[:, prefix:]
    mask = token_mask(targets, eos_id, pad_id)

    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprob = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    seq_logprob = (token_logprob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return seq_logprob, mask, logprobs


def sequence_kl(policy_logprobs: torch.Tensor, reference_logprobs: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """Mean token-level KL(policy || reference) over the scored positions."""
    per_token = (policy_logprobs.exp() * (policy_logprobs - reference_logprobs)).sum(-1)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------
# config + loop
# --------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    init: str = "openai/whisper-tiny.en"
    out: str = "runs/grpo"
    train: DataSpec = field(default_factory=DataSpec)
    dev: DataSpec = field(default_factory=DataSpec)

    steps: int = 300
    batch: int = 4
    group: int = 6
    temperature: float = 0.9
    max_new_tokens: int = 64

    lr: float = 1e-6
    beta: float = 0.04
    grad_clip: float = 1.0
    weights: RewardWeights = field(default_factory=RewardWeights)

    eval_every: int = 50
    dev_batch: int = 8
    dev_max_new_tokens: int = 100
    seed: int = 0
    device: str | None = None


def evaluate_dev(model, processor, pool: UtterancePool, device: torch.device,
                 batch_size: int, max_new_tokens: int) -> dict:
    """Greedy-decode `pool` and return ATC-normalized WER + hallucination rate."""
    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for start in range(0, len(pool), batch_size):
            items = [pool[i] for i in range(start, min(start + batch_size, len(pool)))]
            features = processor([u.audio for u in items], sampling_rate=SR,
                                 return_tensors="pt").input_features.to(device)
            ids = model.generate(features, max_new_tokens=max_new_tokens, do_sample=False)
            hypotheses.extend(processor.batch_decode(ids, skip_special_tokens=True))
            references.extend(u.text for u in items)

    speech = [(normalize_atc(r), normalize_atc(h))
              for r, h in zip(references, hypotheses) if normalize_atc(r).strip()]
    noise = [h for r, h in zip(references, hypotheses) if not normalize_atc(r).strip()]
    return {
        "wer": float(jiwer.wer([r for r, _ in speech], [h for _, h in speech]))
        if speech else None,
        "hallucination_rate": (sum(bool(normalize_atc(h).strip()) for h in noise) / len(noise))
        if noise else None,
        "samples": len(references),
    }


def save_checkpoint(model, processor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)


def run_grpo(cfg: GRPOConfig, *, train_pool: UtterancePool | None = None,
             dev_pool: UtterancePool | None = None) -> dict:
    """Run the GRPO stage. Returns the run summary (also written to run.json).

    `train_pool`/`dev_pool` override `cfg.train`/`cfg.dev`, which is how
    `training/recipe.py` hands GRPO the exact ratio-controlled mixture it used
    for SFT rather than a uniform draw over the union of the sources.
    """
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    processor = WhisperProcessor.from_pretrained(cfg.init)
    policy = load_policy(cfg.init, device)
    reference = load_policy(cfg.init, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    eos_id = policy.config.eos_token_id
    pad_id = policy.config.pad_token_id if policy.config.pad_token_id is not None else eos_id
    special_ids = set(processor.tokenizer.all_special_ids)
    prompt_ids = decoder_prompt_ids(policy)

    train_source = cfg.train.describe() if train_pool is None else "<caller-supplied>"
    dev_source = cfg.dev.describe() if dev_pool is None else "<caller-supplied>"
    if train_pool is None:
        train_pool = load_pool(cfg.train)
    if dev_pool is None and (cfg.dev.manifests or cfg.dev.real_split):
        dev_pool = load_pool(cfg.dev)

    optimizer = AdamW(policy.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(train_pool))
    cursor = 0

    metrics_path = out / "metrics.jsonl"
    metrics_file = metrics_path.open("w")
    best_wer, best_step = None, None
    started = time.monotonic()

    # `eval()` for the whole loop: rollouts must be scored under exactly the
    # distribution that produced them, and eval mode disables dropout without
    # touching autograd.
    policy.eval()

    for step in range(1, cfg.steps + 1):
        step_started = time.monotonic()
        if cursor + cfg.batch > len(train_pool):
            order = rng.permutation(len(train_pool))
            cursor = 0
        indices = order[cursor:cursor + cfg.batch]
        cursor += cfg.batch
        items = [train_pool[i] for i in indices]

        input_features = processor([u.audio for u in items], sampling_rate=SR,
                                   return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            sequences = policy.generate(
                input_features, do_sample=True, temperature=cfg.temperature,
                num_return_sequences=cfg.group, max_new_tokens=cfg.max_new_tokens,
                use_cache=True,
            )
        sequences = ensure_decoder_prompt(sequences, prompt_ids)
        hypotheses = processor.batch_decode(sequences, skip_special_tokens=True)

        rewards = [
            score_hypothesis(items[row // cfg.group].text, hypothesis, cfg.weights)
            for row, hypothesis in enumerate(hypotheses)
        ]
        reward_matrix = np.array([r.total for r in rewards]).reshape(len(items), cfg.group)
        advantages, keep = group_advantages(reward_matrix)

        prefix = prefix_length(sequences, special_ids, eos_id)
        seq_logprob, mask, policy_logprobs = score_sequences(
            policy, input_features, sequences, cfg.group, prefix, eos_id, pad_id,
            cfg.temperature)
        with torch.no_grad():
            _, _, reference_logprobs = score_sequences(
                reference, input_features, sequences, cfg.group, prefix, eos_id,
                pad_id, cfg.temperature)

        keep_rows = torch.as_tensor(keep.reshape(-1), dtype=torch.bool, device=device)
        keep_rows = keep_rows & (mask.sum(dim=1) > 0)
        advantage_tensor = torch.as_tensor(
            advantages.reshape(-1), dtype=torch.float32, device=device)

        kl = sequence_kl(policy_logprobs, reference_logprobs, mask)
        n_kept = int(keep_rows.sum())
        if n_kept:
            policy_loss = -(advantage_tensor * seq_logprob * keep_rows.float()).sum() / n_kept
        else:
            policy_loss = seq_logprob.sum() * 0.0
        loss = policy_loss + cfg.beta * kl

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        group_std = reward_matrix.std(axis=1)
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "kl": float(kl.detach()),
            "grad_norm": float(grad_norm),
            "reward_mean": float(reward_matrix.mean()),
            "reward_max": float(reward_matrix.max()),
            "wer_mean": float(np.mean([r.wer for r in rewards])),
            "repetition_mean": float(np.mean([r.repetition for r in rewards])),
            "length_mean": float(np.mean([r.length for r in rewards])),
            "hallucination_mean": float(np.mean([r.hallucination for r in rewards])),
            # Per-token log-probability of the model's own samples: the canary
            # for a decoder-prompt/teacher-forcing misalignment, which drives it
            # toward log(1/vocab) ~= -10.9 while everything else still looks sane.
            "logp_token_mean": float(seq_logprob.detach().mean()),
            "group_std_mean": float(group_std.mean()),
            "group_std_min": float(group_std.min()),
            "groups_skipped": int((group_std < 1e-6).sum()),
            "rows_kept": n_kept,
            "hyp_words_mean": float(np.mean([len(h.split()) for h in hypotheses])),
            "seconds": round(time.monotonic() - step_started, 3),
        }

        if dev_pool is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
            dev = evaluate_dev(policy, processor, dev_pool, device,
                               cfg.dev_batch, cfg.dev_max_new_tokens)
            row["dev_wer"] = dev["wer"]
            row["dev_hallucination_rate"] = dev["hallucination_rate"]
            if dev["wer"] is not None and (best_wer is None or dev["wer"] < best_wer):
                best_wer, best_step = dev["wer"], step
                save_checkpoint(policy, processor, out / "best")

        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    metrics_file.close()
    save_checkpoint(policy, processor, out / "last")

    summary = {
        "config": _config_json(cfg),
        "device": str(device),
        "train_pool": {"size": len(train_pool), "source": train_source},
        "dev_pool": ({"size": len(dev_pool), "source": dev_source}
                     if dev_pool is not None else None),
        "best": {"dev_wer": best_wer, "step": best_step,
                 "checkpoint": str(out / "best") if best_step else None},
        "last_checkpoint": str(out / "last"),
        "wall_seconds": round(time.monotonic() - started, 2),
        "metrics": str(metrics_path),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _config_json(cfg: GRPOConfig) -> dict:
    payload = asdict(cfg)
    for key in ("train", "dev"):
        indices = payload[key].get("real_indices")
        payload[key]["real_indices"] = list(indices) if indices else None
    return payload


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_indices(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    lo, _, hi = text.partition(":")
    return int(lo), int(hi)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="openai/whisper-tiny.en",
                    help="SFT checkpoint directory or model id to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", action="append", default=[],
                    help="synthetic dataset dir or manifest.jsonl; repeatable")
    ap.add_argument("--real-corpus", default="jacktol/atc-dataset")
    ap.add_argument("--real-split", default=None)
    ap.add_argument("--real-indices", default=None, help="LO:HI slice of --real-split")
    ap.add_argument("--dev-manifest", action="append", default=[])
    ap.add_argument("--dev-split", default=None)
    ap.add_argument("--dev-indices", default=None)

    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4, help="audio clips per update")
    ap.add_argument("--group", type=int, default=6, help="sampled hypotheses per clip")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=64)

    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL-to-reference weight")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--w-rep", type=float, default=0.5)
    ap.add_argument("--w-len", type=float, default=0.3)
    ap.add_argument("--w-hal", type=float, default=1.0)
    ap.add_argument("--wer-clip", type=float, default=2.0)
    ap.add_argument("--len-tol", type=float, default=0.3)

    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--dev-batch", type=int, default=8)
    ap.add_argument("--dev-max-new-tokens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    return ap


def config_from_args(args) -> GRPOConfig:
    return GRPOConfig(
        init=args.init, out=args.out,
        train=DataSpec(manifests=args.manifest, real_corpus=args.real_corpus,
                       real_split=args.real_split,
                       real_indices=parse_indices(args.real_indices)),
        dev=DataSpec(manifests=args.dev_manifest, real_corpus=args.real_corpus,
                     real_split=args.dev_split,
                     real_indices=parse_indices(args.dev_indices)),
        steps=args.steps, batch=args.batch, group=args.group,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        lr=args.lr, beta=args.beta, grad_clip=args.grad_clip,
        weights=RewardWeights(w_rep=args.w_rep, w_len=args.w_len, w_hal=args.w_hal,
                              wer_clip=args.wer_clip, len_tol=args.len_tol),
        eval_every=args.eval_every, dev_batch=args.dev_batch,
        dev_max_new_tokens=args.dev_max_new_tokens,
        seed=args.seed, device=args.device,
    )


def main() -> None:
    args = build_parser().parse_args()
    if not args.manifest and not args.real_split:
        build_parser().error("need --manifest and/or --real-split/--real-indices")
    summary = run_grpo(config_from_args(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
