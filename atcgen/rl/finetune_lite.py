"""Minimal, dependency-light Whisper fine-tuning for the RL reward harness.

`training/finetune_whisper.py` is the heavyweight regime runner (HF Trainer,
multi-epoch, curriculum phases). The reward harness needs something much
smaller: a fixed number of steps, deterministic given a seed, cheap enough to
run once per candidate config inside an outer optimization loop.

Whisper's feature extractor always returns a fixed-shape (n_mels, 3000)
log-mel spectrogram (audio is padded/truncated to 30s), so batches never need
padding on ``input_features`` -- only ``labels`` (token id sequences) vary in
length and need dynamic padding. That means the training step here needs no
`WhisperProcessor` at all: padding is done directly with -100 (the ignored
label id), and the redundant leading start-of-transcript column that
`WhisperTokenizer` bakes into every label sequence is stripped using
`model.config.decoder_start_token_id`, which every Whisper model carries.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

Feature = dict  # {"input_features": np.ndarray[80, 3000], "labels": list[int]}


def prepare_features(dataset, processor) -> list[Feature]:
    """Map an HF dataset (columns 'audio', 'text') to model-ready features."""
    features = []
    for example in dataset:
        audio = example["audio"]
        input_features = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(example["text"]).input_ids
        features.append({"input_features": input_features, "labels": labels})
    return features


def _collate(batch: list[Feature], decoder_start_token_id: int | None) -> dict[str, torch.Tensor]:
    """Stack fixed-shape input features; pad labels to the batch max with -100."""
    input_features = torch.as_tensor(
        np.stack([np.asarray(f["input_features"], dtype=np.float32) for f in batch])
    )
    max_len = max(len(f["labels"]) for f in batch)
    labels = np.full((len(batch), max_len), -100, dtype=np.int64)
    for row, f in enumerate(batch):
        ids = f["labels"]
        labels[row, :len(ids)] = ids
    labels_t = torch.as_tensor(labels)
    if (decoder_start_token_id is not None and labels_t.size(1) > 1
            and (labels_t[:, 0] == decoder_start_token_id).all()):
        labels_t = labels_t[:, 1:]
    return {"input_features": input_features, "labels": labels_t}


def finetune(model, features: list[Feature], *, steps: int, batch_size: int, lr: float,
            seed: int, device, warmup_frac: float = 0.1,
            on_step: Callable[[int, float], None] | None = None) -> object:
    """Train `model` for `steps` optimizer steps against `features`.

    Batches are drawn with a seeded `numpy.random.Generator`: a fresh
    permutation of `features` is shuffled in whenever the cursor runs past the
    end, so training cycles through epochs deterministically. The learning
    rate schedule is linear warmup over `warmup_frac * steps` steps, then held
    constant -- simple and sufficient for the handful of hundred steps this
    harness runs per candidate. Gradients are clipped to norm 1.0.

    The per-step loss curve is stashed as `model._ft_losses` (a plain list of
    floats) for callers that want it in a report; it is not part of the
    return contract.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device)
    model.train()
    model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, int(steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / warmup_steps)

    scheduler = LambdaLR(optimizer, lr_lambda)

    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    rng = np.random.default_rng(seed)
    n = len(features)
    order = rng.permutation(n)
    cursor = 0
    losses: list[float] = []
    t_start = time.monotonic()

    for step in range(steps):
        if cursor + batch_size > n:
            order = rng.permutation(n)
            cursor = 0
        indices = order[cursor:cursor + batch_size]
        cursor += batch_size

        batch = _collate([features[i] for i in indices], decoder_start_token_id)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if on_step is not None:
            on_step(step + 1, loss_value)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            rate = (step + 1) / max(time.monotonic() - t_start, 1e-6)
            recent = sum(losses[-100:]) / len(losses[-100:])
            print(f"[finetune] step {step + 1}/{steps} loss {recent:.3f} "
                  f"{rate:.2f} steps/s", flush=True)

    model._ft_losses = losses
    return model


def transcribe(model, processor, features_or_audio: list, device, batch_size: int = 16,
               max_new_tokens: int = 100) -> list[str]:
    """Batched greedy decode over pre-extracted `input_features`.

    `features_or_audio` accepts either the `Feature` dicts from
    `prepare_features` or raw (n_mels, 3000) arrays directly.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    arrays = [
        np.asarray(item["input_features"] if isinstance(item, dict) else item, dtype=np.float32)
        for item in features_or_audio
    ]
    model.to(device)
    model.eval()
    hypotheses: list[str] = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            chunk = arrays[start:start + batch_size]
            input_features = torch.as_tensor(np.stack(chunk)).to(device)
            ids = model.generate(input_features, max_new_tokens=max_new_tokens)
            hypotheses.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return hypotheses
