"""Paired significance testing for WER comparisons.

The RL loop compares a candidate config's post-fine-tune WER against a
baseline on the same fixed dev set, so a paired bootstrap over utterances is
the right test: it resamples which utterances count, not which system is
"correct", and needs no independence assumption between A and B beyond
utterance-level exchangeability.
"""

from __future__ import annotations

import numpy as np

from training.normalize import normalize_atc

try:
    import jiwer
except ImportError as exc:  # pragma: no cover - jiwer is a base dependency
    raise ImportError("atcgen.rl.stats requires jiwer") from exc


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """Per-utterance (errors, reference words), ATC-normalized, via jiwer."""
    output = jiwer.process_words([normalize_atc(ref)], [normalize_atc(hyp)])
    errors = output.substitutions + output.deletions + output.insertions
    words = output.hits + output.substitutions + output.deletions
    return errors, words


def paired_bootstrap(refs: list[str], hyps_a: list[str], hyps_b: list[str], *,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the corpus WER_a - WER_b gap over utterance resamples.

    Returns ``{"delta", "ci_low", "ci_high", "p_value"}``: the observed gap,
    a 95% percentile CI over `n_boot` resamples, and a two-sided p-value via
    the sign-flip convention ``2 * min(P(delta* <= 0), P(delta* >= 0))``.
    """
    if not (len(refs) == len(hyps_a) == len(hyps_b)):
        raise ValueError("refs, hyps_a, hyps_b must have the same length")
    if not refs:
        raise ValueError("paired_bootstrap needs at least one utterance")

    counts_a = np.array([wer_counts(r, h) for r, h in zip(refs, hyps_a)], dtype=np.int64)
    counts_b = np.array([wer_counts(r, h) for r, h in zip(refs, hyps_b)], dtype=np.int64)
    errors_a, words_a = counts_a[:, 0], counts_a[:, 1]
    errors_b, words_b = counts_b[:, 0], counts_b[:, 1]

    def corpus_wer(errors: np.ndarray, words: np.ndarray) -> np.ndarray:
        denom = np.maximum(words, 1)
        return errors / denom

    observed = float(errors_a.sum() / max(words_a.sum(), 1)
                     - errors_b.sum() / max(words_b.sum(), 1))

    rng = np.random.default_rng(seed)
    n = len(refs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_wer_a = corpus_wer(errors_a[idx].sum(axis=1), words_a[idx].sum(axis=1))
    boot_wer_b = corpus_wer(errors_b[idx].sum(axis=1), words_b[idx].sum(axis=1))
    deltas = boot_wer_a - boot_wer_b

    ci_low, ci_high = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    p_low = float(np.mean(deltas <= 0))
    p_high = float(np.mean(deltas >= 0))
    p_value = min(2.0 * min(p_low, p_high), 1.0)

    return {"delta": observed, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value}
