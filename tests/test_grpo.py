"""Tests for the GRPO post-training stage.

Reward components, advantage normalization and the KL term are pure functions
and are tested directly. The end-to-end smoke runs the real `run_grpo` loop for
two steps on four synthetic tone/noise clips with the cached whisper-tiny.en
checkpoint (nothing is downloaded): it asserts the mechanics -- finite loss,
parameters actually move, metrics rows carry every reward component -- not
that WER improves.
"""

import json

import numpy as np
import pytest
import torch

from training.grpo import (
    DataSpec,
    GRPOConfig,
    RewardWeights,
    Utterance,
    UtterancePool,
    concat_pools,
    decoder_prompt_ids,
    ensure_decoder_prompt,
    group_advantages,
    length_deviation,
    normalized_wer,
    prefix_length,
    repetition_ratio,
    run_grpo,
    score_hypothesis,
    sequence_kl,
    token_mask,
)

MODEL_ID = "openai/whisper-tiny.en"
SR = 16000


# -- reward components -----------------------------------------------------

def test_wer_is_clipped():
    reference = "cleared to land runway two four"
    hypothesis = " ".join(["garbage"] * 60)
    assert normalized_wer(reference, hypothesis) > 2.0
    assert score_hypothesis(reference, hypothesis).wer == 2.0


def test_wer_zero_on_exact_match_after_normalization():
    # niner/tree fold to nine/three, so these are the same transcript.
    assert normalized_wer("descend to niner tree zero", "descend to 930") == 0.0


def test_empty_hypothesis_scores_full_wer():
    assert normalized_wer("radar contact", "") == 1.0


def test_repetition_catches_short_single_token_loop():
    assert repetition_ratio("the the the the") == pytest.approx(1.0)
    assert repetition_ratio("cleared to land runway two four") == 0.0


def test_repetition_catches_phrase_loop():
    looped = "cleared to land cleared to land cleared to land"
    assert repetition_ratio(looped) > 0.4


def test_length_penalty_has_a_tolerance_band():
    reference = "one two three four five six seven eight nine ten"
    # 20% short: inside the 0.3 band.
    assert length_deviation(reference, "one two three four five six seven eight") == 0.0
    # 60% long: 0.3 over the band.
    assert length_deviation(reference, reference + " eleven two three four five six") == \
        pytest.approx(0.3)


def test_hallucination_is_the_only_term_on_empty_references():
    weights = RewardWeights()
    silent = score_hypothesis("", "")
    assert silent.total == 0.0 and silent.hallucination == 0.0

    invented = score_hypothesis("", "the the the the united two three")
    assert invented.hallucination == 1.0
    assert invented.total == pytest.approx(-weights.w_hal)
    # WER/repetition/length are undefined against an empty target.
    assert (invented.wer, invented.repetition, invented.length) == (0.0, 0.0, 0.0)


def test_penalties_stack_into_the_total():
    weights = RewardWeights()
    reference = "united two three heavy contact tower"
    hypothesis = "the the the the the the the the the the the the"
    reward = score_hypothesis(reference, hypothesis, weights)
    expected = -(reward.wer + weights.w_rep * reward.repetition
                 + weights.w_len * reward.length)
    assert reward.total == pytest.approx(expected)
    assert reward.repetition > 0.9 and reward.length > 0.0


# -- advantages ------------------------------------------------------------

def test_group_advantages_are_normalized_per_group():
    rewards = np.array([[-1.0, -2.0, -3.0], [0.0, -10.0, -5.0]])
    advantages, keep = group_advantages(rewards)
    assert keep.all()
    assert advantages.mean(axis=1) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert advantages.std(axis=1) == pytest.approx([1.0, 1.0])


def test_zero_variance_groups_are_skipped():
    rewards = np.array([[-1.0, -1.0, -1.0], [-1.0, -2.0, -3.0]])
    advantages, keep = group_advantages(rewards)
    assert not keep[0].any() and keep[1].all()
    assert advantages[0].tolist() == [0.0, 0.0, 0.0]
    assert advantages[1].std() == pytest.approx(1.0)


# -- KL --------------------------------------------------------------------

def test_kl_is_zero_against_itself_and_positive_otherwise():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    logprobs = torch.log_softmax(logits, dim=-1)
    mask = torch.ones(2, 5)
    assert float(sequence_kl(logprobs, logprobs, mask)) == pytest.approx(0.0, abs=1e-6)

    other = torch.log_softmax(torch.randn(2, 5, 11), dim=-1)
    assert float(sequence_kl(logprobs, other, mask)) > 0.0


def test_kl_ignores_masked_positions():
    torch.manual_seed(0)
    policy = torch.log_softmax(torch.randn(1, 3, 7), dim=-1)
    reference = policy.clone()
    reference[0, 2] = torch.log_softmax(torch.randn(7), dim=-1)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert float(sequence_kl(policy, reference, mask)) == pytest.approx(0.0, abs=1e-6)


# -- sequence bookkeeping --------------------------------------------------

def test_token_mask_keeps_through_the_first_eos():
    targets = torch.tensor([[5, 6, 50256, 50256, 50256], [7, 8, 9, 10, 50256]])
    mask = token_mask(targets, eos_id=50256, pad_id=50256)
    assert mask[0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert mask[1].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_decoder_prompt_is_restored_when_generate_strips_it():
    # Whisper's short-form generate returns only the newly sampled tokens.
    stripped = torch.tensor([[100, 200, 50256], [300, 50256, 50256]])
    restored = ensure_decoder_prompt(stripped, [50257, 50362])
    assert restored.shape == (2, 5)
    assert restored[:, 0].tolist() == [50257, 50257]
    assert restored[:, 1].tolist() == [50362, 50362]
    assert restored[:, 2:].tolist() == stripped.tolist()


def test_decoder_prompt_is_not_duplicated():
    already = torch.tensor([[50257, 50362, 100, 50256]])
    assert torch.equal(ensure_decoder_prompt(already, [50257, 50362]), already)


def test_decoder_prompt_ids_follow_the_generation_config():
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    assert decoder_prompt_ids(model) == [50257, 50362]


def test_prefix_length_counts_shared_leading_special_tokens():
    sequences = torch.tensor([[50257, 50362, 100, 200, 50256],
                              [50257, 50362, 300, 50256, 50256]])
    assert prefix_length(sequences, {50257, 50362, 50256}, eos_id=50256) == 2
    # A content token that happens to be shared is not part of the prefix.
    shared_word = torch.tensor([[50257, 50362, 100, 200], [50257, 50362, 100, 300]])
    assert prefix_length(shared_word, {50257, 50362, 50256}, eos_id=50256) == 2


# -- pools -----------------------------------------------------------------

def _fake_dataset(texts):
    from datasets import Dataset

    rng = np.random.default_rng(0)
    return Dataset.from_list([
        {"audio": {"array": rng.standard_normal(SR).astype(np.float32).tolist(),
                   "sampling_rate": SR}, "text": text}
        for text in texts
    ])


def test_pool_indexing_and_concat():
    pool = concat_pools([UtterancePool([_fake_dataset(["a b", ""])]),
                         UtterancePool([_fake_dataset(["c"])])])
    assert len(pool) == 3
    assert [pool[i].text for i in range(3)] == ["a b", "", "c"]
    assert pool[0].audio.dtype == np.float32
    assert [pool.select([2, 0])[i].text for i in range(2)] == ["c", "a b"]


# -- scoring ---------------------------------------------------------------

def test_shared_encoder_scoring_matches_the_naive_expansion():
    """The G rollouts of a clip reuse one encoder pass; the expansion must line
    up with how `generate` orders `num_return_sequences` (repeat_interleave:
    all G rows of clip 0, then clip 1). Scoring against the naive
    one-encoder-pass-per-row path is the check that they agree -- a transposed
    expansion would silently pair every rollout with the wrong audio.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from training.grpo import score_sequences

    pool = _TonePool()
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.use_cache = False
    model.eval()

    group = 2
    features = processor([pool[i].audio for i in range(2)], sampling_rate=SR,
                         return_tensors="pt").input_features
    torch.manual_seed(0)
    with torch.no_grad():
        sequences = model.generate(features, do_sample=True, temperature=1.0,
                                   num_return_sequences=group, max_new_tokens=8)
        sequences = ensure_decoder_prompt(sequences, decoder_prompt_ids(model))
        prefix = 2
        shared, _, _ = score_sequences(model, features, sequences, group, prefix,
                                       model.config.eos_token_id,
                                       model.config.pad_token_id)
        # Naive: expand the audio itself, one encoder pass per rollout.
        logits = model(input_features=features.repeat_interleave(group, dim=0),
                       decoder_input_ids=sequences[:, :-1]).logits.float()
        logprobs = torch.log_softmax(logits[:, prefix - 1:], dim=-1)
        targets = sequences[:, prefix:]
        mask = token_mask(targets, model.config.eos_token_id, model.config.pad_token_id)
        token_logprob = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        naive = (token_logprob * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    assert torch.allclose(shared, naive, atol=1e-4)


# -- end-to-end smoke ------------------------------------------------------

class _TonePool:
    """Four short clips (three tones, one silence) with fabricated refs.

    The silent row has an empty reference, so the loop's noise-only /
    hallucination path is exercised alongside the WER path.
    """

    def __init__(self):
        time_axis = np.arange(SR) / SR
        self._items = [
            Utterance(np.sin(2 * np.pi * 440 * time_axis).astype(np.float32) * 0.2,
                      "united two three cleared to land"),
            Utterance(np.sin(2 * np.pi * 880 * time_axis).astype(np.float32) * 0.2,
                      "delta four five contact tower"),
            Utterance(np.sin(2 * np.pi * 220 * time_axis).astype(np.float32) * 0.2,
                      "speedbird niner descend flight level three five zero"),
            Utterance(np.zeros(SR, dtype=np.float32), ""),
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[int(index)]


def test_grpo_smoke_updates_the_policy(tmp_path):
    pool = _TonePool()
    cfg = GRPOConfig(
        init=MODEL_ID, out=str(tmp_path / "grpo"), train=DataSpec(), dev=DataSpec(),
        steps=2, batch=4, group=4, max_new_tokens=12, temperature=1.0,
        eval_every=2, dev_max_new_tokens=12, dev_batch=4, lr=1e-4, seed=0,
    )
    summary = run_grpo(cfg, train_pool=pool, dev_pool=pool)

    metrics = (tmp_path / "grpo" / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in metrics]
    assert len(rows) == 2
    for row in rows:
        assert np.isfinite(row["loss"]) and np.isfinite(row["kl"])
        assert row["kl"] >= -1e-6
        for component in ("wer_mean", "repetition_mean", "length_mean",
                          "hallucination_mean", "group_std_mean", "reward_mean"):
            assert component in row and np.isfinite(row[component])
    # KL to the frozen reference is exactly zero before the first update.
    assert rows[0]["kl"] == pytest.approx(0.0, abs=1e-5)
    # Sampled tokens must be probable under the scoring pass. Anything near
    # log(1/51864) = -10.9 means the teacher-forced scoring lost Whisper's
    # decoder prompt and is scoring every token one position out of step.
    assert rows[0]["logp_token_mean"] > -5.0
    assert rows[-1].get("dev_wer") is not None
    # Temperature sampling over four clips should leave at least one group with
    # a spread of rewards; without that there is nothing for GRPO to learn from.
    assert sum(row["rows_kept"] for row in rows) > 0

    assert summary["train_pool"]["size"] == 4
    assert summary["best"]["checkpoint"] is not None

    # The checkpoint must be loadable the way training/evaluate.py loads one:
    # model + processor straight from the directory.
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    checkpoint = tmp_path / "grpo" / "last"
    assert WhisperProcessor.from_pretrained(checkpoint) is not None
    trained = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    original = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    deltas = [
        float((a - b).detach().abs().max())
        for (_, a), (_, b) in zip(trained.named_parameters(), original.named_parameters())
    ]
    assert max(deltas) > 0.0
