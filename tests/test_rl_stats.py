"""Fast tests for the paired-bootstrap WER significance testing utilities.

No models, no audio, no network -- pure string/numpy arithmetic.
"""

from atcgen.rl.stats import paired_bootstrap, wer_counts


def test_wer_counts_exact_match_has_no_errors():
    errors, words = wer_counts("cleared to land", "cleared to land")
    assert (errors, words) == (0, 3)


def test_wer_counts_deletion():
    errors, words = wer_counts("cleared to land", "cleared land")
    assert (errors, words) == (1, 3)


def test_wer_counts_insertion():
    errors, words = wer_counts("cleared to land", "cleared to land now")
    assert (errors, words) == (1, 3)


def test_wer_counts_substitution():
    errors, words = wer_counts("cleared to land", "cleared to taxi")
    assert (errors, words) == (1, 3)


def test_wer_counts_normalizes_atc_phonetics():
    # "niner tree fife" folds to "nine three five" -- should match exactly.
    errors, words = wer_counts("niner tree fife", "nine three five")
    assert (errors, words) == (0, 3)


def test_wer_counts_empty_reference_counts_hallucinated_words_as_errors():
    errors, words = wer_counts("", "roger")
    assert (errors, words) == (1, 0)


def test_paired_bootstrap_detects_a_strictly_better():
    refs = [f"word{i} word{i}" for i in range(60)]
    hyps_a = list(refs)  # perfect transcription
    # half wrong, half perfect -> some variance across utterances
    hyps_b = [
        f"wrong{i} word{i}" if i % 2 == 0 else word
        for i, word in enumerate(refs)
    ]

    result = paired_bootstrap(refs, hyps_a, hyps_b, n_boot=2000, seed=0)

    assert result["delta"] < 0  # A has lower WER than B
    assert result["ci_high"] <= 0
    assert result["p_value"] < 0.05


def test_paired_bootstrap_identical_hypotheses_gives_zero_delta():
    refs = [f"word{i} word{i}" for i in range(20)]
    hyps = [f"wrong{i} word{i}" for i in range(20)]

    result = paired_bootstrap(refs, hyps, hyps, n_boot=500, seed=1)

    assert result["delta"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    assert result["p_value"] == 1.0


def test_paired_bootstrap_ci_bounds_ordering():
    refs = [f"word{i} word{i}" for i in range(30)]
    hyps_a = list(refs)
    hyps_b = [f"wrong{i} word{i}" for i in range(30)]

    result = paired_bootstrap(refs, hyps_a, hyps_b, n_boot=1000, seed=2)

    assert result["ci_low"] <= result["delta"] <= result["ci_high"]
