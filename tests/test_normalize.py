import pytest

from training.normalize import FOLD, normalize_atc


def test_folds_radio_phonetics():
    assert normalize_atc("niner tree fife") == "nine three five"


@pytest.mark.parametrize("spelling", ["xray", "x-ray", "X-Ray", "X ray"])
def test_every_spelling_of_xray_lands_on_one_form(spelling):
    """All four spellings occur in real transcripts and must score as equal.

    `x-ray` appears 11 times in the 200-row KIXD RL dev slice, so a fold that
    makes the hyphenated and unhyphenated spellings disagree is a measurable
    WER penalty, not a curiosity.
    """
    assert normalize_atc(spelling) == "x ray"


@pytest.mark.parametrize("value", sorted(set(FOLD.values())))
def test_fold_targets_are_fixed_points_of_the_normalizer(value):
    """A fold value the normalizer would itself rewrite can never match.

    Folding runs after the hyphen and punctuation passes, so a value carrying
    punctuation is emitted verbatim and compares unequal to the normalization
    of the very spelling it was meant to fold onto.
    """
    assert normalize_atc(value) == value


def test_normalization_is_idempotent():
    for text in ["Cleared to Land, Runway 27L.", "niner tree fife", "xray 118.3",
                 "N123AB hold short RWY 36", ""]:
        assert normalize_atc(normalize_atc(text)) == normalize_atc(text)


def test_reference_and_hypothesis_conventions_meet():
    """Spelled-out references and digit-writing hypotheses must agree.

    The KIXD references are fully spelled out (0 of 200 rows contain a digit)
    while whisper writes digits in 9 of 20 sampled hypotheses; folding the two
    conventions together is the normalizer's whole job.
    """
    assert normalize_atc("1-1-3 kilo romeo") == normalize_atc(
        "one one three kilo romeo")
    assert normalize_atc("runway 18") == normalize_atc("runway one eight")


def test_expands_digits():
    assert normalize_atc("fly heading 067") == "fly heading zero six seven"
    assert normalize_atc("runway 3-5 right") == "runway three five right"


def test_case_and_punct():
    assert normalize_atc("Cleared to Land, Runway 27L.") == "cleared to land runway two seven l"
