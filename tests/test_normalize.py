from training.normalize import normalize_atc


def test_folds_radio_phonetics():
    assert normalize_atc("niner tree fife") == "nine three five"


def test_expands_digits():
    assert normalize_atc("fly heading 067") == "fly heading zero six seven"
    assert normalize_atc("runway 3-5 right") == "runway three five right"


def test_case_and_punct():
    assert normalize_atc("Cleared to Land, Runway 27L.") == "cleared to land runway two seven l"
