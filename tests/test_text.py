import json
import random

import pytest

from atcgen.text import grammar, lexicon
from atcgen.text.sources import GrammarTextSource, JsonlTextSource, make_text_source

ALLOWED_WORDS = None  # built lazily


def _vocab():
    global ALLOWED_WORDS
    if ALLOWED_WORDS is None:
        words = set(lexicon.DIGITS_SPOKEN.values()) | set(lexicon.PHONETIC_ALPHABET.values())
        ALLOWED_WORDS = words
    return ALLOWED_WORDS


def test_spell_digits():
    assert lexicon.spell_digits("359") == "tree fife niner"
    assert lexicon.spell_digits("120") == "one two zero"


def test_spell_alnum():
    assert lexicon.spell_alnum("23AB") == "two tree alpha bravo"


def test_group_number():
    assert lexicon.group_number(412) == "four twelve"
    assert lexicon.group_number(1850) == "eighteen fifty"
    assert lexicon.group_number(20) == "twenty"
    assert lexicon.group_number(7) == "seven"
    assert lexicon.group_number(305) == "tree zero fife"
    assert lexicon.group_number(1013) == "ten thirteen"


def test_frequency_in_airband():
    rng = random.Random(1)
    for _ in range(50):
        freq = lexicon.random_frequency(rng)
        assert " point " in freq


def test_exchange_structure():
    rng = random.Random(42)
    for _ in range(200):
        exchange = grammar.generate_exchange(rng)
        assert len(exchange) >= 2
        roles = {u.role for u in exchange}
        assert roles <= {"controller", "pilot"}
        for u in exchange:
            assert u.spoken == u.transcript
            assert u.spoken.strip() == u.spoken
            assert "  " not in u.spoken
            assert u.spoken  # non-empty


def test_utterances_are_lowercase_words():
    rng = random.Random(7)
    for _ in range(200):
        u = grammar.generate_utterance(rng)
        for ch in u.spoken:
            assert ch.islower() or ch in " ,'", f"bad char {ch!r} in {u.spoken!r}"


def test_deterministic_with_seed():
    a = [grammar.generate_utterance(random.Random(5)).spoken for _ in range(3)]
    b = [grammar.generate_utterance(random.Random(5)).spoken for _ in range(3)]
    assert a == b


def test_grammar_text_source():
    src = GrammarTextSource()
    u = src.sample(random.Random(0))
    assert u.spoken


def test_jsonl_text_source(tmp_path):
    p = tmp_path / "ext.jsonl"
    lines = [
        {"spoken": "delta one two, cleared to land", "transcript": "delta one two, cleared to land"},
        {"text": "united five, contact ground one two one point niner"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))
    src = make_text_source(str(p))
    assert isinstance(src, JsonlTextSource)
    rng = random.Random(3)
    samples = {src.sample(rng).spoken for _ in range(20)}
    assert len(samples) == 2


def test_jsonl_text_source_rejects_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError):
        JsonlTextSource(p)
