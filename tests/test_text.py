import json
import random

import pytest

from atcgen.entities import Entity, extract_entities
from atcgen.text import lexicon
from atcgen.text.grammar import (ScenarioConfig, Utterance, generate_exchange,
                                 generate_utterance, load_vocab,
                                 validate_utterance)
from atcgen.text.sources import GrammarTextSource, JsonlTextSource, make_text_source

#: Hermetic: the builtin lexicon only, so the tests do not depend on whether
#: scripts/harvest_vocab.py has been run on this machine.
BUILTIN = ScenarioConfig(vocab_path=None)


def config(**knobs) -> ScenarioConfig:
    return ScenarioConfig(vocab_path=None, **knobs)


def utterances(count, cfg=BUILTIN, seed=42):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        out.extend(generate_exchange(rng, cfg))
    return out[:count]


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def test_spell_digits():
    assert lexicon.spell_digits("359") == "tree fife niner"
    assert lexicon.spell_digits("120") == "one two zero"


def test_spell_alnum():
    assert lexicon.spell_alnum("23AB") == "two tree alpha bravo"


def test_group_number():
    assert lexicon.group_number(412) == "fower twelve"   # ICAO spelling of 4
    assert lexicon.group_number(1850) == "eighteen fifty"
    assert lexicon.group_number(20) == "twenty"
    assert lexicon.group_number(7) == "seven"
    assert lexicon.group_number(305) == "tree zero fife"
    assert lexicon.group_number(1013) == "ten thirteen"


def test_corrupt_value_stays_legal_and_differs():
    rng = random.Random(0)
    for type_, value in [("runway", "24L"), ("heading", "270"),
                         ("flight_level", "FL350"), ("altitude", "3500ft"),
                         ("frequency", "127.825"), ("squawk", "4521"),
                         ("speed", "250")]:
        for _ in range(20):
            wrong = lexicon.corrupt_value(type_, value, rng)
            assert wrong is not None and wrong != value
            assert (type_, wrong) in [(e.type, e.value) for e in extract_entities(
                {"runway": "runway ", "heading": "heading ",
                 "flight_level": "flight level ", "altitude": "altitude ",
                 "frequency": "", "squawk": "squawk ", "speed": "speed "}[type_]
                + lexicon.make_slot(type_, wrong, lexicon.Style()).spoken)]


def test_abbreviated_callsigns_stay_consistent_with_the_full_one():
    rng = random.Random(3)
    for _ in range(50):
        short = lexicon.abbreviate_callsign("CSA926", rng)
        assert short is None or lexicon.callsigns_consistent("CSA926", short)
    assert lexicon.abbreviate_callsign("CSA3KF", rng) is None
    assert not lexicon.callsigns_consistent("CSA926", "DLH926")
    assert not lexicon.callsigns_consistent("CSA926", "CSA925")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_exchange_structure():
    rng = random.Random(42)
    for _ in range(200):
        exchange = generate_exchange(rng, BUILTIN)
        assert len(exchange) >= 2
        assert {u.role for u in exchange} <= {"controller", "pilot"}
        for u in exchange:
            assert u.spoken.strip() == u.spoken and "  " not in u.spoken
            assert u.transcript and u.display


def test_transcript_is_the_spoken_words_without_punctuation():
    for u in utterances(200):
        assert "," not in u.transcript
        assert u.transcript == " ".join(u.transcript.split())
        assert u.transcript == u.spoken.replace(",", "").replace("  ", " ")


def test_transcripts_are_lowercase_words():
    for u in utterances(300):
        for char in u.transcript:
            assert char.islower() or char in " '", f"bad char in {u.transcript!r}"


def test_deterministic_with_seed():
    first = [u.transcript for u in utterances(20, seed=5)]
    second = [u.transcript for u in utterances(20, seed=5)]
    assert first == second


def test_display_form_uses_canonical_values():
    rng = random.Random(9)
    cfg = config(region="eu")
    seen = 0
    for _ in range(300):
        for u in generate_exchange(rng, cfg):
            for entity in u.entities:
                assert entity.value in u.display
                seen += 1
    assert seen > 100


# ---------------------------------------------------------------------------
# The property the pipeline rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("region", ["us", "eu", "mixed"])
def test_every_generated_utterance_validates(region):
    """500 seeded samples per region: legal values, labels that round-trip."""
    cfg = config(region=region, readback_error_prob=0.1,
                 confusable_callsign_prob=0.1)
    for u in utterances(500, cfg, seed=1234):
        assert validate_utterance(u) == [], u.transcript


def test_ground_truth_matches_a_fresh_parse():
    for u in utterances(300, config(region="eu")):
        parsed = {(e.type, e.value) for e in extract_entities(u.transcript)}
        assert {(e.type, e.value) for e in u.entities} <= parsed


def test_entities_are_labelled_at_all():
    labelled = sum(1 for u in utterances(200) if u.entities)
    assert labelled == 200


# ---------------------------------------------------------------------------
# Scenario knobs
# ---------------------------------------------------------------------------

def test_region_selects_phraseology():
    eu = " ".join(u.transcript for u in utterances(300, config(region="eu")))
    us = " ".join(u.transcript for u in utterances(300, config(region="us")))
    assert "flight level" in eu and "decimal" in eu
    assert "flight level" not in us
    assert "point" in us and "decimal" not in us


def test_phonetic_respelling_probability_controls_radio_variants():
    variants = ("niner", "tree", "fife", "fower")
    plain = " ".join(u.transcript for u in utterances(
        300, config(phonetic_respelling_prob=0.0)))
    spelled = " ".join(u.transcript for u in utterances(
        300, config(phonetic_respelling_prob=1.0)))
    assert not any(word in plain.split() for word in variants)
    assert any(word in spelled.split() for word in variants)


def test_readback_error_labels_what_was_actually_said():
    rng = random.Random(17)
    cfg = config(readback_error_prob=1.0)
    checked = 0
    for _ in range(80):
        exchange = generate_exchange(rng, cfg)
        wrong = [u for u in exchange if "readback_error" in u.meta]
        if not wrong:
            continue
        checked += 1
        error = wrong[0].meta["readback_error"]
        assert error["said"] != error["correct"]
        # the label follows the audio: the wrong value, not the intended one
        values = [e.value for e in wrong[0].entities if e.type == error["type"]]
        assert values == [error["said"]]
        assert validate_utterance(wrong[0]) == []
        # and the controller corrects it in the same exchange
        corrections = [u for u in exchange if u.kind == "correction"]
        assert error["correct"] in [
            e.value for u in corrections for e in u.entities]
    assert checked > 40


def test_no_readback_errors_when_the_knob_is_off():
    for u in utterances(400, config(readback_error_prob=0.0)):
        assert "readback_error" not in u.meta
        assert not u.kind.endswith("_error")


def test_confusable_callsigns_share_a_transmission():
    rng = random.Random(23)
    cfg = config(confusable_callsign_prob=1.0)
    found = 0
    for _ in range(60):
        for u in generate_exchange(rng, cfg):
            callsigns = [e.value for e in u.entities if e.type == "callsign"]
            if len(callsigns) < 2:
                continue
            found += 1
            first, second = callsigns[0], callsigns[1]
            assert first != second
            assert len(first) == len(second)
            differing = sum(a != b for a, b in zip(first, second))
            assert differing == 1, (first, second)
    assert found > 30


def test_invalid_scenario_knobs_are_rejected():
    with pytest.raises(ValueError):
        ScenarioConfig(region="antarctica")
    with pytest.raises(ValueError):
        ScenarioConfig(readback_error_prob=1.5)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def utterance(text, entities):
    return Utterance(spoken=text, transcript=text, role="controller",
                     kind="test", entities=entities)


@pytest.mark.parametrize("text,entity", [
    ("runway four one", Entity("runway", "41", "four one")),
    ("heading four zero zero", Entity("heading", "400", "four zero zero")),
    ("contact tower one five zero decimal zero zero zero",
     Entity("frequency", "150.000", "one five zero decimal zero zero zero")),
    ("squawk eight eight zero zero", Entity("squawk", "8800", "eight eight zero zero")),
    ("climb to flight level five zero zero",
     Entity("flight_level", "FL500", "five zero zero")),
])
def test_validator_rejects_illegal_values(text, entity):
    assert validate_utterance(utterance(text, [entity]))


def test_validator_rejects_a_label_the_text_does_not_say():
    problems = validate_utterance(utterance(
        "csa nine two six roger", [Entity("flight_level", "FL350", "three five zero")]))
    assert any("round-trip lost" in problem for problem in problems)


def test_validator_rejects_an_unlabelled_critical_value():
    problems = validate_utterance(utterance("cleared to land runway two four left", []))
    assert any("unlabelled runway" in problem for problem in problems)


def test_validator_accepts_a_clean_utterance():
    text = "csa nine two six climb to flight level three four zero"
    assert validate_utterance(utterance(text, [
        Entity("callsign", "CSA926", "csa nine two six"),
        Entity("flight_level", "FL340", "three four zero")])) == []


# ---------------------------------------------------------------------------
# Anchored vocabulary
# ---------------------------------------------------------------------------

def anchor_file(tmp_path):
    path = tmp_path / "real_anchor.json"
    path.write_text(json.dumps({
        "meta": {"split": "train[0:8000]"},
        "airlines": {"jobair": {"count": 50, "icao": "JOB"},
                     "nor shuttle": {"count": 30, "icao": "NOS"}},
        "stations": {"praha radar": 502, "ruzyne tower": 253},
        "waypoints": {"rapet": 41, "pepik": 17},
    }))
    return path


def test_grammar_falls_back_to_builtin_vocabulary(tmp_path):
    vocab = load_vocab(tmp_path / "absent.json")
    assert not vocab.anchored
    assert vocab.airlines["lufthansa"] == "DLH"
    for u in utterances(100, ScenarioConfig(vocab_path=tmp_path / "absent.json")):
        assert validate_utterance(u) == []


def test_grammar_uses_the_harvested_vocabulary(tmp_path):
    path = anchor_file(tmp_path)
    vocab = load_vocab(path)
    assert vocab.anchored and vocab.airlines["nor shuttle"] == "NOS"
    cfg = ScenarioConfig(region="eu", vocab_path=path)
    text = " ".join(u.transcript for u in utterances(200, cfg, seed=8))
    assert "jobair" in text or "nor shuttle" in text
    assert "praha radar" in text or "ruzyne tower" in text
    for u in utterances(200, cfg, seed=8):
        assert validate_utterance(u, vocab.airlines) == []


# ---------------------------------------------------------------------------
# Text sources
# ---------------------------------------------------------------------------

def test_generate_utterance_returns_one_validated_line():
    u = generate_utterance(random.Random(4), BUILTIN)
    assert validate_utterance(u) == []
    assert u.role in ("controller", "pilot")


def test_grammar_text_source():
    source = GrammarTextSource(vocab_path=None)
    u = source.sample(random.Random(0))
    assert u.spoken and u.entities


def test_make_text_source_parses_grammar_knobs():
    source = make_text_source("grammar:region=eu,readback_error_prob=0.25")
    assert isinstance(source, GrammarTextSource)
    assert source.config.region == "eu"
    assert source.config.readback_error_prob == 0.25
    assert make_text_source({"kind": "grammar", "region": "us"}).config.region == "us"
    with pytest.raises(ValueError):
        make_text_source("grammar:nonsense=1")


def test_jsonl_text_source(tmp_path):
    path = tmp_path / "ext.jsonl"
    lines = [
        {"spoken": "delta one two, cleared to land",
         "transcript": "delta one two cleared to land"},
        {"text": "united five contact ground one two one point niner"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines))
    source = make_text_source(str(path))
    assert isinstance(source, JsonlTextSource)
    rng = random.Random(3)
    assert len({source.sample(rng).spoken for _ in range(20)}) == 2


def test_jsonl_text_source_reads_entities_and_display(tmp_path):
    path = tmp_path / "ext.jsonl"
    path.write_text(json.dumps({
        "text": "csa nine two six climb to flight level three four zero",
        "display": "CSA926, climb to FL340",
        "entities": [{"type": "callsign", "value": "CSA926"},
                     {"type": "flight_level", "value": "FL340"}],
    }))
    record = JsonlTextSource(path).records[0]
    assert record.display == "CSA926, climb to FL340"
    assert [(e.type, e.value) for e in record.entities] == [
        ("callsign", "CSA926"), ("flight_level", "FL340")]
    assert record.entities[1].critical is True


def test_jsonl_text_source_rejects_illegal_entities(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "runway four one",
                                "entities": [{"type": "runway", "value": "41"}]}))
    with pytest.raises(ValueError):
        JsonlTextSource(path)


def test_jsonl_text_source_rejects_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        JsonlTextSource(path)
