"""Entity schema, spoken<->canonical numbers, parser and scoring."""

import json

import pytest

from atcgen.entities import (CRITICAL_TYPES, DEFAULT_AIRLINES, Entity,
                             aggregate, check_value, digits_to_words,
                             extract_entities, feet_to_words,
                             group_number_words, load_airlines, score_entities,
                             telephony_code)


def types(entities):
    return [(e.type, e.value) for e in entities]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def test_digits_to_words_both_registers():
    assert digits_to_words("359") == "three five nine"
    assert digits_to_words("359", atc_variants=True) == "tree fife niner"
    assert digits_to_words("1204", atc_variants=True) == "one two zero fower"


def test_group_number_words():
    assert group_number_words(412) == "four twelve"
    assert group_number_words(1850) == "eighteen fifty"
    assert group_number_words(127) == "one twenty seven"
    assert group_number_words(20) == "twenty"
    assert group_number_words(305, atc_variants=True) == "tree zero fife"


def test_feet_to_words():
    assert feet_to_words(3500) == "three thousand five hundred"
    assert feet_to_words(10000) == "one zero thousand"
    assert feet_to_words(500) == "five hundred"


@pytest.mark.parametrize("spoken,expected", [
    ("heading two seven zero", ("heading", "270")),
    ("heading zero niner zero", ("heading", "090")),
    ("runway two four left", ("runway", "24L")),
    ("runway zero four", ("runway", "04")),
    ("flight level three five zero", ("flight_level", "FL350")),
    ("level one one zero", ("flight_level", "FL110")),
    ("climb and maintain three thousand five hundred", ("altitude", "3500ft")),
    ("descend and maintain one zero thousand", ("altitude", "10000ft")),
    ("squawk four five two one", ("squawk", "4521")),
    ("qnh one zero one three", ("altimeter", "Q1013")),
    ("altimeter two niner niner two", ("altimeter", "A2992")),
    ("reduce speed to two five zero knots", ("speed", "250")),
    ("contact tower one two seven decimal eight two five", ("frequency", "127.825")),
    ("contact ground one two one point niner", ("frequency", "121.900")),
    # group form: "one twenty seven" must not read as 1207
    ("contact radar one twenty seven decimal eight two five", ("frequency", "127.825")),
])
def test_spoken_to_canonical(spoken, expected):
    assert expected in types(extract_entities(spoken))


def test_words_round_trip_through_the_renderers():
    """Every canonical value the grammar can emit reads back as itself."""
    from atcgen.text.lexicon import Style, make_slot

    cases = [("runway", "24L"), ("runway", "04"), ("heading", "090"),
             ("flight_level", "FL350"), ("altitude", "3500ft"),
             ("frequency", "127.825"), ("frequency", "118.000"),
             ("speed", "250"), ("squawk", "4521"), ("altimeter", "Q1013")]
    anchors = {"runway": "runway ", "heading": "heading ",
               "flight_level": "flight level ", "altitude": "altitude ",
               "frequency": "", "speed": "speed ", "squawk": "squawk ",
               "altimeter": "qnh "}
    for variants in (False, True):
        style = Style(atc_variants=variants)
        for type_, value in cases:
            slot = make_slot(type_, value, style)
            text = anchors[type_] + slot.spoken
            assert (type_, value) in types(extract_entities(text)), text


# ---------------------------------------------------------------------------
# Parser on real-corpus strings
# ---------------------------------------------------------------------------

def test_extracts_european_transmission():
    text = ("sky travel six seven zero contact praha radar "
            "one two seven decimal eight two five")
    assert types(extract_entities(text)) == [
        ("callsign", "SKV670"), ("frequency", "127.825")]


def test_extracts_waypoint_request():
    assert ("waypoint", "PADKA") in types(
        extract_entities("direct padka request three five zero"))


def test_extracts_alphanumeric_european_callsign():
    text = "csa three kilo foxtrot change proceed direct to lodz"
    assert types(extract_entities(text)) == [
        ("callsign", "CSA3KF"), ("waypoint", "LODZ")]


def test_extracts_full_enroute_clearance():
    text = ("csa four two zero praha radar contact climb to "
            "flight level three four zero")
    assert types(extract_entities(text)) == [
        ("callsign", "CSA420"), ("flight_level", "FL340")]


def test_extracts_us_tower_transmission():
    text = ("cessna one two three alpha bravo squawk four five two one "
            "contact departure one two one point niner")
    assert types(extract_entities(text)) == [
        ("callsign", "N123AB"), ("squawk", "4521"), ("frequency", "121.900")]


def test_wind_is_not_mistaken_for_a_heading_or_speed():
    text = ("csa nine two six runway three one cleared for takeoff "
            "wind two zero zero degrees four knots")
    assert types(extract_entities(text)) == [
        ("callsign", "CSA926"), ("runway", "31")]


def test_taxiway_letters_are_not_callsigns():
    text = "speedbird four six two taxi via alpha bravo delta"
    assert types(extract_entities(text)) == [("callsign", "BAW462")]


def test_out_of_range_values_are_dropped_rather_than_guessed():
    assert extract_entities("heading nine nine nine") == []
    assert extract_entities("runway four one") == []


def test_extractor_uses_the_supplied_airline_table():
    text = "moonjet four two one climb to flight level three three zero"
    assert ("callsign", "MOO421") not in types(extract_entities(text))
    airlines = dict(DEFAULT_AIRLINES, **{"moonjet": "MOO"})
    assert ("callsign", "MOO421") in types(extract_entities(text, airlines))


def test_default_airline_table_extends_the_builtin_one():
    from atcgen.entities import default_airlines

    table = default_airlines()
    assert all(table[phrase] == code for phrase, code in DEFAULT_AIRLINES.items())


# ---------------------------------------------------------------------------
# Numeral-form hypotheses (what ASR teachers actually write)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hyp,expected", [
    # ordinals: whisper writes "26th" where the controller said "two six"
    ("Finding up runway 26th center", ("runway", "26C")),
    ("cleared to land runway 1st left", ("runway", "01L")),
    # the side as a bare letter rather than a word
    ("all short of runway 20L", ("runway", "20L")),
    ("runway 03 R line up and wait", ("runway", "03R")),
    # bare numeral altitudes, with and without a thousands comma
    ("Climb and maintain 1500.", ("altitude", "1500ft")),
    ("The send and maintain 2,500 for Singapore", ("altitude", "2500ft")),
    ("descend to 4,000", ("altitude", "4000ft")),
    # a unit suffix anchors on its own
    ("Open to 2000 feet.", ("altitude", "2000ft")),
    ("I'm going 855, reduce heat to 260 knots.", ("speed", "260")),
    ("We use speeds 180 knots.", ("speed", "180")),
    # punctuated idents
    ("Austrian 6-2-1-2, reduce speed to 270 knots", ("callsign", "AUA6212")),
    ("KLM 3996, proceed direct lomki", ("callsign", "KLM3996")),
    ("China Southern 8 Romeo Foxtrot", ("callsign", "CSN8RF")),
    ("maintaining front level 280", ("flight_level", "FL280")),
    ("contact Frankfort radar 126 decimal 865", ("frequency", "126.865")),
    ("turn right heading 030", ("heading", "030")),
    ("Korean Air 7217, squawk 4711", ("squawk", "4711")),
])
def test_numeral_form_hypotheses(hyp, expected):
    assert expected in types(extract_entities(hyp))


@pytest.mark.parametrize("hyp,joined,expected", [
    ("Aero Mexico 3847 climbing", "aeromexico 3847 climbing", "AMX3847"),
    ("k l m 3996 direct", "klm 3996 direct", "KLM3996"),
    ("wizz air 421 roger", "wizzair 421 roger", "WZZ421"),
    ("sky travel 670 standby", "skytravel 670 standby", "SKV670"),
])
def test_telephony_matches_however_the_words_are_split(hyp, joined, expected):
    """ASR splits and joins telephony names freely; both must reach one code."""
    assert ("callsign", expected) in types(extract_entities(hyp))
    assert ("callsign", expected) in types(extract_entities(joined))


def test_knots_beats_an_altitude_anchor():
    """"maintain 250 knots" is a speed wearing an altitude's anchor word."""
    assert types(extract_entities("maintain 250 knots")) == [("speed", "250")]


def test_reported_wind_is_still_not_an_airspeed():
    """The 60 kt floor is what makes the bare "knots" anchor safe."""
    text = "csa nine two six cleared to land wind two zero zero degrees four knots"
    assert types(extract_entities(text)) == [("callsign", "CSA926")]


def test_harvested_name_inherits_a_builtin_designator(tmp_path):
    """'finn air' must not get its own code when 'finnair' is already FIN."""
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(
        {"airlines": {"finn air": {"count": 31, "icao": "FIA"}}}))
    airlines = load_airlines(path)
    assert airlines["finn air"] == airlines["finnair"] == "FIN"
    assert ("callsign", "FIN421") in types(
        extract_entities("finn air 421 roger", airlines))


def test_entity_spoken_is_the_substring_that_says_it():
    entity = extract_entities("contact praha radar one two seven decimal "
                              "eight two five")[0]
    assert entity.spoken == "one two seven decimal eight two five"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_critical_defaults_by_type():
    assert Entity("heading", "270").critical is True
    assert Entity("callsign", "CSA123").critical is True
    assert Entity("waypoint", "PADKA").critical is False
    assert set(CRITICAL_TYPES) == {
        "callsign", "runway", "heading", "altitude", "flight_level",
        "frequency", "speed", "squawk", "altimeter"}


def test_entity_dict_round_trip():
    entity = Entity("frequency", "127.825", "one two seven decimal eight two five")
    assert Entity.from_dict(json.loads(json.dumps(entity.to_dict()))) == entity


@pytest.mark.parametrize("type_,value", [
    ("runway", "24L"), ("runway", "01"), ("heading", "360"),
    ("flight_level", "FL010"), ("altitude", "45000ft"),
    ("frequency", "118.000"), ("frequency", "121.755"), ("speed", "250"),
    ("squawk", "7700"), ("altimeter", "Q1013"), ("altimeter", "A2992"),
])
def test_legal_values(type_, value):
    assert check_value(type_, value) is None


@pytest.mark.parametrize("type_,value", [
    ("runway", "37"), ("runway", "4"), ("heading", "000"), ("heading", "361"),
    ("flight_level", "FL500"), ("altitude", "50000ft"), ("altitude", "3550ft"),
    ("frequency", "137.000"), ("frequency", "127.827"), ("speed", "400"),
    ("squawk", "7800"), ("altimeter", "Q1200"), ("waypoint", "P"),
    ("nonsense", "x"),
])
def test_illegal_values(type_, value):
    assert check_value(type_, value)


def test_telephony_code_is_letters_only_and_stable():
    assert telephony_code("nor shuttle") == "NOS"
    assert telephony_code("jobair") == "JOB"
    assert telephony_code("air malta").isalpha()


def test_load_airlines_falls_back_without_the_vocab_file(tmp_path):
    assert load_airlines(tmp_path / "missing.json") == DEFAULT_AIRLINES
    assert load_airlines(None) == DEFAULT_AIRLINES


def test_load_airlines_merges_harvested_names(tmp_path):
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps({"airlines": {"jobair": {"count": 50, "icao": "JOB"}}}))
    airlines = load_airlines(path)
    assert airlines["jobair"] == "JOB"
    assert airlines["lufthansa"] == "DLH"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_perfect_match_scores_one():
    ref = [Entity("callsign", "CSA123"), Entity("flight_level", "FL350")]
    score = score_entities(ref, list(ref))
    assert score.f1 == 1.0
    assert score.callsign_accuracy == 1.0
    assert score.critical_substitution_rate == 0.0


def test_substituted_digit_is_counted_as_a_critical_substitution():
    ref = [Entity("callsign", "CSA123"), Entity("flight_level", "FL350")]
    hyp = [Entity("callsign", "CSA123"), Entity("flight_level", "FL250")]
    score = score_entities(ref, hyp)
    assert score.critical_substitutions_of("flight_level") == 1
    assert score.critical_substitution_rate == 0.5
    assert score.callsign_accuracy == 1.0
    assert score.per_type()["flight_level"]["f1"] == 0.0


def test_missing_entity_is_a_miss_not_a_substitution():
    ref = [Entity("heading", "270")]
    score = score_entities(ref, [])
    assert score.recall == 0.0
    assert score.critical_sub == 0
    assert score.critical_ref == 1


def test_hallucinated_entity_costs_precision_only():
    score = score_entities([], [Entity("runway", "24L")])
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.critical_ref == 0


def test_scores_aggregate_over_a_corpus():
    good = score_entities([Entity("heading", "270")], [Entity("heading", "270")])
    bad = score_entities([Entity("heading", "270")], [Entity("heading", "280")])
    total = aggregate([good, bad, good])
    assert total.critical_ref == 3
    assert total.critical_sub == 1
    assert total.critical_substitution_rate == pytest.approx(1 / 3)
    assert total.per_type()["heading"]["tp"] == 2
    assert (good + bad + good).to_dict() == total.to_dict()


def test_score_survives_json():
    score = score_entities([Entity("runway", "24L")], [Entity("runway", "24R")])
    assert json.loads(json.dumps(score.to_dict()))["critical_substitutions"] == 1


def test_end_to_end_reference_versus_hypothesis():
    reference = "csa nine two six descend to flight level one zero zero"
    hypothesis = "csa nine two six descend to flight level one zero one"
    score = score_entities(extract_entities(reference), extract_entities(hypothesis))
    assert score.callsign_accuracy == 1.0
    assert score.critical_substitutions_of("flight_level") == 1
