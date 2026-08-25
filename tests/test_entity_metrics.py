"""Tests for the entity/safety panel.

The distinction the panel exists to make is substitution vs. miss: a garbled
runway number is an operational hazard, a dropped one is a dropped word. Both
cost the same F1 and only one moves `critical.substitution_rate`.
"""

import pytest

from atcgen.entities import Entity, extract_entities
from atcgen.eval.entity_metrics import (
    diff_entities,
    entity_panel,
    resolve_ref_entities,
)

REF = "csa one two three cleared to land runway two four left"


def test_substitution_and_miss_score_the_same_f1_but_differ_on_criticality():
    substituted = entity_panel(
        [REF], ["csa one two three cleared to land runway zero six"])
    missed = entity_panel([REF], ["csa one two three cleared to land"])

    for panel in (substituted, missed):
        assert panel["per_type"]["runway"]["recall"] == 0.0
        assert panel["per_type"]["runway"]["support"] == 1

    assert substituted["per_type"]["runway"]["sub"] == 1
    assert substituted["per_type"]["runway"]["fp"] == 1
    assert substituted["critical"]["substitutions"] == 1
    assert substituted["critical"]["reference_slots"] == 2
    assert substituted["critical"]["substitution_rate"] == pytest.approx(0.5)

    assert missed["per_type"]["runway"]["sub"] == 0
    assert missed["per_type"]["runway"]["fp"] == 0
    assert missed["critical"]["substitutions"] == 0
    assert missed["critical"]["substitution_rate"] == 0.0


def test_per_type_precision_recall_f1_and_support():
    panel = entity_panel(
        references=[
            "csa one two three descend flight level two four zero",
            "speedbird four six two contact tower one two seven decimal eight two five",
        ],
        hypotheses=[
            # callsign right, flight level garbled
            "csa one two three descend flight level two four five",
            # callsign garbled, frequency right
            "speedbird four six three contact tower one two seven decimal eight two five",
        ],
    )

    callsign = panel["per_type"]["callsign"]
    assert (callsign["tp"], callsign["fp"], callsign["fn"], callsign["sub"]) == (1, 1, 1, 1)
    assert callsign["support"] == 2
    assert callsign["precision"] == pytest.approx(0.5)
    assert callsign["recall"] == pytest.approx(0.5)
    assert callsign["f1"] == pytest.approx(0.5)

    assert panel["per_type"]["flight_level"]["f1"] == 0.0
    assert panel["per_type"]["frequency"]["f1"] == 1.0

    assert panel["callsign"] == {"accuracy": pytest.approx(0.5), "correct": 1,
                                 "total": 2, "substitutions": 1}
    assert panel["overall"]["f1"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 2))
    assert panel["critical"]["substitutions"] == 2      # one callsign, one level
    assert panel["critical"]["reference_slots"] == 4


def test_utterances_without_reference_entities_are_excluded_but_counted():
    panel = entity_panel(
        references=["roger thank you", REF],
        hypotheses=["turn right heading two seven zero", REF],
    )

    assert panel["utterances"] == {"total": 2, "scored": 1,
                                   "skipped_no_ref_entities": 1}
    # the hypothesis-side heading in the unscorable row is reported, not charged
    assert panel["unscored"] == {"utterances": 1, "hypothesis_entities": 1}
    assert panel["overall"]["fp"] == 0
    assert panel["overall"]["f1"] == 1.0


def test_examples_are_the_worst_utterances_critical_substitutions_first():
    references = [
        REF,                                        # clean
        "csa one two three descend flight level two four zero",   # one sub
        "speedbird four six two runway two four left cleared to land",  # two subs
        "csa one two three contact tower",          # miss only
    ]
    hypotheses = [
        REF,
        "csa one two three descend flight level two four five",
        "speedbird four six three runway zero six cleared to land",
        "contact tower",
    ]
    panel = entity_panel(references, hypotheses, max_examples=2)

    assert [example["index"] for example in panel["examples"]] == [2, 1]
    worst = panel["examples"][0]
    assert worst["critical_substitutions"] == 2
    assert {(item["type"], item["ref"], item["hyp"]) for item in worst["substitutions"]} == {
        ("callsign", "BAW462", "BAW463"), ("runway", "24L", "06")}
    assert worst["reference"] == references[2]
    assert worst["hypothesis"] == hypotheses[2]
    assert worst["missed"] == worst["spurious"] == []

    # the clean utterance never becomes an example, however few there are
    assert all(example["index"] != 0 for example in panel["examples"])


def test_examples_report_misses_and_spurious_extractions_separately():
    panel = entity_panel(
        ["csa one two three contact tower"],
        ["turn right heading two seven zero"],
    )
    example = panel["examples"][0]
    assert example["missed"] == [
        {"type": "callsign", "value": "CSA123", "critical": True}]
    assert example["spurious"] == [
        {"type": "heading", "value": "270", "critical": True}]
    assert example["substitutions"] == []


def test_ground_truth_entities_are_used_instead_of_parsing_the_reference():
    # "cleared to land two four left" has no runway anchor word, so the parser
    # finds nothing; the grammar's own label does.
    reference = "csa one two three cleared to land two four left"
    labels = [Entity(type="runway", value="24L", spoken="two four left")]
    assert not any(entity.type == "runway"
                   for entity in extract_entities(reference))

    panel = entity_panel([reference], ["csa one two three cleared to land runway two four left"],
                         [labels])
    assert panel["per_type"]["runway"]["tp"] == 1
    assert panel["utterances"]["scored"] == 1
    # only the label counts as reference truth: the parsed callsign is not there
    assert "callsign" not in panel["per_type"] or \
        panel["per_type"]["callsign"]["fn"] == 0


def test_manifest_dicts_are_accepted_as_ground_truth():
    labels = [{"type": "runway", "value": "24L", "spoken": "two four left",
               "critical": True}]
    panel = entity_panel(["cleared to land two four left"],
                         ["cleared to land runway two four left"], [labels])
    assert panel["per_type"]["runway"]["tp"] == 1


def test_resolve_ref_entities_parses_only_the_unlabelled_rows():
    resolved = resolve_ref_entities(
        ["cleared to land two four left", "csa one two three contact tower"],
        [[{"type": "runway", "value": "24L"}], None],
    )
    assert [entity.value for entity in resolved[0]] == ["24L"]
    assert [entity.value for entity in resolved[1]] == ["CSA123"]


def test_shared_airline_table_is_applied_to_both_sides():
    spoken = "clarion air four five one contact tower"
    panel = entity_panel([spoken], [spoken], airlines={"clarion air": "CLA"})
    assert panel["callsign"] == {"accuracy": 1.0, "correct": 1, "total": 1,
                                 "substitutions": 0}
    # a carrier missing from the table is not a callsign on either side, so it
    # silently drops out of the panel rather than scoring as an error
    unknown = entity_panel([spoken], [spoken])
    assert unknown["utterances"]["scored"] == 0


def test_diff_pairs_leftovers_of_the_same_type_only():
    diff = diff_entities(
        [Entity("runway", "24L"), Entity("heading", "270")],
        [Entity("runway", "06"), Entity("frequency", "127.825")],
    )
    assert [(reference.value, hypothesis.value)
            for reference, hypothesis in diff.substitutions] == [("24L", "06")]
    assert [entity.value for entity in diff.missed] == ["270"]
    assert [entity.value for entity in diff.spurious] == ["127.825"]
    assert diff.critical_substitutions == 1


def test_empty_corpus_and_length_mismatch():
    empty = entity_panel([], [])
    assert empty["utterances"] == {"total": 0, "scored": 0,
                                   "skipped_no_ref_entities": 0}
    assert empty["overall"]["f1"] == 0.0
    assert empty["critical"]["substitution_rate"] == 0.0
    assert empty["examples"] == []

    with pytest.raises(ValueError, match="same length"):
        entity_panel(["one"], [])
    with pytest.raises(ValueError, match="same length"):
        entity_panel(["one"], ["two"], [])
