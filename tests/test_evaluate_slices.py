"""Tests for the report the evaluator builds.

Model loading is never exercised here: `evaluate_dataset` takes a transcriber
callable, so the integration test injects a fake one over rows of numpy noise.
"""

import json

import numpy as np
import pytest

from atcgen.entities import Entity
from training.evaluate import build_report, evaluate_dataset, write_report
from training.finetune_whisper import curriculum_phases

SR = 16000


def _row(text, seconds=4.0, category="routine", entities=None):
    row = {
        "audio": {"array": np.zeros(int(SR * seconds), dtype=np.float32),
                  "sampling_rate": SR},
        "text": text,
        "category": category,
    }
    if entities is not None:
        row["entities"] = entities
    return row


def test_per_category_wer_and_aggregate_exclude_noise():
    report = build_report(
        references=[
            "cleared to land",
            "contact tower",
            "mayday engine failure",
            "",
        ],
        hypotheses=[
            "cleared to land",
            "contact ground",
            "mayday failure",
            "phantom speech",
        ],
        categories=["routine", "routine", "emergency", "noise"],
    )

    assert report["wer"]["raw"] == pytest.approx(2 / 8)
    assert report["wer"]["atc_normalized"] == pytest.approx(2 / 8)
    assert report["per_category"]["routine"]["wer"]["raw"] == pytest.approx(1 / 5)
    assert report["per_category"]["emergency"]["wer"]["raw"] == pytest.approx(1 / 3)
    assert "noise" not in report["per_category"]


def test_per_category_reports_raw_and_atc_normalized_wer_independently():
    report = build_report(
        ["fly heading 35"],
        ["fly heading three five"],
        ["rare_vocab"],
    )

    wer = report["per_category"]["rare_vocab"]["wer"]
    assert wer["raw"] == pytest.approx(2 / 3)
    assert wer["atc_normalized"] == 0.0


def test_wer_carries_the_substitution_deletion_insertion_split():
    report = build_report(
        references=["csa one two three descend flight level two four zero"],
        hypotheses=["csa one two four descend to flight level two four"],
        categories=["routine"],
    )

    wer = report["wer"]
    assert wer["substitutions"] == 1        # three -> four
    assert wer["deletions"] == 1            # dropped trailing "zero"
    assert wer["insertions"] == 1           # spurious "to"
    assert wer["hits"] == 8
    assert wer["reference_words"] == 10
    assert wer["atc_normalized"] == pytest.approx(3 / 10)


def test_error_counts_come_from_the_normalized_pair_not_the_raw_one():
    report = build_report(["fly heading 35"], ["fly heading three five"],
                          ["rare_vocab"])
    wer = report["wer"]
    assert wer["raw"] == pytest.approx(2 / 3)
    assert (wer["substitutions"], wer["deletions"], wer["insertions"]) == (0, 0, 0)
    assert wer["reference_words"] == 4


def test_entity_panel_reports_callsign_and_critical_substitutions():
    report = build_report(
        references=[
            "csa one two three cleared to land runway two four left",
            "speedbird four six two contact tower one two seven decimal eight two five",
        ],
        hypotheses=[
            "csa one two three cleared to land runway zero six",
            "speedbird four six two contact tower one two seven decimal eight two five",
        ],
        categories=["routine", "routine"],
    )

    entities = report["entities"]
    assert entities["utterances"] == {"total": 2, "scored": 2,
                                      "skipped_no_ref_entities": 0}
    assert entities["callsign"]["accuracy"] == 1.0
    assert entities["critical"]["substitutions"] == 1
    assert entities["critical"]["reference_slots"] == 4
    assert entities["critical"]["substitution_rate"] == pytest.approx(0.25)
    assert entities["per_type"]["runway"]["sub"] == 1
    assert entities["examples"][0]["substitutions"][0]["ref"] == "24L"


def test_callsign_slice_uses_entity_ground_truth_not_a_regex():
    report = build_report(
        references=[
            "csa one two three climb to five thousand",
            "november four five alpha bravo contact tower",
            "cleared for takeoff",
        ],
        hypotheses=[
            "csa one two three climb to five thousand",
            "november four six alpha bravo contact tower",
            "cleared for takeoff",
        ],
        categories=["routine", "routine", "routine"],
    )

    callsign = report["callsign"]
    assert callsign["samples"] == 2                 # the third row has none
    assert callsign["reference_callsigns"] == 2
    assert callsign["exact_matches"] == 1
    assert callsign["substitutions"] == 1
    assert callsign["accuracy"] == pytest.approx(0.5)
    assert callsign["wer"]["atc_normalized"] == pytest.approx(1 / 15)


def test_manifest_entities_are_preferred_over_parsing_the_reference():
    reference = "csa one two three cleared to land two four left"
    labels = [Entity(type="callsign", value="CSA123", spoken="csa one two three"),
              Entity(type="runway", value="24L", spoken="two four left")]

    parsed = build_report([reference], [reference], ["routine"])
    labelled = build_report([reference], [reference], ["routine"],
                            ref_entities=[labels])

    assert "runway" not in parsed["entities"]["per_type"]
    assert labelled["entities"]["per_type"]["runway"]["fn"] == 1
    assert labelled["entities"]["critical"]["reference_slots"] == 2


def test_duration_band_slices():
    report = build_report(
        references=["contact tower", "cleared to land", "hold short of runway two four"],
        hypotheses=["contact tower", "cleared to ground", "hold short runway two four"],
        categories=["routine"] * 3,
        durations=[2.0, 4.5, 9.0],
    )

    duration = report["slices"]["duration"]
    assert set(duration) == {"<3s", "3-6s", ">6s"}
    assert duration["<3s"] == {"samples": 1, "wer": duration["<3s"]["wer"]}
    assert duration["<3s"]["wer"]["atc_normalized"] == 0.0
    assert duration["3-6s"]["wer"]["atc_normalized"] == pytest.approx(1 / 3)
    assert duration[">6s"]["wer"]["atc_normalized"] == pytest.approx(1 / 6)

    # bands are only reported when durations are known
    assert build_report(["contact tower"], ["contact tower"])["slices"]["duration"] == {}


def test_hallucination_rate_treats_whitespace_as_empty_after_normalization():
    report = build_report(
        references=["", "   ", "", "contact tower"],
        hypotheses=["", " \t ", "...", "contact tower"],
        categories=["noise", "noise", "noise", "routine"],
    )
    hallucination = report["hallucination"]
    assert hallucination == {
        "samples": 3,
        "non_empty_hypotheses": 0,
        "rate": 0.0,
    }

    report = build_report(["", ""], ["uh", "  "], ["noise", "noise"])
    assert report["hallucination"]["rate"] == pytest.approx(0.5)


def test_json_report_shape_and_round_trip(tmp_path):
    report = build_report(
        ["delta one two cleared to land", ""],
        ["delta one two cleared to land", "voice"],
        ["routine", "noise"],
        model="runs/candidate",
        dataset="data/eval/manifest.jsonl",
        split={"name": "model_select", "slice": "train[9000:10000]"},
    )
    out = tmp_path / "report.json"
    write_report(report, out)
    decoded = json.loads(out.read_text())

    assert set(decoded) == {
        "schema_version", "model", "dataset", "split", "samples", "wer",
        "entities", "per_category", "slices", "callsign", "hallucination",
    }
    assert decoded["schema_version"] == 2
    assert decoded["samples"] == {"total": 2, "speech": 1, "noise_only": 1}
    assert set(decoded["wer"]) == {"raw", "atc_normalized", "substitutions",
                                   "deletions", "insertions", "hits",
                                   "reference_words"}
    assert set(decoded["per_category"]["routine"]) == {"samples", "wer"}
    assert decoded["split"]["name"] == "model_select"
    assert decoded["callsign"]["accuracy"] == 1.0
    assert decoded["hallucination"]["rate"] == 1.0


def test_evaluate_dataset_with_a_fake_transcriber():
    rows = [
        _row("csa one two three cleared to land runway two four left", seconds=5.0),
        _row("contact tower one two seven decimal eight two five", seconds=2.0),
        _row("", seconds=7.0, category="noise"),
    ]
    hypotheses = [
        "csa one two four cleared to land runway two four left",
        "contact tower one two seven decimal eight two five",
        "",
    ]
    pending = list(hypotheses)
    seen_batches = []

    def transcriber(batch):
        seen_batches.append(len(batch))
        return [pending.pop(0) for _ in batch]

    report = evaluate_dataset(rows, transcriber, batch_size=2,
                              model="fake", dataset_name="unit-test",
                              progress=False)

    assert seen_batches == [2, 1]
    assert report["model"] == "fake"
    assert report["dataset"] == "unit-test"
    assert report["samples"] == {"total": 3, "speech": 2, "noise_only": 1}
    assert report["entities"]["callsign"]["accuracy"] == 0.0
    assert report["entities"]["critical"]["substitutions"] == 1
    assert report["hallucination"]["rate"] == 0.0
    # durations come from the audio when the row does not carry one
    assert set(report["slices"]["duration"]) == {"<3s", "3-6s"}


def test_evaluate_dataset_uses_manifest_entities_when_rows_carry_them():
    labels = [{"type": "runway", "value": "24L", "spoken": "two four left",
               "critical": True}]
    rows = [_row("cleared to land two four left", entities=labels)]
    report = evaluate_dataset(rows, lambda batch: ["cleared to land runway two four left"],
                              progress=False)

    assert report["entities"]["per_type"]["runway"]["tp"] == 1
    assert report["entities"]["utterances"]["scored"] == 1


def test_evaluate_dataset_rejects_a_transcriber_that_drops_rows():
    rows = [_row("contact tower"), _row("cleared to land")]
    with pytest.raises(ValueError, match="hypotheses"):
        evaluate_dataset(rows, lambda batch: ["contact tower"], progress=False)


def test_curriculum_phase_order_is_synthetic_then_real():
    synthetic = object()
    real = object()
    phases = curriculum_phases(synthetic, real)

    assert [phase.name for phase in phases] == ["synthetic", "real"]
    assert [phase.dataset for phase in phases] == [synthetic, real]


def test_report_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        build_report(["one"], [], ["routine"])
    with pytest.raises(ValueError, match="same length"):
        build_report(["one"], ["two"], ["routine"], durations=[])
