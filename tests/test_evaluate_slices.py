import json

import pytest

from training.evaluate import build_report, has_callsign, write_report
from training.finetune_whisper import curriculum_phases


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


@pytest.mark.parametrize("text", [
    "American 123 climb and maintain five thousand",
    "air france two seven contact tower",
    "November one two three alpha bravo cleared to land",
    "N45AB cleared to land",
])
def test_callsign_detection_true(text):
    assert has_callsign(text)


@pytest.mark.parametrize("text", [
    "climb and maintain three thousand",
    "american contact tower",
    "november alpha bravo",
    "taxi via alpha and bravo",
])
def test_callsign_detection_false(text):
    assert not has_callsign(text)


def test_callsign_slice_wer_and_exact_token_accuracy_after_normalization():
    report = build_report(
        references=[
            "American 123 climb to five thousand",
            "november four five alpha bravo contact tower",
            "cleared for takeoff",
        ],
        hypotheses=[
            "american one two three climb to five thousand",
            "november four six alpha bravo contact tower",
            "cleared for takeoff",
        ],
        categories=["routine", "routine", "routine"],
    )

    callsign = report["callsign"]
    assert callsign["samples"] == 2
    assert callsign["reference_sequences"] == 2
    assert callsign["exact_sequences"] == 1
    assert callsign["token_accuracy"] == pytest.approx(0.5)
    assert callsign["wer"]["atc_normalized"] == pytest.approx(1 / 15)


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
    )
    out = tmp_path / "report.json"
    write_report(report, out)
    decoded = json.loads(out.read_text())

    assert set(decoded) == {
        "schema_version", "model", "dataset", "samples", "wer",
        "per_category", "callsign", "hallucination",
    }
    assert decoded["schema_version"] == 1
    assert decoded["samples"] == {"total": 2, "speech": 1, "noise_only": 1}
    assert set(decoded["wer"]) == {"raw", "atc_normalized"}
    assert set(decoded["per_category"]["routine"]) == {"samples", "wer"}
    assert decoded["callsign"]["token_accuracy"] == 1.0
    assert decoded["hallucination"]["rate"] == 1.0


def test_curriculum_phase_order_is_synthetic_then_real():
    synthetic = object()
    real = object()
    phases = curriculum_phases(synthetic, real)

    assert [phase.name for phase in phases] == ["synthetic", "real"]
    assert [phase.dataset for phase in phases] == [synthetic, real]


def test_report_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        build_report(["one"], [], ["routine"])
