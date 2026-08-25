"""Verification-gate tests: mocked teachers, no network, no GPU.

The tier rules are the contract, so most of this pins behaviour at the
threshold boundaries and on the asymmetries that are easy to get backwards --
verification needs *any* teacher, substitution needs *every* teacher that
engaged with the slot, and a noise row is only mislabelled when the teachers
agree they heard speech.

One integration test loads the real whisper-base.en on two clips and is
skipped when the fixture set has not been built; everything else is pure.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from atcgen.entities import Entity, entities_to_dicts
from atcgen.gate import (
    GateConfig,
    audio_checks,
    evaluate_row,
    gate_dataset,
    gate_stats,
    load_gated,
    repeat_score,
    retier,
    select_tiers,
    verify_entities,
)
from atcgen.gate.gate import hypothesis_entities
from atcgen.gate.teachers import Throughput

FIXTURE = Path("runs/smoke_gate_fixture")

CLEAN_AUDIO = {"ok": True, "qc_ok": True, "qc_reason": None,
               "repeat_score": 0.2, "repeated_segment": False, "metrics": {}}

REF_TEXT = "csa one two three contact praha radar one two seven decimal eight two five"
REF_ENTITIES = [Entity("callsign", "CSA123"), Entity("frequency", "127.825")]


def row(text=REF_TEXT, entities=REF_ENTITIES):
    return {"text": text, "entities": entities_to_dicts(entities)}


def audio(**overrides):
    return {**CLEAN_AUDIO, **overrides}


class FakeTeacher:
    """Returns canned text; ignores the audio entirely."""

    def __init__(self, name, texts):
        self.name = name
        self.texts = texts
        self.batches = []

    def transcribe(self, waves, sr):
        self.batches.append(len(waves))
        return [self.texts[index % len(self.texts)] for index in range(len(waves))]


# --------------------------------------------------------------------------
# tier boundaries
# --------------------------------------------------------------------------

def wer_row(wer_text, **config):
    """Tier a sample whose teachers both produce `wer_text`."""
    return evaluate_row(row(), {"a": wer_text, "b": wer_text}, audio(),
                        GateConfig(**config))


def test_perfect_transcription_is_gold():
    tier, gate = wer_row(REF_TEXT)
    assert tier == "gold"
    assert gate["best_wer"] == 0.0 and gate["reasons"] == []
    assert gate["entities"]["critical_recall"] == 1.0


#: A slot-free reference, so the WER boundary tests are not perturbed by
#: entity verdicts: a truncated ATC label parses to a truncated callsign,
#: which is a substitution and rejects for the right reason at the wrong time.
PLAIN_TEXT = ("roger wilco standing by for the next available instruction "
              "from you now please thanks")


def test_tiers_step_at_the_configured_wer_boundaries():
    # 14 reference words, so dropping k trailing words costs exactly k/14
    words = PLAIN_TEXT.split()
    assert len(words) == 14

    def tier_after_dropping(k):
        heard = " ".join(words[:len(words) - k])
        return evaluate_row(row(PLAIN_TEXT, []), {"a": heard, "b": heard},
                            audio(), GateConfig())

    assert tier_after_dropping(3)[0] == "gold"          # 0.214
    assert tier_after_dropping(4)[0] == "silver"        # 0.286
    assert tier_after_dropping(7)[0] == "silver"        # 0.500, on the boundary
    assert tier_after_dropping(8)[0] == "adversarial"   # 0.571
    assert tier_after_dropping(12)[0] == "adversarial"  # 0.857
    tier, gate = tier_after_dropping(13)                # 0.929, over the ceiling
    assert tier == "rejected" and gate["reasons"] == ["teacher_wer_above_ceiling"]


def test_a_truncated_callsign_is_a_substitution_not_a_near_miss():
    """"csa one" against a CSA123 label is a wrong value, and D8 rejects it."""
    tier, gate = evaluate_row(row(), {"a": "csa one", "b": "csa one"}, audio(),
                              GateConfig())
    verdicts = {item["type"]: item["verdict"] for item in gate["entities"]["verdicts"]}
    assert verdicts["callsign"] == "substituted"
    assert tier == "rejected"
    assert "critical_entity_substitution" in gate["reasons"]


def test_gold_needs_entity_recall_as_well_as_low_wer():
    """A clean-WER clip whose critical slots nobody recovered is not gold.

    Two errors in fourteen words, but both land on the slots: the airline word
    is gone and so is the "decimal" the frequency rule anchors on.
    """
    heard = "xray one two three contact praha radar one two seven eight two five"
    tier, gate = evaluate_row(row(), {"a": heard, "b": heard}, audio(),
                              GateConfig(gold_critical_recall=0.5))
    assert gate["best_wer"] < 0.25
    assert gate["entities"]["critical_recall"] == 0.0
    assert tier == "silver"


def test_hard_clip_without_verified_entities_is_rejected_not_adversarial():
    """The adversarial band is for provable labels, not merely hard ones."""
    heard = " ".join(REF_TEXT.split()[:6])       # keeps the callsign, loses the rest
    tier, gate = evaluate_row(row(), {"a": heard, "b": "nothing at all here"},
                              audio(), GateConfig(adversarial_critical_recall=1.0))
    assert 0.5 < gate["best_wer"] <= 0.9
    assert gate["entities"]["critical_recall"] == 0.5
    assert tier == "rejected"
    assert gate["reasons"] == ["hard_clip_unverified_entities"]


def test_hard_clip_with_verified_entities_is_adversarial():
    """The same WER band, but every critical slot survived: hard, still provable."""
    heard = REF_TEXT + " blah blah blah blah blah blah blah blah"
    tier, gate = evaluate_row(row(), {"a": heard, "b": "mush"}, audio(),
                              GateConfig(adversarial_critical_recall=1.0))
    assert 0.5 < gate["best_wer"] <= 0.9
    assert gate["entities"]["critical_recall"] == 1.0
    assert tier == "adversarial" and gate["reasons"] == []


# --------------------------------------------------------------------------
# entity verification: any-teacher verifies, every-teacher substitutes
# --------------------------------------------------------------------------

def test_any_teacher_recovering_a_slot_verifies_it():
    report = verify_entities(REF_ENTITIES, {
        "a": [Entity("callsign", "CSA123")],
        "b": [Entity("callsign", "CSA999"), Entity("frequency", "127.825")],
    })
    verdicts = {item["type"]: item for item in report["verdicts"]}
    assert verdicts["callsign"]["verdict"] == "verified"
    assert verdicts["callsign"]["verified_by"] == ["a"]
    assert verdicts["frequency"]["verdict"] == "verified"
    assert report["critical_recall"] == 1.0
    assert not report["any_critical_substitution"]


def test_unanimous_disagreement_is_a_substitution_and_rejects():
    """Every teacher that heard the slot heard a *different* value: D8 rejects."""
    report = verify_entities(REF_ENTITIES, {
        "a": [Entity("frequency", "127.855")],
        "b": [Entity("frequency", "127.900")],
    })
    verdicts = {item["type"]: item for item in report["verdicts"]}
    assert verdicts["frequency"]["verdict"] == "substituted"
    assert verdicts["frequency"]["heard"] == {"a": ["127.855"], "b": ["127.900"]}
    assert verdicts["callsign"]["verdict"] == "missed"   # nobody produced one
    assert report["any_critical_substitution"]

    tier, gate = evaluate_row(
        row(), {"a": "contact one two seven decimal eight five five",
                "b": "contact one two seven decimal nine zero zero"},
        audio(), GateConfig())
    assert tier == "rejected"
    assert "critical_entity_substitution" in gate["reasons"]


def test_one_teacher_agreeing_defeats_a_substitution():
    report = verify_entities(REF_ENTITIES, {
        "a": [Entity("frequency", "127.855")],
        "b": [Entity("frequency", "127.825")],
    })
    verdicts = {item["type"]: item for item in report["verdicts"]}
    assert verdicts["frequency"]["verdict"] == "verified"
    assert not report["any_critical_substitution"]


def test_a_slot_no_teacher_produced_is_missed_not_substituted():
    """Absence of evidence costs recall; it is not evidence against the label."""
    report = verify_entities(REF_ENTITIES, {"a": [], "b": []})
    assert [item["verdict"] for item in report["verdicts"]] == ["missed", "missed"]
    assert report["critical_missed"] == 2
    assert not report["any_critical_substitution"]


def test_hearing_the_other_reference_value_is_not_a_substitution():
    """Two runways in the label; a teacher that heard one of them is not wrong."""
    ref = [Entity("runway", "24L"), Entity("runway", "07R")]
    report = verify_entities(ref, {"a": [Entity("runway", "24L")]})
    verdicts = [item["verdict"] for item in report["verdicts"]]
    assert verdicts == ["verified", "missed"]
    assert not report["any_critical_substitution"]


def test_our_own_respelling_cannot_contradict_a_label():
    """"flight level 3.0" normalizes to FL030; that is our artifact, not a teacher's."""
    ref = [Entity("flight_level", "FL340")]
    heard = "descend to flight level 3.0"
    tier, gate = evaluate_row({"text": "descend to flight level tree fower zero",
                               "entities": entities_to_dicts(ref)},
                              {"a": heard, "b": heard}, audio(), GateConfig())

    # the re-spelling really does produce the bogus value...
    merged, asserted = hypothesis_entities(heard, "descend to flight level three zero")
    assert ("flight_level", "FL030") in {(e.type, e.value) for e in merged}
    assert asserted == []
    # ...but it is not allowed to reject the sample
    assert gate["entities"]["verdicts"][0]["verdict"] == "missed"
    assert "critical_entity_substitution" not in gate["reasons"]
    assert tier != "rejected"


def test_a_teacher_writing_its_own_wrong_value_still_substitutes():
    """The asymmetry must not disarm real disagreement: "819" is whisper's own."""
    ref = [Entity("callsign", "AIF8118")]
    heard = "air force 819"
    report = verify_entities(
        ref, {"a": hypothesis_entities(heard, heard)[0]},
        {"a": hypothesis_entities(heard, heard)[1]})
    assert report["verdicts"][0]["verdict"] == "substituted"


def test_non_critical_slots_do_not_drive_tiering():
    """A misheard waypoint is noise; only CRITICAL_TYPES gate."""
    ref = [Entity("callsign", "CSA123"), Entity("waypoint", "LOMKI")]
    report = verify_entities(ref, {"a": [Entity("callsign", "CSA123"),
                                         Entity("waypoint", "PADKA")]})
    verdicts = {item["type"]: item["verdict"] for item in report["verdicts"]}
    assert verdicts == {"callsign": "verified", "waypoint": "substituted"}
    assert report["critical_total"] == 1 and report["critical_recall"] == 1.0
    assert not report["any_critical_substitution"]


def test_entities_are_parsed_from_both_hypothesis_renderings():
    """Raw text keeps "127.825"; normalization spells "471.1" back out."""
    from atcgen.eval.qc import normalize_atc

    raw = "Contact Praha 127.825, squawk 471.1"
    merged, asserted = hypothesis_entities(raw, normalize_atc(raw))
    found = {(e.type, e.value) for e in merged}
    assert ("frequency", "127.825") in found     # only the raw rendering has this
    assert ("squawk", "4711") in found           # only the normalized one has this
    # the squawk came from our re-spelling, so it is not something to hold
    # against a label -- only the raw rendering's slots are
    assert {(e.type, e.value) for e in asserted} == {("frequency", "127.825")}


# --------------------------------------------------------------------------
# noise-only rows
# --------------------------------------------------------------------------

def test_silent_teachers_make_a_noise_row_gold():
    tier, gate = evaluate_row(row("", []), {"a": "", "b": ""}, audio(), GateConfig())
    assert tier == "gold" and gate["noise_only"] and gate["reasons"] == []
    assert gate["teachers"]["a"]["wer"] is None      # no reference to score against


def test_a_lone_hallucinating_teacher_does_not_reject_a_noise_row():
    """whisper says "Thanks for watching!" over dead air; the CTC teacher does not."""
    tier, gate = evaluate_row(row("", []),
                              {"a": "Thanks for watching!", "b": ""},
                              audio(), GateConfig())
    assert tier == "gold"
    assert gate["noise_speech_heard_by"] == ["a"]


def test_teachers_agreeing_they_heard_speech_rejects_a_noise_row():
    tier, gate = evaluate_row(
        row("", []), {"a": "csa one two three cleared to land",
                      "b": "csa one two three cleared to land"},
        audio(), GateConfig())
    assert tier == "rejected" and gate["reasons"] == ["noise_row_has_speech"]


def test_strict_noise_mode_rejects_on_a_single_teacher():
    tier, _ = evaluate_row(row("", []), {"a": "Thanks for watching!", "b": ""},
                           audio(), GateConfig(noise_requires_consensus=False))
    assert tier == "rejected"


def test_short_hallucinations_are_under_the_word_allowance():
    tier, _ = evaluate_row(row("", []), {"a": "you", "b": "The End"},
                           audio(), GateConfig())
    assert tier == "gold"


# --------------------------------------------------------------------------
# audio validity
# --------------------------------------------------------------------------

def test_invalid_audio_rejects_whatever_the_teachers_say():
    tier, gate = evaluate_row(row(), {"a": REF_TEXT, "b": REF_TEXT},
                              audio(ok=False, qc_ok=False, qc_reason="clipping"),
                              GateConfig())
    assert tier == "rejected" and gate["reasons"] == ["audio_clipping"]


def test_a_repeated_segment_rejects_a_perfect_transcription():
    tier, gate = evaluate_row(row(), {"a": REF_TEXT, "b": REF_TEXT},
                              audio(ok=False, repeat_score=0.97,
                                    repeated_segment=True), GateConfig())
    assert tier == "rejected" and gate["reasons"] == ["audio_repeated_segment"]


def test_repeat_score_separates_looped_audio_from_speech():
    rng = np.random.default_rng(0)
    sr = 16000
    # amplitude-modulated noise: a speech-like, aperiodic envelope
    n = sr * 3
    envelope = np.abs(rng.normal(size=n // 800).repeat(800))[:n]
    clip = (rng.normal(size=n) * envelope).astype(np.float32)

    assert repeat_score(clip, sr) < 0.8
    assert repeat_score(np.concatenate([clip, clip]), sr) > 0.8


def test_audio_checks_reuse_the_tier_zero_gates():
    sr = 16000
    silence = np.zeros(sr, np.float32)
    result = audio_checks(silence, sr, GateConfig())
    assert not result["ok"] and result["qc_reason"] == "silence"

    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    assert audio_checks(tone, sr, GateConfig())["qc_ok"]


def test_a_clip_too_short_to_score_is_not_called_repetitive():
    assert repeat_score(np.zeros(100, np.float32), 16000) == 0.0


# --------------------------------------------------------------------------
# select_tiers and the adversarial cap
# --------------------------------------------------------------------------

def tiered(**counts):
    rows = []
    for tier, count in counts.items():
        rows.extend({"audio": f"{tier}{i}.wav", "tier": tier} for i in range(count))
    return rows


def test_select_tiers_filters_by_tier():
    rows = tiered(gold=3, silver=2, rejected=4)
    assert len(select_tiers(rows)) == 3
    assert len(select_tiers(rows, tiers=("gold", "silver"))) == 5
    assert all(r["tier"] != "rejected" for r in select_tiers(rows, tiers=("gold", "silver")))


def test_adversarial_is_capped_at_five_percent_of_the_mix():
    mix = select_tiers(tiered(gold=95, adversarial=50), tiers=("gold", "adversarial"))
    adversarial = [r for r in mix if r["tier"] == "adversarial"]
    assert len(adversarial) / len(mix) <= 0.05
    assert len(adversarial) == 5 and len(mix) == 100


def test_the_cap_scales_with_the_rest_of_the_mix():
    for n_other in (20, 200, 1000):
        mix = select_tiers(tiered(gold=n_other, adversarial=n_other),
                           tiers=("gold", "adversarial"), adversarial_cap=0.05)
        share = sum(1 for r in mix if r["tier"] == "adversarial") / len(mix)
        assert share <= 0.05
        assert share > 0.04          # and the cap is reached, not merely respected


def test_a_zero_cap_excludes_adversarial_entirely():
    mix = select_tiers(tiered(gold=10, adversarial=10),
                       tiers=("gold", "adversarial"), adversarial_cap=0.0)
    assert len(mix) == 10 and all(r["tier"] == "gold" for r in mix)


def test_select_tiers_preserves_manifest_order():
    rows = [{"tier": "gold", "i": 0}, {"tier": "rejected", "i": 1},
            {"tier": "silver", "i": 2}, {"tier": "gold", "i": 3}]
    assert [r["i"] for r in select_tiers(rows, tiers=("gold", "silver"))] == [0, 2, 3]


def test_select_tiers_rejects_an_unknown_tier():
    with pytest.raises(ValueError, match="unknown tier"):
        select_tiers([], tiers=("platinum",))


# --------------------------------------------------------------------------
# whole-dataset pass
# --------------------------------------------------------------------------

@pytest.fixture
def built_set(tmp_path):
    """A three-row manifest with real wavs: one clean, one noise, one clipped."""
    import soundfile as sf

    sr = 16000
    root = tmp_path / "set"
    (root / "wavs").mkdir(parents=True)
    rng = np.random.default_rng(1)
    speech = (0.1 * rng.normal(size=sr * 2)).astype(np.float32)
    clipped = np.ones(sr * 2, np.float32)
    rows = [
        {"audio": "wavs/0.wav", "text": REF_TEXT, "duration": 2.0,
         "entities": entities_to_dicts(REF_ENTITIES), "gen": {"mode": "procedural"}},
        {"audio": "wavs/1.wav", "text": "", "duration": 2.0, "entities": []},
        {"audio": "wavs/2.wav", "text": REF_TEXT, "duration": 2.0,
         "entities": entities_to_dicts(REF_ENTITIES)},
    ]
    for index, wav in enumerate((speech, speech, clipped)):
        sf.write(root / f"wavs/{index}.wav", wav, sr)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_gate_dataset_writes_every_row_with_a_tier_and_stats(built_set):
    teachers = [FakeTeacher("seq2seq", [REF_TEXT, "", REF_TEXT]),
                FakeTeacher("ctc", [REF_TEXT, "", REF_TEXT])]
    stats = gate_dataset(built_set, teachers=teachers, batch_size=2, progress=False)

    gated = load_gated(built_set)
    assert len(gated) == 3                       # nothing is dropped, ever
    assert [r["tier"] for r in gated] == ["gold", "gold", "rejected"]
    assert gated[2]["gate"]["reasons"] == ["audio_clipping"]
    assert gated[0]["text"] == REF_TEXT and gated[0]["gen"] == {"mode": "procedural"}

    assert stats["tiers"] == {"gold": 2, "silver": 0, "adversarial": 0, "rejected": 1}
    assert stats["tier_fractions"]["gold"] == pytest.approx(2 / 3, abs=1e-3)
    assert stats["rejection_reasons"] == {"audio_clipping": 1}
    assert stats["teachers"] == ["seq2seq", "ctc"]
    assert stats["throughput"]["clips"] == 3
    assert json.loads((built_set / "gate_stats.json").read_text())["n_samples"] == 3
    assert teachers[0].batches == [2, 1]         # batched, not one clip at a time


def test_gate_dataset_honours_max_samples_and_an_out_dir(built_set, tmp_path):
    out = tmp_path / "elsewhere"
    teachers = [FakeTeacher("seq2seq", [REF_TEXT]), FakeTeacher("ctc", [REF_TEXT])]
    stats = gate_dataset(built_set, out, teachers=teachers, max_samples=2,
                         batch_size=8, progress=False)

    assert stats["n_samples"] == 2
    assert not (built_set / "manifest_gated.jsonl").exists()
    assert len(load_gated(out)) == 2


def test_gated_rows_round_trip_through_json(built_set):
    teachers = [FakeTeacher("seq2seq", [REF_TEXT]), FakeTeacher("ctc", [REF_TEXT])]
    gate_dataset(built_set, teachers=teachers, batch_size=8, progress=False)

    gated = load_gated(built_set / "manifest_gated.jsonl")
    assert json.loads(json.dumps(gated)) == gated
    blob = gated[0]["gate"]
    assert set(blob) >= {"teachers", "audio", "best_wer", "best_teacher",
                         "entities", "reasons", "noise_only"}
    assert set(blob["teachers"]["ctc"]) == {"hyp", "hyp_normalized", "words",
                                            "wer", "entities", "asserted"}


def test_retier_reapplies_thresholds_without_touching_the_teachers(built_set):
    teachers = [FakeTeacher("seq2seq", [REF_TEXT]), FakeTeacher("ctc", [REF_TEXT])]
    gate_dataset(built_set, teachers=teachers, batch_size=8, progress=False)
    calls = [t.batches[:] for t in teachers]

    gated = load_gated(built_set)
    assert [r["tier"] for r in retier(gated, GateConfig())] == \
           [r["tier"] for r in gated]
    strict = retier(gated, GateConfig(gold_wer=0.0, gold_critical_recall=1.0))
    assert [t.batches for t in teachers] == calls      # no new inference
    assert {r["tier"] for r in strict} <= {"gold", "silver", "rejected"}


def test_gate_stats_reports_entity_rates_by_type():
    rows = [
        {"tier": "gold", "gate": {"noise_only": False, "best_wer": 0.1,
         "best_teacher": "a", "reasons": [],
         "teachers": {"a": {"wer": 0.1}, "b": {"wer": 0.4}},
         "entities": {"verdicts": [
             {"type": "callsign", "verdict": "verified"},
             {"type": "frequency", "verdict": "substituted"}]}}},
        {"tier": "rejected", "gate": {"noise_only": False, "best_wer": 0.95,
         "best_teacher": "b", "reasons": ["teacher_wer_above_ceiling"],
         "teachers": {"a": {"wer": 0.99}, "b": {"wer": 0.95}},
         "entities": {"verdicts": [{"type": "callsign", "verdict": "missed"}]}}},
    ]
    stats = gate_stats(rows, GateConfig(), Throughput(clips=2, seconds=1.0), ["a", "b"])

    by_type = stats["entities"]["by_type"]
    assert by_type["callsign"] == {"total": 2, "verified": 1, "substituted": 0,
                                   "missed": 1, "verified_rate": 0.5,
                                   "substitution_rate": 0.0, "critical": True}
    assert by_type["frequency"]["substitution_rate"] == 1.0
    assert stats["entities"]["critical_substitution_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["teacher_wer"]["a"]["n"] == 2
    assert stats["best_teacher_counts"] == {"a": 1, "b": 1}
    assert stats["rejection_reasons"] == {"teacher_wer_above_ceiling": 1}
    assert stats["config"]["silver_wer"] == 0.5


# --------------------------------------------------------------------------
# integration: the real seq2seq teacher on two built clips
# --------------------------------------------------------------------------

@pytest.mark.skipif(not (FIXTURE / "manifest.jsonl").exists(),
                    reason="build runs/smoke_gate_fixture first (see the module docstring "
                           "of scripts/gate_dataset.py)")
def test_real_whisper_teacher_gates_two_clips():
    from atcgen.gate.teachers import WhisperTeacher

    stats = gate_dataset(FIXTURE, teachers=[WhisperTeacher()], max_samples=2,
                         batch_size=2, progress=False)
    gated = load_gated(FIXTURE)

    assert stats["n_samples"] == 2 and len(gated) == 2
    assert all(r["tier"] in ("gold", "silver", "adversarial", "rejected") for r in gated)
    assert stats["throughput"]["clips_per_sec"] > 0
    for gated_row in gated:
        item = gated_row["gate"]["teachers"]["whisper-base.en"]
        assert isinstance(item["hyp"], str)
        assert item["wer"] is None or 0.0 <= item["wer"]
