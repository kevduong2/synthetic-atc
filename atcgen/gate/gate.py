"""The verification gate: does the audio provably say what the label claims?

Research-findings §4.4 and decision D8.  Every candidate sample faces a panel
of frozen teachers before it is allowed to train or reward anything, and the
verdict is **reject, never auto-relabel** — a sample whose label the panel
cannot confirm is dropped, not quietly rewritten to whatever the teachers
heard.  Relabelling would launder generator errors into ground truth and teach
the student the generator's mistakes; that is the failure mode the published
word-aligned filtering result (cross-domain WER 13.54% -> 6.89%, ~34% fewer
training steps) is measured against.

The gate runs on the *built* dataset — post-channel audio, the same waveform
the student will train on, not the clean TTS render.  Gating the clean render
would only prove Kokoro can pronounce; the interesting failure is the channel
destroying a digit.

Four tiers come out, and every row is preserved so downstream can choose:

    gold          the panel reads it nearly cleanly and enough of its critical
                  slots are independently recovered.  Default training material.
    silver        the label is broadly right (best teacher under the published
                  50% WER floor) and nothing critical was misheard as a
                  *different* value.  Usable, weaker evidence.
    adversarial   hard but still provable: the channel crushed the words (WER
                  above the floor) yet the safety numbers came through anyway.
                  In practice these are clips where the flight level or heading
                  is recovered exactly and only the callsign is mush.  Precious
                  and dangerous — capped at 5% of any training mix by
                  `select_tiers`.
    rejected      everything else, with a machine-readable reason list.

"Enough critical slots" is `GateConfig.gold_critical_recall` /
`adversarial_critical_recall` rather than a flat "all of them"; that field's
comment carries the measurement behind the default.

Entity verdicts are the part that does not reduce to WER.  A critical slot is
*verified* when any teacher recovers its exact type and value: one clean read
is proof the audio carries it.  It is *substituted* when every teacher that
heard that slot type at all heard a value the label does not contain anywhere
— unanimous disagreement is semantic ambiguity, not teacher noise, and D8
rejects it.  A slot no teacher produced at all is merely *missed*: absence of
evidence, which costs recall but proves nothing against the label.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from ..entities import (
    CRITICAL_TYPES,
    Entity,
    entities_from_dicts,
    entities_to_dicts,
    extract_entities,
)
from ..eval.qc import QCConfig as QCGates
from ..eval.qc import normalize_atc, qc_sample
from .teachers import Teacher, Throughput, default_teachers, timed

TIERS = ("gold", "silver", "adversarial", "rejected")
TRAINABLE_TIERS = ("gold", "silver", "adversarial")


@dataclass
class GateConfig:
    """Every threshold the gate applies, in one place.

    The WER numbers start from the published coarse floor — discard above 50%
    teacher WER — with `gold_wer` drawn tighter so the default training pool
    is samples the panel reads nearly cleanly, and `adversarial_wer` opening a
    band above the floor for clips that are hard *and* verified.
    """

    gold_wer: float = 0.25
    silver_wer: float = 0.50            # the published floor
    adversarial_wer: float = 0.90
    #: Share of a sample's critical slots that some teacher must recover
    #: exactly.  1.0 demands every one of them, which is the bar to want and
    #: not the bar these teachers can clear: measured on 60 matched-profile
    #: clips, per-slot critical verification runs ~0.28, so with the ~2
    #: critical slots a transmission carries, "all verified" lands near 0.28^2
    #: and gold collapses to 8% — a number set by how well generic ASR reads
    #: ATC callsigns, not by whether the labels are right.  Half the slots,
    #: with nothing contradicted, keeps the tier evidence-based without
    #: demanding unanimity from weak judges.  Raise both toward 1.0 when D4's
    #: stronger teachers land.
    gold_critical_recall: float = 0.5
    adversarial_critical_recall: float = 0.5
    #: Words a teacher may emit on a noise-only row before it counts as
    #: hearing speech.
    noise_max_words: int = 2
    #: ...and whether *every* teacher has to hear speech before the row is
    #: called mislabelled.  It does, because a lone seq2seq teacher talking
    #: over dead air is the best-documented hallucination in ASR, not
    #: evidence: on 12 all-noise matched clips, whisper-base.en produced
    #: "you", "The End" and "Thanks for watching!" on every one while
    #: wav2vec2 — CTC, no decoder LM to confabulate with — produced "" or a
    #: single stray letter on all 12.  Trusting either teacher alone would
    #: reject 42% of the noise arm *because* a teacher hallucinated, which
    #: throws away precisely the anti-hallucination samples the arm exists to
    #: create.  Set False for the strict single-teacher reading.
    noise_requires_consensus: bool = True
    #: Envelope-autocorrelation peak above which a clip looks looped.  0.80
    #: sits in an empty gap: over the 60-clip matched smoke set, honest clips
    #: top out at 0.72 and the same clips concatenated with themselves bottom
    #: out at 0.85, with no error in either direction.
    repeat_threshold: float = 0.80
    repeat_min_lag_sec: float = 0.25
    repeat_frame_sec: float = 0.02
    #: Share of a training mix `select_tiers` lets the adversarial tier hold.
    adversarial_cap: float = 0.05
    # Tier-0 audio validity, mirrored onto `atcgen.eval.qc`
    min_duration: float = 0.5
    max_duration: float = 30.0
    max_clip_frac: float = 0.01
    min_rms_db: float = -40.0
    max_rms_db: float = -8.0

    def audio_gates(self) -> QCGates:
        """The audio half as `atcgen.eval.qc` gate thresholds (no ASR gate)."""
        return QCGates(min_duration=self.min_duration, max_duration=self.max_duration,
                       max_clip_frac=self.max_clip_frac, min_rms_db=self.min_rms_db,
                       max_rms_db=self.max_rms_db, asr_gate=False)

    def to_dict(self) -> dict:
        return {item.name: getattr(self, item.name) for item in fields(self)}


# ---------------------------------------------------------------------------
# Audio validity
# ---------------------------------------------------------------------------

def repeat_score(wav: np.ndarray, sr: int, frame_sec: float = 0.02,
                 min_lag_sec: float = 0.25) -> float:
    """How periodic the clip's loudness envelope is, in [0, 1]-ish.

    A generator that loops a segment leaves the giveaway in the envelope, not
    the spectrum: the same rise-and-fall recurs at a fixed lag.  Speech is
    aperiodic at quarter-second lags and up, so the normalized
    autocorrelation peak over that range stays low for honest clips and
    approaches 1 for a clip that is literally the same audio twice.
    """
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    n_frame = max(1, int(sr * frame_sec))
    n_frames = len(x) // n_frame
    if n_frames < 8:
        return 0.0
    frames = x[:n_frames * n_frame].reshape(n_frames, n_frame)
    envelope = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    envelope = envelope - envelope.mean()
    if not np.any(envelope):
        return 0.0
    min_lag = max(1, round(min_lag_sec / frame_sec))
    max_lag = n_frames // 2          # a loop worth catching spans half the clip
    if max_lag <= min_lag:
        return 0.0

    # Normalize each lag by the energy of the two overlapping halves, not by
    # the whole signal: dividing by total energy shrinks the score in
    # proportion to the lag, so a clip that is literally itself twice — which
    # peaks at exactly lag n/2 — could never score above 0.5 and sat below the
    # p99 of honest speech. Per-overlap normalization makes it 1.0.
    lags = np.arange(min_lag, max_lag + 1)
    correlation = np.correlate(envelope, envelope, mode="full")[n_frames - 1:]
    prefix = np.concatenate([[0.0], np.cumsum(envelope ** 2)])
    head = prefix[n_frames - lags]
    tail = prefix[n_frames] - prefix[lags]
    scale = np.sqrt(head * tail)
    valid = scale > 0
    if not np.any(valid):
        return 0.0
    return float(np.max(correlation[lags][valid] / scale[valid]))


def audio_checks(wav: np.ndarray, sr: int, config: GateConfig) -> dict:
    """Tier-0 QC plus the repeated-segment check, as one verdict blob."""
    result = qc_sample(wav, sr, None, config.audio_gates())
    score = repeat_score(wav, sr, config.repeat_frame_sec, config.repeat_min_lag_sec)
    repeated = score > config.repeat_threshold
    return {
        "ok": bool(result.ok and not repeated),
        "qc_ok": bool(result.ok),
        "qc_reason": result.reason,
        "repeat_score": round(score, 4),
        "repeated_segment": bool(repeated),
        "metrics": {name: round(float(value), 4)
                    for name, value in result.metrics.items()},
    }


# ---------------------------------------------------------------------------
# Entity verification
# ---------------------------------------------------------------------------

def verify_entities(ref: list[Entity],
                    hyp_by_teacher: dict[str, list[Entity]],
                    asserted_by_teacher: dict[str, list[Entity]] | None = None) -> dict:
    """Per-slot verdicts for one sample: verified / substituted / missed.

    Verification is *any* teacher, substitution is *every* teacher that heard
    the slot type — the asymmetry is the point.  One teacher reading the
    frequency correctly proves the audio carries it, whatever the other one
    did with the surrounding words; but a value only counts as misheard when
    no judge that engaged with the slot agreed with the label, which is what
    separates semantic ambiguity from one model's bad day.

    `asserted_by_teacher` carries the same asymmetry one level down.  Positive
    evidence may come from either rendering of a hypothesis (see
    `hypothesis_entities`), but *negative* evidence has to come from what the
    teacher actually wrote, because our own re-spelling invents disagreements:
    Whisper's "flight level 3.0" normalizes to "three zero" and parses to
    FL030, which reads as a teacher contradicting an FL340 label when all the
    teacher really did was render a number ambiguously.  Defaults to
    `hyp_by_teacher`, i.e. both renderings count against the label.
    """
    asserted_by_teacher = asserted_by_teacher or hyp_by_teacher
    ref_values: dict[str, set[str]] = {}
    for entity in ref:
        ref_values.setdefault(entity.type, set()).add(entity.value)

    remaining = {name: Counter(entity.key for entity in entities)
                 for name, entities in hyp_by_teacher.items()}
    #: Values a teacher heard for a type that the label does not contain at
    #: all — the evidence of a genuine mishearing rather than a re-ordering.
    stray: dict[str, dict[str, set[str]]] = {}
    produced: dict[str, set[str]] = {}
    for name in hyp_by_teacher:
        entities = asserted_by_teacher.get(name, [])
        produced[name] = {entity.type for entity in entities}
        strayed: dict[str, set[str]] = {}
        for entity in entities:
            if entity.value not in ref_values.get(entity.type, ()):
                strayed.setdefault(entity.type, set()).add(entity.value)
        stray[name] = strayed

    verdicts = []
    for entity in ref:
        verified_by = [name for name, counts in remaining.items()
                       if counts[entity.key] > 0]
        for name in verified_by:
            remaining[name][entity.key] -= 1
        producers = [name for name in hyp_by_teacher if entity.type in produced[name]]
        if verified_by:
            verdict, heard = "verified", {}
        elif producers and all(stray[name].get(entity.type) for name in producers):
            verdict = "substituted"
            heard = {name: sorted(stray[name][entity.type]) for name in producers}
        else:
            verdict, heard = "missed", {}
        verdicts.append({"type": entity.type, "value": entity.value,
                         "critical": bool(entity.critical), "verdict": verdict,
                         "verified_by": verified_by, "heard": heard})

    critical = [item for item in verdicts if item["critical"]]
    counts = Counter(item["verdict"] for item in critical)
    return {
        "verdicts": verdicts,
        "critical_total": len(critical),
        "critical_verified": counts["verified"],
        "critical_substituted": counts["substituted"],
        "critical_missed": counts["missed"],
        # a sample with no critical slots has nothing to fail: recall is 1.0
        "critical_recall": round(counts["verified"] / len(critical), 4) if critical else 1.0,
        "all_critical_verified": counts["verified"] == len(critical),
        "any_critical_substitution": counts["substituted"] > 0,
    }


# ---------------------------------------------------------------------------
# Per-sample verdict
# ---------------------------------------------------------------------------

def _wer(ref: str, hyp: str) -> float:
    import jiwer

    return float(jiwer.wer(ref, hyp))


def hypothesis_entities(raw: str, normalized: str,
                        airlines: dict[str, str] | None = None
                        ) -> tuple[list[Entity], list[Entity]]:
    """Entities a teacher's hypothesis supports, parsed from both renderings.

    The parser is spoken-form-first but stock ASR writes numerals, and the two
    renderings lose different slots:

        "Squawk 471.1"   raw -> nothing ("471.1" is not a number run)
                         normalized -> "squawk four seven one one" -> 4711
        "contact 127.825"  raw -> frequency 127.825
                           normalized -> "one two seven eight two five", and
                           `normalize_atc` has dropped the decimal point the
                           frequency rule anchors on -> nothing

    So neither is sufficient and the union is taken.  That is safe because
    `extract_entities` favours precision — every slot needs an anchor word and
    passes `check_value` — and it is the conservative direction for a gate
    that *rejects*: more recovered slots can only turn a rejection into a
    verification, never the other way round.

    The second return value is the raw rendering's slots alone, which is what
    `verify_entities` weighs *against* a label; see its docstring.
    """
    asserted = extract_entities(raw or "", airlines)
    merged: dict[tuple[str, str], Entity] = {entity.key: entity for entity in asserted}
    for entity in extract_entities(normalized or "", airlines):
        merged.setdefault(entity.key, entity)
    return list(merged.values()), asserted


def evaluate_row(row: dict, hypotheses: dict[str, str], audio: dict,
                 config: GateConfig | None = None,
                 airlines: dict[str, str] | None = None) -> tuple[str, dict]:
    """Tier one manifest row against the teachers' raw hypotheses.

    `row` is a built manifest record (needs `text` and, for speech, the
    ground-truth `entities`); `hypotheses` maps teacher name to raw text;
    `audio` is what `audio_checks` returned.  Returns `(tier, gate_blob)` —
    the blob is what gets written next to the row, reasons and all.
    """
    config = config or GateConfig()
    reference = normalize_atc(row.get("text") or "")
    noise_only = not reference

    teachers: dict[str, dict] = {}
    asserted: dict[str, list[Entity]] = {}
    for name, raw in hypotheses.items():
        normalized = normalize_atc(raw or "")
        entities, raw_only = (([], []) if noise_only
                              else hypothesis_entities(raw, normalized, airlines))
        asserted[name] = raw_only
        teachers[name] = {
            "hyp": (raw or "").strip(),
            "hyp_normalized": normalized,
            "words": len(normalized.split()),
            "wer": None if noise_only else round(_wer(reference, normalized), 4),
            "entities": entities_to_dicts(entities),
            # what the teacher itself wrote, before our re-spelling: the only
            # slots allowed to count *against* the label
            "asserted": entities_to_dicts(raw_only),
        }

    reasons: list[str] = []
    if not audio["qc_ok"]:
        reasons.append(f"audio_{audio['qc_reason']}")
    if audio["repeated_segment"]:
        reasons.append("audio_repeated_segment")

    gate: dict = {"teachers": teachers, "audio": audio, "noise_only": noise_only}

    if noise_only:
        # An empty label claims nobody transmitted. A teacher reading words off
        # it is either right (the row is mislabelled and must not train) or
        # hallucinating (the row is doing its job). The teachers disagreeing is
        # what tells the two apart -- see `noise_requires_consensus`.
        loud = [name for name, item in teachers.items()
                if item["words"] > config.noise_max_words]
        gate["noise_words"] = {name: item["words"] for name, item in teachers.items()}
        gate["noise_speech_heard_by"] = loud
        mislabelled = (len(loud) == len(teachers) if config.noise_requires_consensus
                       else bool(loud))
        if mislabelled and loud:
            reasons.append("noise_row_has_speech")
        gate["reasons"] = reasons
        return ("rejected" if reasons else "gold"), gate

    scored = {name: item["wer"] for name, item in teachers.items()
              if item["wer"] is not None}
    best_teacher = min(scored, key=lambda name: scored[name]) if scored else None
    best_wer = scored[best_teacher] if best_teacher else 1.0
    gate["best_teacher"] = best_teacher
    gate["best_wer"] = best_wer

    reference_entities = entities_from_dicts(row.get("entities") or [])
    entity_report = verify_entities(
        reference_entities,
        {name: entities_from_dicts(item["entities"]) for name, item in teachers.items()},
        asserted)
    gate["entities"] = entity_report

    recall = entity_report["critical_recall"]
    if entity_report["any_critical_substitution"]:
        reasons.append("critical_entity_substitution")
    if best_wer > config.adversarial_wer:
        reasons.append("teacher_wer_above_ceiling")
    elif best_wer > config.silver_wer and recall < config.adversarial_critical_recall:
        # the adversarial band only admits clips whose label is still provable
        reasons.append("hard_clip_unverified_entities")

    gate["reasons"] = reasons
    if reasons:
        return "rejected", gate
    if best_wer <= config.gold_wer and recall >= config.gold_critical_recall:
        return "gold", gate
    if best_wer <= config.silver_wer:
        return "silver", gate
    return "adversarial", gate


# ---------------------------------------------------------------------------
# Whole-dataset pass
# ---------------------------------------------------------------------------

def gate_dataset(dataset_dir: str | Path, out_dir: str | Path | None = None, *,
                 teachers: list[Teacher] | None = None,
                 config: GateConfig | None = None,
                 max_samples: int | None = None, batch_size: int = 8,
                 airlines: dict[str, str] | None = None,
                 progress: bool = True) -> dict:
    """Gate a built dataset; writes `manifest_gated.jsonl` and `gate_stats.json`.

    Audio is read and transcribed a batch at a time rather than up front: a
    4k-clip set is a couple of gigabytes of float32 and there is no reason for
    more than one batch of it to be resident.  Returns the stats dict.
    """
    import soundfile as sf
    from tqdm import tqdm

    config = config or GateConfig()
    teachers = teachers if teachers is not None else default_teachers()
    source = Path(dataset_dir)
    manifest_path = source / "manifest.jsonl" if source.is_dir() else source
    root = manifest_path.parent
    out = Path(out_dir) if out_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()
            if line.strip()]
    if max_samples is not None:
        rows = rows[:max_samples]

    throughput = Throughput()
    gated: list[dict] = []
    batches = range(0, len(rows), batch_size)
    for start in tqdm(batches, desc="gating", disable=not progress):
        chunk = rows[start:start + batch_size]
        waves, rates = [], []
        for row in chunk:
            wav, sr = sf.read(root / row["audio"], dtype="float32")
            waves.append(np.asarray(wav).reshape(-1))
            rates.append(sr)
        sr = rates[0]

        hypotheses: dict[str, list[str]] = {}
        for teacher in teachers:
            texts, elapsed = timed(teacher.transcribe, waves, sr)
            hypotheses[teacher.name] = texts
            throughput.add(teacher.name, len(waves), elapsed)
        throughput.clips += len(waves)

        for index, row in enumerate(chunk):
            audio = audio_checks(waves[index], rates[index], config)
            tier, blob = evaluate_row(
                row, {name: texts[index] for name, texts in hypotheses.items()},
                audio, config, airlines)
            gated.append({**row, "tier": tier, "gate": blob})

    # the pass costs every teacher's time, so aggregate throughput is clips
    # over the summed inference wall-clock; per-teacher rows keep the split so
    # a slow judge is visible
    throughput.seconds = sum(entry["seconds"]
                             for entry in throughput.per_teacher.values())

    gated_path = out / "manifest_gated.jsonl"
    with open(gated_path, "w") as handle:
        for row in gated:
            handle.write(json.dumps(row) + "\n")

    stats = gate_stats(gated, config, throughput,
                       [teacher.name for teacher in teachers])
    stats["manifest"] = str(gated_path)
    (out / "gate_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def retier(rows: list[dict], config: GateConfig,
           airlines: dict[str, str] | None = None) -> list[dict]:
    """Re-apply the tier rules to already-gated rows, no ASR.

    The expensive half of a gate pass is the teachers, and their hypotheses
    are in the manifest.  Sweeping thresholds is therefore free, which is the
    only honest way to pick them: run the pool once, then look at what each
    cut-off would have kept.
    """
    out = []
    for row in rows:
        gate = row["gate"]
        hypotheses = {name: item["hyp"] for name, item in gate["teachers"].items()}
        tier, blob = evaluate_row(row, hypotheses, gate["audio"], config, airlines)
        out.append({**row, "tier": tier, "gate": blob})
    return out


def gate_stats(rows: list[dict], config: GateConfig, throughput: Throughput,
               teacher_names: list[str]) -> dict:
    """Run summary: tier mix, why rows died, teacher WERs, entity recovery."""
    total = len(rows) or 1
    tiers = Counter(row["tier"] for row in rows)
    reasons: Counter = Counter()
    for row in rows:
        reasons.update(row["gate"].get("reasons", ()))

    wers: dict[str, list[float]] = {name: [] for name in teacher_names}
    best: list[float] = []
    by_type: dict[str, Counter] = {}
    for row in rows:
        gate = row["gate"]
        for name, item in gate["teachers"].items():
            if item["wer"] is not None:
                wers.setdefault(name, []).append(item["wer"])
        if gate.get("best_wer") is not None and not gate["noise_only"]:
            best.append(gate["best_wer"])
        for verdict in gate.get("entities", {}).get("verdicts", ()):
            by_type.setdefault(verdict["type"], Counter())[verdict["verdict"]] += 1

    return {
        "n_samples": len(rows),
        "config": config.to_dict(),
        "teachers": teacher_names,
        "tiers": {name: tiers.get(name, 0) for name in TIERS},
        "tier_fractions": {name: round(tiers.get(name, 0) / total, 4) for name in TIERS},
        "rejection_reasons": dict(reasons.most_common()),
        "rejection_reason_rates": {name: round(count / total, 4)
                                   for name, count in reasons.most_common()},
        "teacher_wer": {name: _distribution(values) for name, values in wers.items()},
        "best_teacher_wer": _distribution(best),
        "best_teacher_counts": dict(Counter(
            row["gate"].get("best_teacher") for row in rows
            if not row["gate"]["noise_only"])),
        "entities": _entity_stats(by_type),
        "throughput": throughput.summary(),
    }


def _entity_stats(by_type: dict[str, Counter]) -> dict:
    """Verification rates per slot type, plus the critical-slot totals."""
    per_type = {}
    for name in sorted(by_type):
        counts = by_type[name]
        total = sum(counts.values()) or 1
        per_type[name] = {
            "total": sum(counts.values()),
            "verified": counts["verified"],
            "substituted": counts["substituted"],
            "missed": counts["missed"],
            "verified_rate": round(counts["verified"] / total, 4),
            "substitution_rate": round(counts["substituted"] / total, 4),
            "critical": name in CRITICAL_TYPES,
        }
    critical = [item for name, item in per_type.items() if item["critical"]]
    total = sum(item["total"] for item in critical) or 1
    return {
        "by_type": per_type,
        "critical_total": sum(item["total"] for item in critical),
        "critical_verified_rate": round(
            sum(item["verified"] for item in critical) / total, 4),
        "critical_substitution_rate": round(
            sum(item["substituted"] for item in critical) / total, 4),
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    p10, p50, p90 = (round(float(v), 4) for v in np.percentile(array, [10, 50, 90]))
    return {"n": len(values), "mean": round(float(array.mean()), 4),
            "p10": p10, "p50": p50, "p90": p90}


# ---------------------------------------------------------------------------
# Training-mix assembly
# ---------------------------------------------------------------------------

def load_gated(path: str | Path) -> list[dict]:
    """Read a `manifest_gated.jsonl` back as rows."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest_gated.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_tiers(rows: list[dict] | str | Path, tiers: tuple[str, ...] = ("gold",),
                 adversarial_cap: float = 0.05) -> list[dict]:
    """Rows of the requested tiers, holding adversarial to `adversarial_cap`.

    The cap is research §5.6's invariant and it is enforced here rather than
    trusted to callers: adversarial samples are the ones whose audio the panel
    could barely read, and past a few percent they stop being hard positives
    and start being a second, noisier training distribution.  Order is
    preserved, and the adversarial rows kept are the first ones — deterministic
    given a manifest, which is what a reproducible mix needs.
    """
    if isinstance(rows, (str, Path)):
        rows = load_gated(rows)
    unknown = sorted(set(tiers) - set(TIERS))
    if unknown:
        raise ValueError(f"unknown tier(s): {unknown}")
    wanted = set(tiers)
    if "adversarial" not in wanted:
        return [row for row in rows if row.get("tier") in wanted]
    if not 0.0 <= adversarial_cap < 1.0:
        raise ValueError("adversarial_cap must be within [0, 1)")

    plain = wanted - {"adversarial"}
    n_other = sum(1 for row in rows if row.get("tier") in plain)
    # n_adv / (n_other + n_adv) <= cap
    allowed = int(n_other * adversarial_cap / (1.0 - adversarial_cap))
    kept = 0
    selected = []
    for row in rows:
        tier = row.get("tier")
        if tier in plain:
            selected.append(row)
        elif tier == "adversarial" and kept < allowed:
            selected.append(row)
            kept += 1
    return selected
