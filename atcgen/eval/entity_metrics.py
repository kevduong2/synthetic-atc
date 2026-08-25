"""Corpus-level entity/safety panel over aligned reference/hypothesis text.

The metric the project actually gates on. Aggregate WER and command accuracy
decouple in ATC (Helmke et al.): a model can shave WER by getting filler words
right while garbling the digits that carry the clearance, so releases read this
panel -- slot F1 per entity type, exact callsign accuracy, and above all the
critical-number substitution rate -- and never aggregate WER alone.

Pure functions over strings and `Entity` lists: nothing here loads a model or
touches the network, so it is cheap to call inside the eval harness, the
verification gate, or a test.

Scoring population
------------------
Only utterances whose *reference* carries at least one entity are scored. On
real transcripts the extractor is precision-first and fires on ~73% of
jacktol utterances; counting the other 27% would charge every hypothesis-side
extraction there as a false positive and turn the panel into a measurement of
the parser's recall instead of the model's fidelity. Synthetic rows carry
grammar-emitted ground truth (`entities` in the manifest) and should pass it in
via `ref_entities` -- a label is never re-parsed when it is already known. What
the exclusion hides is reported anyway, under `unscored`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..entities import (
    Entity,
    aggregate,
    entities_from_dicts,
    extract_entities,
    score_entities,
)


@dataclass
class EntityDiff:
    """Which slots went wrong in one utterance, not just how many.

    Pairing mirrors `score_entities`: equal `(type, value)` slots match as a
    multiset, then leftovers of the same type pair up positionally into
    substitutions (the dangerous class -- the model heard a value of the right
    kind and got it wrong), and whatever is still unpaired is a miss or a
    spurious extraction.
    """

    substitutions: list[tuple[Entity, Entity]] = field(default_factory=list)
    missed: list[Entity] = field(default_factory=list)
    spurious: list[Entity] = field(default_factory=list)

    @property
    def critical_substitutions(self) -> int:
        return sum(1 for reference, _ in self.substitutions if reference.critical)

    def to_dict(self) -> dict:
        return {
            "substitutions": [
                {"type": reference.type, "ref": reference.value,
                 "hyp": hypothesis.value, "critical": bool(reference.critical),
                 "ref_spoken": reference.spoken, "hyp_spoken": hypothesis.spoken}
                for reference, hypothesis in self.substitutions
            ],
            "missed": [
                {"type": entity.type, "value": entity.value,
                 "critical": bool(entity.critical)} for entity in self.missed
            ],
            "spurious": [
                {"type": entity.type, "value": entity.value,
                 "critical": bool(entity.critical)} for entity in self.spurious
            ],
        }


def diff_entities(reference: list[Entity], hypothesis: list[Entity]) -> EntityDiff:
    """Slot-level disagreement between one reference and one hypothesis."""
    diff = EntityDiff()
    by_type: dict[str, tuple[list[Entity], list[Entity]]] = {}
    for entity in reference:
        by_type.setdefault(entity.type, ([], []))[0].append(entity)
    for entity in hypothesis:
        by_type.setdefault(entity.type, ([], []))[1].append(entity)

    for refs, hyps in by_type.values():
        remaining_ref, remaining_hyp = list(refs), list(hyps)
        for entity in list(remaining_ref):
            for candidate in remaining_hyp:
                if candidate.value == entity.value:
                    remaining_hyp.remove(candidate)
                    remaining_ref.remove(entity)
                    break
        paired = min(len(remaining_ref), len(remaining_hyp))
        diff.substitutions.extend(zip(remaining_ref[:paired], remaining_hyp[:paired]))
        diff.missed.extend(remaining_ref[paired:])
        diff.spurious.extend(remaining_hyp[paired:])
    return diff


def _as_entities(labels: list[Entity] | list[dict] | None) -> list[Entity] | None:
    """Accept ground truth as `Entity` objects or as manifest dicts."""
    if labels is None:
        return None
    if labels and isinstance(labels[0], dict):
        return entities_from_dicts(labels)  # type: ignore[arg-type]
    return list(labels)  # type: ignore[arg-type]


def resolve_ref_entities(references: list[str], ref_entities: list | None = None,
                         *, airlines: dict[str, str] | None = None
                         ) -> list[list[Entity]]:
    """Reference ground truth per utterance: given labels, else parsed.

    `ref_entities[i]` may be a list of `Entity`, a list of manifest dicts, or
    `None` for "no label, parse it". Synthetic rows carry grammar-emitted
    entities; real corpus rows do not.
    """
    if ref_entities is not None and len(ref_entities) != len(references):
        raise ValueError("ref_entities must have the same length as references")
    resolved = []
    for index, reference in enumerate(references):
        labels = _as_entities(ref_entities[index]) if ref_entities else None
        resolved.append(labels if labels is not None
                        else extract_entities(reference, airlines))
    return resolved


def entity_panel(references: list[str], hypotheses: list[str],
                 ref_entities: list | None = None, *,
                 airlines: dict[str, str] | None = None,
                 max_examples: int = 5) -> dict:
    """The JSON-ready entity panel for a corpus of aligned transcripts.

    `ref_entities[i]` is the ground truth for utterance `i` -- a list of
    `Entity` or of manifest dicts -- and `None` (or a missing list) means "parse
    the reference". Hypotheses are always parsed, with the *same* airline table
    as the references, otherwise a harvested carrier gets one ICAO code on one
    side and another on the other and every callsign scores as a substitution.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    resolved = resolve_ref_entities(references, ref_entities, airlines=airlines)

    scores, examples = [], []
    skipped, unscored_hyp_entities = 0, 0
    for index, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
        labels = resolved[index]
        found = extract_entities(hypothesis, airlines)
        if not labels:
            skipped += 1
            unscored_hyp_entities += len(found)
            continue
        scores.append(score_entities(labels, found))
        diff = diff_entities(labels, found)
        if diff.substitutions or diff.missed or diff.spurious:
            examples.append((index, reference, hypothesis, diff))

    total = aggregate(scores)
    per_type = total.per_type()
    for counts in per_type.values():
        counts["support"] = counts["tp"] + counts["fn"]

    examples.sort(key=lambda item: (item[3].critical_substitutions,
                                    len(item[3].substitutions),
                                    len(item[3].missed) + len(item[3].spurious)),
                  reverse=True)
    return {
        "utterances": {
            "total": len(references),
            "scored": len(scores),
            "skipped_no_ref_entities": skipped,
        },
        # what the scoring population hides: entities the model produced in
        # utterances whose reference had none. Mostly parser recall, partly
        # invention -- worth watching, not worth charging as precision.
        "unscored": {
            "utterances": skipped,
            "hypothesis_entities": unscored_hyp_entities,
        },
        "overall": {
            "precision": total.precision,
            "recall": total.recall,
            "f1": total.f1,
            "tp": sum(total.tp.values()),
            "fp": sum(total.fp.values()),
            "fn": sum(total.fn.values()),
            "substitutions": sum(total.sub.values()),
        },
        "callsign": {
            "accuracy": total.callsign_accuracy,
            "correct": total.callsign_correct,
            "total": total.callsign_total,
            "substitutions": total.sub.get("callsign", 0),
        },
        "critical": {
            "substitution_rate": total.critical_substitution_rate,
            "substitutions": total.critical_sub,
            "reference_slots": total.critical_ref,
        },
        "per_type": per_type,
        "examples": [
            {"index": index, "reference": reference, "hypothesis": hypothesis,
             "critical_substitutions": diff.critical_substitutions,
             **diff.to_dict()}
            for index, reference, hypothesis, diff in examples[:max_examples]
        ],
    }
