"""Harvest scenario vocabulary from the real-train split (D10 real anchor).

Reads `jacktol/atc-dataset` train[0:8000] and *only* that slice: reward-val,
model-selection and the locked test set stay untouched, so anything the
grammar learns here cannot leak into evaluation.

Counts three things in spoken-form transcripts:
  airlines   tokens/bigrams immediately preceding a callsign ident run
  stations   the facility phrase around "contact"/"radar"/"approach"/...
  waypoints  the name after "direct"/"overhead"/"abeam"/"via"

Writes data/vocab/real_anchor.json (data/ is gitignored -- the file is a
regenerable artifact). The grammar loads it when present and falls back to
its builtin lexicon when it is absent.

    uv run python scripts/harvest_vocab.py [--limit 8000] [--min-count 3]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from string import ascii_uppercase
from pathlib import Path

from atcgen.entities import (DEFAULT_AIRLINES, DEFAULT_VOCAB_PATH, DIGIT_WORDS,
                             LETTER_WORDS, MULTIPLIER_WORDS, TEEN_WORDS,
                             TENS_WORDS, scan_number, telephony_code, tokenize)

CORPUS = "jacktol/atc-dataset"
REAL_TRAIN_END = 8000       # D11 split discipline: never look past this

FACILITY_WORDS = ("radar", "approach", "tower", "ground", "apron", "control",
                  "director", "delivery", "arrival", "departure", "information")
WAYPOINT_ANCHORS = ("direct", "overhead", "abeam", "via")

#: Phraseology that sits next to numbers or facilities but is not a name.
STOPWORDS = frozenset("""
a about above affirm affirmative after afternoon again ahead airbus airborne all also
and antonov any approach approved apron are at altitude available back be behind
below boeing
bye call called can cancel change check cleared clear climb climbing close come
comming coming confirm contact continue control copied copy correct correction
could cross crossing day degrees delivery depart departure descend descending
cessna embraer fokker
direct disregard do down due east en end established etc evening expect feet
final flight flying follow for from further gate get give go going good goodbye
airways break fato fraction localizer maximum munich peed touchdown
ground gusting had have heading hello here hold holding hour how identified if
ils immediate in inbound information initial is it just keep kilometers knots
land landing later leave leaving left level like line long look looking
maintain maintaining make may maybe me meters miles minimum minutes more
morning most much my need negative never new next night no not now number of
off ok okay on one only or other our out outbound over passing per please point
position present proceed qnh radar rate readback ready receive received reduce
reducing regards remain report request requested requesting resume right roger
route routing runway safe say see selected send sequence service short should
side sir slow so some soon speed squawk squak stand standby start stop straight
takeoff taxi taxiway thank thanks that the then there this three threshold
through time to today tower traffic turn two understand until up us via
vectors very visual wait we weather welcome well were west what when where
which while will wind with without would yes yet you your zone
""".split()) | set(DIGIT_WORDS) | set(TEEN_WORDS) | set(TENS_WORDS) \
    | set(MULTIPLIER_WORDS) | set(LETTER_WORDS) | {"decimal"}


def _ident_length(tokens: list[str], start: int) -> int:
    """Length in tokens of the number run starting at `start` (0 if none)."""
    run = scan_number(tokens, start)
    return run.end - run.start if run else 0


def harvest(texts: list[str]) -> dict[str, Counter]:
    airlines: Counter[str] = Counter()
    stations: Counter[str] = Counter()
    waypoints: Counter[str] = Counter()

    for text in texts:
        tokens = tokenize(text)
        for position, token in enumerate(tokens):
            # airline candidates: name(s) directly before a >=2 group ident.
            # A frequency read-out looks the same from here, so runs that feed
            # a "decimal" and anything right after "contact" are skipped.
            run_length = _ident_length(tokens, position)
            is_frequency = tokens[position + run_length:position + run_length + 1] \
                and tokens[position + run_length] in ("decimal", "point")
            if run_length >= 2 and not is_frequency and position >= 1 \
                    and tokens[position - 1] not in STOPWORDS \
                    and tokens[position - 2:position - 1] != ["contact"]:
                airlines[tokens[position - 1]] += 1
                if position >= 2 and tokens[position - 2] not in STOPWORDS:
                    airlines[f"{tokens[position - 2]} {tokens[position - 1]}"] += 1
            # station: "<name> radar", and whatever follows "contact"
            if token in FACILITY_WORDS and position >= 1 \
                    and tokens[position - 1] not in STOPWORDS \
                    and len(tokens[position - 1]) >= 4 \
                    and tokens[position - 1].isalpha():
                stations[f"{tokens[position - 1]} {token}"] += 1
            if token == "contact" and position + 1 < len(tokens):
                name = tokens[position + 1]
                if name in FACILITY_WORDS:
                    stations[name] += 1
            if token in WAYPOINT_ANCHORS:
                name_at = position + 2 if tokens[position + 1:position + 2] == ["to"] \
                    else position + 1
                name = tokens[name_at] if name_at < len(tokens) else ""
                if name.isalpha() and len(name) >= 4 and name not in STOPWORDS:
                    waypoints[name] += 1

    return {"airlines": airlines, "stations": stations, "waypoints": waypoints}


def _prune_airlines(airlines: Counter[str], stations: Counter[str]) -> Counter[str]:
    """Keep the phrases that behave like telephony names.

    Drops the unigram once its bigram is established ('travel' vs 'sky
    travel'), place names already claimed as facilities ('praha'), and
    fragments of a builtin name that survive a clipped transmission ('bird'
    out of 'speedbird').
    """
    bigrams = {name: count for name, count in airlines.items() if " " in name}
    place_words = {word for station in stations for word in station.split()}
    pruned = Counter()
    for name, count in airlines.items():
        if " " in name:
            if any(word in place_words for word in name.split()):
                continue
        else:
            if name in place_words or len(name) < 4 or name.endswith("ing"):
                continue
            if any(bigram.endswith(f" {name}") and bigram_count >= 0.5 * count
                   for bigram, bigram_count in bigrams.items()):
                continue
            if any(known != name and known.endswith(name)
                   for known in DEFAULT_AIRLINES):
                continue
        pruned[name] = count
    return pruned


def _assign_codes(airlines: dict[str, int]) -> dict[str, dict]:
    """Attach an ICAO designator to each telephony name, keeping codes unique.

    Two harvested names must never share a code: the metrics compare codes,
    so a collision would score a genuine carrier confusion as correct.
    """
    out: dict[str, dict] = {}
    taken: set[str] = set()
    for name, count in airlines.items():
        code = DEFAULT_AIRLINES.get(name) or telephony_code(name)
        if code in taken:
            base = code[:2]
            code = next((base + suffix for suffix in ascii_uppercase
                         if base + suffix not in taken), code)
        taken.add(code)
        out[name] = {"count": count, "icao": code}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=REAL_TRAIN_END,
                        help="rows of the train split to read (never exceeds 8000)")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_VOCAB_PATH)
    args = parser.parse_args()

    from datasets import load_dataset

    limit = min(args.limit, REAL_TRAIN_END)
    dataset = load_dataset(CORPUS, split=f"train[:{limit}]").select_columns(["text"])
    texts = [row["text"] for row in dataset]
    counts = harvest(texts)
    counts["airlines"] = _prune_airlines(counts["airlines"], counts["stations"])

    kept = {name: {word: count for word, count in counter.most_common()
                   if count >= args.min_count and all(part.isalpha() for part in word.split())}
            for name, counter in counts.items()}
    blob = {
        "meta": {"corpus": CORPUS, "split": f"train[0:{limit}]",
                 "utterances": len(texts), "min_count": args.min_count,
                 "generated": date.today().isoformat()},
        "airlines": _assign_codes(kept["airlines"]),
        "stations": kept["stations"],
        "waypoints": kept["waypoints"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, sort_keys=False) + "\n")

    print(f"{len(texts)} transcripts -> {args.out}")
    for name in ("airlines", "stations", "waypoints"):
        section = blob[name]
        top = list(section)[:15]
        print(f"  {name}: {len(section)} kept; top: {', '.join(top)}")


if __name__ == "__main__":
    main()
