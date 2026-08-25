"""Structured entity ground truth for ATC transcripts.

Single shared module for the three consumers that must agree on what an
"entity" is: the scenario grammar (`atcgen.text`, which emits ground truth
structurally), the verification gate (which compares teacher hypotheses to
labels) and the evaluation panel (which reports slot F1 and the
critical-number substitution rate).

Everything here is *spoken-form first*: transcripts in this project are
verbatim spoken words in the ATCO2 style ("one two seven decimal eight two
five"), and `value` is the canonical machine reading of that ("127.825").

Canonical value formats
-----------------------
    callsign      CSA123, BAW462, CSA3KF, N123AB   ICAO/ident, upper case
    runway        24L, 04                          two digits + optional side
    heading       270, 090                         three digits, 001-360
    altitude      3500ft                           feet
    flight_level  FL350                            three digits
    frequency     127.825                          MHz, three decimals
    speed         250                              knots
    squawk        4521                             four octal digits
    altimeter     Q1013 / A2992                    hPa (QNH) / inHg
    waypoint      PADKA                            upper case
    atis          C                                single letter
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ENTITY_TYPES = (
    "callsign", "runway", "heading", "altitude", "flight_level",
    "frequency", "speed", "squawk", "altimeter", "waypoint", "atis",
)

#: Types whose corruption is operationally dangerous. One garbled digit here
#: outweighs five garbled filler words, so these drive the headline metric.
CRITICAL_TYPES = frozenset({
    "callsign", "runway", "heading", "altitude", "flight_level",
    "frequency", "speed", "squawk", "altimeter",
})


@dataclass(frozen=True)
class Entity:
    """One labelled slot inside an utterance.

    `spoken` is the substring of the utterance that renders the value.
    `critical` defaults to whether `type` is in `CRITICAL_TYPES`; pass it
    explicitly only to override that.
    """

    type: str
    value: str
    spoken: str = ""
    critical: bool | None = None

    def __post_init__(self) -> None:
        if self.critical is None:
            object.__setattr__(self, "critical", self.type in CRITICAL_TYPES)

    @property
    def key(self) -> tuple[str, str]:
        """Identity used for ref/hyp matching: type and canonical value."""
        return (self.type, self.value)

    def to_dict(self) -> dict:
        return {"type": self.type, "value": self.value, "spoken": self.spoken,
                "critical": bool(self.critical)}

    @classmethod
    def from_dict(cls, obj: dict) -> "Entity":
        return cls(type=obj["type"], value=obj["value"],
                   spoken=obj.get("spoken", ""), critical=obj.get("critical"))


def entities_to_dicts(entities: list[Entity]) -> list[dict]:
    return [e.to_dict() for e in entities]


def entities_from_dicts(objs: list[dict]) -> list[Entity]:
    return [Entity.from_dict(o) for o in objs]


# ---------------------------------------------------------------------------
# Spoken numbers <-> canonical digits
# ---------------------------------------------------------------------------

#: Plain readings. ATC variants (niner/tree/fife/fower) are a style knob.
PLAIN_DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
ATC_DIGITS = dict(PLAIN_DIGITS, **{"3": "tree", "4": "fower", "5": "fife", "9": "niner"})

#: Accepted spellings on the way back in. "oh" is a real zero; bare "o" is
#: not accepted -- in the real corpus it is far more often a truncated word.
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "nought": "0",
    "one": "1", "two": "2",
    "three": "3", "tree": "3",
    "four": "4", "fower": "4",
    "five": "5", "fife": "5",
    "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "niner": "9",
}

TEEN_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
MULTIPLIER_WORDS = {"hundred": 100, "thousand": 1000}
DECIMAL_WORDS = frozenset({"decimal", "point"})

PHONETIC_ALPHABET = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo",
    "F": "foxtrot", "G": "golf", "H": "hotel", "I": "india", "J": "juliett",
    "K": "kilo", "L": "lima", "M": "mike", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey", "X": "xray",
    "Y": "yankee", "Z": "zulu",
}
#: Reverse map, tolerant of the spellings that show up in real transcripts.
LETTER_WORDS = {word: letter for letter, word in PHONETIC_ALPHABET.items()}
LETTER_WORDS.update({"alfa": "A", "juliet": "J", "whisky": "W", "x-ray": "X"})

RUNWAY_SIDES = {"left": "L", "right": "R", "center": "C", "centre": "C",
                "l": "L", "r": "R", "c": "C"}
SIDE_WORDS = {"L": "left", "R": "right", "C": "center"}


def digits_to_words(digits: str, atc_variants: bool = False) -> str:
    """'127' -> 'one two seven' (or 'one two seven' with tree/fife/niner)."""
    table = ATC_DIGITS if atc_variants else PLAIN_DIGITS
    return " ".join(table[c] for c in str(digits) if c in table)


def group_number_words(n: int, atc_variants: bool = False) -> str:
    """Spoken group form: 412 -> 'four twelve', 1850 -> 'eighteen fifty'.

    Controllers use group form for flight numbers and (in Europe) sometimes
    for frequencies; digit-by-digit is the other option.
    """
    table = ATC_DIGITS if atc_variants else PLAIN_DIGITS
    teens = {value: word for word, value in TEEN_WORDS.items()}
    tens = {value: word for word, value in TENS_WORDS.items()}
    text = str(n)

    def two(pair: str) -> str:
        value = int(pair)
        if value in teens:
            return teens[value]
        if pair[0] == "0":
            return "zero " + table[pair[1]]
        if pair[1] == "0":
            return tens[value]
        return tens[int(pair[0]) * 10] + " " + table[pair[1]]

    if len(text) == 1:
        return table[text]
    if len(text) == 2:
        return two(text)
    if len(text) == 3:
        return table[text[0]] + " " + two(text[1:])
    if len(text) == 4:
        return two(text[:2]) + " " + two(text[2:])
    return digits_to_words(text, atc_variants)


def feet_to_words(feet: int, atc_variants: bool = False) -> str:
    """3500 -> 'three thousand five hundred'; 10000 -> 'one zero thousand'."""
    table = ATC_DIGITS if atc_variants else PLAIN_DIGITS
    thousands, remainder = divmod(int(feet), 1000)
    hundreds = remainder // 100
    parts = []
    if thousands:
        parts.append(digits_to_words(str(thousands), atc_variants) + " thousand")
    if hundreds:
        parts.append(table[str(hundreds)] + " hundred")
    return " ".join(parts) if parts else table["0"]


#: Numerals first, so an ordinal suffix binds to its digits rather than
#: splitting off as a word ("runway 26th center").
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?|[a-z]+")
_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower().replace("-", " ")):
        if token[0].isdigit() and token.endswith(_ORDINAL_SUFFIXES):
            token = token[:-2]
        tokens.append(token)
    folded: list[str] = []
    for token in tokens:
        # "x ray" is one letter
        if token == "ray" and folded and folded[-1] == "x":
            folded[-1] = "xray"
            continue
        folded.append(token)
    return folded


@dataclass(frozen=True)
class _Run:
    """A maximal run of number words, read two ways."""

    start: int
    end: int
    digits: str          # concatenated reading, "" when multipliers are present
    value: int | None    # arithmetic reading ("three thousand five hundred")


def _atom_at(tokens: list[str], index: int) -> tuple[str, int] | None:
    """Digits contributed by the token(s) at `index`, and where they end.

    A tens word absorbs a following unit ("twenty seven" -> "27"), which is
    what makes group forms read correctly: "one twenty seven" -> "127", not
    "1207".
    """
    if index >= len(tokens):
        return None
    token = tokens[index]
    if token in TENS_WORDS:
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        unit = DIGIT_WORDS.get(following)
        if unit and unit != "0":
            return str(TENS_WORDS[token] + int(unit)), index + 2
        return str(TENS_WORDS[token]), index + 1
    if token in DIGIT_WORDS:
        return DIGIT_WORDS[token], index + 1
    if token in TEEN_WORDS:
        return str(TEEN_WORDS[token]), index + 1
    if token.isdigit():
        return token, index + 1
    return None


def scan_number(tokens: list[str], start: int) -> _Run | None:
    """Maximal number run at `start`, or None if no number begins there."""
    parts: list[tuple[str, str | int]] = []
    index = start
    while index < len(tokens):
        if tokens[index] in MULTIPLIER_WORDS:
            parts.append(("mult", MULTIPLIER_WORDS[tokens[index]]))
            index += 1
            continue
        atom = _atom_at(tokens, index)
        if atom is None:
            break
        parts.append(("atom", atom[0]))
        index = atom[1]
    if not parts or parts[0][0] == "mult":
        return None

    if any(kind == "mult" for kind, _ in parts):
        total, chunk = 0, ""
        for kind, payload in parts:
            if kind == "atom":
                chunk += str(payload)
            else:
                total += (int(chunk) if chunk else 1) * int(payload)
                chunk = ""
        if chunk:
            total += int(chunk)
        return _Run(start, index, "", total)

    digits = "".join(str(payload) for _, payload in parts)
    return _Run(start, index, digits, int(digits))


def _all_runs(tokens: list[str]) -> dict[int, _Run]:
    """Every maximal number run, keyed by start index."""
    runs: dict[int, _Run] = {}
    index = 0
    while index < len(tokens):
        run = scan_number(tokens, index)
        if run is None:
            index += 1
            continue
        runs[index] = run
        index = run.end
    return runs


# ---------------------------------------------------------------------------
# Airline telephony
# ---------------------------------------------------------------------------

#: Telephony word(s) -> ICAO designator. European carriers first: the eval
#: corpus (jacktol/atc-dataset = ATCO2 + UWB-ATCC) is Czech/European airspace.
DEFAULT_AIRLINES: dict[str, str] = {
    # Europe
    "csa": "CSA", "c s a": "CSA", "czech": "CSA",
    "lufthansa": "DLH", "hansa": "DLH",
    "speedbird": "BAW",
    "ryanair": "RYR",
    "wizz air": "WZZ", "wizzair": "WZZ",
    "easy": "EZY",
    "austrian": "AUA",
    "swiss": "SWR",
    "sky travel": "SKV", "skytravel": "SKV",
    "smartwings": "TVS", "travel service": "TVS",
    "air france": "AFR",
    "k l m": "KLM", "klm": "KLM",
    "alitalia": "AZA",
    "iberia": "IBE",
    "turkish": "THY",
    "aeroflot": "AFL", "aero flot": "AFL",
    "air berlin": "BER",
    "thomson": "TOM",
    "volga dniepr": "VDA",
    "eurowings": "EWG",
    "germanwings": "GWI",
    "condor": "CFG",
    "finnair": "FIN",
    "scandinavian": "SAS",
    "norwegian": "NAX",
    "brussels": "BEL",
    "tarom": "ROT",
    "lot": "LOT",
    "malev": "MAH",
    "croatia": "CTN",
    "adria": "ADR",
    "carpatair": "KRP",
    "aegean": "AEE",
    "transavia": "TRA",
    "vueling": "VLG",
    "jet": "JET",
    # North America / long haul
    "american": "AAL",
    "united": "UAL",
    "delta": "DAL",
    "southwest": "SWA",
    "jetblue": "JBU",
    "alaska": "ASA",
    "skywest": "SKW",
    "envoy": "ENY",
    "republic": "RPA",
    "spirit": "NKS",
    "frontier": "FFT",
    "fedex": "FDX",
    "u p s": "UPS", "ups": "UPS",
    "aeromexico": "AMX",
    "air canada": "ACA",
    "westjet": "WJA",
    "qantas": "QFA",
    "emirates": "UAE",
    "singapore": "SIA",
    "cathay": "CPA",
    "china southern": "CSN",
    "korean air": "KAL",
}

#: Spoken prefixes that introduce a tail number rather than a flight number.
GA_PREFIXES: dict[str, str] = {
    prefix: "N" for prefix in
    ("november", "cessna", "piper", "cirrus", "beechcraft", "mooney",
     "bonanza", "skyhawk", "archer", "citation", "gulfstream", "king air")
}

#: Longest ident a real callsign carries ("N123AB", "CSA1234"). Capping it
#: stops the run from swallowing the number that follows the callsign
#: ("aeroflot seven zero one four, fourteen miles east").
IDENT_MAX = 5

DEFAULT_VOCAB_PATH = Path("data/vocab/real_anchor.json")


def load_airlines(vocab_path: str | Path | None = DEFAULT_VOCAB_PATH) -> dict[str, str]:
    """Builtin telephony table merged with the harvested real-corpus vocab.

    Both the grammar and the metrics must use the *same* table, otherwise a
    harvested carrier gets one code on the reference side and another on the
    hypothesis side. Missing vocab file -> builtin table only.
    """
    airlines = dict(DEFAULT_AIRLINES)
    if vocab_path is None:
        return airlines
    path = Path(vocab_path)
    if not path.exists():
        return airlines
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return airlines
    builtin = {joined_key(phrase): code for phrase, code in DEFAULT_AIRLINES.items()}
    for phrase, entry in (blob.get("airlines") or {}).items():
        code = entry.get("icao") if isinstance(entry, dict) else None
        code = builtin.get(joined_key(phrase), code)
        if code:
            airlines.setdefault(phrase, code)
    return airlines


def telephony_code(phrase: str) -> str:
    """Fallback ICAO-ish code for a telephony name we have no real code for.

    Harvested carriers ("nor shuttle") have no designator in the builtin
    table, so we mint a stable one. It only has to be consistent: the same
    table builds the reference entities and parses the hypothesis.
    """
    words = [word for word in phrase.upper().split() if word.isalpha()]
    if not words:
        return "XXX"
    code = words[0][:2] + words[-1][0] if len(words) > 1 else words[0][:3]
    return (code + "".join(words))[:3]


#: Longest telephony name in words; bounds the join window in the scanner.
AIRLINE_MAX_WORDS = 4


def joined_key(phrase: str) -> str:
    """Telephony name with word boundaries removed: 'aero mexico' -> 'aeromexico'.

    Speech recognizers split and join these names freely -- "aeromexico" /
    "aero mexico", "k l m" / "klm", "wizz air" / "wizzair" -- and all of them
    have to reach the same designator.
    """
    return "".join(phrase.split())


def _airline_index(airlines: dict[str, str]) -> dict[str, str]:
    """Join-insensitive lookup. Earlier entries win, so builtin beats harvested."""
    index: dict[str, str] = {}
    for phrase, code in airlines.items():
        index.setdefault(joined_key(phrase), code)
    return index


_DEFAULT_TABLE: dict[str, str] | None = None
_DEFAULT_INDEX: list[tuple[tuple[str, ...], str]] | None = None


def default_airlines() -> dict[str, str]:
    """The table `extract_entities` uses when the caller passes none.

    Builtin names merged with the harvested anchor file if it exists at
    `DEFAULT_VOCAB_PATH` (relative to the working directory). Everyone --
    grammar, gate, eval -- lands on the same codes this way, which is what
    makes reference and hypothesis entities comparable at all.
    """
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        _DEFAULT_TABLE = load_airlines(DEFAULT_VOCAB_PATH)
    return _DEFAULT_TABLE


def _index_for(airlines: dict[str, str] | None) -> dict[str, str]:
    global _DEFAULT_INDEX
    if airlines is not None:
        return _airline_index(airlines)
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = _airline_index(default_airlines())
    return _DEFAULT_INDEX


# ---------------------------------------------------------------------------
# Value legality (shared by the scenario validator and the gate)
# ---------------------------------------------------------------------------

_RE = {
    "callsign": re.compile(r"^[A-Z]{1,4}\d[A-Z0-9]{0,5}$"),
    "runway": re.compile(r"^\d{2}[LRC]?$"),
    "heading": re.compile(r"^\d{3}$"),
    "altitude": re.compile(r"^[1-9]\d{2,4}ft$"),
    "flight_level": re.compile(r"^FL\d{3}$"),
    "frequency": re.compile(r"^\d{3}\.\d{3}$"),
    "speed": re.compile(r"^[1-9]\d{1,2}$"),
    "squawk": re.compile(r"^[0-7]{4}$"),
    "altimeter": re.compile(r"^[QA]\d{3,4}$"),
    "waypoint": re.compile(r"^[A-Z]{2,10}$"),
    "atis": re.compile(r"^[A-Z]$"),
}


def check_value(type_: str, value: str) -> str | None:
    """Return a violation string for an illegal (type, value), else None.

    Ranges follow ICAO/FAA phraseology limits. Frequencies allow any 5 kHz
    step so that both 25 kHz channels (118.850) and the 8.33 kHz channel
    names that dominate European R/T (121.755) are legal.
    """
    if type_ not in ENTITY_TYPES:
        return f"unknown entity type {type_!r}"
    pattern = _RE[type_]
    if not pattern.match(value):
        return f"{type_} value {value!r} is malformed"
    if type_ == "runway":
        number = int(value[:2])
        if not 1 <= number <= 36:
            return f"runway {value!r} outside 01-36"
    elif type_ == "heading":
        if not 1 <= int(value) <= 360:
            return f"heading {value!r} outside 001-360"
    elif type_ == "flight_level":
        if not 10 <= int(value[2:]) <= 450:
            return f"flight level {value!r} outside FL010-FL450"
    elif type_ == "altitude":
        feet = int(value[:-2])
        if not 100 <= feet <= 45000:
            return f"altitude {value!r} outside 100-45000 ft"
        if feet % 100:
            return f"altitude {value!r} is not a round hundred feet"
    elif type_ == "frequency":
        khz = round(float(value) * 1000)
        if not 118000 <= khz <= 136975:
            return f"frequency {value!r} outside 118.000-136.975"
        if khz % 5:
            return f"frequency {value!r} is not a 5 kHz step"
    elif type_ == "speed":
        if not 60 <= int(value) <= 350:
            return f"speed {value!r} outside 60-350 kt"
    elif type_ == "altimeter":
        number = int(value[1:])
        if value[0] == "Q" and not 900 <= number <= 1100:
            return f"QNH {value!r} outside 900-1100 hPa"
        if value[0] == "A" and not 2800 <= number <= 3150:
            return f"altimeter {value!r} outside 28.00-31.50 inHg"
    return None


def check_entity(entity: Entity) -> str | None:
    """Value legality plus the invariant that `spoken` is really spoken."""
    problem = check_value(entity.type, entity.value)
    if problem:
        return problem
    if entity.critical != (entity.type in CRITICAL_TYPES):
        return f"{entity.type} {entity.value!r} has non-default `critical`"
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_RUNWAY_WORDS = {"runway", "rwy"}
_HEADING_WORDS = {"heading"}
_SQUAWK_WORDS = {"squawk", "sqawk"}
_SPEED_WORDS = {"speed", "speeds"}
_QNH_WORDS = {"qnh"}
_ALTIMETER_WORDS = {"altimeter"}
_ATIS_WORDS = {"information", "atis"}
#: A bare numeral is an altitude only next to a vertical-clearance verb --
#: ASR drops "feet" and writes "climb and maintain 1,500".
_ALTITUDE_WORDS = {"altitude", "climb", "climbing", "descend", "descending",
                   "maintain", "maintaining"}
#: Fillers that may sit between an anchor word and its value.
_FILLERS = ("to", "and", "at", "of")
#: Unit suffixes that anchor a value on their own.
_FEET_SUFFIX = {"feet", "foot", "ft"}
_KNOTS_SUFFIX = {"knots", "knot", "kt", "kts"}
_WAYPOINT_ANCHORS = {"direct", "overhead", "abeam"}
#: Never mistaken for a waypoint after "direct".
_WAYPOINT_STOPWORDS = {
    "to", "the", "a", "is", "and", "we", "you", "now", "when", "able",
    "routing", "route", "clearance", "cleared", "climb", "descend",
    "maintain", "contact", "roger", "approach", "runway", "level", "flight",
}


def _finish(entity_type: str, value: str, tokens: list[str], run_start: int,
            run_end: int) -> Entity | None:
    """Build an entity if the value is legal; an illegal one is a mis-parse."""
    if check_value(entity_type, value):
        return None
    return Entity(type=entity_type, value=value,
                  spoken=" ".join(tokens[run_start:run_end]))


def extract_entities(text: str, airlines: dict[str, str] | None = None) -> list[Entity]:
    """Best-effort entities from spoken-form ATC text.

    Used on ASR hypotheses and on real corpus transcripts; the grammar emits
    its own ground truth structurally instead of parsing itself. `airlines`
    maps telephony words to ICAO codes ("speedbird" -> "BAW"); omitting it
    uses `default_airlines()`, so a reference and a hypothesis scored in the
    same process always agree on codes.

    Precision is favoured over recall throughout -- every extraction needs an
    anchor word, because a wrong slot value costs twice (a false positive
    *and* a missed reference) in `score_entities`.
    """
    index = _index_for(airlines)
    tokens = tokenize(text)
    runs = _all_runs(tokens)
    runs_by_end = {run.end: run for run in runs.values()}
    used = [False] * len(tokens)
    found: list[tuple[int, Entity]] = []

    def claim(start: int, end: int) -> None:
        for position in range(start, end):
            used[position] = True

    def free(start: int, end: int) -> bool:
        return not any(used[start:end])

    def keep(entity: Entity | None, span: tuple[int, int], claim_from: int) -> bool:
        if entity is None or not free(*span):
            return False
        found.append((span[0], entity))
        claim(claim_from, span[1])
        return True

    # 1. Frequencies, anchored on "decimal"/"point" (or a literal 127.825).
    for position, token in enumerate(tokens):
        if token in DECIMAL_WORDS:
            whole = runs_by_end.get(position)
            fraction = runs.get(position + 1)
            if whole is None or fraction is None or len(whole.digits) != 3 \
                    or not fraction.digits:
                continue
            value = f"{whole.digits}.{fraction.digits[:3].ljust(3, '0')}"
            span = (whole.start, min(fraction.end, position + 4))
            keep(_finish("frequency", value, tokens, *span), span, span[0])
        elif re.fullmatch(r"\d{3}\.\d{1,3}", token):
            whole_text, fraction_text = token.split(".")
            span = (position, position + 1)
            keep(_finish("frequency", f"{whole_text}.{fraction_text.ljust(3, '0')}",
                         tokens, *span), span, position)

    # 2. Callsigns: telephony phrase (or GA type prefix) plus an ident run.
    position = 0
    while position < len(tokens):
        if used[position]:
            position += 1
            continue
        code, width = None, 0
        for span in range(min(AIRLINE_MAX_WORDS, len(tokens) - position), 0, -1):
            end = position + span
            words = tokens[position:end]
            if not all(word.isalpha() for word in words) or not free(position, end):
                continue
            code = index.get("".join(words))
            if code is not None:
                width = span
                break
        # "taxi via delta, november five nine six" is a taxiway followed by a
        # tail number, not a Delta flight: a GA prefix ends the airline match.
        if code is not None and tokens[position + width:position + width + 1] \
                and tokens[position + width] in GA_PREFIXES:
            code = None
        if code is None and tokens[position] in GA_PREFIXES:
            code, width = GA_PREFIXES[tokens[position]], 1
        if code is None:
            position += 1
            continue
        ident, cursor, letters, digits = "", position + width, 0, 0
        while cursor < len(tokens) and not used[cursor] and len(ident) < IDENT_MAX:
            atom = _atom_at(tokens, cursor)
            if atom is not None:
                if len(ident) + len(atom[0]) > IDENT_MAX:
                    break
                ident += atom[0]
                digits += 1
                cursor = atom[1]
            elif tokens[cursor] in LETTER_WORDS and letters < 3:
                ident += LETTER_WORDS[tokens[cursor]]
                letters += 1
                cursor += 1
            else:
                break
        entity = _finish("callsign", code + ident, tokens, position, cursor) \
            if digits else None
        if keep(entity, (position, cursor), position):
            position = cursor
        else:
            position += width

    # 3. Keyword-anchored slots. Every value here needs its anchor word, so a
    #    bare number ("wind two zero zero degrees") is never mistaken for one.
    for position, token in enumerate(tokens):
        if used[position]:
            continue
        run = runs.get(position + 1)
        entity: Entity | None = None
        span = (position + 1, run.end if run else position + 1)

        if token in _RUNWAY_WORDS and run and 1 <= len(run.digits) <= 2:
            value = run.digits.zfill(2)
            end = run.end
            if end < len(tokens) and tokens[end] in RUNWAY_SIDES:
                value += RUNWAY_SIDES[tokens[end]]
                end += 1
            span = (run.start, end)
            entity = _finish("runway", value, tokens, *span)
        elif token in _HEADING_WORDS and run and len(run.digits) == 3:
            entity = _finish("heading", run.digits, tokens, *span)
        elif token == "level" and run and 2 <= len(run.digits) <= 3:
            entity = _finish("flight_level", "FL" + run.digits.zfill(3), tokens, *span)
        elif token in _SQUAWK_WORDS and run and len(run.digits) == 4:
            entity = _finish("squawk", run.digits, tokens, *span)
        elif token in _SPEED_WORDS:
            if run is None and tokens[position + 1:position + 2] \
                    and tokens[position + 1] in _FILLERS:
                run = runs.get(position + 2)
                span = (position + 2, run.end if run else position + 2)
            if run and 2 <= len(run.digits) <= 3:
                entity = _finish("speed", str(int(run.digits)), tokens, *span)
        elif token in _QNH_WORDS and run and 3 <= len(run.digits) <= 4:
            entity = _finish("altimeter", "Q" + run.digits, tokens, *span)
        elif token in _ALTIMETER_WORDS and run and len(run.digits) == 4:
            entity = _finish("altimeter", "A" + run.digits, tokens, *span)
        elif token in _ALTITUDE_WORDS:
            if run is None and tokens[position + 1:position + 2] \
                    and tokens[position + 1] in _FILLERS:
                run = runs.get(position + 2)
                span = (position + 2, run.end if run else position + 2)
            # "maintain 250 knots" is a speed wearing an altitude's anchor
            if run and run.value and tokens[run.end:run.end + 1] != [] \
                    and tokens[run.end] in _KNOTS_SUFFIX:
                run = None
            if run and run.value:
                entity = _finish("altitude", f"{run.value}ft", tokens, *span)
        elif token in _ATIS_WORDS and tokens[position + 1:position + 2] \
                and tokens[position + 1] in LETTER_WORDS:
            span = (position + 1, position + 2)
            entity = _finish("atis", LETTER_WORDS[tokens[position + 1]], tokens, *span)
        elif token in _WAYPOINT_ANCHORS:
            name_at = position + 1
            if tokens[name_at:name_at + 1] == ["to"]:
                name_at += 1
            name = tokens[name_at] if name_at < len(tokens) else ""
            if (name.isalpha() and len(name) >= 4
                    and name not in _WAYPOINT_STOPWORDS
                    and name not in DIGIT_WORDS and name not in TEEN_WORDS
                    and name not in TENS_WORDS and name not in MULTIPLIER_WORDS
                    and name not in LETTER_WORDS):
                span = (name_at, name_at + 1)
                entity = _finish("waypoint", name.upper(), tokens, *span)

        keep(entity, span, position)

    # 4. A unit suffix anchors its own value: "2,000 feet", "260 knots".
    #    The legal ranges do the filtering -- a reported wind of "four knots"
    #    is far below the 60 kt floor, so it never reads as an airspeed.
    for start, run in sorted(runs.items()):
        following = tokens[run.end] if run.end < len(tokens) else ""
        if not free(run.start, run.end):
            continue
        if following in _FEET_SUFFIX and run.value:
            keep(_finish("altitude", f"{run.value}ft", tokens, run.start, run.end),
                 (run.start, run.end), run.start)
        elif following in _KNOTS_SUFFIX and run.digits:
            keep(_finish("speed", str(int(run.digits)), tokens, run.start, run.end),
                 (run.start, run.end), run.start)

    # 5. Unanchored altitudes: a number run built with thousand/hundred is an
    #    altitude on its own ("climb and maintain four thousand five hundred").
    for start, run in sorted(runs.items()):
        if run.digits or not run.value or not free(run.start, run.end):
            continue
        if tokens[start - 1:start] and tokens[start - 1] in {"level", "heading"}:
            continue
        keep(_finish("altitude", f"{run.value}ft", tokens, run.start, run.end),
             (run.start, run.end), run.start)

    return [entity for _, entity in sorted(found, key=lambda item: item[0])]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class EntityScore:
    """Slot-level agreement between a reference and a hypothesis.

    Holds raw counts so scores over a corpus are just `sum(...)`; every rate
    is a derived property. `substitutions` counts a critical entity that is
    present in both sides with a *different* value — the operationally
    dangerous error class the research panel leads with.
    """

    tp: dict[str, int] = field(default_factory=dict)
    fp: dict[str, int] = field(default_factory=dict)
    fn: dict[str, int] = field(default_factory=dict)
    sub: dict[str, int] = field(default_factory=dict)
    callsign_correct: int = 0
    callsign_total: int = 0
    critical_ref: int = 0
    critical_sub: int = 0

    def __add__(self, other: "EntityScore") -> "EntityScore":
        merged = EntityScore(
            callsign_correct=self.callsign_correct + other.callsign_correct,
            callsign_total=self.callsign_total + other.callsign_total,
            critical_ref=self.critical_ref + other.critical_ref,
            critical_sub=self.critical_sub + other.critical_sub,
        )
        for name in ("tp", "fp", "fn", "sub"):
            counts = dict(getattr(self, name))
            for key, value in getattr(other, name).items():
                counts[key] = counts.get(key, 0) + value
            setattr(merged, name, counts)
        return merged

    __radd__ = __add__

    @property
    def types(self) -> list[str]:
        return sorted(set(self.tp) | set(self.fp) | set(self.fn))

    def per_type(self) -> dict[str, dict[str, float]]:
        out = {}
        for name in self.types:
            tp, fp, fn = self.tp.get(name, 0), self.fp.get(name, 0), self.fn.get(name, 0)
            out[name] = {
                "tp": tp, "fp": fp, "fn": fn, "sub": self.sub.get(name, 0),
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
                "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            }
        return out

    @property
    def precision(self) -> float:
        return _ratio(sum(self.tp.values()), sum(self.tp.values()) + sum(self.fp.values()))

    @property
    def recall(self) -> float:
        return _ratio(sum(self.tp.values()), sum(self.tp.values()) + sum(self.fn.values()))

    @property
    def f1(self) -> float:
        tp = sum(self.tp.values())
        return _ratio(2 * tp, 2 * tp + sum(self.fp.values()) + sum(self.fn.values()))

    @property
    def callsign_accuracy(self) -> float:
        return _ratio(self.callsign_correct, self.callsign_total)

    def critical_substitutions_of(self, type_: str) -> int:
        """Substituted values for one slot type ("which digits get garbled")."""
        return self.sub.get(type_, 0)

    @property
    def critical_substitution_rate(self) -> float:
        """Share of critical reference slots the hypothesis got *wrong*."""
        return _ratio(self.critical_sub, self.critical_ref)

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "callsign_accuracy": self.callsign_accuracy,
            "callsign_total": self.callsign_total,
            "critical_substitution_rate": self.critical_substitution_rate,
            "critical_substitutions": self.critical_sub,
            "critical_ref": self.critical_ref,
            "per_type": self.per_type(),
        }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def score_entities(ref: list[Entity], hyp: list[Entity]) -> EntityScore:
    """Compare one reference/hypothesis entity set.

    Matching is a multiset intersection on (type, value). Leftovers of the
    same type are paired into substitutions, which count as both a false
    positive and a false negative — and, for critical types, as the
    substitution the safety metric tracks.
    """
    score = EntityScore()
    ref_by_type: dict[str, list[Entity]] = {}
    hyp_by_type: dict[str, list[Entity]] = {}
    for entity in ref:
        ref_by_type.setdefault(entity.type, []).append(entity)
    for entity in hyp:
        hyp_by_type.setdefault(entity.type, []).append(entity)

    for type_ in set(ref_by_type) | set(hyp_by_type):
        refs = list(ref_by_type.get(type_, []))
        hyps = list(hyp_by_type.get(type_, []))
        matched = 0
        remaining = list(hyps)
        for entity in list(refs):
            for candidate in remaining:
                if candidate.value == entity.value:
                    remaining.remove(candidate)
                    refs.remove(entity)
                    matched += 1
                    break
        substitutions = min(len(refs), len(remaining))
        critical = type_ in CRITICAL_TYPES
        if matched:
            score.tp[type_] = matched
        if len(remaining):
            score.fp[type_] = len(remaining)
        if len(refs):
            score.fn[type_] = len(refs)
        if substitutions:
            score.sub[type_] = substitutions
        if critical:
            score.critical_ref += matched + len(refs)
            score.critical_sub += substitutions
        if type_ == "callsign":
            score.callsign_total += matched + len(refs)
            score.callsign_correct += matched
    return score


def aggregate(scores: list[EntityScore]) -> EntityScore:
    """Corpus-level score from per-utterance scores."""
    total = EntityScore()
    for score in scores:
        total = total + score
    return total
