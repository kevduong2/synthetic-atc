"""Vocabulary and spoken renderers for ATC phraseology.

Two halves:

* Constants -- phonetic alphabet, telephony names, taxiways, traffic types.
* `Slot` renderers -- given a canonical value ("24L") and a `Style`, produce
  the spoken words, the display form and the `Entity` that labels them. The
  grammar builds every utterance out of slots, so ground truth is emitted
  structurally instead of being parsed back out of the text.

Spoken forms follow FAA Order 7110.65 / ICAO Annex 10. Whether a transmission
uses the radio variants ("tree", "fife", "niner") or plain digits is a style
draw, not a constant: the research finding is that pre-synthesis pronunciation
variation is worth consistent WER gains downstream.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..entities import (ATC_DIGITS, DEFAULT_AIRLINES, DEFAULT_VOCAB_PATH,
                        PHONETIC_ALPHABET, PLAIN_DIGITS, SIDE_WORDS, Entity,
                        check_value, digits_to_words, feet_to_words,
                        group_number_words, telephony_code)

DIGITS_SPOKEN = ATC_DIGITS

AIRLINE_TELEPHONY = sorted(DEFAULT_AIRLINES)

GA_TYPES = ["cessna", "piper", "cirrus", "beechcraft", "mooney", "bonanza",
            "skyhawk", "archer"]

HOLDING_POINTS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                  "golf", "hotel", "juliett", "kilo", "mike", "papa"]

ATIS_LETTERS = list(PHONETIC_ALPHABET.values())

#: Deliberately digit-free: "a cessna one seventy two" is indistinguishable
#: from a callsign once the audio is gone, and a traffic call is not a
#: clearance, so it must not put an unlabelled ident into the transcript.
AIRCRAFT_TRAFFIC_TYPES = [
    "a cessna", "a boeing", "an airbus", "a piper cherokee", "a regional jet",
    "a king air", "a helicopter", "a citation jet", "a business jet",
    "a light twin",
]

DIRECTIONS = ["north", "south", "east", "west", "northeast", "northwest",
              "southeast", "southwest"]

#: European facilities, the flavour of the eval corpus (Czech/central Europe).
EU_STATIONS = ["praha radar", "praha approach", "praha control", "ruzyne tower",
               "ruzyne ground", "wien approach", "wien radar", "munchen radar",
               "budapest radar", "warsaw radar", "bratislava tower",
               "zurich tower", "frankfurt radar", "apron"]

US_STATIONS = ["tower", "ground", "departure", "approach", "center",
               "clearance delivery", "ramp"]

#: Five-letter ICAO area navigation points; the harvest adds the real ones.
EU_WAYPOINTS = ["rapet", "baltu", "vozit", "kenok", "arvek", "lomki", "netvi",
                "sopav", "dibet", "tomti", "budex", "agava", "melun"]

US_WAYPOINTS = ["mocha", "buffy", "grice", "hobbs", "kayoh", "riivr", "tandy"]

GREETINGS = ["good morning", "good afternoon", "good evening", "good day"]
SIGNOFFS = ["good day", "bye bye", "servus", "cheers", "auf wiederhoren"]


# ---------------------------------------------------------------------------
# Style: how this transmission pronounces its numbers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Style:
    """Per-transmission pronunciation draw."""

    atc_variants: bool = True    # tree/fife/niner/fower instead of plain digits
    group_numbers: bool = False  # "one twenty seven" instead of "one two seven"
    decimal_word: str = "decimal"  # ICAO "decimal" vs FAA "point"

    @classmethod
    def draw(cls, rng: random.Random, respelling_prob: float,
             region: str = "eu") -> "Style":
        return cls(
            atc_variants=rng.random() < respelling_prob,
            group_numbers=rng.random() < respelling_prob * 0.5,
            decimal_word="point" if region == "us" else "decimal",
        )


# ---------------------------------------------------------------------------
# Anchored vocabulary
# ---------------------------------------------------------------------------

@dataclass
class Vocab:
    """Telephony/station/waypoint pools: builtin, plus the real-train anchor.

    `scripts/harvest_vocab.py` writes the anchor file from jacktol train
    [0:8000]. Without it everything falls back to the builtin lists, so the
    grammar never depends on a generated artifact being present.
    """

    airlines: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_AIRLINES))
    anchored_airlines: list[tuple[str, float]] = field(default_factory=list)
    stations: list[str] = field(default_factory=list)
    waypoints: list[str] = field(default_factory=list)
    anchored: bool = False

    @classmethod
    def load(cls, path: str | Path | None = DEFAULT_VOCAB_PATH) -> "Vocab":
        vocab = cls()
        blob = None
        if path is not None and Path(path).exists():
            try:
                blob = json.loads(Path(path).read_text())
            except (json.JSONDecodeError, OSError):
                blob = None
        if not blob:
            return vocab
        for phrase, entry in (blob.get("airlines") or {}).items():
            count = entry.get("count", 1) if isinstance(entry, dict) else 1
            code = entry.get("icao") if isinstance(entry, dict) else None
            code = code or DEFAULT_AIRLINES.get(phrase) or telephony_code(phrase)
            vocab.airlines.setdefault(phrase, code)
            vocab.anchored_airlines.append((phrase, float(count)))
        vocab.stations = list(blob.get("stations") or {})
        vocab.waypoints = list(blob.get("waypoints") or {})
        vocab.anchored = bool(vocab.anchored_airlines)
        return vocab

    def phrase_for(self, code: str) -> str:
        """A telephony phrase that renders `code` (first match wins)."""
        for phrase, icao in self.airlines.items():
            if icao == code:
                return phrase
        return code.lower()

    def pick_airline(self, rng: random.Random, anchor_prob: float = 0.7
                     ) -> tuple[str, str]:
        """(telephony phrase, ICAO code), preferring real harvested carriers."""
        if self.anchored_airlines and rng.random() < anchor_prob:
            phrases, weights = zip(*self.anchored_airlines)
            phrase = rng.choices(phrases, weights=weights, k=1)[0]
            return phrase, self.airlines[phrase]
        phrase = rng.choice(AIRLINE_TELEPHONY)
        return phrase, self.airlines[phrase]

    def pick_station(self, rng: random.Random, region: str) -> str:
        builtin = EU_STATIONS if region == "eu" else US_STATIONS
        if self.stations and region == "eu" and rng.random() < 0.6:
            return rng.choice(self.stations)
        return rng.choice(builtin)

    def pick_waypoint(self, rng: random.Random, region: str) -> str:
        builtin = EU_WAYPOINTS if region == "eu" else US_WAYPOINTS
        if self.waypoints and rng.random() < 0.6:
            return rng.choice(self.waypoints)
        return rng.choice(builtin)


# ---------------------------------------------------------------------------
# Slots: canonical value -> spoken words + display form + ground-truth entity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    """A rendered value. `str(slot)` is what the microphone hears."""

    spoken: str
    display: str
    entity: Entity | None = None

    def __str__(self) -> str:
        return self.spoken


def plain(spoken: str, display: str | None = None) -> Slot:
    """A slot with no ground truth attached (stations, winds, greetings)."""
    return Slot(spoken=spoken, display=display if display is not None else spoken)


def _digits(value: str, style: Style) -> str:
    return digits_to_words(value, style.atc_variants)


def _render_runway(value: str, style: Style) -> str:
    number, side = value[:2], value[2:]
    spoken = _digits(number, style)
    return f"{spoken} {SIDE_WORDS[side]}" if side else spoken


def _render_frequency(value: str, style: Style) -> str:
    whole, fraction = value.split(".")
    fraction = fraction.rstrip("0") or "0"
    head = group_number_words(int(whole), style.atc_variants) if style.group_numbers \
        else _digits(whole, style)
    return f"{head} {style.decimal_word} {_digits(fraction, style)}"


def _render_altimeter(value: str, style: Style) -> str:
    return _digits(value[1:], style)


def _render_callsign(value: str, style: Style, vocab: Vocab) -> str:
    code = value[:_ident_start(value)]
    ident = value[_ident_start(value):]
    phrase = vocab.phrase_for(code)
    if code == "N" and phrase == "n":
        phrase = "november"
    if ident.isdigit() and style.group_numbers and not ident.startswith("0"):
        words = [group_number_words(int(ident), style.atc_variants)]
    else:
        table = ATC_DIGITS if style.atc_variants else PLAIN_DIGITS
        words = [table[char] if char.isdigit() else PHONETIC_ALPHABET[char]
                 for char in ident]
    return f"{phrase} {' '.join(words)}"


def _ident_start(value: str) -> int:
    """Index where the ident begins: 'CSA3KF' -> 3, 'N123AB' -> 1."""
    for index, char in enumerate(value):
        if char.isdigit():
            return index
    return len(value)


RENDERERS = {
    "runway": _render_runway,
    "heading": lambda value, style: _digits(value, style),
    "flight_level": lambda value, style: _digits(value[2:], style),
    "altitude": lambda value, style: feet_to_words(int(value[:-2]), style.atc_variants),
    "frequency": _render_frequency,
    "speed": lambda value, style: _digits(value, style),
    "squawk": lambda value, style: _digits(value, style),
    "altimeter": _render_altimeter,
    "waypoint": lambda value, style: value.lower(),
    "atis": lambda value, style: PHONETIC_ALPHABET[value],
}


def make_slot(type_: str, value: str, style: Style,
              vocab: Vocab | None = None) -> Slot:
    """Render `value` and attach its ground-truth entity."""
    if type_ == "callsign":
        spoken = _render_callsign(value, style, vocab or Vocab())
    else:
        spoken = RENDERERS[type_](value, style)
    return Slot(spoken=spoken, display=value,
                entity=Entity(type=type_, value=value, spoken=spoken))


# ---------------------------------------------------------------------------
# Value pickers
# ---------------------------------------------------------------------------

def pick_callsign(rng: random.Random, vocab: Vocab, region: str) -> str:
    if region == "us" and rng.random() < 0.3:
        body = "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 3)))
        suffix = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
        return "N" + body + suffix
    _, code = vocab.pick_airline(rng)
    if region == "eu" and rng.random() < 0.2:      # alphanumeric European ident
        return f"{code}{rng.randint(1, 9)}" + "".join(
            rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(2))
    return f"{code}{rng.randint(1, 9999)}"


def abbreviate_callsign(value: str, rng: random.Random) -> str | None:
    """Shortened readback form ('CSA926' -> 'CSA26'), or None if unsafe.

    Only purely numeric idents abbreviate: dropping digits from an
    alphanumeric ident can leave something that is no longer a legal
    callsign, and the label has to stay legal because it describes audio
    that was really spoken.
    """
    start = _ident_start(value)
    code, ident = value[:start], value[start:]
    if not ident.isdigit() or len(ident) < 3:
        return None
    keep = rng.randint(1, len(ident) - 1)
    tail = ident[-keep:].lstrip("0") or ident[-1]
    return code + tail


def callsigns_consistent(a: str, b: str) -> bool:
    """Whether two callsigns can be the same aircraft (one may be abbreviated)."""
    if a == b:
        return True
    start_a, start_b = _ident_start(a), _ident_start(b)
    if a[:start_a] != b[:start_b]:
        return False
    ident_a, ident_b = a[start_a:], b[start_b:]
    return ident_a.endswith(ident_b) or ident_b.endswith(ident_a)


def confusable_callsign(value: str, rng: random.Random) -> str:
    """A callsign a controller could mix up with `value`: same airline, one
    digit apart. This is the confusion pair the research wants represented."""
    start = _ident_start(value)
    code, ident = value[:start], value[start:]
    positions = [index for index, char in enumerate(ident) if char.isdigit()]
    position = rng.choice(positions)
    digit = rng.choice([d for d in "0123456789" if d != ident[position]])
    return code + ident[:position] + digit + ident[position + 1:]


def corrupt_value(type_: str, value: str, rng: random.Random,
                  attempts: int = 20) -> str | None:
    """A legal but different value, one digit away -- a plausible misheard
    readback. Returns None when no legal neighbour exists."""
    positions = [index for index, char in enumerate(value) if char.isdigit()]
    if not positions:
        return None
    for _ in range(attempts):
        position = rng.choice(positions)
        digit = rng.choice([d for d in "0123456789" if d != value[position]])
        candidate = value[:position] + digit + value[position + 1:]
        if check_value(type_, candidate) is None:
            return candidate
    return None


def pick_runway(rng: random.Random) -> str:
    number = f"{rng.randint(1, 36):02d}"
    return number + rng.choice(["", "", "", "L", "R", "C"])


def pick_heading(rng: random.Random) -> str:
    heading = rng.randrange(10, 360, 10) if rng.random() < 0.5 else rng.randint(1, 360)
    return f"{heading:03d}"


def pick_flight_level(rng: random.Random) -> str:
    return f"FL{rng.randrange(80, 410, 10):03d}"


def pick_altitude(rng: random.Random) -> str:
    feet = rng.randrange(1000, 13000, 500) if rng.random() < 0.7 \
        else rng.randrange(300, 3000, 100)
    return f"{feet}ft"


def pick_frequency(rng: random.Random, region: str) -> str:
    mhz = rng.randint(118, 136)
    if region == "eu" and rng.random() < 0.5:      # 8.33 kHz channel names
        khz = rng.choice([0, 5, 10, 15, 30, 35, 40, 55, 60, 65, 80, 85, 90])
        khz += rng.choice([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    else:
        khz = rng.choice([0, 25, 50, 75]) + rng.choice(range(0, 1000, 100))
    return f"{mhz}.{min(khz, 975):03d}"


def pick_speed(rng: random.Random) -> str:
    return str(rng.randrange(180, 320, 10))


def pick_squawk(rng: random.Random) -> str:
    return "".join(rng.choice("01234567") for _ in range(4))


def pick_altimeter(rng: random.Random, region: str) -> str:
    if region == "us":
        return f"A{rng.randint(2892, 3095)}"
    return f"Q{rng.randint(985, 1035)}"


def pick_atis(rng: random.Random) -> str:
    return rng.choice(sorted(PHONETIC_ALPHABET))


# ---------------------------------------------------------------------------
# Non-entity phrases
# ---------------------------------------------------------------------------

def wind_phrase(rng: random.Random, style: Style, region: str) -> Slot:
    """Wind is read as a bare number, never labelled: it carries no clearance."""
    heading = rng.randrange(10, 360, 10)
    speed = rng.randint(3, 28)
    spoken = f"wind {_digits(f'{heading:03d}', style)}"
    display = f"wind {heading:03d}"
    if region == "eu":
        spoken += f" degrees {group_number_words(speed, style.atc_variants)} knots"
        display += f" degrees {speed} knots"
    else:
        spoken += f" at {group_number_words(speed, style.atc_variants)}"
        display += f" at {speed}"
        if rng.random() < 0.2:
            gust = speed + rng.randint(4, 12)
            spoken += f" gusting {group_number_words(gust, style.atc_variants)}"
            display += f" gusting {gust}"
    return plain(spoken, display)


def taxi_route(rng: random.Random) -> Slot:
    route = " ".join(rng.sample(HOLDING_POINTS, rng.randint(1, 3)))
    return plain(route, route.title())


def station_slot(rng: random.Random, vocab: Vocab, region: str) -> Slot:
    name = vocab.pick_station(rng, region)
    return plain(name, name.title())


# ---------------------------------------------------------------------------
# Compatibility helpers (kept for training/evaluate.py and existing tests)
# ---------------------------------------------------------------------------

def spell_digits(text: str) -> str:
    """'123' -> 'one two tree'."""
    return digits_to_words(text, atc_variants=True)


def spell_alnum(text: str) -> str:
    """'23AB' -> 'two tree alpha bravo'."""
    out = []
    for char in text.upper():
        if char.isdigit():
            out.append(ATC_DIGITS[char])
        elif char in PHONETIC_ALPHABET:
            out.append(PHONETIC_ALPHABET[char])
    return " ".join(out)


def group_number(number: int) -> str:
    """Spoken group form: 412 -> 'four twelve', 1850 -> 'eighteen fifty'."""
    return group_number_words(number, atc_variants=True)
