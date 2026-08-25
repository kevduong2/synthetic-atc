"""Template grammar for controller/pilot radio exchanges.

Every utterance carries four things:

* `spoken`     -- what the TTS voice says, commas and all (prosody matters).
* `transcript` -- the ASR label: verbatim spoken words, lowercase, no
  punctuation. This is the ATCO2 convention the real eval corpus uses.
* `entities`   -- structured ground truth, built *structurally* while the
  text is composed. The grammar never parses its own output to label it.
* `display`    -- the human/controller-display rendering, "CSA123, contact
  Praha Radar 127.825".

Nothing leaves this module unvalidated: `generate_exchange` re-derives the
entities from the text with `atcgen.entities.extract_entities` and refuses to
emit an utterance whose label does not round-trip, whose values are out of
legal range, or whose callsigns are inconsistent across the exchange.

Two phraseology regions are supported. `eu` follows ICAO/European R/T (flight
levels, "decimal" frequencies, QNH, station names like "praha radar"), which
is what the evaluation corpus contains; `us` is FAA tower/approach style.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..entities import (CRITICAL_TYPES, DEFAULT_VOCAB_PATH, Entity,
                        check_entity, extract_entities)
from . import lexicon as lx
from .lexicon import Slot, Style, Vocab, make_slot, plain

REGIONS = ("us", "eu", "mixed")


@dataclass
class Utterance:
    spoken: str
    transcript: str
    role: str  # "controller" | "pilot"
    kind: str
    meta: dict = field(default_factory=dict)
    weight: float = 1.0        # relative sampling weight within its category
    category: str = "routine"  # "routine" | "emergency" | "rare_vocab" | ...
    entities: list[Entity] = field(default_factory=list)
    display: str = ""          # "" means: same as the transcript


@dataclass
class ScenarioConfig:
    """Generation knobs for the scenario service."""

    region: str = "mixed"
    #: Pilot reads a value back wrong; the controller corrects it. Labels
    #: follow the audio, so the wrong readback is labelled with what was said.
    readback_error_prob: float = 0.05
    #: Two callsigns one digit apart inside a single transmission.
    confusable_callsign_prob: float = 0.05
    #: How often a transmission uses radio variants (niner/tree/fife/fower)
    #: and grouped digits ("one twenty seven") instead of plain readings.
    phonetic_respelling_prob: float = 0.5
    vocab_path: str | Path | None = DEFAULT_VOCAB_PATH
    max_retries: int = 25

    def __post_init__(self) -> None:
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}: {self.region!r}")
        for name in ("readback_error_prob", "confusable_callsign_prob",
                     "phonetic_respelling_prob"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]: {value!r}")


@lru_cache(maxsize=8)
def _cached_vocab(path: str | None) -> Vocab:
    return Vocab.load(path)


def load_vocab(path: str | Path | None = DEFAULT_VOCAB_PATH) -> Vocab:
    """Anchored vocabulary, cached per path. Missing file -> builtin lists."""
    return _cached_vocab(str(path) if path is not None else None)


# ---------------------------------------------------------------------------
# Line composition
# ---------------------------------------------------------------------------

def _transcript(spoken: str) -> str:
    """ASR label form: the spoken words, without the prosody punctuation."""
    stripped = "".join(" " if char in ",.!?;:" else char for char in spoken)
    return " ".join(stripped.split())


@dataclass
class Line:
    """One transmission under construction: literals and slots in order."""

    role: str
    kind: str
    category: str = "routine"
    parts: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, *parts) -> "Line":
        self.parts.extend(parts)
        return self

    def prepend(self, *parts) -> "Line":
        self.parts[:0] = parts
        return self

    def build(self) -> Utterance:
        spoken, display, entities = [], [], []
        for part in self.parts:
            if isinstance(part, Slot):
                spoken.append(part.spoken)
                display.append(part.display)
                if part.entity is not None:
                    entities.append(part.entity)
            else:
                spoken.append(str(part))
                display.append(str(part))
        text = " ".join("".join(spoken).split())
        return Utterance(
            spoken=text, transcript=_transcript(text), role=self.role,
            kind=self.kind, category=self.category, meta=dict(self.meta),
            entities=entities, display=" ".join("".join(display).split()),
        )


# ---------------------------------------------------------------------------
# Scenario: the per-exchange draw of callsign, style and slots
# ---------------------------------------------------------------------------

#: Phrase that re-anchors a corrected value so the parser can find it again.
CORRECTION_ANCHORS = {
    "runway": "runway ",
    "heading": "heading ",
    "flight_level": "flight level ",
    "altitude": "altitude ",
    "frequency": "frequency ",
    "speed": "speed ",
    "squawk": "squawk ",
}


class Scenario:
    """Mutable state for one exchange: who is talking, how, and about what."""

    def __init__(self, rng: random.Random, config: ScenarioConfig, vocab: Vocab):
        self.rng = rng
        self.config = config
        self.vocab = vocab
        self.region = config.region if config.region != "mixed" else (
            "eu" if rng.random() < 0.6 else "us")
        self.style = Style.draw(rng, config.phonetic_respelling_prob, self.region)
        self.airlines = vocab.airlines
        self._callsign_value = lx.pick_callsign(rng, vocab, self.region)
        self.expected_callsigns = [self._callsign_value]
        self._error_budget = 1 if rng.random() < config.readback_error_prob else 0
        self.corrections: list[tuple[str, str, Slot]] = []
        self.readback_error: dict | None = None

    # -- slots ------------------------------------------------------------
    def slot(self, type_: str, value: str) -> Slot:
        return make_slot(type_, value, self.style, self.vocab)

    @property
    def callsign(self) -> Slot:
        return self.slot("callsign", self._callsign_value)

    def readback_callsign(self) -> Slot:
        """The callsign as the pilot says it -- often abbreviated.

        An abbreviated form is labelled with what was actually spoken
        ("CSA26"), not with the aircraft's full ident: the label describes
        the audio. Exchange-level validation checks the two are compatible.
        """
        if self.rng.random() < 0.35:
            short = lx.abbreviate_callsign(self._callsign_value, self.rng)
            if short:
                return self.slot("callsign", short)
        return self.callsign

    def confusable_callsign(self) -> Slot:
        value = lx.confusable_callsign(self._callsign_value, self.rng)
        self.expected_callsigns.append(value)
        return self.slot("callsign", value)

    def runway(self) -> Slot:
        return self.slot("runway", lx.pick_runway(self.rng))

    def heading(self) -> Slot:
        return self.slot("heading", lx.pick_heading(self.rng))

    def flight_level(self) -> Slot:
        return self.slot("flight_level", lx.pick_flight_level(self.rng))

    def altitude(self) -> Slot:
        return self.slot("altitude", lx.pick_altitude(self.rng))

    def frequency(self) -> Slot:
        return self.slot("frequency", lx.pick_frequency(self.rng, self.region))

    def speed(self) -> Slot:
        return self.slot("speed", lx.pick_speed(self.rng))

    def squawk(self) -> Slot:
        return self.slot("squawk", lx.pick_squawk(self.rng))

    def altimeter(self) -> Slot:
        return self.slot("altimeter", lx.pick_altimeter(self.rng, self.region))

    def waypoint(self) -> Slot:
        name = self.vocab.pick_waypoint(self.rng, self.region)
        return self.slot("waypoint", name.upper())

    def atis(self) -> Slot:
        return self.slot("atis", lx.pick_atis(self.rng))

    def station(self) -> Slot:
        return lx.station_slot(self.rng, self.vocab, self.region)

    def wind(self) -> Slot:
        return lx.wind_phrase(self.rng, self.style, self.region)

    def spoken_number(self, value: int) -> str:
        """An unlabelled quantity (miles, clock position) in this style."""
        return lx.group_number_words(value, self.style.atc_variants)

    def greeting(self) -> str:
        return self.rng.choice(lx.GREETINGS)

    def signoff(self) -> str:
        return self.rng.choice(lx.SIGNOFFS)

    def maybe(self, probability: float) -> bool:
        return self.rng.random() < probability

    # -- readback errors --------------------------------------------------
    def rb(self, slot: Slot) -> Slot:
        """Readback of `slot`: usually itself, occasionally a wrong value.

        Spending the error budget also queues the controller's correction,
        which `build_exchange` appends after the template's own lines.
        """
        entity = slot.entity
        if (not self._error_budget or entity is None
                or entity.type not in CORRECTION_ANCHORS):
            return slot
        wrong = lx.corrupt_value(entity.type, entity.value, self.rng)
        if wrong is None:
            return slot
        self._error_budget = 0
        wrong_slot = self.slot(entity.type, wrong)
        self.corrections.append((entity.type, entity.value, wrong_slot))
        self.readback_error = {"type": entity.type, "said": wrong,
                               "correct": entity.value}
        return wrong_slot


# ---------------------------------------------------------------------------
# European / ICAO exchanges (the flavour of the evaluation corpus)
# ---------------------------------------------------------------------------

def eu_initial_contact(ctx: Scenario) -> list[Line]:
    cs, station, level = ctx.callsign, ctx.station(), ctx.flight_level()
    pilot = Line("pilot", "checkin").add(
        station, " ", ctx.greeting(), " ", cs,
        plain(" maintaining flight level ", " maintaining "), level)
    ctrl = Line("controller", "checkin_reply").add(
        cs, " ", station, ", identified")
    return [pilot, ctrl]


def eu_climb(ctx: Scenario) -> list[Line]:
    cs, level = ctx.callsign, ctx.flight_level()
    ctrl = Line("controller", "climb").add(
        cs, plain(", climb to flight level ", ", climb to "), level)
    pilot = Line("pilot", "climb_readback").add(
        plain("climbing to flight level ", "climbing to "), ctx.rb(level),
        ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_descend(ctx: Scenario) -> list[Line]:
    cs, level = ctx.callsign, ctx.flight_level()
    ctrl = Line("controller", "descend").add(
        cs, plain(", descend to flight level ", ", descend to "), level)
    if ctx.maybe(0.3):
        ctrl.add(", no atc speed restriction")
    pilot = Line("pilot", "descend_readback").add(
        plain("descending to flight level ", "descending to "), ctx.rb(level),
        ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_direct(ctx: Scenario) -> list[Line]:
    cs, point = ctx.callsign, ctx.waypoint()
    ctrl = Line("controller", "direct").add(cs, ", proceed direct ", point)
    pilot = Line("pilot", "direct_readback").add(
        "direct ", point, ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_frequency_change(ctx: Scenario) -> list[Line]:
    cs, station, freq = ctx.callsign, ctx.station(), ctx.frequency()
    ctrl = Line("controller", "freq_change").add(cs, ", contact ", station, " ", freq)
    if ctx.maybe(0.4):
        ctrl.add(", ", ctx.signoff())
    pilot = Line("pilot", "freq_change_readback").add(
        station, " ", ctx.rb(freq), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_landing(ctx: Scenario) -> list[Line]:
    cs, rwy = ctx.callsign, ctx.runway()
    ctrl = Line("controller", "landing").add(
        cs, ", runway ", rwy, ", cleared to land, ", ctx.wind())
    pilot = Line("pilot", "landing_readback").add(
        "cleared to land runway ", ctx.rb(rwy), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_takeoff(ctx: Scenario) -> list[Line]:
    cs, rwy = ctx.callsign, ctx.runway()
    ctrl = Line("controller", "takeoff").add(
        cs, ", line up runway ", rwy, " and wait")
    pilot = Line("pilot", "takeoff_readback").add(
        "lining up runway ", ctx.rb(rwy), ", ", ctx.readback_callsign())
    clear = Line("controller", "takeoff_clearance").add(
        cs, ", runway ", rwy, ", ", ctx.wind(), ", cleared for takeoff")
    return [ctrl, pilot, clear]


def eu_heading(ctx: Scenario) -> list[Line]:
    cs, hdg = ctx.callsign, ctx.heading()
    turn = ctx.rng.choice(["turn left heading ", "turn right heading ",
                           "fly heading "])
    ctrl = Line("controller", "heading").add(cs, ", ", turn, hdg)
    if ctx.maybe(0.3):
        ctrl.add(", vectoring for the ils")
    pilot = Line("pilot", "heading_readback").add(
        turn, ctx.rb(hdg), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_speed(ctx: Scenario) -> list[Line]:
    cs, speed = ctx.callsign, ctx.speed()
    ctrl = Line("controller", "speed").add(
        cs, ", reduce speed to ", speed, " knots")
    pilot = Line("pilot", "speed_readback").add(
        "speed ", ctx.rb(speed), " knots, ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_qnh(ctx: Scenario) -> list[Line]:
    cs, alt, qnh = ctx.callsign, ctx.altitude(), ctx.altimeter()
    ctrl = Line("controller", "qnh").add(
        cs, ", descend to altitude ", alt, plain(" feet", ""),
        plain(", qnh ", ", "), qnh)
    pilot = Line("pilot", "qnh_readback").add(
        "altitude ", alt, plain(" feet", ""), plain(", qnh ", ", "), ctx.rb(qnh),
        ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_squawk(ctx: Scenario) -> list[Line]:
    cs, code = ctx.callsign, ctx.squawk()
    ctrl = Line("controller", "squawk").add(cs, ", squawk ", code)
    pilot = Line("pilot", "squawk_readback").add(
        "squawk ", ctx.rb(code), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def eu_ils(ctx: Scenario) -> list[Line]:
    cs, rwy = ctx.callsign, ctx.runway()
    ctrl = Line("controller", "ils", category="rare_vocab").add(
        cs, ", cleared ils approach runway ", rwy, ", report established")
    pilot = Line("pilot", "ils_readback", category="rare_vocab").add(
        "cleared ils approach runway ", ctx.rb(rwy), ", ",
        ctx.readback_callsign())
    return [ctrl, pilot]


def eu_report_level(ctx: Scenario) -> list[Line]:
    cs, leaving, cleared = ctx.callsign, ctx.flight_level(), ctx.flight_level()
    pilot = Line("pilot", "level_report").add(
        cs, plain(", leaving flight level ", ", leaving "), leaving,
        plain(" for flight level ", " for "), cleared)
    ctrl = Line("controller", "level_ack").add(cs, ", roger")
    return [pilot, ctrl]


def eu_emergency(ctx: Scenario) -> list[Line]:
    cs, alt = ctx.callsign, ctx.altitude()
    kind = ctx.rng.choice(["mayday mayday mayday", "pan pan pan pan pan pan"])
    pilot = Line("pilot", "emergency", category="emergency").add(
        kind, ", ", cs, ", ", ctx.rng.choice([
            "engine failure", "smoke in the cockpit", "medical emergency on board",
            "hydraulic failure"]),
        plain(", descending to altitude ", ", descending to "), alt,
        plain(" feet", ""))
    ctrl = Line("controller", "emergency_reply", category="emergency").add(
        cs, ", roger mayday, ",
        plain("descend to altitude ", "descend to "), alt,
        plain(" feet, report your intentions", ", report your intentions"))
    return [pilot, ctrl]


# ---------------------------------------------------------------------------
# US / FAA exchanges
# ---------------------------------------------------------------------------

def us_takeoff(ctx: Scenario) -> list[Line]:
    cs, rwy = ctx.callsign, ctx.runway()
    ctrl = Line("controller", "takeoff").add(
        cs, ", runway ", rwy, ", ", ctx.wind(), ", cleared for takeoff")
    if ctx.maybe(0.3):
        hdg = ctx.heading()
        ctrl = Line("controller", "takeoff").add(
            cs, ", runway ", rwy, ", fly heading ", hdg, ", cleared for takeoff")
    pilot = Line("pilot", "takeoff_readback").add(
        "cleared for takeoff runway ", ctx.rb(rwy), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_landing(ctx: Scenario) -> list[Line]:
    cs, rwy = ctx.callsign, ctx.runway()
    ctrl = Line("controller", "landing").add(
        cs, ", runway ", rwy, ", ", ctx.wind(), ", cleared to land")
    pilot = Line("pilot", "landing_readback").add(
        "cleared to land runway ", ctx.rb(rwy), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_lineup_wait(ctx: Scenario) -> list[Line]:
    cs, rwy = ctx.callsign, ctx.runway()
    ctrl = Line("controller", "lineup").add(cs, ", runway ", rwy, ", line up and wait")
    reason = ctx.rng.choice(["traffic on final", "traffic crossing downfield",
                             "landing traffic", ""])
    if reason:
        ctrl.add(", ", reason)
    pilot = Line("pilot", "lineup_readback").add(
        "line up and wait runway ", ctx.rb(rwy), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_taxi(ctx: Scenario) -> list[Line]:
    cs, rwy, route = ctx.callsign, ctx.runway(), lx.taxi_route(ctx.rng)
    hold = ctx.maybe(0.4)
    ctrl = Line("controller", "taxi").add(cs, ", runway ", rwy, ", taxi via ", route)
    pilot = Line("pilot", "taxi_readback").add("taxi via ", route)
    if hold:
        held = ctx.runway()
        ctrl.add(", hold short of runway ", held)
        pilot.add(", hold short runway ", held)
    pilot.add(", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_frequency_change(ctx: Scenario) -> list[Line]:
    cs, station, freq = ctx.callsign, ctx.station(), ctx.frequency()
    ctrl = Line("controller", "freq_change").add(cs, ", contact ", station, " ", freq)
    if ctx.maybe(0.2):
        ctrl.add(", good day")
    pilot = Line("pilot", "freq_change_readback").add(
        "over to ", station, " ", ctx.rb(freq), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_traffic(ctx: Scenario) -> list[Line]:
    cs = ctx.callsign
    clock = ctx.spoken_number(ctx.rng.randint(1, 12))
    miles = ctx.spoken_number(ctx.rng.randint(1, 9))
    ctrl = Line("controller", "traffic").add(
        cs, f", traffic {clock} o'clock, {miles} miles, ",
        ctx.rng.choice(lx.DIRECTIONS), "bound, ",
        ctx.rng.choice(lx.AIRCRAFT_TRAFFIC_TYPES))
    pilot = Line("pilot", "traffic_reply").add(
        ctx.rng.choice(["looking for traffic", "traffic in sight",
                        "negative contact"]), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_altitude(ctx: Scenario) -> list[Line]:
    cs, alt = ctx.callsign, ctx.altitude()
    verb = ctx.rng.choice(["climb and maintain ", "descend and maintain ",
                           "maintain "])
    ctrl = Line("controller", "altitude").add(cs, ", ", verb, alt)
    pilot = Line("pilot", "altitude_readback").add(
        verb, ctx.rb(alt), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_heading(ctx: Scenario) -> list[Line]:
    cs, hdg = ctx.callsign, ctx.heading()
    turn = ctx.rng.choice(["turn left heading ", "turn right heading ",
                           "fly heading "])
    ctrl = Line("controller", "heading").add(cs, ", ", turn, hdg)
    pilot = Line("pilot", "heading_readback").add(
        turn, ctx.rb(hdg), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_squawk(ctx: Scenario) -> list[Line]:
    cs, code = ctx.callsign, ctx.squawk()
    ctrl = Line("controller", "squawk").add(cs, ", squawk ", code)
    pilot = Line("pilot", "squawk_readback").add(
        "squawk ", ctx.rb(code), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_checkin(ctx: Scenario) -> list[Line]:
    cs, atis = ctx.callsign, ctx.atis()
    style = ctx.rng.random()
    if style < 0.4:
        miles = ctx.spoken_number(ctx.rng.randint(4, 15))
        rwy = ctx.runway()
        pilot = Line("pilot", "checkin").add(
            "tower, ", cs, f", we are {miles} miles ",
            ctx.rng.choice(lx.DIRECTIONS), ", inbound with information ", atis)
        ctrl = Line("controller", "checkin_reply").add(
            cs, ", tower, enter left downwind runway ", rwy, ", report midfield")
    elif style < 0.7:
        pilot = Line("pilot", "checkin").add(
            "tower, ", cs, ", ready for departure, with information ", atis)
        ctrl = Line("controller", "checkin_reply").add(
            cs, ", tower, hold short, landing traffic")
    else:
        alt = ctx.altitude()
        pilot = Line("pilot", "checkin").add(
            "approach, ", cs, ", altitude ", alt, ", information ", atis)
        ctrl = Line("controller", "checkin_reply").add(
            cs, ", approach, expect vectors, descend and maintain ", ctx.altitude())
    return [pilot, ctrl]


def us_go_around(ctx: Scenario) -> list[Line]:
    cs, alt = ctx.callsign, ctx.altitude()
    ctrl = Line("controller", "go_around", category="emergency").add(
        cs, ", go around, fly runway heading, climb and maintain ", alt)
    pilot = Line("pilot", "go_around_readback", category="emergency").add(
        "going around, ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_altimeter(ctx: Scenario) -> list[Line]:
    cs, setting = ctx.callsign, ctx.altimeter()
    ctrl = Line("controller", "altimeter").add(cs, ", altimeter ", setting)
    pilot = Line("pilot", "altimeter_readback").add(
        "altimeter ", ctx.rb(setting), ", ", ctx.readback_callsign())
    return [ctrl, pilot]


def us_cancel_takeoff(ctx: Scenario) -> list[Line]:
    cs = ctx.callsign
    ctrl = Line("controller", "cancel_takeoff", category="emergency").add(
        cs, ", hold position, cancel takeoff clearance, vehicle on the runway")
    pilot = Line("pilot", "cancel_readback", category="emergency").add(
        "holding position, ", ctx.readback_callsign())
    return [ctrl, pilot]


EU_EXCHANGES = [
    (eu_climb, 0.13),
    (eu_descend, 0.13),
    (eu_frequency_change, 0.14),
    (eu_initial_contact, 0.10),
    (eu_direct, 0.09),
    (eu_heading, 0.09),
    (eu_landing, 0.08),
    (eu_takeoff, 0.06),
    (eu_report_level, 0.05),
    (eu_speed, 0.05),
    (eu_qnh, 0.03),
    (eu_squawk, 0.03),
    (eu_ils, 0.01),
    (eu_emergency, 0.01),
]

US_EXCHANGES = [
    (us_takeoff, 0.13),
    (us_landing, 0.13),
    (us_taxi, 0.12),
    (us_frequency_change, 0.12),
    (us_altitude, 0.10),
    (us_traffic, 0.09),
    (us_heading, 0.08),
    (us_lineup_wait, 0.07),
    (us_checkin, 0.06),
    (us_squawk, 0.05),
    (us_go_around, 0.02),
    (us_altimeter, 0.02),
    (us_cancel_takeoff, 0.01),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_utterance(u: Utterance, airlines: dict[str, str] | None = None
                       ) -> list[str]:
    """Deterministic checks on one utterance. Empty list means valid.

    Beyond value ranges this asserts the property the whole pipeline rests
    on: parsing the transcript back recovers exactly the labelled entities.
    A label that cannot be recovered is a label the gate and the eval panel
    would silently disagree about.
    """
    problems: list[str] = []
    if not u.spoken.strip():
        problems.append("empty spoken text")
    if u.transcript != " ".join(u.transcript.split()):
        problems.append("transcript is not single-spaced")
    if u.role not in ("controller", "pilot", "unknown"):
        problems.append(f"unknown role {u.role!r}")

    for entity in u.entities:
        problem = check_entity(entity)
        if problem:
            problems.append(problem)
        elif entity.spoken and entity.spoken not in u.transcript:
            problems.append(
                f"{entity.type} {entity.value!r} claims spoken form "
                f"{entity.spoken!r}, absent from the transcript")

    recovered = Counter(e.key for e in extract_entities(u.transcript, airlines))
    for entity in u.entities:
        if recovered[entity.key] > 0:
            recovered[entity.key] -= 1
        else:
            problems.append(f"round-trip lost {entity.type} {entity.value!r}")
    for (type_, value), count in recovered.items():
        if count > 0 and type_ in CRITICAL_TYPES:
            problems.append(f"unlabelled {type_} {value!r} recoverable from text")
    return problems


def validate_exchange(utterances: list[Utterance],
                      expected_callsigns: list[str]) -> list[str]:
    """Controller and pilot must be talking about the same aircraft."""
    problems = []
    for u in utterances:
        for entity in u.entities:
            if entity.type != "callsign":
                continue
            if not any(lx.callsigns_consistent(entity.value, expected)
                       for expected in expected_callsigns):
                problems.append(
                    f"callsign {entity.value!r} is inconsistent with "
                    f"{expected_callsigns}")
    return problems


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_exchange(ctx: Scenario) -> list[Line]:
    exchanges = EU_EXCHANGES if ctx.region == "eu" else US_EXCHANGES
    fns, weights = zip(*exchanges)
    template = ctx.rng.choices(fns, weights=weights, k=1)[0]
    lines = template(ctx)

    for type_, correct, wrong_slot in ctx.corrections:
        at = len(lines)
        for position, line in enumerate(lines):
            if any(part is wrong_slot for part in line.parts):
                line.kind = f"{line.kind}_error"
                line.meta["readback_error"] = dict(ctx.readback_error or {})
                at = position + 1
        anchor = CORRECTION_ANCHORS[type_]
        lines[at:at] = [
            Line("controller", "correction").add(
                "negative ", ctx.callsign, ", ", anchor, ctx.slot(type_, correct)),
            Line("pilot", "corrected_readback").add(
                anchor, ctx.slot(type_, correct), ", ", ctx.readback_callsign()),
        ]

    if ctx.rng.random() < ctx.config.confusable_callsign_prob:
        for line in lines:
            if line.role == "controller":
                line.prepend(ctx.confusable_callsign(), ", standby, ")
                line.meta["confusable"] = True
                break

    for line in lines:
        line.meta.setdefault("region", ctx.region)
        line.meta.setdefault("template", template.__name__)
    return lines


def generate_exchange(rng: random.Random | None = None,
                      config: ScenarioConfig | None = None,
                      vocab: Vocab | None = None) -> list[Utterance]:
    """One controller/pilot exchange, validated before it is returned.

    Retries on violation (each attempt redraws from `rng`) and raises if the
    grammar cannot produce a clean exchange -- an invalid utterance must
    never reach the dataset, because its label would not describe its audio.
    """
    rng = rng or random.Random()
    config = config or ScenarioConfig()
    if vocab is None:
        vocab = load_vocab(config.vocab_path)

    problems: list[str] = []
    for _ in range(config.max_retries):
        ctx = Scenario(rng, config, vocab)
        utterances = [line.build() for line in _build_exchange(ctx)]
        problems = [problem for u in utterances
                    for problem in validate_utterance(u, ctx.airlines)]
        problems += validate_exchange(utterances, ctx.expected_callsigns)
        if not problems:
            return utterances
    raise ValueError(
        f"no valid exchange after {config.max_retries} attempts: {problems}")


def generate_utterance(rng: random.Random | None = None,
                       config: ScenarioConfig | None = None,
                       vocab: Vocab | None = None) -> Utterance:
    """A single utterance sampled from a random exchange."""
    rng = rng or random.Random()
    return rng.choice(generate_exchange(rng, config, vocab))
