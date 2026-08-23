"""Template grammar for tower-control radio exchanges.

Each generated utterance carries a spoken form (fed to TTS) and a transcript
label. The default convention is verbatim spoken words (ATCO2 style), so
transcript == spoken; eval-time normalization handles "niner"->"nine" etc.
"""

import random
from dataclasses import dataclass, field

from . import lexicon as lx


@dataclass
class Utterance:
    spoken: str
    transcript: str
    role: str  # "controller" | "pilot"
    kind: str
    meta: dict = field(default_factory=dict)
    weight: float = 1.0        # relative sampling weight within its category
    category: str = "routine"  # "routine" | "emergency" | "rare_vocab" | ...


def _utt(spoken: str, role: str, kind: str, **meta) -> Utterance:
    spoken = " ".join(spoken.split())
    return Utterance(spoken=spoken, transcript=spoken, role=role, kind=kind, meta=meta)


# ---------------------------------------------------------------------------
# Exchange templates. Each returns a list[Utterance] (controller + readback).
# ---------------------------------------------------------------------------

def takeoff_clearance(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    rwy = lx.random_runway(rng)
    wind = lx.random_wind(rng)
    ctrl = f"{cs}, runway {rwy}, {wind}, cleared for takeoff"
    if rng.random() < 0.3:
        hdg = lx.random_heading(rng)
        ctrl = f"{cs}, runway {rwy}, fly heading {hdg}, cleared for takeoff"
    pilot = f"cleared for takeoff runway {rwy}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "takeoff"), _utt(pilot, "pilot", "takeoff_readback")]


def landing_clearance(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    rwy = lx.random_runway(rng)
    wind = lx.random_wind(rng)
    ctrl = f"{cs}, runway {rwy}, {wind}, cleared to land"
    pilot = f"cleared to land runway {rwy}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "landing"), _utt(pilot, "pilot", "landing_readback")]


def lineup_wait(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    rwy = lx.random_runway(rng)
    reason = rng.choice(["traffic on final", "traffic crossing downfield", "landing traffic", ""])
    ctrl = f"{cs}, runway {rwy}, line up and wait"
    if reason:
        ctrl += f", {reason}"
    pilot = f"line up and wait runway {rwy}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "lineup"), _utt(pilot, "pilot", "lineup_readback")]


def taxi_instruction(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    rwy = lx.random_runway(rng)
    route = lx.random_taxiways(rng)
    hold = rng.random() < 0.4
    ctrl = f"{cs}, runway {rwy}, taxi via {route}"
    if hold:
        ctrl += f", hold short of runway {rwy}"
    pilot = f"taxi via {route}"
    if hold:
        pilot += f", hold short runway {rwy}"
    pilot += f", {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "taxi"), _utt(pilot, "pilot", "taxi_readback")]


def frequency_change(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    facility = rng.choice(["departure", "approach", "ground", "tower", "center"])
    freq = lx.random_frequency(rng)
    ctrl = f"{cs}, contact {facility} {freq}"
    if rng.random() < 0.2:
        ctrl += ", good day"
    pilot = f"over to {facility} {freq}, {lx.short_callsign(cs)}" if rng.random() < 0.5 \
        else f"{facility} {freq}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "freq_change"), _utt(pilot, "pilot", "freq_change_readback")]


def traffic_advisory(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    clock = rng.randint(1, 12)
    miles = rng.randint(1, 9)
    ac = rng.choice(lx.AIRCRAFT_TRAFFIC_TYPES)
    direction = rng.choice(lx.DIRECTIONS)
    ctrl = (f"{cs}, traffic {lx.group_number(clock)} o'clock, {lx.group_number(miles)} miles, "
            f"{direction}bound, {ac}")
    pilot = rng.choice(["looking for traffic", "traffic in sight", "negative contact"]) \
        + f", {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "traffic"), _utt(pilot, "pilot", "traffic_reply")]


def altitude_assignment(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    verb = rng.choice(["climb and maintain", "descend and maintain", "maintain"])
    alt = lx.random_altitude(rng)
    ctrl = f"{cs}, {verb} {alt}"
    pilot = f"{verb} {alt}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "altitude"), _utt(pilot, "pilot", "altitude_readback")]


def heading_assignment(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    turn = rng.choice(["turn left heading", "turn right heading", "fly heading"])
    hdg = lx.random_heading(rng)
    ctrl = f"{cs}, {turn} {hdg}"
    pilot = f"{turn} {hdg}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "heading"), _utt(pilot, "pilot", "heading_readback")]


def squawk_assignment(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    code = lx.random_squawk(rng)
    ctrl = f"{cs}, squawk {code}"
    pilot = f"squawk {code}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "squawk"), _utt(pilot, "pilot", "squawk_readback")]


def pilot_checkin(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    atis = lx.random_atis(rng)
    style = rng.random()
    if style < 0.4:
        miles = rng.randint(4, 15)
        direction = rng.choice(lx.DIRECTIONS)
        pilot = f"tower, {cs}, {lx.group_number(miles)} miles {direction}, inbound with {atis}"
        rwy = lx.random_runway(rng)
        ctrl = f"{cs}, tower, enter left downwind runway {rwy}, report midfield"
    elif style < 0.7:
        pilot = f"tower, {cs}, ready for departure, with {atis}"
        ctrl = f"{cs}, tower, hold short, landing traffic"
    else:
        alt = lx.random_altitude(rng)
        pilot = f"approach, {cs}, level {alt}, information {atis}"
        alt2 = lx.random_altitude(rng)
        ctrl = f"{cs}, approach, expect vectors, descend and maintain {alt2}"
    return [_utt(pilot, "pilot", "checkin"), _utt(ctrl, "controller", "checkin_reply")]


def go_around(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    ctrl = f"{cs}, go around, fly runway heading, climb and maintain {lx.random_altitude(rng)}"
    pilot = f"going around, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "go_around"), _utt(pilot, "pilot", "go_around_readback")]


def altimeter_report(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    setting = lx.random_altimeter(rng)
    ctrl = f"{cs}, altimeter {setting}"
    pilot = f"altimeter {setting}, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "altimeter"), _utt(pilot, "pilot", "altimeter_readback")]


def position_and_hold_cancel(rng: random.Random) -> list[Utterance]:
    cs = lx.random_callsign(rng)
    ctrl = f"{cs}, hold position, cancel takeoff clearance, vehicle on the runway"
    pilot = f"holding position, {lx.short_callsign(cs)}"
    return [_utt(ctrl, "controller", "cancel_takeoff"), _utt(pilot, "pilot", "cancel_readback")]


EXCHANGES = [
    (takeoff_clearance, 0.13),
    (landing_clearance, 0.13),
    (lineup_wait, 0.07),
    (taxi_instruction, 0.12),
    (frequency_change, 0.12),
    (traffic_advisory, 0.09),
    (altitude_assignment, 0.10),
    (heading_assignment, 0.08),
    (squawk_assignment, 0.05),
    (pilot_checkin, 0.06),
    (go_around, 0.02),
    (altimeter_report, 0.02),
    (position_and_hold_cancel, 0.01),
]


def generate_exchange(rng: random.Random | None = None) -> list[Utterance]:
    """One controller/pilot exchange (usually 2 utterances)."""
    rng = rng or random.Random()
    fns, weights = zip(*EXCHANGES)
    fn = rng.choices(fns, weights=weights, k=1)[0]
    return fn(rng)


def generate_utterance(rng: random.Random | None = None) -> Utterance:
    """A single utterance sampled from a random exchange."""
    rng = rng or random.Random()
    exchange = generate_exchange(rng)
    return rng.choice(exchange)
