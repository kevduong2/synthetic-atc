"""Vocabulary for ATC phraseology: spoken digits, phonetic alphabet, callsigns.

Spoken forms follow FAA Order 7110.65 / ICAO Annex 10 conventions:
"tree" for 3, "fife" for 5, "niner" for 9.
"""

import random

DIGITS_SPOKEN = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "tree",
    "4": "four",
    "5": "fife",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "niner",
}

PHONETIC_ALPHABET = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo",
    "F": "foxtrot", "G": "golf", "H": "hotel", "I": "india", "J": "juliett",
    "K": "kilo", "L": "lima", "M": "mike", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey", "X": "xray",
    "Y": "yankee", "Z": "zulu",
}

AIRLINE_TELEPHONY = [
    "american", "united", "delta", "southwest", "jetblue", "alaska",
    "skywest", "envoy", "republic", "spirit", "frontier", "cargo jet",
    "fedex", "u p s", "speedbird", "lufthansa", "air france", "k l m",
    "aeromexico", "air canada", "westjet", "qantas", "emirates",
    "singapore", "cathay", "china southern", "korean air", "ryanair",
    "easy", "wizz air", "citation", "gulfstream test",
]

GA_TYPES = ["cessna", "piper", "cirrus", "beechcraft", "mooney", "bonanza", "skyhawk", "archer"]

HOLDING_POINTS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "juliett", "kilo", "mike", "papa"]

ATIS_LETTERS = list(PHONETIC_ALPHABET.values())

AIRCRAFT_TRAFFIC_TYPES = [
    "a cessna one seventy two", "a boeing seven thirty seven", "an airbus a three twenty",
    "a piper cherokee", "a regional jet", "a king air", "a helicopter", "a citation jet",
]

DIRECTIONS = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]


def spell_digits(s: str) -> str:
    """'123' -> 'one two tree'."""
    return " ".join(DIGITS_SPOKEN[c] for c in s if c in DIGITS_SPOKEN)


def spell_alnum(s: str) -> str:
    """'23AB' -> 'two tree alpha bravo'."""
    out = []
    for c in s.upper():
        if c in DIGITS_SPOKEN:
            out.append(DIGITS_SPOKEN[c])
        elif c in PHONETIC_ALPHABET:
            out.append(PHONETIC_ALPHABET[c])
    return " ".join(out)


def group_number(n: int) -> str:
    """Spoken group form for flight numbers, e.g. 412 -> 'four twelve', 1850 -> 'eighteen fifty'.

    Controllers often use group form for airline flight numbers; digit-by-digit
    is also used. We pick group form when it reads naturally.
    """
    s = str(n)
    teens = {
        "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
        "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
        "18": "eighteen", "19": "nineteen",
    }
    tens = {
        "2": "twenty", "3": "thirty", "4": "forty", "5": "fifty",
        "6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety",
    }

    def two_digit(t: str) -> str:
        if t in teens:
            return teens[t]
        if t[0] == "0":
            return "zero " + DIGITS_SPOKEN[t[1]]
        if t[1] == "0":
            return tens[t[0]]
        return tens[t[0]] + " " + DIGITS_SPOKEN[t[1]]

    if len(s) == 1:
        return DIGITS_SPOKEN[s]
    if len(s) == 2:
        return two_digit(s)
    if len(s) == 3:
        return DIGITS_SPOKEN[s[0]] + " " + two_digit(s[1:])
    if len(s) == 4:
        return two_digit(s[:2]) + " " + two_digit(s[2:])
    return spell_digits(s)


def random_airline_callsign(rng: random.Random) -> str:
    airline = rng.choice(AIRLINE_TELEPHONY)
    number = rng.randint(1, 9999)
    spoken = group_number(number) if rng.random() < 0.6 else spell_digits(str(number))
    return f"{airline} {spoken}"


def random_ga_callsign(rng: random.Random) -> str:
    """GA N-number, spoken with type prefix or 'november'."""
    body = "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 3)))
    suffix = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    tail = spell_alnum(body + suffix)
    if rng.random() < 0.7:
        return f"{rng.choice(GA_TYPES)} {tail}"
    return f"november {tail}"


def random_callsign(rng: random.Random) -> str:
    return random_airline_callsign(rng) if rng.random() < 0.65 else random_ga_callsign(rng)


def short_callsign(callsign: str) -> str:
    """Abbreviated readback form: last chunk of the callsign."""
    words = callsign.split()
    if len(words) <= 3:
        return callsign
    return " ".join(words[-3:])


def random_runway(rng: random.Random) -> str:
    num = rng.randint(1, 36)
    side = rng.choice(["", "", "", " left", " right", " center"])
    return spell_digits(str(num)) + side


def random_frequency(rng: random.Random) -> str:
    """VHF airband 118.0-136.975, spoken e.g. 'one one eight point seven'."""
    mhz = rng.randint(118, 136)
    khz = rng.choice(["0", "1", "2", "25", "3", "4", "5", "55", "6", "7", "75", "8", "9", "97"])
    return spell_digits(str(mhz)) + " point " + spell_digits(khz)


def random_altimeter(rng: random.Random) -> str:
    setting = rng.randint(2892, 3095)
    return spell_digits(str(setting))


def random_wind(rng: random.Random) -> str:
    heading = rng.randrange(10, 360, 10)
    speed = rng.randint(3, 28)
    wind = f"wind {spell_digits(str(heading).zfill(3))} at {group_number(speed)}"
    if rng.random() < 0.2:
        wind += f" gusting {group_number(speed + rng.randint(4, 12))}"
    return wind


def random_altitude(rng: random.Random) -> str:
    """Spoken altitude, e.g. 'four thousand fife hundred' or flight level."""
    if rng.random() < 0.25:
        fl = rng.randrange(180, 400, 10)
        return "flight level " + spell_digits(str(fl))
    thousands = rng.randint(2, 17)
    hundreds = rng.choice([0, 0, 5])
    alt = group_number(thousands) + " thousand"
    if hundreds:
        alt += f" {DIGITS_SPOKEN[str(hundreds)]} hundred"
    return alt


def random_heading(rng: random.Random) -> str:
    heading = rng.randrange(10, 360, 10) if rng.random() < 0.5 else rng.randint(1, 359)
    return spell_digits(str(heading).zfill(3))


def random_squawk(rng: random.Random) -> str:
    code = "".join(rng.choice("01234567") for _ in range(4))
    return spell_digits(code)


def random_taxiways(rng: random.Random) -> str:
    n = rng.randint(1, 3)
    return " ".join(rng.sample(HOLDING_POINTS, n))


def random_atis(rng: random.Random) -> str:
    return rng.choice(ATIS_LETTERS)
