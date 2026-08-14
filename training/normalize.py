"""ATC-aware text normalization for WER computation.

Folds radio-phonetic spellings to plain words so synthetic-transcript
conventions and real-corpus conventions score comparably.
"""

import re

FOLD = {
    "niner": "nine",
    "tree": "three",
    "fife": "five",
    "fower": "four",
    "juliett": "juliet",
    "xray": "x-ray",
    "okay": "ok",
}

_PUNCT = re.compile(r"[^\w\s']")

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _expand_token(tok: str) -> list[str]:
    """'067' -> ['zero','six','seven']; '35r' -> ['three','five','r']."""
    if not any(c.isdigit() for c in tok):
        return [tok]
    out = []
    for c in tok:
        if c.isdigit():
            out.append(_DIGIT_WORDS[c])
        else:
            out.append(c)
    return out


def normalize_atc(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = _PUNCT.sub(" ", text)
    words = []
    for tok in text.split():
        for w in _expand_token(tok):
            words.append(FOLD.get(w, w))
    return " ".join(words)
