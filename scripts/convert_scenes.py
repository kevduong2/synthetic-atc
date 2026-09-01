#!/usr/bin/env python
"""Flatten the scene corpus into the per-line JSONL `JsonlTextSource` reads.

`synthetic_generation_deployed_airports_v2.0.1.jsonl` is not JSONL: it is
concatenated *pretty-printed* JSON scene objects, so one scene spans hundreds of
lines and `JsonlTextSource`'s per-line `json.loads` fails on line 1.  This walks
the file with `json.JSONDecoder.raw_decode` and emits one utterance per line:

    uv run python scripts/convert_scenes.py \
        --scenes synthetic_generation_deployed_airports_v2.0.1.jsonl \
        --out data/text/scenes_v2.0.1.jsonl

    uv run python scripts/generate_dataset.py --config configs/mode1_matched.yaml \
        --n-samples 8 --out runs/smoke --text data/text/scenes_v2.0.1.jsonl

Field mapping, per message: `text` -> both `spoken` and `transcript` (the scene
text is already in spoken form), `speaker` -> `role`, `category` -> `category`,
and the scene's `airport_ident` -> `kind`, which `build_dataset` copies onto
every manifest row and is therefore the handle for per-airport bookkeeping over
a generated set.  `role` matters functionally: only `"pilot"` is eligible for
`dataset.pilot_double_hop_prob`'s ground relay.  ATIS broadcasts keep their own
`atis` role — they are neither, and nothing downstream validates the value.

`entities` is deliberately **not** emitted.  Scenes carry loose `slots`, not
`atcgen.entities` values, and `check_entity` raises on anything outside its
domains — one bad slot would kill the whole build.  Without entities the
verification gate still tiers rows (critical recall defaults to 1.0); only the
entity panel goes blind.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENES = ROOT / "synthetic_generation_deployed_airports_v2.0.1.jsonl"
DEFAULT_OUT = ROOT / "data" / "text" / "scenes_v2.0.1.jsonl"


def iter_scenes(path: Path):
    """Yield each scene object from a stream of concatenated JSON values."""
    text = path.read_text()
    decoder = json.JSONDecoder()
    index = 0
    end = len(text)
    while index < end:
        while index < end and text[index].isspace():
            index += 1
        if index >= end:
            return
        scene, index = decoder.raw_decode(text, index)
        yield scene


def utterances(scene: dict):
    """One record per message in `scene`, in the schema JsonlTextSource reads."""
    airport = scene.get("airport_ident") or "unknown"
    for message in scene.get("messages", []):
        spoken = (message.get("text") or "").strip()
        if not spoken:
            continue
        yield {
            "spoken": spoken,
            "transcript": spoken,
            "role": message.get("speaker") or "unknown",
            "kind": airport,
            "category": message.get("category") or "routine",
            "weight": 1.0,
            # Ignored by JsonlTextSource (it reads a fixed key set); kept so the
            # text pool itself is greppable per airport without re-deriving it.
            "airport": airport,
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    airports: Counter = Counter()
    roles: Counter = Counter()
    categories: Counter = Counter()
    scenes = 0
    lines = 0
    with args.out.open("w") as handle:
        for scene in iter_scenes(args.scenes):
            scenes += 1
            for record in utterances(scene):
                handle.write(json.dumps(record) + "\n")
                lines += 1
                airports[record["kind"]] += 1
                roles[record["role"]] += 1
                categories[record["category"]] += 1

    summary = {
        "scenes": scenes,
        "utterances": lines,
        "airports": dict(airports.most_common()),
        "roles": dict(roles.most_common()),
        "top_categories": dict(categories.most_common(10)),
        "out": str(args.out.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
