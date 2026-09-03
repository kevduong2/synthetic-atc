"""Data-plumbing scripts: transcript join, scene flattening, corpus export.

The invariants under test are the ones that fail silently. A clip leaking out of
the locked capture day into the RL dev slice, the same transcript landing on both
sides of a synthetic split, or a scheduled two-view render quietly wrapping
around and repeating itself all produce a number that looks fine and is wrong.
"""

import csv
import json
import random
from pathlib import Path

import pytest

from atcgen.text.sources import (
    SequentialTextSource,
    TextSourceExhausted,
    WeightedSampler,
    make_text_source,
)
from scripts import (
    convert_scenes,
    expand_text_views,
    export_corpus_csv,
    join_kixd_transcripts,
)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# ticket 1: transcript join and session splits
# --------------------------------------------------------------------------

DAYS = {"20250801": 6, "20250802": 5, "20250803": 4}   # train / dev / locked


def build_kixd_fixture(tmp_path):
    """Three capture days of clips plus a V2.1.2-shaped corpus snapshot."""
    clips = tmp_path / "clips"
    clips.mkdir()
    names = [f"KIXD_TOWER_{day}_{100000 + i:06d}.wav"
             for day, count in DAYS.items() for i in range(count)]
    for name in names:
        (clips / name).write_bytes(b"RIFF")

    # one clip is reachable only through the legacy-name rename log
    legacy, renamed = "KIXD_8-1-2025_clip0000.wav", names[0]
    log = tmp_path / "rename.log"
    log.write_text(f"=== KIXD rename run ===\nRENAME [0000] {legacy} -> {renamed}\n")

    corpus = tmp_path / "V2.1.2"
    corpus.mkdir()
    rows = [{"audio": f"/mnt/data0/kixd/{name}", "text": f"utterance {i}",
             "suspect": "False"} for i, name in enumerate(names[1:], start=1)]
    rows.append({"audio": f"/mnt/data0/kixd/{legacy}", "text": "legacy row",
                 "suspect": "False"})
    write_csv(corpus / "corpus_train.csv", rows, ["audio", "text", "suspect"])
    # the test half is the same shape; membership is deliberately ignored, and
    # each of these three rows is dropped by a different filter
    write_csv(corpus / "corpus_test.csv", [
        {"audio": f"/mnt/data0/kixd/{names[1]}", "text": "doubted", "suspect": "True"},
        {"audio": f"/mnt/data0/kixd/{names[2]}", "text": "   ", "suspect": "False"},
        {"audio": f"/mnt/data0/kixd/{names[3]}", "text": "say again [inaudible]",
         "suspect": "False"},
        {"audio": "/mnt/data0/kixd/KIXD_TOWER_20250801_999999.wav",
         "text": "no local audio", "suspect": "False"},
    ], ["audio", "text", "suspect"])
    return clips, corpus, log


def run_join(tmp_path, *extra):
    clips, corpus, log = build_kixd_fixture(tmp_path)
    out = tmp_path / "out"
    summary = join_kixd_transcripts.main([
        "--clips", str(clips), "--corpus", str(corpus), "--rename-log", str(log),
        "--out", str(out), "--dev-day", "20250802", "--locked-day", "20250803",
        "--dev-rows", "3", *extra])
    splits = {}
    for stem in ("kixd_labeled", "kixd_train", "kixd_heldout", "kixd_dev",
                 "kixd_locked_day"):
        rows = [json.loads(line) for line in (out / f"{stem}.jsonl").open()]
        splits[stem] = {Path(row["audio"]).name for row in rows}
    return summary, splits, out


def day_of(name):
    return name.split("_")[2]


def test_join_cuts_splits_by_whole_capture_day(tmp_path):
    summary, splits, _ = run_join(tmp_path)

    assert {day_of(n) for n in splits["kixd_train"]} == {"20250801"}
    assert {day_of(n) for n in splits["kixd_dev"]} == {"20250802"}
    assert {day_of(n) for n in splits["kixd_locked_day"]} == {"20250803"}
    assert {day_of(n) for n in splits["kixd_heldout"]} == {"20250802", "20250803"}
    assert summary["train_days"] == ["20250801"]


def test_the_locked_day_never_reaches_the_rl_dev_slice(tmp_path):
    _, splits, _ = run_join(tmp_path)
    assert splits["kixd_locked_day"].isdisjoint(splits["kixd_dev"])
    assert splits["kixd_locked_day"].isdisjoint(splits["kixd_train"])
    assert splits["kixd_dev"] <= splits["kixd_heldout"]
    assert splits["kixd_train"].isdisjoint(splits["kixd_heldout"])


def test_join_drops_suspect_empty_and_bracket_tagged_rows(tmp_path):
    summary, splits, _ = run_join(tmp_path)
    tally = summary["join"]
    assert tally["suspect"] == 1
    assert tally["empty_text"] == 1
    assert tally["bracket_tag"] == 1
    assert tally["no_local_audio"] == 1
    assert not any("inaudible" in text for text in splits["kixd_labeled"])
    assert summary["counts"]["kixd_labeled"] == tally["clean"]


def test_join_falls_back_to_the_rename_log_for_legacy_names(tmp_path):
    summary, splits, out = run_join(tmp_path)
    assert summary["join"]["matched_via_rename"] == 1
    texts = {row["text"] for row in
             csv.DictReader((out / "kixd_labeled.csv").open(newline=""))}
    assert "legacy row" in texts


def test_join_csv_and_jsonl_agree_and_paths_are_absolute(tmp_path):
    _, _, out = run_join(tmp_path)
    rows = [json.loads(line) for line in (out / "kixd_train.jsonl").open()]
    csv_rows = list(csv.DictReader((out / "kixd_train.csv").open(newline="")))
    assert rows == [dict(row) for row in csv_rows]
    assert all(Path(row["audio"]).is_absolute() for row in rows)


def test_join_refuses_a_dev_slice_larger_than_its_day(tmp_path):
    with pytest.raises(ValueError, match="fewer than --dev-rows"):
        run_join(tmp_path, "--dev-rows", "99")


# --------------------------------------------------------------------------
# ticket 2: scene flattening
# --------------------------------------------------------------------------

def test_convert_scenes_reads_concatenated_pretty_json(tmp_path):
    scenes = [
        {"scene_id": "a", "airport_ident": "KIXD", "messages": [
            {"speaker": "controller", "category": "taxi_clearance", "text": "taxi via alpha"},
            {"speaker": "pilot", "category": "readback", "text": "via alpha"},
            {"speaker": "pilot", "category": "readback", "text": "   "}]},
        {"scene_id": "b", "airport_ident": "KOJC", "messages": [
            {"speaker": "atis", "category": "atis_broadcast", "text": "information bravo"}]},
    ]
    source = tmp_path / "scenes.jsonl"
    source.write_text("\n".join(json.dumps(scene, indent=2) for scene in scenes))

    out = tmp_path / "flat.jsonl"
    summary = convert_scenes.main(["--scenes", str(source), "--out", str(out)])
    records = [json.loads(line) for line in out.open()]

    assert summary["scenes"] == 2
    assert summary["utterances"] == 3          # the blank message is dropped
    assert [r["kind"] for r in records] == ["KIXD", "KIXD", "KOJC"]
    assert [r["role"] for r in records] == ["controller", "pilot", "atis"]
    assert records[0]["spoken"] == records[0]["transcript"] == "taxi via alpha"
    assert all(r["weight"] == 1.0 and "entities" not in r for r in records)


def test_converted_scenes_load_as_a_text_source(tmp_path):
    from atcgen.text.sources import JsonlTextSource

    scene = {"airport_ident": "S50", "messages": [
        {"speaker": "pilot", "category": "readback", "text": "roger"}]}
    source = tmp_path / "scenes.jsonl"
    source.write_text(json.dumps(scene, indent=2))
    out = tmp_path / "flat.jsonl"
    convert_scenes.main(["--scenes", str(source), "--out", str(out)])

    records = JsonlTextSource(out).records
    assert len(records) == 1
    assert records[0].kind == "S50" and records[0].role == "pilot"
    assert records[0].extra == {"airport": "S50"}


# --------------------------------------------------------------------------
# ticket 3: corpus CSV export
# --------------------------------------------------------------------------

def write_manifest(dataset, rows):
    dataset = Path(dataset)
    (dataset / "wavs").mkdir(parents=True)
    gated_rows = []
    with (dataset / "manifest.jsonl").open("w") as handle:
        for index, (kind, text) in enumerate(rows):
            (dataset / "wavs" / f"{index:06d}.wav").write_bytes(b"RIFF")
            row = {
                "audio": f"wavs/{index:06d}.wav", "text": text, "kind": kind,
                "lineage": {"config_hash": "abc", "git_revision": "def"}}
            handle.write(json.dumps(row) + "\n")
            if text:
                gated_rows.append({**row, "tier": "gold"})
    if gated_rows:
        (dataset / "manifest_gated.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in gated_rows))
    return dataset


def test_export_merges_several_runs_before_splitting(tmp_path):
    speech = write_manifest(tmp_path / "a",
                            [("KIXD", f"utterance {i}") for i in range(20)])
    noise = write_manifest(tmp_path / "b", [("noise", "")] * 5)
    out = tmp_path / "corpus"

    manifest = export_corpus_csv.main([
        "--dataset", str(speech), "--dataset", str(noise), "--out", str(out),
        "--test-frac", "0.2", "--include-noise-only"])

    train = list(csv.DictReader((out / "corpus_train.csv").open(newline="")))
    test = list(csv.DictReader((out / "corpus_test.csv").open(newline="")))
    assert manifest["source"]["rows_in_manifest"] == 25
    assert len(manifest["source"]["datasets"]) == 2
    assert manifest["source"]["noise_only_in_train"] == 5
    # hallucination control belongs in training and nowhere near the held-out set
    assert sum(1 for row in train if not row["text"].strip()) == 5
    assert all(row["text"].strip() for row in test)
    assert {row["gate_tier"] for row in train + test} == {"gold", "noise"}
    assert manifest["source"]["gate_tiers"] == {"gold": 20, "noise": 5}
    assert len(train) + len(test) == 25


def test_export_rejects_unmatched_gate_clip_ids(tmp_path):
    dataset = write_manifest(tmp_path / "run", [("KIXD", "roger")])
    (dataset / "manifest_gated.jsonl").write_text("")

    with pytest.raises(ValueError, match="unmatched clip ids.*1 missing"):
        export_corpus_csv.main([
            "--dataset", str(dataset), "--out", str(tmp_path / "corpus")])


def test_export_preserves_all_speech_gate_tiers(tmp_path):
    dataset = write_manifest(
        tmp_path / "run",
        [("KIXD", f"transmission {index}") for index in range(4)],
    )
    tiers = ["gold", "silver", "adversarial", "rejected"]
    gated_rows = [
        {**json.loads(line), "tier": tier}
        for line, tier in zip(
            (dataset / "manifest.jsonl").read_text().splitlines(), tiers,
            strict=True,
        )
    ]
    (dataset / "manifest_gated.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gated_rows)
    )

    export_corpus_csv.main([
        "--dataset", str(dataset), "--out", str(tmp_path / "corpus"),
    ])

    with (tmp_path / "corpus" / "corpus_train.csv").open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert [row["gate_tier"] for row in exported] == tiers


def test_noise_only_rows_do_not_move_the_speech_split(tmp_path):
    """Adding hallucination control must not perturb what is held out."""
    speech = write_manifest(tmp_path / "a",
                            [("KIXD", f"utterance {i}") for i in range(20)])
    noise = write_manifest(tmp_path / "b", [("noise", "")] * 5)

    alone = export_corpus_csv.main([
        "--dataset", str(speech), "--out", str(tmp_path / "one"),
        "--test-frac", "0.2"])
    merged = export_corpus_csv.main([
        "--dataset", str(speech), "--dataset", str(noise),
        "--out", str(tmp_path / "two"), "--test-frac", "0.2",
        "--include-noise-only"])
    assert alone["sha256"]["test_csv"] == merged["sha256"]["test_csv"]


def test_export_rejects_runs_that_share_audio_paths(tmp_path):
    dataset = write_manifest(tmp_path / "a", [("KIXD", "roger")])
    with pytest.raises(ValueError, match="share audio paths"):
        export_corpus_csv.main([
            "--dataset", str(dataset), "--dataset", str(dataset),
            "--out", str(tmp_path / "corpus")])


def test_export_drops_noise_only_rows_and_never_shares_a_transcript(tmp_path):
    rows = [(kind, f"{kind} utterance {i % 5}")
            for kind in ("KIXD", "KOJC") for i in range(20)]
    rows += [("noise", ""), ("noise", "")]
    dataset = write_manifest(tmp_path / "run", rows)
    out = tmp_path / "corpus"

    manifest = export_corpus_csv.main([
        "--dataset", str(dataset), "--out", str(out), "--test-frac", "0.2"])

    train = list(csv.DictReader((out / "corpus_train.csv").open(newline="")))
    test = list(csv.DictReader((out / "corpus_test.csv").open(newline="")))
    assert manifest["source"]["noise_only_dropped"] == 2
    assert len(train) + len(test) == 40
    assert {r["text"] for r in train}.isdisjoint({r["text"] for r in test})
    assert {r["suspect"] for r in train + test} == {"False"}
    assert all(Path(r["audio"]).is_absolute() for r in train + test)
    # both airports are represented in the held-out slice
    assert len({r["text"].split()[0] for r in test}) == 2


def test_export_is_byte_identical_across_runs(tmp_path):
    dataset = write_manifest(tmp_path / "run",
                             [("KIXD", f"utterance {i}") for i in range(30)])
    first = export_corpus_csv.main([
        "--dataset", str(dataset), "--out", str(tmp_path / "a"), "--test-frac", "0.2"])
    second = export_corpus_csv.main([
        "--dataset", str(dataset), "--out", str(tmp_path / "b"), "--test-frac", "0.2"])
    assert first["sha256"] == second["sha256"]


def test_export_keeps_noise_only_rows_on_request(tmp_path):
    dataset = write_manifest(tmp_path / "run", [("KIXD", "hello"), ("noise", "")])
    manifest = export_corpus_csv.main([
        "--dataset", str(dataset), "--out", str(tmp_path / "corpus"),
        "--include-noise-only"])
    assert manifest["source"]["noise_only_dropped"] == 0


# --------------------------------------------------------------------------
# ticket 4: deterministic two-view scheduling
# --------------------------------------------------------------------------

def write_text(tmp_path, n, name="text.jsonl"):
    path = tmp_path / name
    path.write_text("".join(
        json.dumps({"text": f"utterance {i}", "kind": "KIXD"}) + "\n"
        for i in range(n)))
    return path


def test_expand_gives_every_text_the_same_number_of_views(tmp_path):
    source = write_text(tmp_path, 50)
    out = tmp_path / "views.jsonl"
    summary = expand_text_views.main([
        "--text", str(source), "--out", str(out), "--views", "2"])
    records = [json.loads(line) for line in out.open()]

    assert summary["lines"] == 100 and summary["distinct_base_ids"] == 50
    by_base = {}
    for record in records:
        by_base.setdefault(record["base_id"], []).append(record)
    assert {len(views) for views in by_base.values()} == {2}
    assert all(sorted(v["view_index"] for v in views) == [0, 1]
               for views in by_base.values())
    # the views of one base_id carry identical text
    assert all(len({v["text"] for v in views}) == 1 for views in by_base.values())


def test_expand_shuffles_deterministically_and_seed_matters(tmp_path):
    source = write_text(tmp_path, 40)
    order = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        out = tmp_path / f"{name}.jsonl"
        expand_text_views.main(["--text", str(source), "--out", str(out),
                                "--seed", str(seed)])
        order.append([json.loads(line)["base_id"] for line in out.open()])
    assert order[0] == order[1]
    assert order[0] != order[2]


def test_sequential_source_reads_in_order_exactly_once(tmp_path):
    source = write_text(tmp_path, 5)
    sequential = SequentialTextSource(source)
    rng = random.Random(0)

    assert len(sequential) == 5 and sequential.remaining == 5
    drawn = [sequential.sample(rng).transcript for _ in range(5)]
    assert drawn == [f"utterance {i}" for i in range(5)]
    assert sequential.remaining == 0
    with pytest.raises(TextSourceExhausted):
        sequential.sample(rng)


def test_sequential_source_escapes_the_weighted_sampler(tmp_path):
    source = write_text(tmp_path, 5)
    # `for_source` wraps anything exposing `records`; order must survive
    assert WeightedSampler.for_source(SequentialTextSource(source)) is None
    assert WeightedSampler.for_source(make_text_source(str(source))) is not None


def test_make_text_source_routes_sequential_specs(tmp_path):
    source = write_text(tmp_path, 3)
    assert isinstance(make_text_source(f"sequential:{source}"), SequentialTextSource)
    assert isinstance(make_text_source({"kind": "sequential", "path": str(source)}),
                      SequentialTextSource)
    assert not isinstance(make_text_source(str(source)), SequentialTextSource)


def test_unknown_keys_become_scalar_passthrough(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text(json.dumps(
        {"text": "roger", "base_id": "t000001", "view_index": 1}) + "\n")
    record = make_text_source(str(path)).records[0]
    assert record.extra == {"base_id": "t000001", "view_index": 1}


def test_passthrough_rejects_manifest_collisions_and_nested_values(tmp_path):
    reserved = tmp_path / "reserved.jsonl"
    reserved.write_text(json.dumps({"text": "roger", "duration": 3.0}) + "\n")
    with pytest.raises(ValueError, match="collides with a manifest column"):
        make_text_source(str(reserved))

    nested = tmp_path / "nested.jsonl"
    nested.write_text(json.dumps({"text": "roger", "slots": {"runway": "18"}}) + "\n")
    with pytest.raises(ValueError, match="must be a JSON scalar"):
        make_text_source(str(nested))


def test_mixed_dev_rows_carry_their_region(tmp_path):
    """The region label is what a per-region WER breakdown groups on."""
    from atcgen.dataset.real_atc import load_local_corpus
    from scripts.build_rl_dev_mixed import write_rows

    stem = tmp_path / "rl_dev_mixed"
    rows = ([{"audio": f"/clips/kixd_{i}.wav", "text": f"kixd {i}", "source": "kixd"}
             for i in range(3)]
            + [{"audio": f"/clips/eu_{i}.wav", "text": f"eu {i}", "source": "eu"}
               for i in range(3)])
    write_rows(stem, rows)

    written = list(csv.DictReader(stem.with_suffix(".csv").open(newline="")))
    assert list(written[0]) == ["audio", "text", "source"]
    assert [row["source"] for row in written] == ["kixd"] * 3 + ["eu"] * 3
    # order is what --dev-indices slices on, so it has to survive the round trip
    assert [json.loads(line)["audio"] for line in stem.with_suffix(".jsonl").open()] \
        == [row["audio"] for row in rows]
    # the extra column must not break the loader the reward harness uses
    assert load_local_corpus(stem.with_suffix(".csv"),
                             cast_audio=False).num_rows == 6


def test_write_rows_omits_the_region_column_when_absent(tmp_path):
    from scripts.build_rl_dev_mixed import write_rows

    stem = tmp_path / "plain"
    write_rows(stem, [{"audio": "/clips/a.wav", "text": "roger"}])
    written = list(csv.DictReader(stem.with_suffix(".csv").open(newline="")))
    assert list(written[0]) == ["audio", "text"]


def test_generate_dataset_parses_set_overrides():
    """`--set` types values through YAML; the config validators are strict."""
    from scripts.generate_dataset import parse_set

    parsed = parse_set(["dataset.noise_only_frac=1.0", "seed=7", "qc.enabled=false"])
    assert parsed == {"dataset.noise_only_frac": 1.0, "seed": 7, "qc.enabled": False}
    assert isinstance(parsed["dataset.noise_only_frac"], float)
    assert isinstance(parsed["seed"], int)
    with pytest.raises(ValueError, match="dotted.path=value"):
        parse_set(["noise_only_frac"])


def test_set_override_reaches_the_resolved_config(tmp_path):
    from atcgen.config import load_config
    from scripts.generate_dataset import parse_set

    config = tmp_path / "c.yaml"
    config.write_text("mode: procedural\nseed: 1\ndataset: {noise_only_frac: 0.03}\n")
    resolved = load_config(config, parse_set(["dataset.noise_only_frac=1.0"]))
    assert resolved.dataset.noise_only_frac == 1.0
