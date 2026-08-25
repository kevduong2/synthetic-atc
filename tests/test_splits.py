"""Tests for the split registry (D11).

The registry's whole value is that overlap is checkable rather than
remembered, so the disjointness assertions here are computed from `SPLITS`
itself -- adding a split with a bad range fails these tests (and import).
No test touches the network: `load_split` is exercised against a stub corpus.
"""

import itertools

import pytest

from atcgen.dataset import splits as splits_mod
from atcgen.dataset.splits import (
    CORPUS,
    SPLIT_NAMES,
    SPLITS,
    SplitSpec,
    _check_disjoint,
    describe,
    load_split,
    split_spec,
)


class _StubDataset:
    """Enough of a HF Dataset for `load_split`: length and `select`."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return _StubDataset([self.rows[index] for index in indices])


def test_no_two_splits_share_a_row():
    for left, right in itertools.combinations(SPLITS.values(), 2):
        assert not left.overlaps(right), f"{left.name} overlaps {right.name}"
        if left.source_split != right.source_split:
            continue
        # and again by materializing the indices, in case `overlaps` is wrong
        assert not set(left.indices(20000)) & set(right.indices(20000))


def test_locked_test_excludes_the_spent_first_five_hundred_test_rows():
    locked, spent = SPLITS["locked_test"], SPLITS["model_select"]
    assert locked.source_split == "test"
    assert min(locked.indices(20000)) == 500
    assert set(locked.indices(20000)).isdisjoint(range(500))
    assert SPLITS["spent_test"].indices(20000) == range(500)
    assert spent.source_split == "train"


def test_registry_matches_the_documented_experiment_protocol():
    assert {name: (spec.source_split, spec.start, spec.stop)
            for name, spec in SPLITS.items()} == {
        "real_train": ("train", 0, 8000),
        "reward_val": ("train", 8000, 9000),
        "model_select": ("train", 9000, 10000),
        "train_tail": ("train", 10000, None),
        "locked_test": ("test", 500, 2500),
        "spent_test": ("test", 0, 500),
    }
    assert SPLIT_NAMES == tuple(SPLITS)


def test_check_disjoint_rejects_an_overlapping_registry(monkeypatch):
    clashing = dict(SPLITS)
    clashing["oops"] = SplitSpec("oops", "train", 7999, 8001,
                                 purpose="", policy="")
    monkeypatch.setattr(splits_mod, "SPLITS", clashing)
    with pytest.raises(ValueError, match="not disjoint"):
        _check_disjoint()


def test_open_ended_split_overlaps_anything_after_its_start():
    tail = SPLITS["train_tail"]
    assert tail.overlaps(SplitSpec("later", "train", 50000, None,
                                   purpose="", policy=""))
    assert not tail.overlaps(SplitSpec("earlier", "train", 0, 10,
                                       purpose="", policy=""))
    # a different source split can never collide
    assert not tail.overlaps(SPLITS["locked_test"])


def test_indices_clamp_to_the_available_rows():
    assert SPLITS["reward_val"].indices(8500) == range(8000, 8500)
    assert SPLITS["reward_val"].indices(100) == range(100, 100)
    assert SPLITS["train_tail"].indices(10500) == range(10000, 10500)


def test_slice_notation_and_dataset_name():
    assert SPLITS["reward_val"].slice_str == "train[8000:9000]"
    assert SPLITS["train_tail"].slice_str == "train[10000:]"
    assert SPLITS["locked_test"].dataset_name() == f"{CORPUS}:test[500:2500]"


def test_split_spec_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown split"):
        split_spec("dev")


def test_load_split_slices_the_named_rows(monkeypatch):
    seen = {}

    def fake_load(split, corpus, cast_audio):
        seen.update(split=split, corpus=corpus, cast_audio=cast_audio)
        return _StubDataset(range(12000))

    monkeypatch.setattr(splits_mod, "load_real_atc",
                        lambda split, corpus, cast_audio: fake_load(split, corpus, cast_audio))

    dataset = load_split("model_select", cast_audio=False)
    assert seen == {"split": "train", "corpus": CORPUS, "cast_audio": False}
    assert dataset.rows == list(range(9000, 10000))


def test_describe_is_json_ready_and_carries_the_policy():
    described = describe()
    assert set(described) == set(SPLIT_NAMES)
    locked = described["locked_test"]
    assert locked["slice"] == "test[500:2500]"
    assert locked["corpus"] == CORPUS
    assert "FINAL REPORT" in locked["policy"].upper()
