import json

import numpy as np
import pytest
import yaml

from atcgen.rl.loop import format_trial, load_trials, run_loop, top_deviations
from atcgen.rl.policy import RandomSearch
from atcgen.rl.space import SearchSpace, scalar_knob
from atcgen.rl.types import RewardResult, Trial


BASE = {"dataset": {"noise_only_frac": 0.03, "pilot_double_hop_prob": 0.5},
        "channel": {"clean_arm_prob": 0.03}}


def toy_space():
    return SearchSpace([
        scalar_knob("noise_only_frac", "dataset.noise_only_frac", 0.0, 0.10),
        scalar_knob("double_hop", "dataset.pilot_double_hop_prob", 0.0, 0.8),
        scalar_knob("clean_arm", "channel.clean_arm_prob", 0.0, 0.10),
    ])


class FakeReward:
    """Deterministic quadratic on the vector; touches no audio and no model."""

    def __init__(self, fail_on=(), target=0.75):
        self.fail_on = set(fail_on)
        self.target = target
        self.calls = []

    def __call__(self, config, trial_dir):
        assert isinstance(config, dict)
        vector = [config["dataset"]["noise_only_frac"] / 0.10,
                  config["dataset"]["pilot_double_hop_prob"] / 0.8,
                  config["channel"]["clean_arm_prob"] / 0.10]
        self.calls.append(tuple(round(value, 6) for value in vector))
        if len(self.calls) - 1 in self.fail_on:
            raise RuntimeError("synthetic generation blew up")
        reward = -float(np.sum((np.asarray(vector) - self.target) ** 2))
        return RewardResult(reward=reward, wer_after=0.30 - reward,
                            wer_baseline=0.30, metrics={"n_synth": 4})


def test_loop_writes_trials_best_and_state(tmp_path):
    space, reward = toy_space(), FakeReward()
    trials = run_loop(space, RandomSearch(space.dim, seed=0), reward, BASE, tmp_path,
                      iterations=3, pop_size=2)

    assert len(trials) == 1 + 3 * 2      # anchor + 3 batches
    assert [trial.index for trial in trials] == list(range(7))
    assert len(reward.calls) == 7

    rows = [json.loads(line) for line in
            (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len(rows) == 7
    assert rows[0]["overrides"]["noise_only_frac"] == pytest.approx(0.03)
    assert rows[0]["wer_baseline"] == pytest.approx(0.30)
    assert rows[0]["metrics"] == {"n_synth": 4}
    assert all(row["wall_time_sec"] >= 0 for row in rows)

    best = json.loads((tmp_path / "best.json").read_text())
    assert best["reward"] == pytest.approx(max(t.result.reward for t in trials))
    assert best["index"] == max(trials, key=lambda t: t.result.reward).index

    winner = yaml.safe_load((tmp_path / "best_config.yaml").read_text())
    assert winner["dataset"]["noise_only_frac"] == pytest.approx(
        best["overrides"]["noise_only_frac"])

    state = json.loads((tmp_path / "optimizer_state.json").read_text())
    assert state["loop"]["iterations_done"] == 3
    assert state["loop"]["next_index"] == 7
    assert state["optimizer"]["kind"] == "random"

    assert [t.index for t in load_trials(tmp_path)] == [t.index for t in trials]
    assert all((tmp_path / "trials" / f"{index:03d}").is_dir() for index in range(7))


def test_base_config_is_not_mutated(tmp_path):
    base = {"dataset": {"noise_only_frac": 0.03, "pilot_double_hop_prob": 0.5},
            "channel": {"clean_arm_prob": 0.03}}
    space = toy_space()
    run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(), base, tmp_path,
             iterations=1, pop_size=2)
    assert base == BASE


def test_anchor_uses_the_hand_tuned_config_and_is_kept_out_of_tell(tmp_path):
    space = toy_space()
    told = []

    class Recording(RandomSearch):
        def tell(self, vectors, rewards):
            told.append(len(vectors))

    run_loop(space, Recording(space.dim, seed=0), FakeReward(), BASE, tmp_path,
             iterations=2, pop_size=3)
    assert told == [3, 3]        # the anchor never joins a tell batch

    rows = [json.loads(line) for line in
            (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert rows[0]["vector"] == pytest.approx(list(space.default_vector(BASE)))


def test_seed_default_first_can_be_disabled(tmp_path):
    space = toy_space()
    trials = run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(), BASE,
                      tmp_path, iterations=1, pop_size=2, seed_default_first=False)
    assert len(trials) == 2


def test_resume_continues_numbering_without_re_evaluating(tmp_path):
    space = toy_space()
    first = FakeReward()
    run_loop(space, RandomSearch(space.dim, seed=0), first, BASE, tmp_path,
             iterations=2, pop_size=2)
    assert len(first.calls) == 5

    second = FakeReward()
    trials = run_loop(space, RandomSearch(space.dim, seed=0), second, BASE, tmp_path,
                      iterations=4, pop_size=2)

    assert len(second.calls) == 4                      # only the two new batches
    assert [t.index for t in trials] == list(range(9))
    rows = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert len(rows) == 9

    # A third call with nothing left to do is a no-op.
    third = FakeReward()
    run_loop(space, RandomSearch(space.dim, seed=0), third, BASE, tmp_path,
             iterations=4, pop_size=2)
    assert third.calls == []


def test_resume_restores_optimizer_state(tmp_path):
    """Resuming must not replay the RNG: the second half is fresh candidates."""
    space = toy_space()
    run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(), BASE, tmp_path,
             iterations=1, pop_size=3, seed_default_first=False)
    run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(), BASE, tmp_path,
             iterations=2, pop_size=3, seed_default_first=False)

    vectors = [tuple(row.vector) for row in load_trials(tmp_path)]
    assert len(set(vectors)) == 6

    reference = RandomSearch(space.dim, seed=0)
    expected = [tuple(v) for v in reference.ask(6)]
    assert [pytest.approx(list(v)) for v in vectors] == [list(v) for v in expected]


def test_no_resume_restarts_the_log(tmp_path):
    space = toy_space()
    run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(), BASE, tmp_path,
             iterations=1, pop_size=2)
    trials = run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(), BASE,
                      tmp_path, iterations=1, pop_size=2, resume=False)
    assert [t.index for t in trials] == [0, 1, 2]
    assert len((tmp_path / "trials.jsonl").read_text().splitlines()) == 3


def test_failed_candidate_is_logged_and_dropped_from_tell(tmp_path):
    space = toy_space()
    told = []

    class Recording(RandomSearch):
        def tell(self, vectors, rewards):
            told.append((len(vectors), list(rewards)))

    reward = FakeReward(fail_on={2})      # the second candidate of batch one
    trials = run_loop(space, Recording(space.dim, seed=0), reward, BASE, tmp_path,
                      iterations=2, pop_size=2)

    assert [count for count, _ in told] == [1, 2]
    assert len(trials) == 4               # 5 evaluations, one failed
    assert all(np.isfinite(t.result.reward) for t in trials)

    rows = [json.loads(line) for line in
            (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert rows[2]["error"].startswith("RuntimeError:")
    assert "reward" not in rows[2]
    assert [row["index"] for row in rows] == [0, 1, 2, 3, 4]

    # The failed index still owns its slot, so a resume does not reuse it.
    run_loop(space, Recording(space.dim, seed=0), FakeReward(), BASE, tmp_path,
             iterations=3, pop_size=2)
    rows = [json.loads(line) for line in
            (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [row["index"] for row in rows] == list(range(7))


def test_best_ignores_proxy_rewards(tmp_path):
    space = toy_space()

    def reward_fn(config, trial_dir):
        frac = config["dataset"]["noise_only_frac"]
        # The proxy score is enormous but must never win.
        return RewardResult(reward=99.0 if frac > 0.05 else frac,
                            wer_after=0.1, wer_baseline=0.3, proxy=frac > 0.05)

    run_loop(space, RandomSearch(space.dim, seed=0), reward_fn, BASE, tmp_path,
             iterations=3, pop_size=3, seed_default_first=False)
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["reward"] < 0.05


def test_on_trial_callback_sees_every_success(tmp_path):
    space = toy_space()
    seen = []
    run_loop(space, RandomSearch(space.dim, seed=0), FakeReward(fail_on={1}), BASE,
             tmp_path, iterations=2, pop_size=2, on_trial=seen.append)
    assert [trial.index for trial in seen] == [0, 2, 3, 4]


def test_top_deviations_and_format_trial():
    space = toy_space()
    defaults = space.describe(space.default_vector(BASE))
    moved = dict(defaults)
    moved["clean_arm"] = 0.10          # from 0.03 -> the largest unit-cube move
    moved["double_hop"] = 0.6          # 0.5 -> 0.6, a small one

    ranked = top_deviations(moved, defaults, space, count=3)
    # noise_only_frac did not move, so it is dropped rather than padding the list.
    assert [name for name, _, _ in ranked] == ["clean_arm", "double_hop"]
    assert top_deviations(defaults, defaults, space) == []

    line = format_trial(
        Trial(index=4, vector=[0.0, 0.0, 0.0], overrides=moved,
              result=RewardResult(reward=0.021, wer_after=0.28, wer_baseline=0.30),
              wall_time_sec=612.0),
        space, defaults)
    assert "trial   4" in line
    assert "+0.0210" in line
    assert "clean_arm" in line
