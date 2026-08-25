"""L3 bandit: Thompson math, the hardness window, routing, and resume.

Nothing here loads a model or renders audio — the expensive halves sit behind
the `PullEngine`/`Counterfactual` protocols precisely so they can be faked.
"""

import json
import random

import pytest

from atcgen.rl.bandit import (
    DROPPED,
    SELECTED,
    SPILLOVER,
    BetaPosteriors,
    HardnessWindow,
    RecipeBandit,
    format_posteriors,
    format_pull,
    sample_wer,
)
from atcgen.rl.recipes import RECIPES, Recipe

# Three buckets with clearly different in-window rates, so a run of a few pulls
# has an unambiguous best arm.
FAKE_RECIPES = {
    "easy": Recipe("easy", "snr", "student nails it"),
    "sweet": Recipe("sweet", "snr", "right in the window"),
    "garbled": Recipe("garbled", "channel", "teacher cannot read it either"),
}

# The fixture pins its own thresholds rather than tracking the module defaults,
# which are calibrated against a particular teacher/student pair and move when
# that pair does.
FIXTURE_WINDOW = HardnessWindow(tau1=0.5, tau2=0.3, tau3=1.2)

# (teacher WER, student WER) drawn round-robin per bucket.
FAKE_WERS = {
    "easy":    [(0.05, 0.10), (0.05, 0.05), (0.05, 0.40)],   # 1/3 in window
    "sweet":   [(0.10, 0.50), (0.10, 0.60), (0.10, 0.20)],   # 2/3 in window
    "garbled": [(0.80, 0.90), (0.70, 0.95), (0.10, 0.55)],   # 1/3, rest dropped
}


class FakeEngine:
    """Deterministic stand-in: no audio, no ASR, cycling WER pairs per bucket."""

    def __init__(self):
        self.rendered = []
        self.scored = []

    def render(self, recipe, dest, n, seed):
        self.rendered.append((recipe.name, str(dest), n, seed))
        return [
            {"audio": f"{dest}/synth/wavs/{index:06d}.wav",
             "text": f"{recipe.name} utterance {index}",
             "role": "controller", "kind": "climb", "category": "routine",
             "duration": 3.0, "_recipe": recipe.name, "_index": index}
            for index in range(n)
        ]

    def wers(self, rows, which):
        self.scored.append((which, len(rows)))
        column = 0 if which == "teacher" else 1
        return [FAKE_WERS[row["_recipe"]][row["_index"] % 3][column] for row in rows]


class FakeCounterfactual:
    def __init__(self):
        self.calls = []

    def run(self, round_dir, selected, m):
        self.calls.append((str(round_dir), len(selected), m))
        return {"status": "ok", "n": min(len(selected), m),
                "wer_selected": 0.40, "wer_uniform": 0.45,
                "delta_wer_selected_vs_uniform": 0.05}


def make_bandit(tmp_path, **kwargs):
    kwargs.setdefault("recipes", FAKE_RECIPES)
    kwargs.setdefault("window", FIXTURE_WINDOW)
    kwargs.setdefault("n_batch", 9)
    kwargs.setdefault("seed", 3)
    kwargs.setdefault("counterfactual_every", 0)
    return RecipeBandit(tmp_path, kwargs.pop("engine", FakeEngine()), **kwargs)


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


def test_sample_wer_ignores_unscoreable_references():
    assert sample_wer("cleared to land", "cleared to land") == 0.0
    assert sample_wer("one two three", "one two four") == pytest.approx(1 / 3)
    # a hallucinating decode inserts words and scores above 1 -- what tau3 is for
    assert sample_wer("roger", "roger roger roger roger") == pytest.approx(3.0)
    assert sample_wer("", "anything") is None
    assert sample_wer("   ", "anything") is None


@pytest.mark.parametrize("teacher,student,route,reason", [
    (0.10, 0.50, SELECTED, "in_window"),
    (0.10, 0.31, SELECTED, "in_window"),
    (0.10, 0.20, SPILLOVER, "too_easy"),
    (0.10, 1.90, SPILLOVER, "too_hard"),
    (0.90, 0.50, DROPPED, "teacher_untrusted"),
    (None, 0.50, DROPPED, "no_reference"),
    (0.10, None, DROPPED, "no_reference"),
    # boundaries: every comparison in section 4.7 is strict
    (0.50, 0.50, DROPPED, "teacher_untrusted"),
    (0.49, 0.30, SPILLOVER, "too_easy"),
    (0.49, 1.20, SPILLOVER, "too_hard"),
])
def test_hardness_window_boundaries(teacher, student, route, reason):
    window = HardnessWindow(tau1=0.5, tau2=0.3, tau3=1.2)
    assert window.classify(teacher, student) == (route, reason)


def test_untrustworthy_teacher_beats_a_perfect_hardness_score():
    """D5's bound: a hard sample with an unreadable label is dropped, not kept."""
    window = HardnessWindow()
    route, reason = window.classify(window.tau1 + 0.05, 0.60)
    assert (route, reason) == (DROPPED, "teacher_untrusted")


def test_bad_window_declarations_raise():
    with pytest.raises(ValueError):
        HardnessWindow(tau2=0.9, tau3=0.5)
    with pytest.raises(ValueError):
        HardnessWindow(tau1=0.0)


# --------------------------------------------------------------------------
# Thompson sampling
# --------------------------------------------------------------------------


def test_posterior_updates_and_mean():
    posteriors = BetaPosteriors(["a", "b"], seed=0)
    assert posteriors.mean("a") == pytest.approx(0.5)      # Beta(1,1)
    posteriors.update("a", 30, 10)
    assert posteriors.alpha["a"] == pytest.approx(31.0)
    assert posteriors.beta["a"] == pytest.approx(11.0)
    assert posteriors.mean("a") == pytest.approx(31 / 42)
    assert posteriors.observations("a") == pytest.approx(40.0)
    assert posteriors.pulls == {"a": 1, "b": 0}
    # more evidence, tighter posterior
    wide = posteriors.stdev("b")
    posteriors.update("b", 300, 100)
    assert posteriors.stdev("b") < wide
    with pytest.raises(KeyError):
        posteriors.update("nope", 1, 0)


def test_posterior_state_round_trip_continues_the_same_stream():
    original = BetaPosteriors(["a", "b", "c"], seed=5)
    for _ in range(4):
        original.update(original.sample_arm(), 3, 7)
    restored = BetaPosteriors(["a", "b", "c"], seed=999)
    restored.load_state_dict(json.loads(json.dumps(original.state_dict())))

    assert restored.alpha == original.alpha
    assert restored.beta == original.beta
    assert restored.pulls == original.pulls
    assert [restored.sample_arm() for _ in range(6)] == \
           [original.sample_arm() for _ in range(6)]


def test_thompson_beats_uniform_on_distinct_bernoulli_arms():
    """The point of the algorithm: concentrate pulls on the best bucket."""
    rates = {"poor": 0.10, "middling": 0.35, "good": 0.65}
    rounds = 200

    posteriors = BetaPosteriors(sorted(rates), seed=7)
    outcomes = random.Random(11)
    thompson_hits = 0
    for _ in range(rounds):
        arm = posteriors.sample_arm()
        hit = outcomes.random() < rates[arm]
        posteriors.update(arm, int(hit), int(not hit))
        thompson_hits += hit

    arms = sorted(rates)
    control = random.Random(11)
    uniform_hits = sum(control.random() < rates[control.choice(arms)]
                       for _ in range(rounds))

    assert thompson_hits > uniform_hits
    assert thompson_hits >= 1.4 * uniform_hits
    assert posteriors.pulls["good"] > posteriors.pulls["poor"]
    assert max(arms, key=posteriors.mean) == "good"


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def test_pull_routes_samples_and_updates_the_posterior(tmp_path):
    engine = FakeEngine()
    bandit = make_bandit(tmp_path, engine=engine, n_batch=9)
    row = bandit.pull()

    arm = row["recipe"]
    # each bucket's WER table repeats every 3 samples over 9 clips
    expected = {"easy": (3, 6, 0), "sweet": (6, 3, 0), "garbled": (3, 0, 6)}[arm]
    assert (row[SELECTED], row[SPILLOVER], row[DROPPED]) == expected
    assert row["in_window"] == expected[0]
    assert row["n"] == 9
    assert row["seed"] == bandit.pull_seed(0)
    assert sum(row["reasons"].values()) == 9

    posteriors = bandit.posteriors
    assert posteriors.alpha[arm] == pytest.approx(1.0 + expected[0])
    assert posteriors.beta[arm] == pytest.approx(1.0 + 9 - expected[0])

    selected = bandit.buffer_rows(SELECTED)
    spillover = bandit.buffer_rows(SPILLOVER)
    assert len(selected) == expected[0]
    assert len(spillover) == expected[1]
    assert all(sample["reason"] == "in_window" for sample in selected)
    assert all(sample["reason"] in {"too_easy", "too_hard"} for sample in spillover)
    # dropped samples reach neither buffer
    assert len(selected) + len(spillover) == 9 - expected[2]
    assert all(sample["pull"] == 0 for sample in selected + spillover)
    # every kept sample carries both WERs and a resolvable audio path
    for sample in selected:
        assert 0.0 <= sample["wer_teacher"] < 0.5
        assert 0.3 < sample["wer_student"] < 1.2
        assert str(tmp_path) in sample["audio"]

    assert engine.scored == [("teacher", 9), ("student", 9)]
    assert format_pull(row).startswith("pull   0")


def test_run_logs_state_and_resumes_where_it_stopped(tmp_path):
    first = make_bandit(tmp_path)
    first.run(3)

    assert first.pulls_done == 3
    rows = [json.loads(line) for line in
            (tmp_path / "pulls.jsonl").read_text().splitlines()]
    assert [row["pull"] for row in rows] == [0, 1, 2]
    assert sum(row["n"] for row in rows) == 27
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["loop"]["pulls_done"] == 3
    assert state["window"] == FIXTURE_WINDOW.as_dict()

    resumed = make_bandit(tmp_path)
    assert resumed.pulls_done == 3
    assert resumed.posteriors.alpha == first.posteriors.alpha
    assert resumed.posteriors.beta == first.posteriors.beta
    assert resumed.totals == first.totals
    assert len(resumed.buffer_rows(SELECTED)) == len(first.buffer_rows(SELECTED))

    resumed.run(5)
    assert resumed.pulls_done == 5
    rows = [json.loads(line) for line in
            (tmp_path / "pulls.jsonl").read_text().splitlines()]
    assert [row["pull"] for row in rows] == [0, 1, 2, 3, 4]
    # the buffer grew, it did not restart
    assert len(resumed.buffer_rows(SELECTED)) >= len(first.buffer_rows(SELECTED))


def test_resume_replays_the_log_when_the_checkpoint_is_lost(tmp_path):
    first = make_bandit(tmp_path)
    first.run(4)
    (tmp_path / "state.json").unlink()

    replayed = make_bandit(tmp_path)
    assert replayed.pulls_done == 4
    assert replayed.posteriors.alpha == first.posteriors.alpha
    assert replayed.posteriors.beta == first.posteriors.beta
    assert replayed.totals == first.totals


def test_resume_drops_buffer_rows_from_a_pull_that_never_finished(tmp_path):
    bandit = make_bandit(tmp_path)
    bandit.run(2)
    kept = len(bandit.buffer_rows(SELECTED))

    # a hard kill after the samples were appended but before the pull row was
    with (tmp_path / "selected" / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps({"pull": 2, "recipe": "sweet", "audio": "x.wav",
                                 "text": "orphan", "reason": "in_window"}) + "\n")

    resumed = make_bandit(tmp_path)
    assert len(resumed.buffer_rows(SELECTED)) == kept
    assert all(sample["text"] != "orphan"
               for sample in resumed.buffer_rows(SELECTED))


def test_counterfactual_runs_on_schedule_and_at_the_end(tmp_path):
    counterfactual = FakeCounterfactual()
    bandit = make_bandit(tmp_path, counterfactual=counterfactual,
                         counterfactual_every=2, cf_m=5)
    bandit.run(5)

    # after pulls 2 and 4 on schedule, plus one final round at pull 5
    assert [call[2] for call in counterfactual.calls] == [5, 5, 5]
    assert len(counterfactual.calls) == 3
    rows = [json.loads(line) for line in
            (tmp_path / "counterfactuals.jsonl").read_text().splitlines()]
    assert [row["after_pull"] for row in rows] == [2, 4, 5]
    assert [row["round"] for row in rows] == [0, 1, 2]
    assert all(row["delta_wer_selected_vs_uniform"] == 0.05 for row in rows)
    assert bandit.cf_rounds_done == 3

    # a finished run that is re-run adds no further rounds
    RecipeBandit(tmp_path, FakeEngine(), recipes=FAKE_RECIPES, n_batch=9, seed=3,
                 counterfactual=counterfactual, counterfactual_every=2,
                 cf_m=5).run(5)
    assert len(counterfactual.calls) == 3


def test_thompson_concentrates_on_the_best_bucket_end_to_end(tmp_path):
    bandit = make_bandit(tmp_path, n_batch=9)
    bandit.run(25)

    assert bandit.posteriors.pulls["sweet"] > bandit.posteriors.pulls["garbled"]
    assert max(bandit.posteriors.arms, key=bandit.posteriors.mean) == "sweet"
    assert bandit.posteriors.mean("sweet") == pytest.approx(2 / 3, abs=0.05)
    assert bandit.posteriors.mean("garbled") == pytest.approx(1 / 3, abs=0.15)
    assert bandit.totals[SELECTED] == len(bandit.buffer_rows(SELECTED))
    assert sum(bandit.totals.values()) == 25 * 9

    table = format_posteriors(bandit)
    assert table.splitlines()[2].startswith("sweet")
    assert "buffers: selected" in table


def test_pull_seeds_vary_but_are_recorded(tmp_path):
    bandit = make_bandit(tmp_path)
    bandit.run(4)
    rows = [json.loads(line) for line in
            (tmp_path / "pulls.jsonl").read_text().splitlines()]
    seeds = [row["seed"] for row in rows]
    assert len(set(seeds)) == 4
    assert seeds == [bandit.pull_seed(index) for index in range(4)]


def test_the_real_recipe_set_is_a_usable_action_space(tmp_path):
    """The default arms must all be pullable, not just the fakes."""
    bandit = RecipeBandit(tmp_path, FakeEngine(), n_batch=1, seed=1)
    assert set(bandit.posteriors.arms) == set(RECIPES)
    assert len(bandit.posteriors.arms) >= 10


# --------------------------------------------------------------------------
# the counterfactual's uniform arm
# --------------------------------------------------------------------------


class GatingEngine:
    """Renders rows a fixed fraction of which pass the teacher gate."""

    def __init__(self, keep_rate):
        self.keep_rate = keep_rate
        self.generated = 0
        self.device = None

    def render(self, recipe, dest, n, seed):
        rows = [{"audio": f"{dest}/{i}.wav", "text": "t", "_n": self.generated + i}
                for i in range(n)]
        self.generated += n
        return rows

    def wers(self, rows, which):
        # deterministic: every 1/keep_rate-th sample is trustworthy
        stride = round(1 / self.keep_rate)
        return [0.1 if row["_n"] % stride == 0 else 0.9 for row in rows]


def test_uniform_arm_tops_up_to_size_when_the_teacher_gate_bites(tmp_path):
    from atcgen.rl.bandit import AsrCounterfactual

    engine = GatingEngine(keep_rate=0.25)
    runner = AsrCounterfactual(engine, recipes=FAKE_RECIPES,
                               window=FIXTURE_WINDOW)
    pool = runner.uniform_pool(tmp_path, m=48, seed=1)

    assert len(pool) == 48
    assert all(row["wer_teacher"] < 0.5 for row in pool)
    # a uniform arm has to actually be uniform over the buckets
    assert {row["recipe"] for row in pool} == set(FAKE_RECIPES)
    # ... and must not generate without bound to get there
    assert engine.generated <= 4.0 * 48


def test_uniform_arm_gives_up_at_the_budget_rather_than_looping(tmp_path):
    from atcgen.rl.bandit import AsrCounterfactual

    engine = GatingEngine(keep_rate=0.02)
    runner = AsrCounterfactual(engine, recipes=FAKE_RECIPES,
                               window=FIXTURE_WINDOW)
    pool = runner.uniform_pool(tmp_path, m=100, seed=1)

    assert len(pool) < 100
    assert engine.generated <= 4.0 * 100 + len(FAKE_RECIPES)


def test_counterfactual_skips_rather_than_comparing_unequal_arms(tmp_path):
    from atcgen.rl.bandit import AsrCounterfactual

    runner = AsrCounterfactual(GatingEngine(keep_rate=1.0), recipes=FAKE_RECIPES,
                               ft_batch=8)
    result = runner.run(tmp_path / "000", selected=[], m=32)
    assert result["status"] == "skipped"
    assert result["n_selected"] == 0
