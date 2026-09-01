"""The E0 power check: named fixed arms through the reward harness.

The heavy call (`TrueRewardHarness.__call__`) is stubbed; what is tested is
that each arm mutates the profile the way its name claims, that a mutated arm
still parses and renders, and that the summary's separation arithmetic is
right -- the number a go/no-go on the search will be read off.
"""

import json
from typing import ClassVar

import numpy as np
import pytest
import yaml

from atcgen.config import load_config
from atcgen.dataset.build import build_dataset
from scripts.rl_power_check import ARMS, build_arm, main, summarize
from tests.test_rl_space import BASE_CONFIG, FakeSource, FakeTTS, base, step


def procedural_base():
    config = base()
    config["channel"]["noise"]["beds_dir"] = None    # absent in a checkout
    return config


# -- the arms --------------------------------------------------------------

def test_base_arm_is_the_profile_untouched():
    config = procedural_base()
    assert build_arm(config, "base") == config


def test_build_arm_does_not_mutate_the_caller_s_config():
    config = procedural_base()
    snapshot = json.loads(json.dumps(config))
    build_arm(config, "degraded")
    assert config == snapshot


def test_aug_off_arm_disables_augmentation_and_pins_the_rate():
    arm = build_arm(procedural_base(), "aug_off")
    assert arm["voice_augment"]["pitch_semitones"]["prob"] == 0.0
    assert arm["voice_augment"]["tempo"]["prob"] == 0.0
    assert arm["voice_augment"]["eq_tilt_db"]["prob"] == 0.0
    assert arm["tts"]["speed"]["uniform"] == [1.0, 1.0]


def test_degraded_arm_is_actually_degraded():
    config = procedural_base()
    arm = build_arm(config, "degraded")
    _, _, snr_lo, snr_hi = step(arm, "additive_noise")["snr_db"]["beta_scaled"]
    assert (snr_lo, snr_hi) == (0.0, 6.0)
    # ... and worse than the profile it started from, which is the whole point
    _, _, base_lo, base_hi = step(config, "additive_noise")["snr_db"]["beta_scaled"]
    assert snr_hi < base_hi and snr_lo <= base_lo
    assert step(arm, "dropouts")["prob"] == pytest.approx(0.4)
    assert step(arm, "am_distortion")["depth"]["uniform"][1] == pytest.approx(0.35)


def test_unknown_arm_names_the_known_ones():
    with pytest.raises(KeyError, match="ghost"):
        build_arm(procedural_base(), "ghost")


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_parses_and_renders(tmp_path, arm):
    """An arm that cannot render is a wasted cell discovered 7 minutes in."""
    path = tmp_path / f"{arm}.yaml"
    path.write_text(yaml.safe_dump(build_arm(procedural_base(), arm)), encoding="utf-8")
    loaded = load_config(path)

    out = tmp_path / f"render_{arm}"
    manifest = build_dataset(loaded, out, 2, FakeSource(), tts=FakeTTS())
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(np.isfinite(np.asarray(row["duration"])) for row in rows)


# -- the summary -----------------------------------------------------------

def _rows(**arms):
    return [{"arm": arm, "seed": seed, "reward": reward}
            for arm, rewards in arms.items()
            for seed, reward in enumerate(rewards)]


def test_summary_reports_gap_against_base_in_units_of_seed_spread():
    summary = summarize(_rows(base=[0.10, 0.14], degraded=[-0.10, -0.06]))
    arms = summary["arms"]

    assert arms["base"]["mean_reward"] == pytest.approx(0.12)
    assert arms["degraded"]["mean_reward"] == pytest.approx(-0.08)
    assert arms["base"]["gap_vs_base"] == pytest.approx(0.0)
    assert arms["degraded"]["gap_vs_base"] == pytest.approx(-0.20)

    # both arms have the same 0.04 spread, so the pooled stdev is that stdev
    pooled = summary["pooled_stdev"]
    assert pooled == pytest.approx(np.std([0.10, 0.14], ddof=1))
    assert arms["degraded"]["separation"] == pytest.approx(0.20 / pooled)


def test_summary_survives_a_single_seed():
    """One seed gives no spread, so separation is unreported rather than fake."""
    summary = summarize(_rows(base=[0.1], degraded=[-0.1]))
    assert summary["pooled_stdev"] is None
    assert summary["arms"]["base"]["stdev_reward"] is None
    assert summary["arms"]["degraded"]["separation"] is None
    assert summary["arms"]["degraded"]["gap_vs_base"] == pytest.approx(-0.2)


def test_summary_without_a_base_arm_reports_no_gap():
    summary = summarize(_rows(degraded=[-0.1, -0.2]))
    assert summary["arms"]["degraded"]["gap_vs_base"] is None
    assert summary["arms"]["degraded"]["separation"] is None


# -- the loop --------------------------------------------------------------

class _FakeResult:
    proxy = False

    def __init__(self, reward):
        self.reward = reward
        self.wer_after = 0.5 - reward
        self.wer_baseline = 0.5
        self.hallucination_rate = 0.0
        self.metrics = {"by_source": {"kixd": {"wer": 0.4, "samples": 2}}}


class _FakeHarness:
    """Records the (arm, gen_seed, ft_seed) each cell ran at."""

    calls: ClassVar[list[tuple]] = []

    def __init__(self, work_dir, **kwargs):
        self.gen_seed = 0
        self.ft_seed = 0
        # the banner and the reward both read the bounded aggregate
        self.baseline_report = {"wer": {"atc_normalized": 0.9},
                                "wer_bounded": {"atc_normalized": 0.5}}

    def __call__(self, config, trial_dir):
        arm = trial_dir.rsplit("/", 1)[-1]
        _FakeHarness.calls.append((arm, self.gen_seed, self.ft_seed))
        return _FakeResult(-0.2 if arm.startswith("degraded") else 0.1)


def test_main_writes_results_and_summary_and_resumes(tmp_path, monkeypatch):
    import scripts.rl_power_check as power

    _FakeHarness.calls = []
    monkeypatch.setattr(power, "TrueRewardHarness", _FakeHarness)
    argv = ["--out", str(tmp_path / "run"), "--base-config", str(BASE_CONFIG),
            "--arms", "base,degraded", "--seeds", "0,1"]

    summary = power.main(argv)
    assert len(_FakeHarness.calls) == 4
    # each seed moves the generator draw *and* the fine-tune batch order
    assert {(arm, ft) for arm, _, ft in _FakeHarness.calls} == {
        ("base_s0", 0), ("base_s1", 1), ("degraded_s0", 0), ("degraded_s1", 1)}
    assert len({gen for _, gen, _ in _FakeHarness.calls}) == 2

    results = tmp_path / "run" / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    assert rows[0]["by_source"] == {"kixd": {"wer": 0.4, "samples": 2}}
    assert summary["arms"]["degraded"]["gap_vs_base"] == pytest.approx(-0.3)
    assert json.loads((tmp_path / "run" / "summary.json").read_text()) == summary

    # a rerun recomputes nothing and produces the same summary
    again = power.main(argv)
    assert len(_FakeHarness.calls) == 4
    assert again["arms"] == summary["arms"]
    assert len(results.read_text().strip().splitlines()) == 4


def test_main_refuses_the_degraded_arm_on_a_calibrated_profile(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "run"),
              "--base-config", "configs/mode2_fastcut_kixd.yaml",
              "--arms", "base,degraded"])


def test_main_rejects_an_unknown_arm(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "run"), "--arms", "base,nonsense"])
