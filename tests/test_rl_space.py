import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from atcgen.config import load_config
from atcgen.rl.space import (
    SearchSpace,
    chain_center_knob,
    chain_param_knob,
    chain_prob_knob,
    chain_scalar_knob,
    default_atc_space,
    dist_bound_knob,
    dist_prob_knob,
    scalar_knob,
)


ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "configs/mode1_matched.yaml"


def base():
    with BASE_CONFIG.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def step(config, primitive):
    return next(item for item in config["channel"]["chain"]
                if item["primitive"] == primitive)


def test_knob_value_mapping_kinds():
    linear = scalar_knob("lin", "dataset.noise_only_frac", 0.0, 0.10)
    assert linear.value(0.0) == pytest.approx(0.0)
    assert linear.value(0.5) == pytest.approx(0.05)
    assert linear.value(2.0) == pytest.approx(0.10)   # clipped into the cube

    log = scalar_knob("log", "seed", 1.0, 100.0, kind="log")
    assert log.value(0.5) == pytest.approx(10.0)
    assert log.unit(10.0) == pytest.approx(0.5)


def test_bad_knob_declarations_raise():
    with pytest.raises(ValueError):
        scalar_knob("k", "seed", 0.0, 1.0, kind="quadratic")
    with pytest.raises(ValueError):
        scalar_knob("k", "seed", 1.0, 1.0)
    with pytest.raises(ValueError):
        scalar_knob("k", "seed", 0.0, 1.0, kind="log")
    with pytest.raises(ValueError):
        SearchSpace([scalar_knob("k", "seed", 0.0, 1.0),
                     scalar_knob("k", "seed", 0.0, 2.0)])


def test_to_config_leaves_the_base_untouched():
    config = base()
    snapshot = copy.deepcopy(config)
    space = default_atc_space()
    mutated = space.to_config(config, np.full(space.dim, 0.9))

    assert config == snapshot
    assert mutated != config
    assert mutated["dataset"]["noise_only_frac"] == pytest.approx(0.09)


def test_chain_knobs_address_steps_by_primitive_name():
    config = base()
    space = SearchSpace([
        chain_prob_knob("codec", "codec_roundtrip", 0.3, 1.0),
        chain_param_knob("snr_hi", "additive_noise", "snr_db", 3, 15.0, 35.0),
        chain_scalar_knob("bed", "additive_noise", "bed_prob", 0.0, 1.0, kind="prob"),
        chain_center_knob("low", "bandpass", "low", 180.0, 380.0),
    ])
    mutated = space.to_config(config, np.array([0.0, 1.0, 0.25, 1.0]))

    assert step(mutated, "codec_roundtrip")["prob"] == pytest.approx(0.3)
    assert step(mutated, "additive_noise")["snr_db"]["beta_scaled"] == [2.0, 1.8, 5, 35.0]
    assert step(mutated, "additive_noise")["bed_prob"] == pytest.approx(0.25)
    # centre slides to 380, the 120 Hz width of [200, 320] is preserved
    assert step(mutated, "bandpass")["low"]["uniform"] == [320.0, 440.0]


def test_unknown_primitive_raises_with_a_readable_message():
    space = SearchSpace([chain_prob_knob("ghost", "no_such_primitive")])
    with pytest.raises(KeyError, match="no_such_primitive"):
        space.to_config(base(), np.array([0.5]))

    missing_param = SearchSpace([
        chain_param_knob("p", "codec_roundtrip", "bitrate_kbps", 0, 1.0, 2.0)])
    with pytest.raises(KeyError, match="uniform/beta_scaled"):
        missing_param.to_config(base(), np.array([0.5]))

    with pytest.raises(KeyError, match="voice_augment.nope"):
        SearchSpace([dist_prob_knob("p", "voice_augment.nope.deeper")]).to_config(
            base(), np.array([0.5]))


def test_uniform_pair_stays_ordered_when_a_knob_crosses_it():
    config = base()
    # Drive the upper bandpass edge below the profile's lower one (2500).
    space = SearchSpace([chain_param_knob("high", "bandpass", "high", 1, 2400.0, 3400.0)])
    mutated = space.to_config(config, np.array([0.0]))
    low, high = step(mutated, "bandpass")["high"]["uniform"]
    assert low <= high
    assert [low, high] == [2400.0, 2500]

    # ... and a beta_scaled low bound pushed above its high bound.
    space = SearchSpace([chain_param_knob("lo", "additive_noise", "snr_db", 2, 0.0, 40.0)])
    mutated = space.to_config(config, np.array([1.0]))
    _, _, low, high = step(mutated, "additive_noise")["snr_db"]["beta_scaled"]
    assert low <= high
    assert [low, high] == [26, 40.0]


def test_default_vector_round_trips_the_hand_tuned_profile():
    config = base()
    space = default_atc_space()
    vector = space.default_vector(config)

    assert vector.shape == (space.dim,)
    assert np.all((vector >= 0.0) & (vector <= 1.0))

    described = space.describe(vector)
    rebuilt = space.to_config(config, vector)
    for name, value in space.describe(space.default_vector(rebuilt)).items():
        assert value == pytest.approx(described[name], abs=1e-9)

    # A handful of leaves checked against the YAML by hand.
    assert described["dataset.noise_only_frac"] == pytest.approx(0.03)
    assert described["tts.speed_lo"] == pytest.approx(1.0)
    assert described["tts.speed_hi"] == pytest.approx(1.4)
    assert described["additive_noise.bed_prob"] == pytest.approx(0.8)
    assert described["codec_roundtrip.prob"] == pytest.approx(0.8)
    assert described["voice_augment.pitch_prob"] == pytest.approx(0.5)
    assert described["bandpass.low_center"] == pytest.approx(260.0)


def test_default_vector_falls_back_to_the_midpoint_for_absent_leaves():
    space = SearchSpace([
        scalar_knob("missing", "dataset.does_not_exist", 0.0, 4.0),
        dist_bound_knob("speed_lo", "tts.speed", 0, 0.9, 1.2),
    ])
    vector = space.default_vector(base())
    assert vector[0] == pytest.approx(0.5)
    assert vector[1] == pytest.approx((1.0 - 0.9) / 0.3)


def test_mutated_configs_still_load(tmp_path):
    """The whole point of the ordering guards: every corner must parse."""
    space = default_atc_space()
    config = base()
    # A run-machine path that is not present in a test checkout.
    config["channel"]["noise"]["beds_dir"] = None

    for vector in (np.zeros(space.dim), np.ones(space.dim), np.full(space.dim, 0.5),
                   space.default_vector(config)):
        path = tmp_path / "candidate.yaml"
        path.write_text(yaml.safe_dump(space.to_config(config, vector)), encoding="utf-8")
        loaded = load_config(path)
        assert loaded.channel is not None
        assert len(loaded.channel.chain) == len(config["channel"]["chain"])


def test_vector_length_is_checked():
    space = default_atc_space()
    with pytest.raises(ValueError, match="entries"):
        space.to_config(base(), np.zeros(space.dim - 1))
