"""Recipe buckets: override plumbing, config validity, and the text filters."""

import random
from collections import Counter
from pathlib import Path

import pytest
import yaml

from atcgen.channel.envelope import load_envelope
from atcgen.rl.recipes import (
    NUMERIC_KINDS,
    RECIPES,
    FilteredTextSource,
    Recipe,
    check_recipe,
    set_path,
    write_config,
)

ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "configs/mode1_matched.yaml"


def base():
    with BASE_CONFIG.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def step(config, primitive):
    return next(item for item in config["channel"]["chain"]
                if item["primitive"] == primitive)


# --------------------------------------------------------------------------
# override plumbing
# --------------------------------------------------------------------------


def test_set_path_walks_mappings_and_chain_steps():
    config = base()
    set_path(config, "tts.speed", {"uniform": [1.3, 1.55]})
    set_path(config, "dataset.noise_only_frac", 0.0)
    set_path(config, "chain.additive_noise.snr_db", {"beta_scaled": [2.0, 2.5, 0, 14]})
    set_path(config, "chain.crackle.prob", 0.95)

    assert config["tts"]["speed"] == {"uniform": [1.3, 1.55]}
    assert config["dataset"]["noise_only_frac"] == 0.0
    assert step(config, "additive_noise")["snr_db"] == {"beta_scaled": [2.0, 2.5, 0, 14]}
    assert step(config, "crackle")["prob"] == 0.95


def test_set_path_rejects_paths_the_profile_does_not_have():
    config = base()
    with pytest.raises(KeyError):
        set_path(config, "chain.no_such_primitive.prob", 1.0)
    with pytest.raises(KeyError):
        set_path(config, "dataset.invented.leaf", 1.0)
    with pytest.raises(KeyError):
        set_path(config, "chain.crackle", 1.0)      # primitive without a param


def test_apply_leaves_the_base_untouched():
    original = base()
    config = base()
    recipe = RECIPES["low_snr"]
    produced = recipe.apply(config)

    assert config == original
    assert produced != config
    # the value written is a copy, not a reference into the recipe declaration
    step(produced, "additive_noise")["snr_db"]["beta_scaled"][0] = 99.0
    assert recipe.overrides["chain.additive_noise.snr_db"]["beta_scaled"][0] == 2.0


def test_recipes_only_touch_what_they_declare():
    """Everything outside a bucket's overrides keeps its fitted value."""
    config = base()
    produced = RECIPES["eu_fast_speech"].apply(config)
    assert produced["tts"]["speed"] == {"uniform": [1.30, 1.55]}
    produced["tts"]["speed"] = config["tts"]["speed"]
    assert produced == config


# --------------------------------------------------------------------------
# the action space
# --------------------------------------------------------------------------


def test_the_action_space_is_the_right_size_and_covers_the_axes():
    assert 10 <= len(RECIPES) <= 12
    assert all(name == recipe.name for name, recipe in RECIPES.items())
    assert {"scenario", "rate", "difficulty", "snr", "channel", "entity_type"} \
        <= {recipe.axis for recipe in RECIPES.values()}


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_every_recipe_produces_a_config_the_loader_accepts(name, tmp_path):
    config = write_config(RECIPES[name].apply(base()), tmp_path / f"{name}.yaml")
    assert config.mode == "procedural"
    assert config.channel is not None
    assert (tmp_path / f"{name}.yaml").exists()


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_every_recipe_stays_inside_the_measured_envelope(name):
    """§4.3's cap is a report, but every shipped bucket should still pass it."""
    assert check_recipe(RECIPES[name], base()) == []


def test_check_recipe_reports_rather_than_raises_when_a_bucket_runs_wide():
    wide = Recipe("wide", "snr", "deliberately past the cap",
                  overrides={"chain.additive_noise.snr_db":
                             {"beta_scaled": [2.0, 2.0, -40, 80]}})
    findings = check_recipe(wide, base(), load_envelope())
    assert findings and any("additive_noise.snr_db" in text for text in findings)


def test_snr_buckets_actually_separate():
    low = step(RECIPES["low_snr"].apply(base()), "additive_noise")["snr_db"]
    high = step(RECIPES["high_snr_clean"].apply(base()), "additive_noise")["snr_db"]
    assert low["beta_scaled"][3] < high["beta_scaled"][2]


# --------------------------------------------------------------------------
# text sources
# --------------------------------------------------------------------------


def test_scenario_knobs_reach_the_grammar():
    source = RECIPES["eu_readback_errors"].text_source()
    assert source.config.region == "eu"
    assert source.config.readback_error_prob == pytest.approx(0.25)
    assert RECIPES["us_routine"].text_source().config.region == "us"
    assert RECIPES["mixed_phonetic_respell"].text_source() \
        .config.phonetic_respelling_prob == pytest.approx(1.0)


def test_filtered_sources_hit_their_target():
    for name, expected in (("dense_numerics", set(NUMERIC_KINDS)),
                           ("rare_and_emergency", None)):
        source = RECIPES[name].text_source()
        assert isinstance(source, FilteredTextSource)
        rng = random.Random(0)
        drawn = [source.sample(rng) for _ in range(40)]
        assert all(source.matches(utterance) for utterance in drawn)
        if expected is not None:
            assert {utterance.kind for utterance in drawn} <= expected
        else:
            assert {utterance.category for utterance in drawn} \
                <= {"rare_vocab", "emergency"}
        assert 0.0 < source.hit_rate <= 1.0


def test_dense_numerics_is_denser_in_numbers_than_the_plain_grammar():
    rng = random.Random(1)
    filtered = RECIPES["dense_numerics"].text_source()
    plain = RECIPES["eu_routine"].text_source()
    counts = Counter()
    for label, source in (("filtered", filtered), ("plain", plain)):
        for _ in range(60):
            counts[label] += sum(
                token.isdigit() or token in {"one", "two", "three", "four", "five",
                                             "six", "seven", "eight", "nine", "zero",
                                             "niner", "tree", "fife", "fower"}
                for token in source.sample(rng).transcript.split())
    assert counts["filtered"] > counts["plain"]


def test_filter_returns_a_draw_rather_than_hanging_when_nothing_matches():
    source = FilteredTextSource(RECIPES["eu_routine"].text_source(),
                                kinds=("no_such_kind",), max_tries=5)
    utterance = source.sample(random.Random(0))
    assert utterance is not None
    assert source.accepted == 0 and source.rejected == 5
    assert source.hit_rate == 0.0


def test_a_filter_needs_something_to_filter_on():
    with pytest.raises(ValueError):
        FilteredTextSource(RECIPES["eu_routine"].text_source())
