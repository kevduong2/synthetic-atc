import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from scipy import signal

from atcgen.channel.learned.preset import (
    BAND_EDGES,
    Preset,
    band_centers,
    write_presets,
)
from atcgen.config import load_config
from atcgen.dataset.build import build_dataset
from atcgen.rl.space import (
    SPACE_MODES,
    SPACES,
    Knob,
    SearchSpace,
    chain_center_knob,
    chain_param_knob,
    chain_prob_knob,
    chain_scalar_knob,
    default_atc_space,
    dist_bound_knob,
    dist_prob_choice_knob,
    dist_prob_knob,
    mode2_safe_space,
    scalar_knob,
    talker_only_space,
)
from atcgen.text.grammar import Utterance

ROOT = Path(__file__).parents[1]
BASE_CONFIG = ROOT / "configs/mode1_matched.yaml"
MODE2_CONFIG = ROOT / "configs/mode2_fastcut_kixd.yaml"
SR = 16000
STATION = "KIXD_TOWER"


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
    # centre slides to 380, the 120 Hz width of [130, 250] is preserved
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
    # Drive the upper bandpass edge below the profile's lower one (3200).
    space = SearchSpace([chain_param_knob("high", "bandpass", "high", 1, 2400.0, 3400.0)])
    mutated = space.to_config(config, np.array([0.0]))
    low, high = step(mutated, "bandpass")["high"]["uniform"]
    assert low <= high
    assert [low, high] == [2400.0, 3200]

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
    assert described["bandpass.low_center"] == pytest.approx(190.0)


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


# -- the Mode 2 (calibrated) space ----------------------------------------
#
# `mode2_safe_space` exists because the Mode 1 space cannot address a
# calibrated profile at all, so an "it resolves" assertion would miss the
# point: the knobs also have to survive `load_config` and produce audio. These
# tests render, off fixture presets and a fixture noise bank, with a fake TTS.


class FakeTTS:
    """Deterministic tone whose length tracks the text; honours voice/speed."""

    sample_rate = 24000

    def synthesize(self, text, rng, voice="af_heart", speed=1.0):
        seconds = min(0.8 + 0.02 * len(text), 3.0) / speed
        t = np.arange(int(self.sample_rate * seconds)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeSource:
    """A `TextSource` with no record pool, so the builder samples it directly."""

    def sample(self, rng):
        text = rng.choice(["cleared to land runway one eight",
                           "roger hold short"])
        return Utterance(spoken=text, transcript=text,
                         role=rng.choice(["pilot", "controller"]), kind="test")


@pytest.fixture
def calibration(tmp_path):
    """Fitted presets and a station-keyed noise bank, as M2.1/M2.2 write them.

    Single-station, like the KIXD capture this space was built for.
    """
    centres = np.asarray(band_centers(BAND_EDGES))
    gains = np.where((centres >= 300.0) & (centres <= 2800.0), 0.0, -60.0)
    presets = [
        Preset(clip_id=f"{STATION}_{index}", station=STATION,
               band_gains_db=[float(value) for value in gains], drive=1.5,
               poly=[0.0, 0.0], agc_tau_ms=60.0, agc_strength=0.2,
               noise_gain=10.0 ** (-20.0 / 20.0), snr_est=20.0, fit_loss=1.0,
               passband_hz=[300.0, 2800.0])
        for index in range(3)
    ]
    presets_path = write_presets(tmp_path / "presets.jsonl", presets)

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    rng = np.random.default_rng(0)
    sos = signal.butter(4, [300.0, 2800.0], btype="bandpass", fs=SR, output="sos")
    rows = []
    for index in range(4):
        # band-limited like a real harvest: the receiver already filtered these
        segment = signal.sosfilt(
            sos, rng.standard_normal(int(0.3 * SR))).astype(np.float32) * 0.01
        sf.write(noise_dir / f"{index:04d}.wav", segment, SR)
        rows.append({"source_clip": "c", "station": STATION, "duration": 0.3,
                     "rms_db": -40.0, "ltas_centroid_hz": 1000.0,
                     "squelch_gated": index == 0})
    (noise_dir / "noise_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return presets_path, noise_dir


def mode2_base(calibration):
    """`configs/mode2_fastcut_kixd.yaml`, repointed at the fixture artifacts."""
    presets_path, noise_dir = calibration
    with MODE2_CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["calibrated"]["calibration"]["presets"] = str(presets_path)
    config["calibrated"]["calibration"]["noise_bank"] = str(noise_dir)
    return config


def test_mode2_safe_knob_paths_all_resolve_against_the_shipped_profile(calibration):
    """Every knob reads a real leaf: a `None` here is a silent 0.5 anchor."""
    space = mode2_safe_space()
    config = mode2_base(calibration)
    missing = [knob.name for knob in space.knobs if knob.read(config) is None]
    assert missing == []


def test_mode2_safe_space_renders_sampled_configs(tmp_path, calibration):
    """Five cube points, resolved and rendered. The blocker this space fixes.

    `_chain_step` raises on a calibrated profile before any audio exists, so
    resolving is necessary but not sufficient -- a knob can resolve and still
    push a value the config parser or the backend rejects.
    """
    space = mode2_safe_space()
    config = mode2_base(calibration)
    rng = np.random.default_rng(0)

    for trial in range(5):
        vector = rng.random(space.dim)
        candidate = space.to_config(config, vector)
        path = tmp_path / f"candidate_{trial}.yaml"
        path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        loaded = load_config(path)
        assert loaded.mode == "calibrated"

        out = tmp_path / f"render_{trial}"
        manifest = build_dataset(loaded, out, 2, FakeSource(), tts=FakeTTS())
        rows = [json.loads(line) for line in
                manifest.read_text().splitlines() if line.strip()]
        assert len(rows) == 2
        for row in rows:
            wav, sample_rate = sf.read(out / row["audio"])
            assert sample_rate == loaded.output.sample_rate
            assert len(wav) > 0
            assert np.isfinite(wav).all()
            # the fitted preset fired, which is what makes this Mode 2
            assert any(step["primitive"] == "calibrated_preset"
                       for step in row["gen"]["channel"]["steps"])




def test_mode2_safe_extremes_still_load(tmp_path, calibration):
    """Both corners of the cube, where a bound-pair knob could invert a range."""
    space = mode2_safe_space()
    config = mode2_base(calibration)
    for name, vector in (("low", np.zeros(space.dim)),
                         ("high", np.ones(space.dim)),
                         ("default", space.default_vector(config))):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(space.to_config(config, vector)),
                        encoding="utf-8")
        loaded = load_config(path)
        assert loaded.calibrated is not None
        jitter = loaded.calibrated.calibration.snr_jitter_db.value
        assert jitter[0] <= jitter[1]
        speed = loaded.tts.speed.value
        assert speed[0] <= speed[1]


def test_mode2_safe_default_vector_round_trips_the_profile(calibration):
    config = mode2_base(calibration)
    space = mode2_safe_space()
    restored = space.to_config(config, space.default_vector(config))

    assert (restored["calibrated"]["post_effects"]["squelch"]["prob"]
            == pytest.approx(config["calibrated"]["post_effects"]["squelch"]["prob"]))
    assert (restored["calibrated"]["residual"]["apply_prob"]
            == pytest.approx(config["calibrated"]["residual"]["apply_prob"]))
    assert (restored["calibrated"]["calibration"]["snr_jitter_db"]["uniform"]
            == pytest.approx(
                config["calibrated"]["calibration"]["snr_jitter_db"]["uniform"]))
    assert (restored["dataset"]["pilot_double_hop_prob"]
            == pytest.approx(config["dataset"]["pilot_double_hop_prob"]))


def test_mode2_safe_reaches_the_calibrated_block_talker_only_leaves_alone(calibration):
    """The two spaces are for different jobs, and the diff shows which."""
    config = mode2_base(calibration)
    wide = mode2_safe_space().to_config(config, np.zeros(mode2_safe_space().dim))
    narrow = talker_only_space(config).to_config(
        config, np.zeros(talker_only_space(config).dim))

    assert wide["calibrated"] != config["calibrated"]
    assert wide["dataset"] != config["dataset"]
    assert narrow["calibrated"] == config["calibrated"]
    assert narrow["dataset"] == config["dataset"]


# -- talker_only: tonight's short-budget space -----------------------------

def test_talker_only_is_four_knobs_and_nothing_below_a_backend():
    space = talker_only_space()
    assert space.dim == 4
    assert {knob.name for knob in space.knobs} == {
        "tts.speed_lo", "tts.speed_hi",
        "voice_augment.tempo_prob", "voice_augment.pitch_prob",
    }
    # every name is one Mode 1 also knows, because none of them is channel-side
    assert {knob.name for knob in space.knobs} <= {
        knob.name for knob in default_atc_space().knobs}


def test_talker_only_renders_on_a_calibrated_profile(tmp_path, calibration):
    space = talker_only_space()
    config = mode2_base(calibration)
    rng = np.random.default_rng(0)

    for trial in range(5):
        candidate = space.to_config(config, rng.random(space.dim))
        path = tmp_path / f"talker_{trial}.yaml"
        path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        loaded = load_config(path)
        assert loaded.mode == "calibrated"

        out = tmp_path / f"talker_render_{trial}"
        manifest = build_dataset(loaded, out, 2, FakeSource(), tts=FakeTTS())
        rows = [json.loads(line) for line in
                manifest.read_text().splitlines() if line.strip()]
        assert len(rows) == 2
        for row in rows:
            wav, sample_rate = sf.read(out / row["audio"])
            assert sample_rate == loaded.output.sample_rate and len(wav) > 0
            assert np.isfinite(wav).all()
            assert any(step["primitive"] == "calibrated_preset"
                       for step in row["gen"]["channel"]["steps"])


def test_talker_only_renders_on_a_procedural_profile_too(tmp_path):
    """Mode-agnostic, which is what lets tonight's Mode 1 base arm use it."""
    space = talker_only_space()
    config = base()
    config["channel"]["noise"]["beds_dir"] = None      # not present in a checkout
    path = tmp_path / "mode1_candidate.yaml"
    path.write_text(yaml.safe_dump(space.to_config(config, np.full(space.dim, 0.7))),
                    encoding="utf-8")
    loaded = load_config(path)
    assert loaded.mode == "procedural"

    out = tmp_path / "render_mode1"
    manifest = build_dataset(loaded, out, 2, FakeSource(), tts=FakeTTS())
    assert len([line for line in manifest.read_text().splitlines() if line.strip()]) == 2


def test_talker_only_default_vector_round_trips_the_profile(calibration):
    config = mode2_base(calibration)
    space = talker_only_space(config)
    described = space.describe(space.default_vector(config))

    assert described["tts.speed_lo"] == pytest.approx(config["tts"]["speed"]["uniform"][0])
    assert described["tts.speed_hi"] == pytest.approx(config["tts"]["speed"]["uniform"][1])
    assert described["voice_augment.tempo_prob"] == pytest.approx(
        config["voice_augment"]["tempo"]["prob"])
    # the categorical knob snaps back onto the profile's own arm, not a midpoint
    assert described["voice_augment.pitch_prob"] == pytest.approx(
        config["voice_augment"]["pitch_semitones"]["prob"])


def test_pitch_prob_is_a_two_armed_categorical_anchored_on_the_base(calibration):
    """The open question is whether to pitch-shift at all, not by how much."""
    config = mode2_base(calibration)
    config["voice_augment"]["pitch_semitones"]["prob"] = 0.4
    space = talker_only_space(config)
    knob = next(k for k in space.knobs if k.name == "voice_augment.pitch_prob")

    assert knob.kind == "choice"
    assert knob.values == (0.0, 0.4)
    # the whole cube collapses onto the two arms, with no value in between
    assert {knob.value(u / 20) for u in range(21)} == {0.0, 0.4}
    assert space.to_config(config, np.array([0.5, 0.5, 0.5, 0.0]))[
        "voice_augment"]["pitch_semitones"]["prob"] == pytest.approx(0.0)
    assert space.to_config(config, np.array([0.5, 0.5, 0.5, 1.0]))[
        "voice_augment"]["pitch_semitones"]["prob"] == pytest.approx(0.4)


def test_pitch_prob_falls_back_when_the_profile_has_no_gate():
    space = talker_only_space(
        {"voice_augment": {"pitch_semitones": {"uniform": [-2, 2]}}})
    knob = next(k for k in space.knobs if k.name == "voice_augment.pitch_prob")
    # a profile with no `prob` key reads as 1.0 (always on), which is a valid arm
    assert knob.values == (0.0, 1.0)
    assert talker_only_space().knobs[-1].values == (0.0, 0.5)


def test_choice_knob_rejects_a_degenerate_value_list():
    with pytest.raises(ValueError, match="two or more distinct"):
        dist_prob_choice_knob("k", "voice_augment.tempo", (0.5, 0.5))
    with pytest.raises(ValueError, match="only for choice knobs"):
        Knob("k", 0.0, 1.0, "linear", lambda c, v: None, None, (0.0, 1.0))


def test_default_space_cannot_address_a_calibrated_profile(calibration):
    """The documented blocker, pinned so the spaces cannot be swapped."""
    space = default_atc_space()
    config = mode2_base(calibration)
    with pytest.raises(KeyError, match="channel.chain"):
        space.to_config(config, np.full(space.dim, 0.5))


def test_mode2_safe_cannot_address_a_procedural_profile():
    """... and the converse, which is why SPACE_MODES confines it."""
    space = mode2_safe_space()
    with pytest.raises(KeyError, match="calibrated"):
        space.to_config(base(), np.full(space.dim, 0.5))


def test_space_registry_covers_every_space_with_a_mode_and_a_budget():
    assert set(SPACES) == set(SPACE_MODES) == {"default", "mode2_safe", "talker_only"}
    assert SPACES["default"]().dim == default_atc_space().dim == 19
    assert SPACES["mode2_safe"]().dim == mode2_safe_space().dim == 15
    assert SPACES["talker_only"]().dim == talker_only_space().dim == 4
    # every constructor takes the base profile, whether or not it uses it
    assert SPACES["default"](base()).dim == default_atc_space().dim

    assert SPACE_MODES["default"] == {"procedural"}
    assert SPACE_MODES["mode2_safe"] == {"calibrated"}
    assert SPACE_MODES["talker_only"] == {"procedural", "calibrated"}
