import random

import numpy as np
import pytest

from atcgen.channel.chain import (HOP2_SNR_DB, PAD_SEC, RECEIVER_END,
                                  SOURCE_ONCE, ChannelRecord, ProceduralChannel,
                                  UtteranceMeta, mild_chain)
from atcgen.channel.primitives import TARGET_SR
from atcgen.config import ChainStep, DistSpec, load_config

CONFIG = "configs/mode1_default.yaml"


def _tone(sr=24000, sec=1.0, f=440.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _channel(**kwargs):
    return ProceduralChannel.from_config(load_config(CONFIG).channel, **kwargs)


def _step(primitive, prob=1.0, **params):
    return ChainStep(primitive, prob, {k: DistSpec.parse(v) for k, v in params.items()})


def test_output_sr_range_and_padding():
    sim = _channel()
    for seed in range(10):
        wav, rec = sim(_tone(), 24000, random.Random(seed), UtteranceMeta(role="pilot"))
        assert wav.dtype == np.float32
        assert np.abs(wav).max() <= 1.0
        assert np.isfinite(wav).all()
        assert len(wav) == pytest.approx(TARGET_SR * (1 + 2 * PAD_SEC), abs=50)
        assert isinstance(rec, ChannelRecord)


def test_deterministic_given_seed():
    sim = _channel()
    a, ra = sim(_tone(), 24000, random.Random(3))
    b, rb = sim(_tone(), 24000, random.Random(3))
    assert np.array_equal(a, b)
    assert ra.as_dict() == rb.as_dict()


def test_record_lists_applied_steps_in_chain_order():
    sim = _channel()
    _, rec = sim(_tone(), 24000, random.Random(0))
    declared = [s.primitive for s in sim.steps]
    applied = rec.applied()
    assert set(applied) <= set(declared)
    assert applied == sorted(applied, key=declared.index)
    assert "bandpass" in applied and "additive_noise" in applied
    assert rec.hops == 1 and rec.clean_arm is False
    noise = next(s for s in rec.steps if s["primitive"] == "additive_noise")
    assert rec.snr_db == pytest.approx(noise["snr_db"], abs=0.01)
    assert 3 <= rec.snr_db <= 25          # the config's declared range


def test_step_prob_gates_application():
    x = _tone(sr=TARGET_SR)
    always = ProceduralChannel([_step("heterodyne", prob=1.0)])
    never = ProceduralChannel([_step("heterodyne", prob=0.0)])
    assert always(x, TARGET_SR, random.Random(0))[1].applied() == ["heterodyne"]
    assert never(x, TARGET_SR, random.Random(0))[1].applied() == []


def test_params_are_redrawn_per_sample():
    sim = ProceduralChannel([_step("additive_noise", snr_db={"uniform": [3, 25]})])
    snrs = {sim(_tone(sr=TARGET_SR), TARGET_SR, random.Random(s))[1].snr_db
            for s in range(20)}
    assert len(snrs) == 20
    assert all(3 <= v <= 25 for v in snrs)


def test_double_hop_runs_transmit_path_twice_and_receiver_once():
    sim = _channel()
    wav1, rec1 = sim(_tone(), 24000, random.Random(7), hops=1)
    wav2, rec2 = sim(_tone(), 24000, random.Random(7), hops=2)
    assert np.isfinite(wav2).all()
    assert abs(len(wav2) - len(wav1)) < 100
    assert rec2.hops == 2
    assert {s["hop"] for s in rec2.steps} == {0, 1}
    for step in rec2.steps:
        if step["primitive"] in RECEIVER_END:
            assert step["hop"] == 0          # receiver end runs once, after the hops
    hop_counts = [sum(s["primitive"] == "bandpass" and s["hop"] == h for s in rec2.steps)
                  for h in (0, 1)]
    assert hop_counts == [1, 1]


def test_second_hop_uses_relay_snr_range():
    sim = _channel()
    for seed in range(10):
        _, rec = sim(_tone(), 24000, random.Random(seed), hops=2)
        hop2 = [s for s in rec.steps
                if s["primitive"] == "additive_noise" and s["hop"] == 1]
        assert hop2 and all(HOP2_SNR_DB[0] <= s["snr_db"] <= HOP2_SNR_DB[1] for s in hop2)


def test_source_stage_runs_once_and_first_whatever_the_hop_count():
    steps = [_step("bandpass", low=300.0, high=3400.0),
             _step("mic_coloration", tilt_db=3.0),
             _step("ptt_truncation", head_ms=50.0),
             _step("additive_noise", snr_db=20.0),
             _step("squelch_gate", floor_db=-40.0, tail_burst_prob=0.0)]
    sim = ProceduralChannel(steps)
    _, rec = sim(_tone(), 24000, random.Random(0), hops=2)
    applied = rec.applied()
    assert [s for s in applied if s in SOURCE_ONCE] == ["mic_coloration", "ptt_truncation"]
    assert applied[:2] == ["mic_coloration", "ptt_truncation"]     # before any hop
    assert all(s["hop"] == 0 for s in rec.steps if s["primitive"] in SOURCE_ONCE)
    assert applied.count("bandpass") == 2                          # path side, per hop


def test_receiver_stage_gates_the_padding_after_the_noise():
    steps = [_step("additive_noise", snr_db=6.0),
             _step("squelch_gate", floor_db=-45.0, tail_burst_prob=0.0)]
    gated = ProceduralChannel(steps)
    plain = ProceduralChannel(steps[:1])
    pad = int(TARGET_SR * PAD_SEC)
    quiet, _ = gated(_tone(), 24000, random.Random(2))
    noisy, _ = plain(_tone(), 24000, random.Random(2))
    head = slice(pad // 4, pad - 400)
    assert np.std(noisy[head]) > 1e-3                              # continuous noise bed
    assert np.std(quiet[head]) < 0.1 * np.std(noisy[head])         # ... gated away
    assert {"squelch_gate", "agc_attack"} <= RECEIVER_END


def test_bandpass_removes_out_of_band_energy_through_the_chain():
    # noise/artifacts off so the filter is what is being measured
    steps = [_step("bandpass", low=300.0, high=3400.0)]
    sim = ProceduralChannel(steps)
    low, _ = sim(_tone(f=100), 24000, random.Random(4))
    mid, _ = sim(_tone(f=1000), 24000, random.Random(4))
    assert np.mean(mid ** 2) > 10 * np.mean(low ** 2)


def test_noise_floor_fills_the_padding_at_low_snr():
    sim = ProceduralChannel([_step("additive_noise", snr_db=3.0)])
    wav, _ = sim(_tone(f=1000), 24000, random.Random(1))
    assert np.std(wav[:int(TARGET_SR * 0.1)]) > 1e-4


def test_interference_is_mixed_in_only_when_supplied():
    sim = ProceduralChannel([_step("cochannel_mix", level=0.2)])
    interference = np.random.default_rng(0).standard_normal(TARGET_SR).astype(np.float32) * 0.3
    with_i, _ = sim(_tone(), 24000, random.Random(5), interference=interference)
    without, _ = sim(_tone(), 24000, random.Random(5))
    assert np.isfinite(with_i).all()
    assert not np.array_equal(with_i, without)


def test_noise_bank_is_passed_through_to_additive_noise(tmp_path):
    import soundfile as sf

    from atcgen.channel.primitives import NoiseBank

    bed = np.random.default_rng(0).standard_normal(TARGET_SR).astype(np.float32) * 0.05
    sf.write(tmp_path / "noise_00000.wav", bed, TARGET_SR)
    sim = _channel(noise_bank=NoiseBank(tmp_path))
    wav, rec = sim(_tone(), 24000, random.Random(8))
    assert np.isfinite(wav).all()
    assert rec.snr_db is not None


def test_clean_arm_bypasses_everything_but_bandpass():
    cfg = load_config(CONFIG).channel
    cfg.clean_arm_prob = 1.0
    sim = ProceduralChannel.from_config(cfg)
    wav, rec = sim(_tone(), 24000, random.Random(0), hops=2)
    assert rec.clean_arm is True
    assert rec.applied() == ["bandpass"]
    assert rec.hops == 1
    assert rec.snr_db is None
    pad = int(TARGET_SR * PAD_SEC)
    assert np.abs(wav[:pad]).max() < 1e-3      # padding stays silent: no noise added


def test_shuffle_groups_reorder_within_the_group():
    steps = [_step("hum", amp=0.01), _step("crackle", rate=5.0), _step("heterodyne")]
    group = [["hum", "crackle", "heterodyne"]]
    plain = ProceduralChannel(steps)
    shuffled = ProceduralChannel(steps, shuffle_groups=group)
    assert all(plain(_tone(sr=TARGET_SR), TARGET_SR, random.Random(s))[1].applied()
               == ["hum", "crackle", "heterodyne"] for s in range(5))
    orders = {tuple(shuffled(_tone(sr=TARGET_SR), TARGET_SR, random.Random(s))[1].applied())
              for s in range(20)}
    assert len(orders) > 1


def test_unknown_primitive_rejected():
    with pytest.raises(ValueError, match="unknown channel primitive"):
        ProceduralChannel([_step("teleporter")])


def test_mild_chain_is_lighter_than_the_default():
    mild = ProceduralChannel(mild_chain())
    wav, rec = mild(_tone(), 24000, random.Random(9))
    assert np.isfinite(wav).all() and np.abs(wav).max() <= 1.0
    assert rec.snr_db >= 15
    assert not {"am_distortion", "dropouts", "hum", "heterodyne",
                "cochannel_mix"} & set(rec.applied())
