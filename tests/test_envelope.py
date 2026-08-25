import json

import pytest

from atcgen.channel.envelope import (RULES, SNAPSHOT, calibrate, check_profile,
                                     load_envelope, measure_envelope, report)
from atcgen.config import ChainStep, ChannelConfig, DistSpec, load_config

ENVELOPE = {"n_real": 100, "sources": ["fake.json"], "metrics": {
    "snr_db": {"p10": 14.0, "p90": 36.0},
    "spectral_edge_hz": {"p10": 1800.0, "p90": 2700.0},
    "spectral_low_hz": {"p10": 156.0, "p90": 265.0},
    "rms_db": {"p10": -25.0, "p90": -19.0},
}}


def _stats(metrics, n_real=100):
    return {"comparison": {"n_real": n_real, "stats": {
        name: {"real_p10": low, "real_p90": high} for name, (low, high) in metrics.items()
    }}}


def _channel(snr=(5.0, 26.0), low=(130.0, 250.0), high=(3200.0, 3400.0)):
    """A `matched`-shaped profile: the one the rule offsets are calibrated on."""
    spec = DistSpec.parse
    return ChannelConfig(profile="test", chain=[
        ChainStep("additive_noise", 1.0, {"snr_db": spec({"uniform": list(snr)})}),
        ChainStep("bandpass", 1.0, {"low": spec({"uniform": list(low)}),
                                    "high": spec({"uniform": list(high)})}),
    ])


def test_measure_envelope_reads_the_real_percentiles(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(_stats({"snr_db": (14.0, 36.0),
                                       "spectral_edge_hz": (1800.0, 2700.0)})))
    envelope = measure_envelope([path])
    assert envelope["metrics"]["snr_db"] == {"p10": 14.0, "p90": 36.0}
    assert envelope["n_real"] == 100 and envelope["sources"] == [str(path)]


def test_several_stats_files_merge_by_widening(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(_stats({"snr_db": (14.0, 36.0)})))
    second.write_text(json.dumps(_stats({"snr_db": (11.0, 40.0)}, n_real=120)))
    envelope = measure_envelope([first, second])
    assert envelope["metrics"]["snr_db"] == {"p10": 11.0, "p90": 40.0}
    assert envelope["n_real"] == 120


def test_stats_without_a_real_reference_are_rejected(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"summary": {"snr_db": {"p50": 20}}}))
    with pytest.raises(ValueError, match="real percentiles"):
        measure_envelope([path])


def test_a_profile_inside_the_envelope_reports_nothing():
    assert check_profile(_channel(), ENVELOPE) == []


def test_ranges_past_the_envelope_are_reported_with_the_distance():
    findings = check_profile(_channel(snr=(-15.0, 26.0), high=(3200.0, 4200.0)), ENVELOPE)
    assert len(findings) == 2
    assert "additive_noise.snr_db low edge -15" in findings[0]
    assert "15.0 dB past the real p10" in findings[0]      # 14.0 - 14.0 offset = 0.0
    assert "bandpass.high high edge 4200" in findings[1]


def test_slack_is_per_rule_and_overridable():
    tight = _channel(snr=(-1.5, 26.0))
    assert check_profile(tight, ENVELOPE) == []            # 1.5 dB out, slack 5
    findings = check_profile(tight, ENVELOPE, slack={"additive_noise.snr_db": 1.0})
    assert [f.split(" edge")[0] for f in findings] == ["additive_noise.snr_db low",
                                                       "additive_noise.snr_db high"]


def test_a_generator_config_also_checks_the_output_level():
    config = load_config("configs/mode1_matched.yaml")
    assert check_profile(config, ENVELOPE) == []
    config.output.loudness_db = DistSpec.parse({"uniform": [-40, -35]})
    findings = check_profile(config, ENVELOPE)
    assert len(findings) == 1 and findings[0].startswith("output.loudness_db low edge -40")
    # a bare ChannelConfig carries no output section, so that rule is skipped
    assert check_profile(config.channel, ENVELOPE) == []


def test_undeclared_parameters_are_skipped_not_guessed():
    bare = ChannelConfig(profile="bare", chain=[ChainStep("bandpass", 1.0, {})])
    assert check_profile(bare, ENVELOPE) == []


def test_calibrate_rederives_the_offsets_from_a_measured_run():
    """`measured - injected`, so the rules can be re-fitted after a chain change."""
    stats = {"summary": {"snr_db": {"p50": 30.0}, "spectral_edge_hz": {"p50": 2300.0},
                         "spectral_low_hz": {"p50": 220.0}, "rms_db": {"p50": -22.5}}}
    offsets = calibrate(stats, load_config("configs/mode1_matched.yaml"))
    assert offsets["bandpass.high"] == pytest.approx(2300.0 - 3300.0)   # corner midpoint
    assert offsets["bandpass.low"] == pytest.approx(220.0 - 190.0)
    assert offsets["output.loudness_db"] == pytest.approx(0.0)          # applied literally
    assert offsets["additive_noise.snr_db"] == pytest.approx(13.8, abs=0.1)  # beta median


def test_the_committed_snapshot_covers_every_rule():
    envelope = load_envelope()
    assert envelope is not None, f"missing {SNAPSHOT}"
    assert {rule.metric for rule in RULES} <= set(envelope["metrics"])
    assert envelope["n_real"] > 0
    for entry in envelope["metrics"].values():
        assert entry["p10"] < entry["p90"]


def test_missing_snapshot_reads_as_no_envelope(tmp_path):
    assert load_envelope(tmp_path / "nope.json") is None


def test_wide_is_reported_as_the_profile_that_explores_past_the_cap():
    envelope = load_envelope()
    matched = load_config("configs/mode1_matched.yaml")
    wide = load_config("configs/mode1_wide.yaml")
    assert check_profile(matched, envelope) == []
    assert any("bandpass.high" in finding for finding in check_profile(wide, envelope))
    text = report(wide, envelope)
    assert "WARN" in text and "bandpass.high" in text
