"""Builder tests: fake TTS, fake text sources, fake transcriber -- no Kokoro,
no network, no GPU."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from atcgen.config import config_hash, dump_resolved, load_config
from atcgen.dataset import build as build_mod
from atcgen.dataset.build import DEFAULT_CONFIG, build_dataset, make_backend
from atcgen.text.grammar import Utterance
from atcgen.text.sources import WeightedSampler

LIGHT_CHANNEL = """
channel:
  profile: test
  clean_arm_prob: 0.0
  chain:
    - primitive: bandpass
      prob: 1.0
      low: {uniform: [250, 400]}
      high: {uniform: [2800, 3600]}
    - primitive: additive_noise
      prob: 1.0
      snr_db: {uniform: [8, 25]}
      color: {choice: [white, pink]}
"""


def write_config(tmp_path: Path, body: str = "", channel: str = LIGHT_CHANNEL) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("mode: procedural\nseed: 3\n" + body + channel)
    return path


def config_for(tmp_path: Path, body: str = "", channel: str = LIGHT_CHANNEL):
    return load_config(write_config(tmp_path, body, channel))


class FakeTTS:
    """Deterministic tone whose length tracks the text; honours voice/speed."""

    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def synthesize(self, text, rng, voice="af_heart", speed=1.0):
        self.calls.append((text, voice, speed))
        seconds = min(0.8 + 0.02 * len(text), 3.0) / speed
        t = np.arange(int(self.sample_rate * seconds)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class PoolTTS(FakeTTS):
    """A Kokoro-shaped engine: no voice/speed keywords, pools instead."""

    def __init__(self, voices, speed_range=(0.9, 1.6)):
        super().__init__()
        self.voices = list(voices)
        self.speed_range = speed_range

    def synthesize(self, text, rng):
        return FakeTTS.synthesize(self, text, rng, rng.choice(self.voices),
                                  rng.uniform(*self.speed_range))


class SilentTTS(FakeTTS):
    def synthesize(self, text, rng, voice="af_heart", speed=1.0):
        self.calls.append((text, voice, speed))
        return np.zeros(int(self.sample_rate * 1.5), np.float32)


class PoolSource:
    """A text source that exposes its records (quota/weight sampling applies)."""

    def __init__(self, records):
        self.records = records

    def sample(self, rng):
        return rng.choice(self.records)


class StreamSource:
    """A source with no record pool: the builder just asks for one at a time."""

    def __init__(self):
        self.n = 0

    def sample(self, rng):
        self.n += 1
        return Utterance(spoken=f"delta {self.n} cleared to land",
                         transcript=f"delta {self.n} cleared to land",
                         role="pilot" if self.n % 2 else "controller", kind="landing")


def utterances(spec):
    """spec: [(category, weight, count), ...] -> flat list of utterances."""
    out = []
    for category, weight, count in spec:
        for index in range(count):
            text = f"{category} utterance {index}"
            out.append(Utterance(spoken=text, transcript=text, role="controller",
                                 kind=category, weight=weight, category=category))
    return out


def read_manifest(manifest_path):
    return [json.loads(line) for line in Path(manifest_path).read_text().splitlines()
            if line.strip()]


def read_stats(manifest_path):
    return json.loads((Path(manifest_path).parent / "stats.json").read_text())


# --------------------------------------------------------------------------
# manifest / provenance
# --------------------------------------------------------------------------

def test_manifest_record_schema_and_gen_blob(tmp_path):
    config = config_for(tmp_path, "dataset: {noise_only_frac: 0.0}\n")
    manifest = build_dataset(config, tmp_path / "out", 6, PoolSource(utterances(
        [("routine", 1.0, 3)])), FakeTTS())
    records = read_manifest(manifest)

    assert len(records) == 6
    for index, record in enumerate(records):
        assert set(record) == {"audio", "text", "role", "kind", "category",
                               "duration", "gen"}
        assert record["audio"] == f"wavs/{index:06d}.wav"
        assert (tmp_path / "out" / record["audio"]).exists()
        gen = record["gen"]
        assert set(gen) == {"mode", "config_hash", "seed_index", "voice", "speed",
                            "pitch", "channel", "qc"}
        assert gen["mode"] == "procedural"
        assert gen["seed_index"] == index
        assert gen["voice"] in config.tts.voices
        assert 0.95 <= gen["speed"] <= 1.55
        assert gen["pitch"] is None                      # lands in P3
        assert set(gen["channel"]) == {"hops", "clean_arm", "snr_db", "steps"}
        assert 8 <= gen["channel"]["snr_db"] <= 25
        assert gen["qc"] == {"ok": True, "reason": None, "attempts": 1}
        wav, sr = sf.read(tmp_path / "out" / record["audio"], dtype="float32")
        assert sr == config.output.sample_rate
        assert record["duration"] == pytest.approx(len(wav) / sr, abs=0.001)


def test_config_hash_matches_the_dumped_resolved_config(tmp_path):
    config = config_for(tmp_path)
    out = tmp_path / "out"
    manifest = build_dataset(config, out, 2, PoolSource(utterances([("routine", 1.0, 2)])),
                             FakeTTS())
    _, digest = dump_resolved(config, tmp_path / "elsewhere")

    assert (out / "config.resolved.yaml").exists()
    assert digest == config_hash(config)
    assert {record["gen"]["config_hash"] for record in read_manifest(manifest)} == {digest}
    assert read_stats(manifest)["config_hash"] == digest
    assert load_config(out / "config.resolved.yaml") == config


def test_same_seed_reproduces_the_run(tmp_path):
    def run(name):
        return build_dataset(config_for(tmp_path), tmp_path / name, 4,
                             PoolSource(utterances([("routine", 1.0, 4)])), FakeTTS())

    first, second = run("a"), run("b")
    assert first.read_text() == second.read_text()
    assert (first.parent / "wavs/000002.wav").read_bytes() == \
           (second.parent / "wavs/000002.wav").read_bytes()


def test_default_profile_builds(tmp_path):
    config = load_config(DEFAULT_CONFIG)
    manifest = build_dataset(config, tmp_path / "out", 3, StreamSource(), FakeTTS())
    records = read_manifest(manifest)
    assert len(records) == 3
    assert all(np.isfinite(sf.read(tmp_path / "out" / r["audio"], dtype="float32")[0]).all()
               for r in records)


# --------------------------------------------------------------------------
# TTS draws
# --------------------------------------------------------------------------

def test_voice_and_speed_come_from_config(tmp_path):
    config = config_for(tmp_path, "tts:\n  voices: [am_adam, bm_george]\n"
                                  "  speed: {uniform: [1.2, 1.3]}\n")
    tts = FakeTTS()
    manifest = build_dataset(config, tmp_path / "out", 8,
                             PoolSource(utterances([("routine", 1.0, 2)])), tts)
    records = read_manifest(manifest)

    assert {record["gen"]["voice"] for record in records} == {"am_adam", "bm_george"}
    assert all(1.2 <= record["gen"]["speed"] <= 1.3 for record in records)
    assert [(voice, round(speed, 3)) for _, voice, speed in tts.calls] == \
           [(record["gen"]["voice"], record["gen"]["speed"]) for record in records]


def test_pool_style_engine_is_pinned_to_the_drawn_voice(tmp_path):
    config = config_for(tmp_path, "tts:\n  voices: [am_adam, bm_george]\n")
    tts = PoolTTS(voices=["not_used"], speed_range=(9.0, 9.0))
    manifest = build_dataset(config, tmp_path / "out", 4,
                             PoolSource(utterances([("routine", 1.0, 2)])), tts)

    for (_, voice, speed), record in zip(tts.calls, read_manifest(manifest)):
        assert voice == record["gen"]["voice"] and voice != "not_used"
        assert speed == pytest.approx(record["gen"]["speed"], abs=0.001)
    assert tts.voices == ["not_used"] and tts.speed_range == (9.0, 9.0)


# --------------------------------------------------------------------------
# sampling: weights and category quotas
# --------------------------------------------------------------------------

def test_weighted_sampling_within_a_category():
    records = [Utterance("heavy", "heavy", "controller", "k", weight=9.0),
               Utterance("light", "light", "controller", "k", weight=1.0)]
    sampler = WeightedSampler(records)
    rng = random.Random(0)
    drawn = [sampler.sample(rng).spoken for _ in range(2000)]
    assert 0.85 < drawn.count("heavy") / len(drawn) < 0.95


def test_uniform_is_the_default_without_weights_or_quotas():
    sampler = WeightedSampler(utterances([("routine", 1.0, 4)]))
    rng = random.Random(1)
    counts = Counter(sampler.sample(rng).spoken for _ in range(4000))
    assert len(counts) == 4
    assert all(0.2 < count / 4000 < 0.3 for count in counts.values())
    assert sampler.achieved() == {"routine": 1.0}


def test_quota_top_up_reaches_target_fractions(tmp_path):
    config = config_for(tmp_path, "dataset:\n  noise_only_frac: 0.0\n"
                                  "  category_quotas: {emergency: 0.3, rare_vocab: 0.2}\n")
    source = PoolSource(utterances([("routine", 1.0, 40), ("emergency", 1.0, 3),
                                    ("rare_vocab", 1.0, 2)]))
    manifest = build_dataset(config, tmp_path / "out", 400, source, FakeTTS())
    stats = read_stats(manifest)

    assert stats["quotas"]["targets"] == {"emergency": 0.3, "rare_vocab": 0.2}
    assert stats["quotas"]["achieved"]["emergency"] == pytest.approx(0.3, abs=0.06)
    assert stats["quotas"]["achieved"]["rare_vocab"] == pytest.approx(0.2, abs=0.06)
    assert stats["category_fractions"]["routine"] == pytest.approx(0.5, abs=0.08)
    assert stats["quotas"]["unavailable_categories"] == []


def test_quota_for_a_category_the_source_lacks_is_best_effort(tmp_path):
    config = config_for(tmp_path, "dataset:\n  noise_only_frac: 0.0\n"
                                  "  category_quotas: {emergency: 0.5}\n")
    manifest = build_dataset(config, tmp_path / "out", 10,
                             PoolSource(utterances([("routine", 1.0, 3)])), FakeTTS())
    stats = read_stats(manifest)

    assert stats["quotas"]["unavailable_categories"] == ["emergency"]
    assert stats["quotas"]["achieved"] == {"emergency": 0.0}
    assert stats["category_fractions"] == {"routine": 1.0}


def test_streaming_source_without_a_record_pool(tmp_path):
    config = config_for(tmp_path, "dataset: {noise_only_frac: 0.0}\n")
    source = StreamSource()
    records = read_manifest(build_dataset(config, tmp_path / "out", 5, source, FakeTTS()))
    assert source.n == 5
    assert [record["text"] for record in records] == \
           [f"delta {i} cleared to land" for i in range(1, 6)]
    assert {record["category"] for record in records} == {"routine"}


# --------------------------------------------------------------------------
# Tier 0 QC gates
# --------------------------------------------------------------------------

def test_qc_rejects_silent_samples_and_records_the_reason(tmp_path):
    channel = """
channel:
  chain:
    - primitive: bandpass
      prob: 1.0
      low: 300.0
      high: 3400.0
"""     # no additive noise: silence in stays silence out
    config = config_for(tmp_path, "dataset: {noise_only_frac: 0.0}\n", channel)
    manifest = build_dataset(config, tmp_path / "out", 3,
                             PoolSource(utterances([("routine", 1.0, 2)])), SilentTTS())
    stats = read_stats(manifest)

    assert stats["qc"]["reasons"] == {"silence": 3 * (config.qc.max_retries + 1)}
    assert stats["qc"]["kept"] == 0
    assert stats["qc"]["discard_rate"] == 1.0
    assert stats["qc"]["kept_with_flag"] == 3          # kept flagged, never dropped
    for record in read_manifest(manifest):
        assert record["gen"]["qc"] == {"ok": False, "reason": "silence",
                                       "attempts": config.qc.max_retries + 1}


def test_asr_roundtrip_gate_uses_the_injected_transcriber(tmp_path):
    config = config_for(tmp_path, "dataset: {noise_only_frac: 0.0}\n"
                                  "qc: {asr_roundtrip: true, max_retries: 1}\n")
    source = PoolSource(utterances([("routine", 1.0, 1)]))
    transcript = source.records[0].transcript

    deaf = build_dataset(config, tmp_path / "bad", 2, source, FakeTTS(),
                         transcriber=lambda wav, sr: "")
    assert read_stats(deaf)["qc"]["reasons"] == {"asr_wer": 4}

    good = build_dataset(config, tmp_path / "good", 2, source, FakeTTS(),
                         transcriber=lambda wav, sr: transcript)
    stats = read_stats(good)
    assert stats["qc"]["reasons"] == {} and stats["qc"]["kept"] == 2
    assert stats["qc"]["asr_roundtrip"] is True


def test_qc_can_be_disabled(tmp_path):
    config = config_for(tmp_path, "dataset: {noise_only_frac: 0.0}\nqc: {enabled: false}\n")
    manifest = build_dataset(config, tmp_path / "out", 2,
                             PoolSource(utterances([("routine", 1.0, 2)])), SilentTTS())
    stats = read_stats(manifest)
    assert stats["qc"]["total"] == 0 and stats["qc"]["kept_with_flag"] == 0
    assert all("qc" not in record["gen"] for record in read_manifest(manifest))


# --------------------------------------------------------------------------
# noise-only samples, backends, run outputs
# --------------------------------------------------------------------------

def test_noise_only_fraction_is_honoured(tmp_path):
    all_noise = build_dataset(config_for(tmp_path, "dataset: {noise_only_frac: 1.0}\n"),
                              tmp_path / "noise", 4,
                              PoolSource(utterances([("routine", 1.0, 2)])), FakeTTS())
    records = read_manifest(all_noise)
    assert [(r["text"], r["role"], r["kind"], r["category"]) for r in records] == \
           [("", "none", "noise", "noise")] * 4
    assert all(r["gen"]["voice"] is None and 2.0 <= r["duration"] <= 6.5 for r in records)
    assert read_stats(all_noise)["noise_only"]["achieved"] == 1.0
    for record in records:                 # a continuous bed, not silence + clicks
        wav, sr = sf.read(all_noise.parent / record["audio"], dtype="float32")
        frames = wav[:len(wav) // 160 * 160].reshape(-1, 160)
        quiet_frame_db = np.percentile(10 * np.log10((frames ** 2).mean(1) + 1e-20), 10)
        assert quiet_frame_db > -50.0

    none = build_dataset(config_for(tmp_path, "dataset: {noise_only_frac: 0.0}\n"),
                         tmp_path / "speech", 4,
                         PoolSource(utterances([("routine", 1.0, 2)])), FakeTTS())
    assert read_stats(none)["noise_only"]["achieved"] == 0.0
    assert all(record["text"] for record in read_manifest(none))


def test_mix_mode_draws_both_backends(tmp_path, monkeypatch):
    path = tmp_path / "mix.yaml"
    path.write_text("mode: mix\nseed: 5\ndataset: {noise_only_frac: 0.0}\n"
                    "backends:\n  - {backend: procedural, weight: 0.5}\n"
                    "  - {backend: calibrated, weight: 0.5}\n" + LIGHT_CHANNEL)
    config = load_config(path)

    real_backend = make_backend(config, "procedural")
    monkeypatch.setattr(build_mod, "make_backend",
                        lambda cfg, name=None: real_backend)   # calibrated is M2.3
    manifest = build_dataset(config, tmp_path / "out", 40,
                             PoolSource(utterances([("routine", 1.0, 2)])), FakeTTS())

    drawn = read_stats(manifest)["backends"]
    assert set(drawn) == {"procedural", "calibrated"}
    assert min(drawn.values()) > 5
    assert {record["gen"]["mode"] for record in read_manifest(manifest)} == set(drawn)


def test_calibrated_backend_is_not_implemented_yet(tmp_path):
    path = tmp_path / "mode2.yaml"
    path.write_text("mode: calibrated\n")
    with pytest.raises(NotImplementedError, match="M2.3"):
        make_backend(load_config(path))


def test_run_writes_resolved_config_and_stats(tmp_path):
    config = config_for(tmp_path, "dataset: {noise_only_frac: 0.5}\n")
    out = tmp_path / "out"
    manifest = build_dataset(config, out, 12,
                             PoolSource(utterances([("routine", 1.0, 4)])), FakeTTS())
    stats = read_stats(manifest)

    assert (out / "config.resolved.yaml").exists()
    assert stats["n_samples"] == 12 and stats["mode"] == "procedural"
    assert stats["seed"] == config.seed
    assert sum(stats["categories"].values()) == 12
    assert sum(stats["duration"]["histogram"]["counts"]) == 12
    assert stats["duration"]["p50"] > 0
    assert sum(stats["snr_db"]["histogram"]["counts"]) == stats["snr_db"]["n"]
    assert 8 <= stats["snr_db"]["p50"] <= 25
    assert set(stats["voices"]) <= set(config.tts.voices)
    assert stats["qc"]["total"] >= 12
