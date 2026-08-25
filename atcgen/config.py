"""Typed, strict configuration loading for dataset generation.

``pilot_double_hop_prob`` deliberately lives only under ``dataset``.  The
shared architecture is authoritative; the older Mode 1 ``channel.hops``
sketch is not duplicated here.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_DIST_KINDS = {"uniform", "choice", "const", "beta_scaled"}
_SCALAR_TYPES = (str, int, float, bool, type(None))
_DEFAULT_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_eric", "am_onyx",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _probability(value: Any, path: str) -> float:
    number = _number(value, path)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1")
    return number


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted((key for key in data if key not in allowed), key=str)
    if unknown:
        name = unknown[0]
        full_path = f"{path}.{name}" if path else name
        raise ValueError(f"unknown config key: {full_path}")


@dataclass(frozen=True)
class DistSpec:
    """A constant, uniform, choice, or scaled-beta random value."""

    kind: str
    value: Any
    prob: float | None = None

    @classmethod
    def parse(cls, value: Any, path: str = "distribution") -> "DistSpec":
        if isinstance(value, _SCALAR_TYPES):
            return cls("const", value)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a scalar or distribution mapping")

        _reject_unknown(value, _DIST_KINDS | {"prob"}, path)
        kinds = [key for key in _DIST_KINDS if key in value]
        if len(kinds) != 1:
            raise ValueError(f"{path} must contain exactly one distribution kind")
        kind = kinds[0]
        payload = value[kind]
        prob = None
        if "prob" in value:
            prob = _probability(value["prob"], f"{path}.prob")

        if kind == "uniform":
            payload = cls._numeric_sequence(payload, 2, f"{path}.uniform")
            if payload[0] > payload[1]:
                raise ValueError(f"{path}.uniform lower bound exceeds upper bound")
        elif kind == "choice":
            if not isinstance(payload, list) or not payload:
                raise ValueError(f"{path}.choice must be a non-empty list")
            payload = list(payload)
        elif kind == "beta_scaled":
            payload = cls._numeric_sequence(payload, 4, f"{path}.beta_scaled")
            alpha, beta, low, high = payload
            if alpha <= 0 or beta <= 0:
                raise ValueError(f"{path}.beta_scaled alpha and beta must be positive")
            if low > high:
                raise ValueError(f"{path}.beta_scaled lower bound exceeds upper bound")
        return cls(kind, payload, prob)

    @staticmethod
    def _numeric_sequence(value: Any, length: int, path: str) -> list[int | float]:
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"{path} must be a {length}-item list")
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{path} items must be numbers")
        return list(value)

    def sample(self, rng: random.Random) -> Any:
        if self.prob is not None and rng.random() >= self.prob:
            return None
        if self.kind == "const":
            return self.value
        if self.kind == "uniform":
            return rng.uniform(*self.value)
        if self.kind == "choice":
            return rng.choice(self.value)
        if self.kind == "beta_scaled":
            alpha, beta, low, high = self.value
            return low + rng.betavariate(alpha, beta) * (high - low)
        raise ValueError(f"unknown distribution kind: {self.kind}")

    def as_dict(self) -> dict[str, Any]:
        result = {self.kind: self.value}
        if self.prob is not None:
            result["prob"] = self.prob
        return result


@dataclass
class OutputConfig:
    """Delivery format and level.

    ``loudness_mode`` picks which level target the post stage honours: ``rms``
    draws ``loudness_db`` and normalizes RMS to it (what every profile does
    today), ``lufs`` normalizes EBU R128 integrated loudness to
    ``loudness_lufs`` instead (research-findings §4.3, via
    ``atcgen.channel.loudness``).
    """

    sample_rate: int = 16000
    format: str = "wav"
    loudness_db: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"uniform": [-23, -17]})
    )
    loudness_mode: str = "rms"
    loudness_lufs: float = -23.0


@dataclass
class TTSConfig:
    voices: list[str] = field(default_factory=lambda: list(_DEFAULT_VOICES))
    speed: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"uniform": [0.95, 1.55]})
    )


@dataclass
class VoiceAugmentConfig:
    pitch_semitones: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"prob": 0.5, "uniform": [-2, 2]})
    )
    tempo: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"prob": 0.3, "uniform": [0.9, 1.1]})
    )
    eq_tilt_db: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"prob": 0.4, "uniform": [-3, 3]})
    )


@dataclass
class DatasetConfig:
    noise_only_frac: float = 0.03
    pilot_double_hop_prob: float = 0.5
    category_quotas: dict[str, float] = field(default_factory=dict)


@dataclass
class QCConfig:
    """Tier 0 gate settings (05 §2), mirrored onto `atcgen.eval.qc.QCConfig`.

    ``asr_roundtrip`` is off by default: the gate loads a pretrained Whisper,
    which is far too heavy for a routine generation run.  Turn it on for
    release-candidate sets, where discarding ~a third of the samples is the
    point.  ``max_retries`` regenerations are attempted before a failing
    sample is kept with `gen.qc.ok = false`.
    """

    enabled: bool = True
    max_retries: int = 3
    asr_roundtrip: bool = False
    min_duration: float = 0.5
    max_duration: float = 30.0
    max_clip_frac: float = 0.01
    min_rms_db: float = -40.0
    max_rms_db: float = -8.0
    max_wer: float = 0.5


@dataclass
class ChainStep:
    primitive: str
    prob: float
    params: dict[str, DistSpec] = field(default_factory=dict)


@dataclass
class ChannelNoiseConfig:
    beds_dir: str | None = None


@dataclass
class ChannelConfig:
    """The Mode 1 chain.

    ``reapply_bandpass`` re-runs the drawn passband wherever the signal crosses
    a real filter downstream of something that splatters out of band; see
    ``atcgen.channel.chain``'s module docstring.  It is on by default because
    it is physics, and off only for an ablation.
    """

    profile: str = "wide"
    clean_arm_prob: float = 0.07
    chain: list[ChainStep] = field(default_factory=list)
    shuffle_groups: list[list[str]] = field(default_factory=list)
    noise: ChannelNoiseConfig = field(default_factory=ChannelNoiseConfig)
    reapply_bandpass: bool = True


@dataclass
class CalibrationConfig:
    corpus_dir: str = "data/real/calibration"
    presets: str = "runs/calib_v1/presets.jsonl"
    noise_bank: str = "runs/calib_v1/noise/"
    station_mix: dict[str, float] | None = None
    snr_jitter_db: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"uniform": [-3, 3]})
    )
    cross_station_prob: float = 0.1


@dataclass
class ResidualConfig:
    enabled: bool = False
    checkpoint: str = "runs/cut_v1/G_ema.pt"
    apply_prob: float = 0.5
    alpha: DistSpec = field(default_factory=lambda: DistSpec.parse(1.0))
    residual_scale_max: float = 0.35


@dataclass
class SquelchEffectConfig:
    prob: float = 0.8
    gated_floor_prob: float = 0.6


@dataclass
class DropoutsEffectConfig:
    prob: float = 0.15


@dataclass
class CodecEffectConfig:
    prob: float = 0.5
    kind: str = "mp3"
    quality: DistSpec = field(
        default_factory=lambda: DistSpec.parse({"uniform": [0.75, 0.95]})
    )


@dataclass
class PostEffectsConfig:
    squelch: SquelchEffectConfig = field(default_factory=SquelchEffectConfig)
    dropouts: DropoutsEffectConfig = field(default_factory=DropoutsEffectConfig)
    codec: CodecEffectConfig = field(default_factory=CodecEffectConfig)


@dataclass
class ExpansionConfig:
    real_manifest: str = "data/real/labeled/manifest.jsonl"
    target_total: int = 10000
    category_quotas: dict[str, float] = field(default_factory=dict)
    holdout_frac: float = 0.15
    external_texts: str | None = None


@dataclass
class CalibratedConfig:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    post_effects: PostEffectsConfig = field(default_factory=PostEffectsConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)


@dataclass
class BackendConfig:
    backend: str
    weight: float


@dataclass
class GeneratorConfig:
    mode: str = "procedural"
    seed: int = 0
    output: OutputConfig = field(default_factory=OutputConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    voice_augment: VoiceAugmentConfig = field(default_factory=VoiceAugmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    channel: ChannelConfig | None = field(default_factory=ChannelConfig)
    calibrated: CalibratedConfig | None = None
    backends: list[BackendConfig] = field(default_factory=list)


def _parse_output(value: Any, path: str) -> OutputConfig:
    data = _mapping(value, path)
    _reject_unknown(data, {"sample_rate", "format", "loudness_db", "loudness_mode",
                           "loudness_lufs"}, path)
    default = OutputConfig()
    sample_rate = data.get("sample_rate", default.sample_rate)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{path}.sample_rate must be a positive integer")
    format_value = data.get("format", default.format)
    if not isinstance(format_value, str) or not format_value:
        raise ValueError(f"{path}.format must be a non-empty string")
    loudness = DistSpec.parse(data.get("loudness_db", default.loudness_db.as_dict()),
                              f"{path}.loudness_db")
    mode = data.get("loudness_mode", default.loudness_mode)
    if mode not in {"rms", "lufs"}:
        raise ValueError(f"{path}.loudness_mode must be rms or lufs")
    lufs = _number(data.get("loudness_lufs", default.loudness_lufs),
                   f"{path}.loudness_lufs")
    return OutputConfig(sample_rate, format_value, loudness, mode, lufs)


def _parse_tts(value: Any, path: str) -> TTSConfig:
    data = _mapping(value, path)
    _reject_unknown(data, {"voices", "speed"}, path)
    voices = data.get("voices", _DEFAULT_VOICES)
    if (not isinstance(voices, list) or not voices
            or not all(isinstance(v, str) and v for v in voices)):
        raise ValueError(f"{path}.voices must be a non-empty list of strings")
    speed = DistSpec.parse(data.get("speed", {"uniform": [0.95, 1.55]}), f"{path}.speed")
    return TTSConfig(list(voices), speed)


def _parse_voice_augment(value: Any, path: str) -> VoiceAugmentConfig:
    data = _mapping(value, path)
    names = {"pitch_semitones", "tempo", "eq_tilt_db"}
    _reject_unknown(data, names, path)
    default = VoiceAugmentConfig()
    return VoiceAugmentConfig(**{
        name: DistSpec.parse(data.get(name, getattr(default, name).as_dict()), f"{path}.{name}")
        for name in names
    })


def _quota_map(value: Any, path: str) -> dict[str, float]:
    data = _mapping(value, path)
    result = {}
    for name, amount in data.items():
        if not isinstance(name, str):
            raise ValueError(f"{path} keys must be strings")
        result[name] = _probability(amount, f"{path}.{name}")
    return result


def _parse_dataset(value: Any, path: str) -> DatasetConfig:
    data = _mapping(value, path)
    names = {"noise_only_frac", "pilot_double_hop_prob", "category_quotas"}
    _reject_unknown(data, names, path)
    default = DatasetConfig()
    return DatasetConfig(
        _probability(data.get("noise_only_frac", default.noise_only_frac),
                     f"{path}.noise_only_frac"),
        _probability(data.get("pilot_double_hop_prob", default.pilot_double_hop_prob),
                     f"{path}.pilot_double_hop_prob"),
        _quota_map(data.get("category_quotas", {}), f"{path}.category_quotas"),
    )


def _parse_qc(value: Any, path: str) -> QCConfig:
    data = _mapping(value, path)
    default = QCConfig()
    names = {item.name for item in fields(QCConfig)}
    _reject_unknown(data, names, path)
    kwargs: dict[str, Any] = {}
    for name in names:
        item = data.get(name, getattr(default, name))
        if name in {"enabled", "asr_roundtrip"}:
            if not isinstance(item, bool):
                raise ValueError(f"{path}.{name} must be a boolean")
        elif name == "max_retries":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{path}.max_retries must be a non-negative integer")
        elif name in {"max_clip_frac", "max_wer"}:
            item = _probability(item, f"{path}.{name}")
        else:
            item = _number(item, f"{path}.{name}")
        kwargs[name] = item
    if kwargs["min_duration"] > kwargs["max_duration"]:
        raise ValueError(f"{path}.min_duration exceeds max_duration")
    if kwargs["min_rms_db"] > kwargs["max_rms_db"]:
        raise ValueError(f"{path}.min_rms_db exceeds max_rms_db")
    return QCConfig(**kwargs)


def _parse_channel(value: Any, path: str) -> ChannelConfig:
    data = _mapping(value, path)
    _reject_unknown(data, {"profile", "clean_arm_prob", "chain", "shuffle_groups",
                           "noise", "reapply_bandpass"}, path)
    profile = data.get("profile", "wide")
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"{path}.profile must be a non-empty string")
    raw_chain = data.get("chain", [])
    if not isinstance(raw_chain, list):
        raise ValueError(f"{path}.chain must be a list")
    chain = []
    for index, raw_step in enumerate(raw_chain):
        step_path = f"{path}.chain[{index}]"
        step = _mapping(raw_step, step_path)
        if "primitive" not in step:
            raise ValueError(f"{step_path}.primitive is required")
        primitive = step["primitive"]
        if not isinstance(primitive, str) or not primitive:
            raise ValueError(f"{step_path}.primitive must be a non-empty string")
        prob = _probability(step.get("prob", 1.0), f"{step_path}.prob")
        params = {
            name: DistSpec.parse(spec, f"{step_path}.{name}")
            for name, spec in step.items() if name not in {"primitive", "prob"}
        }
        chain.append(ChainStep(primitive, prob, params))
    groups = data.get("shuffle_groups", [])
    if (not isinstance(groups, list)
            or not all(isinstance(group, list)
                       and all(isinstance(name, str) and name for name in group)
                       for group in groups)):
        raise ValueError(f"{path}.shuffle_groups must be a list of string lists")
    noise_path = f"{path}.noise"
    noise_data = _mapping(data.get("noise", {}), noise_path)
    _reject_unknown(noise_data, {"beds_dir"}, noise_path)
    beds_dir = noise_data.get("beds_dir")
    if beds_dir is not None and (not isinstance(beds_dir, str) or not beds_dir):
        raise ValueError(f"{noise_path}.beds_dir must be a non-empty string or null")
    reapply = data.get("reapply_bandpass", True)
    if not isinstance(reapply, bool):
        raise ValueError(f"{path}.reapply_bandpass must be a boolean")
    return ChannelConfig(
        profile=profile,
        clean_arm_prob=_probability(
            data.get("clean_arm_prob", 0.07), f"{path}.clean_arm_prob"),
        chain=chain,
        shuffle_groups=[list(group) for group in groups],
        noise=ChannelNoiseConfig(beds_dir),
        reapply_bandpass=reapply,
    )


def _parse_calibration(value: Any, path: str) -> CalibrationConfig:
    data = _mapping(value, path)
    names = {"corpus_dir", "presets", "noise_bank", "station_mix",
             "snr_jitter_db", "cross_station_prob"}
    _reject_unknown(data, names, path)
    default = CalibrationConfig()
    strings = {}
    for name in ("corpus_dir", "presets", "noise_bank"):
        item = data.get(name, getattr(default, name))
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path}.{name} must be a non-empty string")
        strings[name] = item
    station_mix = data.get("station_mix", default.station_mix)
    if station_mix is not None:
        station_mix = _quota_map(station_mix, f"{path}.station_mix")
    return CalibrationConfig(
        **strings,
        station_mix=station_mix,
        snr_jitter_db=DistSpec.parse(
            data.get("snr_jitter_db", default.snr_jitter_db.as_dict()),
            f"{path}.snr_jitter_db"),
        cross_station_prob=_probability(
            data.get("cross_station_prob", default.cross_station_prob),
            f"{path}.cross_station_prob"),
    )


def _parse_residual(value: Any, path: str) -> ResidualConfig:
    data = _mapping(value, path)
    names = {"enabled", "checkpoint", "apply_prob", "alpha", "residual_scale_max"}
    _reject_unknown(data, names, path)
    default = ResidualConfig()
    enabled = data.get("enabled", default.enabled)
    checkpoint = data.get("checkpoint", default.checkpoint)
    if not isinstance(enabled, bool):
        raise ValueError(f"{path}.enabled must be a boolean")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{path}.checkpoint must be a non-empty string")
    scale = _number(data.get("residual_scale_max", default.residual_scale_max),
                    f"{path}.residual_scale_max")
    if scale < 0:
        raise ValueError(f"{path}.residual_scale_max must be non-negative")
    return ResidualConfig(
        enabled=enabled,
        checkpoint=checkpoint,
        apply_prob=_probability(data.get("apply_prob", default.apply_prob),
                                f"{path}.apply_prob"),
        alpha=DistSpec.parse(data.get("alpha", default.alpha.as_dict()),
                             f"{path}.alpha"),
        residual_scale_max=scale,
    )


def _effect(value: Any, path: str, cls: type, names: set[str]) -> Any:
    data = _mapping(value, path)
    _reject_unknown(data, names, path)
    default = cls()
    kwargs = {}
    for name in names:
        item = data.get(name, getattr(default, name))
        if name == "quality":
            item = DistSpec.parse(item.as_dict() if isinstance(item, DistSpec) else item,
                                  f"{path}.{name}")
        elif name in {"prob", "gated_floor_prob"}:
            item = _probability(item, f"{path}.{name}")
        elif name == "kind" and (not isinstance(item, str) or not item):
            raise ValueError(f"{path}.kind must be a non-empty string")
        kwargs[name] = item
    return cls(**kwargs)


def _parse_post_effects(value: Any, path: str) -> PostEffectsConfig:
    data = _mapping(value, path)
    _reject_unknown(data, {"squelch", "dropouts", "codec"}, path)
    return PostEffectsConfig(
        _effect(data.get("squelch", {}), f"{path}.squelch", SquelchEffectConfig,
                {"prob", "gated_floor_prob"}),
        _effect(data.get("dropouts", {}), f"{path}.dropouts", DropoutsEffectConfig,
                {"prob"}),
        _effect(data.get("codec", {}), f"{path}.codec", CodecEffectConfig,
                {"prob", "kind", "quality"}),
    )


def _parse_expansion(value: Any, path: str) -> ExpansionConfig:
    data = _mapping(value, path)
    names = {"real_manifest", "target_total", "category_quotas", "holdout_frac",
             "external_texts"}
    _reject_unknown(data, names, path)
    default = ExpansionConfig()
    manifest = data.get("real_manifest", default.real_manifest)
    total = data.get("target_total", default.target_total)
    if not isinstance(manifest, str) or not manifest:
        raise ValueError(f"{path}.real_manifest must be a non-empty string")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError(f"{path}.target_total must be a positive integer")
    external = data.get("external_texts", default.external_texts)
    if external is not None and (not isinstance(external, str) or not external):
        raise ValueError(f"{path}.external_texts must be a non-empty string or null")
    return ExpansionConfig(
        manifest, total,
        _quota_map(data.get("category_quotas", {}), f"{path}.category_quotas"),
        _probability(data.get("holdout_frac", default.holdout_frac),
                     f"{path}.holdout_frac"),
        external,
    )


def _parse_calibrated(value: Any, path: str) -> CalibratedConfig:
    data = _mapping(value, path)
    names = {"calibration", "residual", "post_effects", "expansion"}
    _reject_unknown(data, names, path)
    return CalibratedConfig(
        _parse_calibration(data.get("calibration", {}), f"{path}.calibration"),
        _parse_residual(data.get("residual", {}), f"{path}.residual"),
        _parse_post_effects(data.get("post_effects", {}), f"{path}.post_effects"),
        _parse_expansion(data.get("expansion", {}), f"{path}.expansion"),
    )


def _apply_overrides(data: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ValueError("override paths must be non-empty strings")
        parts = dotted_path.split(".")
        target = data
        for index, part in enumerate(parts[:-1]):
            current_path = ".".join(parts[:index + 1])
            if part not in target:
                target[part] = {}
            if not isinstance(target[part], dict):
                raise ValueError(f"override path {current_path} is not a mapping")
            target = target[part]
        target[parts[-1]] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> GeneratorConfig:
    """Load YAML, apply dot-path overrides, merge defaults, and validate."""
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("config must be a mapping")
    data = dict(loaded)
    if overrides:
        _apply_overrides(data, overrides)

    names = {"mode", "seed", "output", "tts", "voice_augment", "dataset", "qc",
             "channel", "calibrated", "backends"}
    _reject_unknown(data, names, "")
    mode = data.get("mode", "procedural")
    if mode not in {"procedural", "calibrated", "mix"}:
        raise ValueError("mode must be procedural, calibrated, or mix")
    seed = data.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    raw_channel = data.get("channel")
    raw_calibrated = data.get("calibrated")
    channel = (_parse_channel({} if raw_channel is None else raw_channel, "channel")
               if raw_channel is not None or mode == "procedural" else None)
    calibrated = (_parse_calibrated(
        {} if raw_calibrated is None else raw_calibrated, "calibrated")
        if raw_calibrated is not None or mode == "calibrated" else None)

    raw_backends = data.get("backends", [])
    if not isinstance(raw_backends, list):
        raise ValueError("backends must be a list")
    backends = []
    for index, raw in enumerate(raw_backends):
        item_path = f"backends[{index}]"
        item = _mapping(raw, item_path)
        _reject_unknown(item, {"backend", "weight"}, item_path)
        if "backend" not in item or "weight" not in item:
            raise ValueError(f"{item_path} requires backend and weight")
        backend = item["backend"]
        weight = _number(item["weight"], f"{item_path}.weight")
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"{item_path}.backend must be a non-empty string")
        if weight <= 0:
            raise ValueError(f"{item_path}.weight must be positive")
        backends.append(BackendConfig(backend, weight))
    if mode == "mix" and not backends:
        raise ValueError("backends must be non-empty for mix mode")

    return GeneratorConfig(
        mode=mode,
        seed=seed,
        output=_parse_output(data.get("output", {}), "output"),
        tts=_parse_tts(data.get("tts", {}), "tts"),
        voice_augment=_parse_voice_augment(data.get("voice_augment", {}), "voice_augment"),
        dataset=_parse_dataset(data.get("dataset", {}), "dataset"),
        qc=_parse_qc(data.get("qc", {}), "qc"),
        channel=channel,
        calibrated=calibrated,
        backends=backends,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, DistSpec):
        return value.as_dict()
    if isinstance(value, ChainStep):
        return {
            "primitive": value.primitive,
            "prob": value.prob,
            **{name: _plain(spec) for name, spec in value.params.items()},
        }
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _resolved_bytes(config: GeneratorConfig) -> bytes:
    text = yaml.safe_dump(_plain(config), sort_keys=True, allow_unicode=True)
    return text.encode("utf-8")


def config_hash(config: GeneratorConfig) -> str:
    """Return the SHA-256 of the canonical resolved YAML."""
    return hashlib.sha256(_resolved_bytes(config)).hexdigest()


def dump_resolved(config: GeneratorConfig, out_dir: str | Path) -> tuple[Path, str]:
    """Write canonical fully resolved YAML and return its path and SHA-256."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.resolved.yaml"
    payload = _resolved_bytes(config)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()
