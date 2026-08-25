"""L3: a Thompson-sampling bandit over data recipes, gated by a hardness window.

research-findings §4.7 puts a *constrained contextual bandit* at L3 — the layer
that decides which synthetic recipe to generate next — and §4.7's "making it
computable" paragraph is the whole design constraint: per-action retraining is
impossible, so the loop runs on a cheap proxy and is recalibrated by scheduled
counterfactual runs.  Three properties follow, and they are the reason this
module looks the way it does.

**The proxy is the hardness window, not student loss.**  A pull generates a
batch from one recipe, transcribes it with a frozen *teacher* (whisper-base.en)
and with the current *student*, and counts the samples satisfying
``WER_teacher < τ1`` (the label is trustworthy) and ``τ2 < WER_student < τ3``
(challenging, not hopeless).  That count is the Bernoulli reward the Beta
posteriors are updated with.  D5 is structural here, not a convention: student
hardness selects *which samples to keep* and never reaches a generator
objective, and any sample the teacher cannot transcribe is dropped outright
rather than being kept as a "hard" example — rewarding student failure without
that bound is how a generator learns to garble "seven" into something
mishearable, which is label corruption wearing a hard-case-mining costume.

**Sample routing is three-way, not two.**  In-window samples land in
``selected/``; teacher-trustworthy but out-of-window samples land in
``spillover/`` (they are perfectly good training data, just not *targeted*
data, and the counterfactual needs comparison material); teacher-untrustworthy
samples are dropped and only counted.

The τ1 half of that decision is a single-teacher stand-in for what
``atcgen/gate/`` is being built to do properly — multi-teacher consensus plus
entity fidelity, tiering samples gold/silver/adversarial/rejected (§4.4, D8).
When the gate lands, this module should read its verdict instead of running
its own teacher pass: same bound, better evidence, and one less Whisper
forward per pull.  Until then τ1 is calibrated directly against the teacher's
own floor — see `HardnessWindow`.

**The proxy is not trusted.**  Every ``counterfactual_every`` pulls, the same
frozen init is fine-tuned twice for the same number of steps — once on a sample
of the selected buffer, once on a freshly generated *uniform* mixture over all
recipes — and both are scored on a real reward-validation slice.  That ΔWER is
the number P4c's exit bar ("bandit beats uniform sampling in counterfactual
runs") is read off, and it is logged separately from the pull log because it is
the only measurement here that is not a proxy.

Everything is crash-idempotent in the same way `atcgen.rl.loop` is: pulls are
appended to ``pulls.jsonl`` as they finish, ``state.json`` is rewritten after
each one, and re-running the same command resumes at the next un-pulled index.
Buffer manifests carry their pull index so a crash between "append the samples"
and "append the pull row" is repaired by truncation on resume rather than
silently double-counting.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jiwer
import numpy as np

from training.normalize import normalize_atc

from .loop import _append_row, _read_rows  # same jsonl idempotency helpers
from .recipes import RECIPES, Recipe, write_config

PULLS_FILE = "pulls.jsonl"
STATE_FILE = "state.json"
COUNTERFACTUALS_FILE = "counterfactuals.jsonl"
SELECTED_DIR = "selected"
SPILLOVER_DIR = "spillover"
MANIFEST = "manifest.jsonl"

SELECTED = "selected"
SPILLOVER = "spillover"
DROPPED = "dropped"

#: The bandit's own generator seed base.  A pull deliberately draws *fresh*
#: text and channel values (it is accumulating a buffer, not comparing two
#: configs), so unlike `atcgen.rl.reward` the seed varies — but it is derived
#: from one recorded base and written into every pull row, so any pull can be
#: regenerated exactly.
GEN_SEED = 20260824

#: The counterfactual's real slice.  `atcgen.dataset.splits` is the registry
#: that keeps it disjoint from real-train and locked-test (D11); naming the
#: split rather than the row range is what makes that check binding.
REWARD_VAL = "reward_val"


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


def sample_wer(reference: str, hypothesis: str) -> float | None:
    """ATC-normalized WER for one (reference, hypothesis) pair.

    ``None`` when the reference normalizes to nothing — a noise-only clip has
    no error rate, so it can be neither trustworthy nor challenging and takes
    no part in the window.  The value is uncapped: a hallucinating decode
    inserts words and scores well above 1.0, which is exactly what τ3 is for.
    """
    ref = normalize_atc(reference).strip()
    if not ref:
        return None
    return float(jiwer.wer(ref, normalize_atc(hypothesis).strip()))


@dataclass(frozen=True)
class HardnessWindow:
    """§4.7's safe form of hard-case mining.

    ``tau1`` bounds the teacher: above it the label is not trustworthy and the
    sample is dropped (D5).  ``tau2``/``tau3`` bound the student: below τ2 the
    sample teaches nothing it does not already know, above τ3 it is hopeless
    (usually a hallucination loop or an unintelligible clip) and training on it
    is closer to label noise than to a hard example.  All three comparisons are
    strict, matching the paper's ``WER_teacher < τ1`` and ``τ2 < WER_student <
    τ3``.

    Calibration
    -----------
    The defaults are 0.8/0.4/1.2 rather than §4.7's illustrative 0.5/0.3/1.2,
    because on *synthetic* data the teacher is not producing the label — the
    grammar already did, structurally.  Its job is narrower: decide whether
    degradation has destroyed intelligibility (§4.3's "unlimited distortion
    manufactures audio whose transcript is no longer recoverable").  So τ1 has
    to sit above the teacher's own out-of-domain floor, or it fires on the
    teacher's ATC inexperience instead of on the audio.

    Measured on 24 clips per bucket plus an undegraded control (clean arm,
    every degradation bypassed), whisper-base.en teacher / whisper-tiny.en
    student, both zero-shot:

    =====================  ==================  ==================
    bucket                 teacher p50 / p90   student p50 / p90
    =====================  ==================  ==================
    undegraded control          0.24 / 0.41         0.37 / 0.67
    us_routine                  0.30 / 0.85         0.40 / 1.07
    dense_numerics              0.38 / 0.68         0.52 / 1.04
    high_snr_clean              0.41 / 0.84         0.56 / 0.98
    eu_routine                  0.50 / 0.97         0.61 / 1.00
    noise_heavy_channel         0.55 / 1.10         0.62 / 1.17
    eu_fast_speech              0.63 / 1.00         0.77 / 1.00
    low_snr                     0.72 / 1.32         0.85 / 1.26
    =====================  ==================  ==================

    The teacher scores 0.24 median on audio with *no* degradation at all, so
    τ1=0.5 dropped 8% of the undegraded control and 52% of everything — and it
    inverted the ranking, handing the highest in-window rate to the cleanest
    bucket because "the teacher can read it" outvoted "the student cannot".
    τ1=0.8 drops none of the control and 21% overall.

    τ2 is set the same way, against the student: 0.4 sits just above the
    student's undegraded median of 0.37, so "challenging" means "harder than
    this student finds clean audio" rather than an absolute number.  That drops
    the control's in-window rate to 0.25 (it lands in spillover, not in the
    bin) while the hard-phraseology buckets rise to 0.5-0.62.

    **Re-calibrate τ2 whenever the student is refreshed.**  It is defined
    against the student's error distribution, and a fine-tuned checkpoint's is
    much tighter; leaving τ2 at a stale value quietly narrows the window toward
    nothing.  Measure the student on a clean-arm batch and put τ2 at its median.
    """

    tau1: float = 0.8
    tau2: float = 0.4
    tau3: float = 1.2

    def __post_init__(self) -> None:
        if not 0.0 < self.tau1:
            raise ValueError("tau1 must be positive")
        if not 0.0 <= self.tau2 < self.tau3:
            raise ValueError("need 0 <= tau2 < tau3")

    def classify(self, teacher: float | None, student: float | None) -> tuple[str, str]:
        """``(route, reason)`` for one sample's teacher/student WER pair."""
        if teacher is None or student is None:
            return DROPPED, "no_reference"
        if not teacher < self.tau1:
            return DROPPED, "teacher_untrusted"
        if not student > self.tau2:
            return SPILLOVER, "too_easy"
        if not student < self.tau3:
            return SPILLOVER, "too_hard"
        return SELECTED, "in_window"

    def as_dict(self) -> dict[str, float]:
        return {"tau1": self.tau1, "tau2": self.tau2, "tau3": self.tau3}


# --------------------------------------------------------------------------
# Thompson sampling
# --------------------------------------------------------------------------


class BetaPosteriors:
    """Beta-Bernoulli posteriors over the recipe arms, sampled Thompson-style.

    The arm is chosen by drawing one θ per arm from its posterior and taking
    the argmax — no explicit exploration parameter, the posterior width does
    that job and narrows on its own as the counts grow.  A pull contributes
    ``in_window`` successes and ``n - in_window`` failures rather than a single
    Bernoulli trial, so one pull of 60 clips is worth 60 observations and the
    posteriors tighten in tens of pulls rather than thousands.

    The generator state round-trips through ``state_dict`` so a resumed run
    continues the same sampling stream instead of replaying it.
    """

    def __init__(self, arms: Sequence[str], *, seed: int = 0,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        if not arms:
            raise ValueError("BetaPosteriors needs at least one arm")
        self.arms = list(arms)
        self.prior = (float(prior_alpha), float(prior_beta))
        self.alpha = {arm: float(prior_alpha) for arm in self.arms}
        self.beta = {arm: float(prior_beta) for arm in self.arms}
        self.pulls = {arm: 0 for arm in self.arms}
        self.rng = np.random.default_rng(seed)

    def sample_arm(self) -> str:
        draws = [self.rng.beta(self.alpha[arm], self.beta[arm]) for arm in self.arms]
        return self.arms[int(np.argmax(draws))]

    def update(self, arm: str, successes: float, failures: float) -> None:
        if arm not in self.alpha:
            raise KeyError(f"unknown arm {arm!r}")
        self.alpha[arm] += float(successes)
        self.beta[arm] += float(failures)
        self.pulls[arm] += 1

    def mean(self, arm: str) -> float:
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def stdev(self, arm: str) -> float:
        a, b = self.alpha[arm], self.beta[arm]
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))

    def observations(self, arm: str) -> float:
        """Samples this arm's posterior has actually seen (priors removed)."""
        return (self.alpha[arm] - self.prior[0]) + (self.beta[arm] - self.prior[1])

    def state_dict(self) -> dict:
        return {"arms": self.arms, "prior": list(self.prior),
                "alpha": self.alpha, "beta": self.beta, "pulls": self.pulls,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.arms = list(state["arms"])
        self.prior = tuple(float(v) for v in state["prior"])  # type: ignore[assignment]
        self.alpha = {arm: float(v) for arm, v in state["alpha"].items()}
        self.beta = {arm: float(v) for arm, v in state["beta"].items()}
        self.pulls = {arm: int(v) for arm, v in state.get("pulls", {}).items()}
        if state.get("rng"):
            self.rng.bit_generator.state = state["rng"]


# --------------------------------------------------------------------------
# the expensive halves, behind protocols so tests can fake them
# --------------------------------------------------------------------------


class PullEngine(Protocol):
    """Generation + ASR for one pull.  `AsrPullEngine` is the real one."""

    def render(self, recipe: Recipe, dest: Path, n: int, seed: int) -> list[dict]:
        """Build ``n`` clips into ``dest``; manifest rows with absolute paths."""

    def wers(self, rows: Sequence[Mapping[str, Any]], which: str) -> list[float | None]:
        """Per-sample normalized WER from ``which`` in {"teacher", "student"}."""


class Counterfactual(Protocol):
    """One scheduled recalibration run."""

    def run(self, round_dir: Path, selected: Sequence[Mapping[str, Any]],
            m: int) -> dict: ...


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


class RecipeBandit:
    """Pull recipes, route samples through the window, recalibrate on schedule.

    ``out_dir`` layout::

        pulls.jsonl            one row per pull (arm, seed, window counts, WERs)
        state.json             posteriors + loop bookkeeping, rewritten per pull
        counterfactuals.jsonl  one row per recalibration round
        pulls/NNN/             config.yaml + synth/ for that pull
        selected/manifest.jsonl    in-window samples (audio refs, not copies)
        spillover/manifest.jsonl   teacher-trustworthy, out-of-window samples
        cf/NNN/                the uniform arm's clips for round NNN
    """

    def __init__(self, out_dir: str | Path, engine: PullEngine, *,
                 recipes: Mapping[str, Recipe] | None = None,
                 window: HardnessWindow | None = None,
                 n_batch: int = 60, seed: int = GEN_SEED,
                 counterfactual: Counterfactual | None = None,
                 counterfactual_every: int = 8, cf_m: int = 150,
                 prior: tuple[float, float] = (1.0, 1.0),
                 on_pull: Callable[[dict], None] | None = None,
                 on_counterfactual: Callable[[dict], None] | None = None) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.recipes = dict(recipes if recipes is not None else RECIPES)
        self.window = window or HardnessWindow()
        self.n_batch = int(n_batch)
        self.seed = int(seed)
        self.cf = counterfactual
        self.cf_every = int(counterfactual_every)
        self.cf_m = int(cf_m)
        self.on_pull = on_pull
        self.on_counterfactual = on_counterfactual

        self.posteriors = BetaPosteriors(sorted(self.recipes), seed=seed,
                                         prior_alpha=prior[0], prior_beta=prior[1])
        self.pulls_done = 0
        self.cf_rounds_done = 0
        self.last_cf_pull = -1
        self.totals = {SELECTED: 0, SPILLOVER: 0, DROPPED: 0}
        self._resume()

    # -- paths ------------------------------------------------------------

    @property
    def pulls_path(self) -> Path:
        return self.out / PULLS_FILE

    @property
    def state_path(self) -> Path:
        return self.out / STATE_FILE

    def buffer_path(self, route: str) -> Path:
        return self.out / (SELECTED_DIR if route == SELECTED else SPILLOVER_DIR) / MANIFEST

    def pull_seed(self, index: int) -> int:
        """Fresh per pull, derived from one recorded base so it reproduces."""
        return self.seed + 7919 * (index + 1)

    # -- state ------------------------------------------------------------

    def _resume(self) -> None:
        rows = _read_rows(self.pulls_path)
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            loop = state.get("loop", {})
            self.pulls_done = int(loop.get("pulls_done", 0))
            self.cf_rounds_done = int(loop.get("cf_rounds_done", 0))
            self.last_cf_pull = int(loop.get("last_cf_pull", -1))
            self.totals = {route: int(loop.get("totals", {}).get(route, 0))
                           for route in (SELECTED, SPILLOVER, DROPPED)}
            if state.get("posteriors"):
                self.posteriors.load_state_dict(state["posteriors"])
        elif rows:
            # A log without a checkpoint: the posteriors are a pure function of
            # the pull log, so replay rather than throwing the run away.
            self._replay(rows)
        if self.pulls_done:
            self._truncate_buffers(self.pulls_done)

    def _replay(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            arm = row.get("recipe")
            if arm not in self.posteriors.alpha:
                continue
            selected = int(row.get("selected", 0))
            self.posteriors.update(arm, selected, int(row.get("n", 0)) - selected)
            for route in (SELECTED, SPILLOVER, DROPPED):
                self.totals[route] += int(row.get(route, 0))
        self.pulls_done = max((int(row["pull"]) for row in rows), default=-1) + 1

    def _truncate_buffers(self, next_pull: int) -> None:
        """Drop buffer rows from pulls that never made it into ``pulls.jsonl``.

        Samples are appended before the pull row is, so a hard kill in between
        leaves orphans that would be counted twice when the pull is redone.
        """
        for route in (SELECTED, SPILLOVER):
            path = self.buffer_path(route)
            rows = _read_rows(path)
            kept = [row for row in rows if int(row.get("pull", -1)) < next_pull]
            if len(kept) != len(rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in kept),
                                encoding="utf-8")

    def checkpoint(self) -> None:
        payload = {
            "loop": {"pulls_done": self.pulls_done,
                     "cf_rounds_done": self.cf_rounds_done,
                     "last_cf_pull": self.last_cf_pull,
                     "totals": self.totals},
            "window": self.window.as_dict(),
            "n_batch": self.n_batch,
            "seed": self.seed,
            "posteriors": self.posteriors.state_dict(),
        }
        self.state_path.write_text(json.dumps(payload, default=str, indent=2),
                                   encoding="utf-8")

    # -- buffers ----------------------------------------------------------

    def buffer_rows(self, route: str) -> list[dict]:
        """Buffer manifest rows with ``audio`` re-absolutized against ``out``."""
        rows = _read_rows(self.buffer_path(route))
        for row in rows:
            row["audio"] = str(self.out / row["audio"])
        return rows

    def _append_sample(self, route: str, row: Mapping[str, Any]) -> None:
        path = self.buffer_path(route)
        path.parent.mkdir(parents=True, exist_ok=True)
        _append_row(path, row)

    # -- one pull ---------------------------------------------------------

    def pull(self) -> dict:
        """Generate, score, route, update.  Returns the pull row."""
        index = self.pulls_done
        arm = self.posteriors.sample_arm()
        recipe = self.recipes[arm]
        seed = self.pull_seed(index)
        dest = self.out / "pulls" / f"{index:03d}"

        started = time.perf_counter()
        rows = self.engine.render(recipe, dest, self.n_batch, seed)
        teacher = self.engine.wers(rows, "teacher")
        student = self.engine.wers(rows, "student")

        counts = {SELECTED: 0, SPILLOVER: 0, DROPPED: 0}
        reasons: dict[str, int] = {}
        for row, t_wer, s_wer in zip(rows, teacher, student):
            route, reason = self.window.classify(t_wer, s_wer)
            counts[route] += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            if route == DROPPED:
                continue
            # The manifest row rides along whole (minus engine bookkeeping), so
            # fields the dataset builder grows later -- entities, lineage --
            # reach the buffer without this module having to know about them.
            sample = {key: value for key, value in row.items()
                      if not key.startswith("_")}
            sample.update({
                "pull": index,
                "recipe": arm,
                "audio": _relative(row["audio"], self.out),
                "wer_teacher": t_wer,
                "wer_student": s_wer,
                "reason": reason,
            })
            self._append_sample(route, sample)

        in_window = counts[SELECTED]
        self.posteriors.update(arm, in_window, len(rows) - in_window)
        for route, count in counts.items():
            self.totals[route] += count

        pull_row = {
            "pull": index,
            "recipe": arm,
            "axis": recipe.axis,
            "seed": seed,
            "n": len(rows),
            "in_window": in_window,
            **counts,
            "reasons": reasons,
            "wer_teacher": _summary(teacher),
            "wer_student": _summary(student),
            "posterior_mean": round(self.posteriors.mean(arm), 4),
            "wall_time_sec": round(time.perf_counter() - started, 2),
        }
        self.pulls_done = index + 1
        _append_row(self.pulls_path, pull_row)
        self.checkpoint()
        if self.on_pull is not None:
            self.on_pull(pull_row)
        return pull_row

    # -- recalibration ----------------------------------------------------

    def counterfactual(self) -> dict | None:
        """One scheduled recalibration; ``None`` when nothing is configured."""
        if self.cf is None:
            return None
        index = self.cf_rounds_done
        round_dir = self.out / "cf" / f"{index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        selected = self.buffer_rows(SELECTED)

        started = time.perf_counter()
        result = self.cf.run(round_dir, selected, self.cf_m)
        row = {"round": index, "after_pull": self.pulls_done,
               "n_selected_buffer": len(selected),
               "wall_time_sec": round(time.perf_counter() - started, 2),
               **result}

        self.cf_rounds_done = index + 1
        self.last_cf_pull = self.pulls_done
        _append_row(self.out / COUNTERFACTUALS_FILE, row)
        self.checkpoint()
        if self.on_counterfactual is not None:
            self.on_counterfactual(row)
        return row

    def run(self, pulls: int) -> list[dict]:
        """Pull until ``pulls`` total, recalibrating on schedule and at the end."""
        done: list[dict] = []
        while self.pulls_done < int(pulls):
            done.append(self.pull())
            if (self.cf is not None and self.cf_every > 0
                    and self.pulls_done % self.cf_every == 0
                    and self.pulls_done != self.last_cf_pull):
                self.counterfactual()
        if (self.cf is not None and self.pulls_done > 0
                and self.last_cf_pull != self.pulls_done):
            self.counterfactual()
        return done


def _relative(path: str | Path, root: Path) -> str:
    """``path`` relative to ``root`` when it is inside it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _summary(values: Iterable[float | None]) -> dict:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return {"n": 0}
    array = np.asarray(numbers, dtype=float)
    return {"n": len(numbers),
            "mean": round(float(array.mean()), 4),
            "p50": round(float(np.percentile(array, 50)), 4),
            "p90": round(float(np.percentile(array, 90)), 4)}


# --------------------------------------------------------------------------
# the real engine and the real counterfactual
# --------------------------------------------------------------------------


def _hf_dataset(rows: Sequence[Mapping[str, Any]]):
    """Manifest/buffer rows as an HF dataset with a decoded audio column."""
    from datasets import Audio, Dataset

    from ..channel.primitives import TARGET_SR

    payload = [{"audio": str(row["audio"]), "text": row.get("text", "") or ""}
               for row in rows]
    return Dataset.from_list(payload).cast_column("audio", Audio(sampling_rate=TARGET_SR))


class AsrPullEngine:
    """Generation + teacher/student transcription on real models.

    The teacher is frozen and architecturally separate from the student (D4);
    it is never the model being trained, and nothing here can make it one.  The
    student is swapped between rounds by the caller via `set_student` — the
    window is defined against the *current* student, so a stale one would keep
    selecting samples the model has already learned.

    Log-mel features are extracted once per row set with the teacher's feature
    extractor and reused for both decodes (Whisper's 80-bin extractor is shared
    across tiny/base), so a pull pays for feature extraction once rather than
    twice; only the last row set is memoized, which is all a pull needs.
    """

    def __init__(self, base_config: Mapping[str, Any], *,
                 teacher: str = "openai/whisper-base.en",
                 student: str = "openai/whisper-tiny.en",
                 device: str | None = None,
                 noise_only_frac: float | None = 0.0,
                 asr_batch: int = 16) -> None:
        from training.evaluate import pick_device

        self.base_config = dict(base_config)
        self.teacher_id = teacher
        self.student_id = student
        self.device = pick_device(device)
        self.noise_only_frac = noise_only_frac
        self.asr_batch = int(asr_batch)
        self._models: dict[str, Any] = {}
        self._processors: dict[str, Any] = {}
        self._features_key: tuple | None = None
        self._features: list | None = None

    # -- models -----------------------------------------------------------

    def model_id(self, which: str) -> str:
        if which == "teacher":
            return self.teacher_id
        if which == "student":
            return self.student_id
        raise ValueError(f"which must be 'teacher' or 'student': {which!r}")

    def set_student(self, student: str) -> None:
        """Point the student at a refreshed checkpoint; drops the cached model."""
        self.student_id = student
        self._models.pop("student", None)
        self._processors.pop("student", None)

    def processor(self, which: str):
        from transformers import WhisperProcessor

        if which not in self._processors:
            name = self.model_id(which)
            try:
                processor = WhisperProcessor.from_pretrained(name)
            except (OSError, ValueError):
                # A bare fine-tune checkpoint dir carries weights but no
                # tokenizer/feature-extractor files; its base model's do.
                processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
            self._processors[which] = processor
        return self._processors[which]

    def model(self, which: str):
        from transformers import WhisperForConditionalGeneration

        if which not in self._models:
            model = WhisperForConditionalGeneration.from_pretrained(self.model_id(which))
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            self._models[which] = model.to(self.device).eval()
        return self._models[which]

    # -- the protocol -----------------------------------------------------

    def render(self, recipe: Recipe, dest: Path, n: int, seed: int) -> list[dict]:
        from ..dataset.build import build_dataset

        dest = Path(dest)
        raw = recipe.apply(self.base_config)
        raw["seed"] = int(seed)
        if self.noise_only_frac is not None:
            # Selection scores samples by WER and a noise-only clip has none.
            # Hallucination control is a *mixture* concern, applied when the
            # selected buffer is blended for training, not a selection one.
            raw.setdefault("dataset", {})["noise_only_frac"] = float(self.noise_only_frac)
        config = write_config(raw, dest / "config.yaml")

        synth = dest / "synth"
        manifest = build_dataset(config, synth, int(n), recipe.text_source())
        rows = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["audio"] = str((synth / record["audio"]).resolve())
            record.pop("gen", None)
            rows.append(record)
        return rows

    def wers(self, rows: Sequence[Mapping[str, Any]], which: str) -> list[float | None]:
        from .finetune_lite import transcribe

        if not rows:
            return []
        features = self.features(rows)
        hypotheses = transcribe(self.model(which), self.processor(which), features,
                                self.device, batch_size=self.asr_batch)
        return [sample_wer(row.get("text", "") or "", hypothesis)
                for row, hypothesis in zip(rows, hypotheses)]

    def features(self, rows: Sequence[Mapping[str, Any]]) -> list:
        from .finetune_lite import prepare_features

        key = tuple(str(row["audio"]) for row in rows)
        if key != self._features_key:
            self._features = prepare_features(_hf_dataset(rows),
                                              self.processor("teacher"))
            self._features_key = key
        return self._features  # type: ignore[return-value]

    def release(self) -> None:
        import torch

        self._models.clear()
        self._features = self._features_key = None
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


class AsrCounterfactual:
    """§4.7's scheduled recalibration: selected-buffer data vs uniform data.

    Both arms fine-tune *the same frozen init* for the same number of steps on
    the same number of clips and are scored on the same real reward-validation
    slice, so the only thing that differs is which synthetic clips they saw.

    The uniform arm is generated fresh — an equal share of clips from every
    recipe — rather than drawn from this run's own spillover.  Spillover is
    already shaped by the bandit's bucket preferences, so comparing against it
    would measure only the hardness filter and hide the arm-selection half of
    the policy.  What the uniform arm *does* keep is the teacher gate
    (``WER_teacher < τ1``): system invariant 5 forbids training anything on
    samples that failed verification, so an ungated baseline would be a
    strawman rather than the honest "uniform sampling" of P4c's exit bar.  The
    comparison therefore isolates exactly what Thompson's bucket choice plus
    the student-hardness window buy, holding label trust fixed.
    """

    def __init__(self, engine: AsrPullEngine, *,
                 init: str = "openai/whisper-tiny.en",
                 recipes: Mapping[str, Recipe] | None = None,
                 window: HardnessWindow | None = None,
                 ft_steps: int = 300, ft_batch: int = 8, ft_lr: float = 1e-5,
                 ft_seed: int = 0, eval_n: int = 400,
                 eval_split: str = REWARD_VAL,
                 seed: int = 0, oversample: float = 1.8,
                 oversample_cap: float = 4.0) -> None:
        self.engine = engine
        self.init = init
        self.recipes = dict(recipes if recipes is not None else RECIPES)
        self.window = window or HardnessWindow()
        self.ft_steps = int(ft_steps)
        self.ft_batch = int(ft_batch)
        self.ft_lr = float(ft_lr)
        self.ft_seed = int(ft_seed)
        self.eval_n = int(eval_n)
        self.eval_split = eval_split
        self.seed = int(seed)
        self.oversample = float(oversample)
        self.oversample_cap = float(oversample_cap)
        self._eval_features: list | None = None
        self._eval_refs: list[str] | None = None
        self._eval_categories: list | None = None
        self._eval_name = eval_split
        self._processor = None
        self._wer_init: float | None = None

    # -- the fixed real slice ---------------------------------------------

    @property
    def processor(self):
        from transformers import WhisperProcessor

        if self._processor is None:
            try:
                self._processor = WhisperProcessor.from_pretrained(self.init)
            except (OSError, ValueError):
                self._processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        return self._processor

    def _ensure_eval(self) -> None:
        """Load and featurize the real slice once; every arm reuses it."""
        if self._eval_features is not None:
            return
        from ..dataset.splits import load_split, split_spec
        from .finetune_lite import prepare_features

        dataset = load_split(self.eval_split)
        # The first `eval_n` rows of the split, not a fresh sample of it: both
        # arms and the init baseline have to be scored on identical audio.
        dataset = dataset.select(range(min(self.eval_n, len(dataset))))
        self._eval_refs = list(dataset["text"])
        self._eval_categories = (list(dataset["category"])
                                 if "category" in dataset.column_names
                                 else [None] * len(dataset))
        self._eval_features = prepare_features(dataset, self.processor)
        self._eval_name = f"{split_spec(self.eval_split).dataset_name()}[:{self.eval_n}]"

    def evaluate(self, model) -> float:
        """ATC-normalized WER of ``model`` on the reward-validation slice."""
        from training.evaluate import build_report

        from .finetune_lite import transcribe

        self._ensure_eval()
        hypotheses = transcribe(model, self.processor, self._eval_features,
                                self.engine.device, batch_size=self.engine.asr_batch)
        report = build_report(self._eval_refs, hypotheses, self._eval_categories,
                              model=self.init, dataset=self._eval_name)
        return float(report["wer"]["atc_normalized"])

    # -- the arms ---------------------------------------------------------

    def uniform_pool(self, round_dir: Path, m: int, seed: int) -> list[dict]:
        """``m`` clips spread evenly over every recipe, then teacher-gated.

        The gate rejects a substantial fraction — a zero-shot teacher on
        out-of-domain phraseology is not a generous grader — so the pool is
        over-generated by ``oversample`` and topped up once against the keep
        rate actually observed, rather than being allowed to come back short
        and silently shrink both arms.  Each recipe's clips are scored right
        after they are rendered, which keeps peak feature memory at one
        recipe's worth instead of the whole pool's.
        """
        names = sorted(self.recipes)
        gated: list[dict] = []
        generated = 0
        budget = int(self.oversample_cap * m)
        target_per = max(1, math.ceil(m * self.oversample / len(names)))

        for attempt in range(2):
            if len(gated) >= m or generated >= budget:
                break
            if attempt:  # top up against the keep rate we just measured
                keep = max(len(gated) / generated, 0.05)
                short = m - len(gated)
                target_per = max(1, math.ceil(min(short / keep, budget - generated)
                                              / len(names)))
            for offset, name in enumerate(names):
                dest = round_dir / "uniform" / f"{attempt}_{name}"
                rows = self.engine.render(self.recipes[name], dest, target_per,
                                          seed + 104729 * (offset + 1) + 13 * attempt)
                generated += len(rows)
                for row, wer in zip(rows, self.engine.wers(rows, "teacher")):
                    if wer is not None and wer < self.window.tau1:
                        gated.append(dict(row, recipe=name, wer_teacher=wer))

        rng = random.Random(seed)
        rng.shuffle(gated)
        return gated[:m]

    def _finetune_and_score(self, rows: Sequence[Mapping[str, Any]]) -> tuple[float, list[float]]:
        from transformers import WhisperForConditionalGeneration

        from .finetune_lite import finetune, prepare_features

        features = prepare_features(_hf_dataset(rows), self.processor)
        model = WhisperForConditionalGeneration.from_pretrained(self.init)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = False
        model.to(self.engine.device)
        finetune(model, features, steps=self.ft_steps, batch_size=self.ft_batch,
                 lr=self.ft_lr, seed=self.ft_seed, device=self.engine.device)
        losses = [round(value, 4) for value in list(getattr(model, "_ft_losses", []))[-10:]]
        wer = self.evaluate(model)
        del model
        self.engine.release()
        return wer, losses

    def run(self, round_dir: Path, selected: Sequence[Mapping[str, Any]],
            m: int) -> dict:
        round_dir = Path(round_dir)
        seed = self.seed + 7907 * (int(round_dir.name) + 1)
        rng = random.Random(seed)

        uniform = self.uniform_pool(round_dir, m, seed)
        picked = list(selected)
        rng.shuffle(picked)
        picked = picked[:m]

        # Equal-size arms: a data-quantity difference would confound the
        # comparison with the data-quality one it is supposed to measure.
        k = min(len(picked), len(uniform))
        if k < self.ft_batch:
            return {"status": "skipped", "reason": "not enough data for either arm",
                    "n_selected": len(picked), "n_uniform": len(uniform)}
        picked, uniform = picked[:k], uniform[:k]

        if self._wer_init is None:
            from transformers import WhisperForConditionalGeneration

            model = WhisperForConditionalGeneration.from_pretrained(self.init)
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            self._wer_init = self.evaluate(model.to(self.engine.device).eval())
            del model
            self.engine.release()

        wer_selected, losses_selected = self._finetune_and_score(picked)
        wer_uniform, losses_uniform = self._finetune_and_score(uniform)

        (round_dir / "arms.json").write_text(json.dumps(
            {"selected": [row.get("audio") for row in picked],
             "uniform": [row.get("audio") for row in uniform]}, indent=2),
            encoding="utf-8")

        return {
            "status": "ok",
            "n": k,
            "n_uniform_generated": len(uniform),
            "init": self.init,
            "ft_steps": self.ft_steps,
            "eval_n": len(self._eval_refs or []),
            "eval_split": self._eval_name,
            "wer_init": round(self._wer_init, 5),
            "wer_selected": round(wer_selected, 5),
            "wer_uniform": round(wer_uniform, 5),
            # positive => the bandit's selection beat uniform sampling (P4c)
            "delta_wer_selected_vs_uniform": round(wer_uniform - wer_selected, 5),
            "loss_tail_selected": losses_selected,
            "loss_tail_uniform": losses_uniform,
            "recipe_mix_selected": _mix(picked),
        }


def _mix(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("recipe", "?"))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def format_pull(row: Mapping[str, Any]) -> str:
    """One compact console line per pull."""
    teacher = row.get("wer_teacher", {}) or {}
    student = row.get("wer_student", {}) or {}
    n = max(int(row.get("n", 0)), 1)
    return (f"pull {int(row['pull']):>3}  {row['recipe']:<24}"
            f" window {int(row['in_window']):>3}/{int(row['n']):<3}"
            f" ({row['in_window'] / n:.2f})"
            f"  spill {int(row.get(SPILLOVER, 0)):>3}"
            f"  drop {int(row.get(DROPPED, 0)):>3}"
            f"  wer_t {teacher.get('p50', float('nan')):.2f}"
            f"  wer_s {student.get('p50', float('nan')):.2f}"
            f"  {float(row.get('wall_time_sec', 0.0)):.0f}s")


def format_posteriors(bandit: RecipeBandit) -> str:
    """The live table: one row per bucket, best posterior mean first."""
    header = (f"{'recipe':<24} {'axis':<12} {'pulls':>5} {'obs':>6} {'in-win':>7}"
              f" {'mean':>7} {'sd':>6}")
    lines = [header, "-" * len(header)]
    for arm in sorted(bandit.posteriors.arms, key=bandit.posteriors.mean, reverse=True):
        observations = bandit.posteriors.observations(arm)
        in_window = bandit.posteriors.alpha[arm] - bandit.posteriors.prior[0]
        recipe = bandit.recipes.get(arm)
        lines.append(
            f"{arm:<24} {(recipe.axis if recipe else ''):<12}"
            f" {bandit.posteriors.pulls[arm]:>5} {observations:>6.0f} {in_window:>7.0f}"
            f" {bandit.posteriors.mean(arm):>7.3f} {bandit.posteriors.stdev(arm):>6.3f}")
    totals = bandit.totals
    lines.append(f"buffers: selected {totals[SELECTED]}  spillover "
                 f"{totals[SPILLOVER]}  dropped {totals[DROPPED]}")
    return "\n".join(lines)


__all__ = ["GEN_SEED", "REWARD_VAL", "AsrCounterfactual", "AsrPullEngine",
           "BetaPosteriors", "Counterfactual", "HardnessWindow", "PullEngine",
           "RecipeBandit", "format_posteriors", "format_pull", "sample_wer"]
