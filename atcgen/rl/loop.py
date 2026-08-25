"""The outer loop: ask the optimizer, evaluate, tell, checkpoint, repeat.

Everything here is written for a loop whose single iteration costs the better
part of an hour, which pushes three properties to the front:

*   **Crash-idempotence.**  Every trial is appended to ``trials.jsonl`` as it
    finishes and the optimizer is checkpointed after every batch, so a restart
    resumes at the next un-evaluated trial rather than paying for the whole run
    again.  Restarting is the expected recovery path, not an edge case.
*   **A failed candidate must not poison the search.**  Generation can fail on
    a pathological config (a squashed band, a TTS crash).  That is recorded as
    an ``error`` row and its vector is *dropped* from the batch before
    ``tell`` — feeding a sentinel reward would teach the policy that the region
    is bad when all we know is that the harness fell over.
*   **The hand-tuned config is the thing to beat.**  With ``seed_default_first``
    trial 0 is the base profile itself, so every later reward reads against a
    number from the same harness on the same day.

The anchor trial is deliberately *not* fed to ``tell``: it is not a sample from
the optimizer's proposal distribution, and REINFORCE's score function is only
valid for its own samples.  It is a reference point in the log, nothing more.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .types import RewardFn, RewardResult, Trial

TRIALS_FILE = "trials.jsonl"
STATE_FILE = "optimizer_state.json"
BEST_FILE = "best.json"
BEST_CONFIG_FILE = "best_config.yaml"


def _row_to_trial(row: Mapping[str, Any]) -> Trial:
    return Trial(
        index=int(row["index"]),
        vector=[float(value) for value in row["vector"]],
        overrides=dict(row.get("overrides", {})),
        result=RewardResult(
            reward=float(row["reward"]),
            wer_after=float(row["wer_after"]),
            wer_baseline=float(row["wer_baseline"]),
            hallucination_rate=row.get("hallucination_rate"),
            proxy=bool(row.get("proxy", False)),
            metrics=dict(row.get("metrics", {})),
        ),
        wall_time_sec=float(row.get("wall_time_sec", 0.0)),
    )


def _trial_to_row(trial: Trial) -> dict:
    result = trial.result
    return {
        "index": trial.index,
        "vector": list(trial.vector),
        "overrides": trial.overrides,
        "reward": result.reward,
        "wer_after": result.wer_after,
        "wer_baseline": result.wer_baseline,
        "hallucination_rate": result.hallucination_rate,
        "proxy": result.proxy,
        "metrics": result.metrics,
        "wall_time_sec": trial.wall_time_sec,
    }


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line and flush — the log has to survive a hard kill."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()


def run_loop(
    space,
    optimizer,
    reward_fn: RewardFn,
    base_config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    iterations: int,
    pop_size: int,
    seed_default_first: bool = True,
    resume: bool = True,
    on_trial: Callable[[Trial], None] | None = None,
) -> list[Trial]:
    """Run ``iterations`` batches of ``pop_size`` candidates and return the trials.

    ``space`` is a :class:`~atcgen.rl.space.SearchSpace`, ``optimizer`` anything
    satisfying :class:`~atcgen.rl.policy.Optimizer`.  Artifacts under
    ``out_dir``: ``trials.jsonl`` (one row per evaluation, successes and
    errors), ``optimizer_state.json`` (checkpoint + loop bookkeeping),
    ``best.json`` and ``best_config.yaml`` (the best true, non-proxy reward so
    far).  Each candidate gets ``out_dir/trials/NNN`` as its ``trial_dir``.

    With ``resume=True`` an existing ``out_dir`` is continued: prior trials are
    loaded, the optimizer state is restored, and only the remaining iterations
    run.  With ``resume=False`` the log is truncated and numbering restarts.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trials_path = out_path / TRIALS_FILE
    state_path = out_path / STATE_FILE

    trials: list[Trial] = []
    next_index = 0
    iterations_done = 0
    seeded_default = False
    best: dict[str, Any] | None = None

    if resume:
        rows = _read_rows(trials_path)
        trials = [_row_to_trial(row) for row in rows if "error" not in row]
        next_index = max((int(row["index"]) for row in rows), default=-1) + 1
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            loop_state = state.get("loop", {})
            iterations_done = int(loop_state.get("iterations_done", 0))
            seeded_default = bool(loop_state.get("seeded_default", False))
            best = loop_state.get("best")
            if state.get("optimizer"):
                optimizer.load_state_dict(state["optimizer"])
        elif rows:
            # Log without a checkpoint: keep the numbering, restart the search.
            seeded_default = seed_default_first
    else:
        trials_path.unlink(missing_ok=True)

    def checkpoint() -> None:
        payload = {
            "loop": {
                "iterations_done": iterations_done,
                "next_index": next_index,
                "seeded_default": seeded_default,
                "best": best,
            },
            "optimizer": optimizer.state_dict(),
        }
        state_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    def evaluate(vector: np.ndarray) -> Trial | None:
        """Run one candidate.  Returns ``None`` (and logs an error row) on failure."""
        nonlocal next_index, best
        index = next_index
        next_index += 1
        config = space.to_config(base_config, vector)
        overrides = space.describe(vector)
        trial_dir = out_path / "trials" / f"{index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            result = reward_fn(config, str(trial_dir))
        except Exception as error:  # noqa: BLE001 - one bad config must not end the run
            _append_row(trials_path, {
                "index": index,
                "vector": [float(value) for value in vector],
                "overrides": overrides,
                "error": f"{type(error).__name__}: {error}",
                "wall_time_sec": time.perf_counter() - started,
            })
            return None

        trial = Trial(
            index=index,
            vector=[float(value) for value in np.asarray(vector, dtype=float)],
            overrides=overrides,
            result=result,
            wall_time_sec=time.perf_counter() - started,
        )
        _append_row(trials_path, _trial_to_row(trial))
        trials.append(trial)

        # Only true fine-tune rewards define "best"; a cheap proxy score is not
        # comparable to one and must never overwrite a config we would ship.
        if not result.proxy and (best is None or result.reward > best["reward"]):
            best = {"index": index, "reward": result.reward, "overrides": overrides}
            (out_path / BEST_FILE).write_text(
                json.dumps(best, default=str, indent=2), encoding="utf-8")
            (out_path / BEST_CONFIG_FILE).write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8")
        if on_trial is not None:
            on_trial(trial)
        return trial

    if seed_default_first and not seeded_default and next_index == 0:
        evaluate(space.default_vector(base_config))
        seeded_default = True
        checkpoint()

    for _ in range(int(iterations) - iterations_done):
        vectors: list[np.ndarray] = list(optimizer.ask(int(pop_size)))
        kept_vectors: list[np.ndarray] = []
        kept_rewards: list[float] = []
        for vector in vectors:
            trial = evaluate(vector)
            if trial is not None:
                kept_vectors.append(np.asarray(vector, dtype=float))
                kept_rewards.append(trial.result.reward)
        if kept_vectors:
            optimizer.tell(kept_vectors, kept_rewards)
        iterations_done += 1
        checkpoint()

    return trials


def top_deviations(
    overrides: Mapping[str, float],
    default_overrides: Mapping[str, float],
    space,
    count: int = 3,
    min_move: float = 1e-6,
) -> list[tuple[str, float, float]]:
    """The ``count`` knobs furthest from the hand-tuned config, for log lines.

    Distance is measured in unit-cube terms so knobs with wildly different
    physical units (hertz vs probability) stay comparable.  Knobs that moved by
    less than ``min_move`` are dropped, so the anchor trial reports no
    deviations at all instead of three arbitrary zeroes.  Returns
    ``(name, value, default_value)`` triples.
    """
    ranges = {knob.name: knob for knob in space.knobs}
    scored: list[tuple[float, str, float, float]] = []
    for name, value in overrides.items():
        knob = ranges.get(name)
        base = default_overrides.get(name)
        if knob is None or base is None:
            continue
        move = abs(knob.unit(value) - knob.unit(base))
        if move >= min_move:
            scored.append((move, name, value, base))
    scored.sort(reverse=True)
    return [(name, value, base) for _, name, value, base in scored[:count]]


def format_trial(
    trial: Trial,
    space,
    default_overrides: Mapping[str, float] | None = None,
) -> str:
    """One compact console line per trial."""
    result = trial.result
    tag = " (proxy)" if result.proxy else ""
    line = (f"trial {trial.index:>3}  reward {result.reward:+.4f}{tag}  "
            f"wer {result.wer_after:.4f} vs {result.wer_baseline:.4f} baseline  "
            f"{trial.wall_time_sec / 60:.1f} min")
    if default_overrides:
        moved = top_deviations(trial.overrides, default_overrides, space)
        if moved:
            line += "  | " + ", ".join(
                f"{name}={value:.3g} (was {base:.3g})" for name, value, base in moved)
    return line


def load_trials(out_dir: str | Path) -> list[Trial]:
    """Read back the successful trials of a finished or in-flight run."""
    rows = _read_rows(Path(out_dir) / TRIALS_FILE)
    return [_row_to_trial(row) for row in rows if "error" not in row]


__all__ = ["format_trial", "load_trials", "run_loop", "top_deviations"]
