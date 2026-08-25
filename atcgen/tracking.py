"""Optional MLBucket experiment tracking — one thin, best-effort seam.

Trainers and orchestrators log through `start_run`, which returns a real
MLBucket run (server: http://localhost:8484) when the SDK is importable and
tracking is not disabled, and a do-nothing stand-in otherwise.  A training run
must never fail, block, or change behavior because the tracker is missing or
the server is down: the SDK is already best-effort over the network, and this
module makes the import and every call site best-effort too.

Disable with ATCGAN_TRACKING=off (also 0/false/no) — useful for tests and
smoke runs that should not create tracker runs.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

_DISABLE = {"0", "false", "off", "no"}


def tracking_enabled() -> bool:
    return os.environ.get("ATCGAN_TRACKING", "").lower() not in _DISABLE


class NoopRun:
    """API-compatible stand-in for `mlbucket.Run` when tracking is off."""

    def log(self, data: dict[str, Any], step: int | None = None) -> int:
        return 0

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def __enter__(self) -> "NoopRun":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.finish()
        return False


def start_run(project: str = "atcgan-fastcut", name: str | None = None,
              config: dict[str, Any] | None = None, tags: tuple[str, ...] = (),
              notes: str = ""):
    """An MLBucket run, or a `NoopRun` when tracking is unavailable/disabled."""
    if not tracking_enabled():
        return NoopRun()
    try:
        import mlbucket
    except ImportError:
        warnings.warn("mlbucket not installed; experiment tracking is off",
                      stacklevel=2)
        return NoopRun()
    try:
        return mlbucket.init(project=project, name=name, tags=list(tags),
                             notes=notes, config=config or {})
    except Exception as error:  # the tracker must never kill a run
        warnings.warn(f"mlbucket.init failed ({error}); tracking is off",
                      stacklevel=2)
        return NoopRun()


def log_audio(run, key: str, path_or_wave, sample_rate: int | None = None,
              caption: str | None = None, step: int | None = None) -> None:
    """Log one audio clip to `run`, silently skipping when unavailable."""
    if isinstance(run, NoopRun):
        return
    try:
        import mlbucket
        run.log({key: mlbucket.Audio(path_or_wave, sample_rate=sample_rate,
                                     caption=caption)}, step=step)
    except Exception:
        return
