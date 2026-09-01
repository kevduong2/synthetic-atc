"""Optional MLBucket experiment tracking — one thin, best-effort seam.

Trainers and orchestrators log through `start_run`, which returns a real
MLBucket run when the SDK is importable, its server is reachable, and tracking
is not disabled, and a do-nothing stand-in otherwise.  A training run must
never fail, block, or change behavior because the tracker is missing or the
server is down: the SDK is already best-effort over the network, and this
module makes the import and every call site best-effort too.

The SDK is not a project dependency: on a machine with a sibling MLBucket
checkout, opt in with `uv pip install -e ../MLBucket/sdk`.  The server URL
follows the SDK's own resolution (`MLBUCKET_SERVER_URL`, then
`~/.mlbucket/config.json`, then http://localhost:8484).  It is probed once with
a short connect timeout; an unreachable server turns tracking off for the rest
of the process with a single warning, instead of the SDK's local-only runs and
background sync threads.

Disable with ATCGAN_TRACKING=off (also 0/false/no) — useful for tests and
smoke runs that should not create tracker runs.
"""

from __future__ import annotations

import os
import socket
import warnings
from typing import Any
from urllib.parse import urlsplit

_DISABLE = {"0", "false", "off", "no"}
_PROBE_TIMEOUT_SECS = 1.0

#: Set once tracking proves unusable (no SDK, dead server, failed init); every
#: later `start_run` in this process then returns a `NoopRun` silently.
_off_reason: str | None = None


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


def _turn_off(reason: str) -> NoopRun:
    global _off_reason
    _off_reason = reason
    warnings.warn(f"{reason}; experiment tracking is off", stacklevel=3)
    return NoopRun()


def _server_url_if_reachable() -> tuple[str, bool]:
    """The SDK-resolved server URL and whether a TCP connect succeeds quickly."""
    from mlbucket import store

    url = store.resolve_server_url(None, store.read_config(store.local_root(None)))
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname or "localhost", port),
                                 timeout=_PROBE_TIMEOUT_SECS).close()
    except OSError:
        return url, False
    return url, True


def start_run(project: str = "atcgan-fastcut", name: str | None = None,
              config: dict[str, Any] | None = None, tags: tuple[str, ...] = (),
              notes: str = ""):
    """An MLBucket run, or a `NoopRun` when tracking is unavailable/disabled."""
    if not tracking_enabled() or _off_reason is not None:
        return NoopRun()
    try:
        import mlbucket
    except ImportError:
        return _turn_off("mlbucket not installed")
    try:
        url, reachable = _server_url_if_reachable()
        if not reachable:
            return _turn_off(f"MLBucket server unreachable at {url}")
        return mlbucket.init(project=project, name=name, tags=list(tags),
                             notes=notes, config=config or {})
    except Exception as error:  # the tracker must never kill a run
        return _turn_off(f"mlbucket setup failed ({error})")


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
