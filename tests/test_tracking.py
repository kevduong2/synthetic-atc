"""The MLBucket seam degrades to a no-op — once, with one warning — never a crash."""

import socket
import sys
import warnings

import pytest

from atcgen import tracking
from atcgen.tracking import NoopRun, log_audio, start_run


@pytest.fixture
def fresh_tracking(monkeypatch):
    monkeypatch.delenv("ATCGAN_TRACKING", raising=False)
    monkeypatch.setattr(tracking, "_off_reason", None)


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _assert_silent_noop():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(start_run(name="again"), NoopRun)


def test_missing_sdk_falls_back_to_noop_once(fresh_tracking, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlbucket", None)  # `import mlbucket` raises
    with pytest.warns(UserWarning, match="mlbucket not installed"):
        run = start_run(name="t", config={"a": 1}, tags=("x",))
    assert isinstance(run, NoopRun)
    # Everything the call sites do on a run keeps working.
    assert run.log({"loss": 1.0}, step=0) == 0
    log_audio(run, "val/audio", [0.0, 0.0], sample_rate=16000, step=0)
    run.finish()
    _assert_silent_noop()


def test_unreachable_server_falls_back_to_noop_once(fresh_tracking, monkeypatch):
    pytest.importorskip("mlbucket")
    monkeypatch.setenv("MLBUCKET_SERVER_URL", f"http://127.0.0.1:{_closed_port()}")
    with pytest.warns(UserWarning, match="server unreachable"):
        run = start_run(name="t")
    assert isinstance(run, NoopRun)
    _assert_silent_noop()


def test_env_switch_is_silent(fresh_tracking, monkeypatch):
    monkeypatch.setenv("ATCGAN_TRACKING", "off")
    _assert_silent_noop()
