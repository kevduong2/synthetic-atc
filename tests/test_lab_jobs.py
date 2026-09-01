"""scripts/lab/jobs.py and scripts/lab/relocate.py: detached jobs, watch events, lock, path rewrite."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.lab import jobs, relocate

PY = sys.executable


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("ATCGAN_LAB_DIR", str(tmp_path / "lab"))
    return tmp_path / "lab"


def _run(argv, capsys) -> tuple[int, dict]:
    try:
        rc = jobs.main(argv)
    except SystemExit as exc:      # _die() paths
        rc = int(exc.code)
    out = capsys.readouterr().out.strip()
    return rc, (json.loads(out) if out else {})


def _wait_state(job_dir: Path, states, timeout=20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jobs.read_status(job_dir) or {}
        if st.get("state") in states:
            return st
        time.sleep(0.05)
    raise AssertionError(f"job never reached {states}: {jobs.read_status(job_dir)}")


def test_launch_watch_finished_and_status(lab, capsys):
    script = "import time\nfor i in range(3):\n    print(f'step {i+1}/3', flush=True); time.sleep(0.1)\n"
    rc, out = _run(["launch", "--id", "ok", "--", PY, "-c", script], capsys)
    assert rc == 0 and out["state"] == "running" and out["child_pid"]
    rc, ev = _run(["watch", "ok", "--interval", "0.1", "--max-wait", "15",
                   "--stall-min", "0"], capsys)
    assert rc == 0 and ev["event"] == "finished" and ev["exit_code"] == 0
    assert ev["last_progress"] == "step 3/3"
    assert ev["tail"][-1] == "step 3/3"
    assert len((lab / "jobs" / "ok" / "status.md").read_text(encoding="utf-8").splitlines()) == 1
    rc, st = _run(["status", "ok", "--tail", "2"], capsys)
    assert st["state"] == "finished" and st["tail"] == ["step 2/3", "step 3/3"]
    assert (lab / "jobs" / "ok" / "cmd.txt").read_text(encoding="utf-8").strip()


def test_failed_exit_code_is_recorded(lab, capsys):
    _run(["launch", "--id", "bad", "--", PY, "-c", "import sys; print('x'); sys.exit(7)"], capsys)
    rc, ev = _run(["watch", "bad", "--interval", "0.1", "--max-wait", "15"], capsys)
    assert rc == 1 and ev["event"] == "failed" and ev["exit_code"] == 7


def test_error_pattern_ends_watch_while_running(lab, capsys):
    script = ("import time\nprint('warming', flush=True)\n"
              "print('RuntimeError: CUDA out of memory. Tried to allocate', flush=True)\n"
              "time.sleep(5)\n")
    _run(["launch", "--id", "oom", "--gpu", "--", PY, "-c", script], capsys)
    rc, ev = _run(["watch", "oom", "--interval", "0.1", "--max-wait", "15"], capsys)
    assert rc == 1 and ev["event"] == "error_pattern"
    assert "CUDA out of memory" in ev["matched"][-1]
    rc, killed = _run(["kill", "oom"], capsys)
    assert killed["state"] == "killed"
    st = _wait_state(lab / "jobs" / "oom", {"killed"})
    assert st["state"] == "killed"
    rc, lk = _run(["lock", "status"], capsys)
    assert lk["held"] is False


def test_stall_and_timeout_events(lab, capsys):
    _run(["launch", "--id", "quiet", "--", PY, "-c", "import time; print('hi', flush=True); time.sleep(6)"], capsys)
    rc, ev = _run(["watch", "quiet", "--interval", "0.1", "--max-wait", "15",
                   "--stall-min", "0.005"], capsys)          # 0.3 s of silence
    assert rc == 3 and ev["event"] == "stalled" and ev["state"] == "running"
    rc, ev = _run(["watch", "quiet", "--interval", "0.1", "--max-wait", "0.3",
                   "--stall-min", "0"], capsys)
    assert rc == 4 and ev["event"] == "timeout"
    _run(["kill", "quiet"], capsys)


def test_gpu_lock_serializes_and_releases(lab, capsys):
    rc, a = _run(["launch", "--id", "g1", "--gpu", "--", PY, "-c", "import time; time.sleep(4)"], capsys)
    assert rc == 0
    rc, lk = _run(["lock", "status"], capsys)
    assert lk["held"] and lk["holder"]["id"] == "g1" and lk["holder"]["pid"] == a["child_pid"]
    rc, _ = _run(["launch", "--id", "g2", "--gpu", "--", PY, "-c", "print(1)"], capsys)
    assert rc == 4                                           # refused: one GPU stream
    rc, _ = _run(["launch", "--id", "cpu", "--", PY, "-c", "print(1)"], capsys)
    assert rc == 0                                           # non-GPU jobs are fine
    _run(["kill", "g1"], capsys)
    _wait_state(lab / "jobs" / "g1", {"killed"})
    rc, lk = _run(["lock", "status"], capsys)
    assert lk["held"] is False
    rc, _ = _run(["launch", "--id", "g2", "--gpu", "--", PY, "-c", "print(1)"], capsys)
    assert rc == 0
    _wait_state(lab / "jobs" / "g2", {"finished"})
    rc, lk = _run(["lock", "status"], capsys)
    assert lk["held"] is False and lk["holder"] is None


def test_launch_refuses_running_duplicate_and_bad_id(lab, capsys):
    _run(["launch", "--id", "dup", "--", PY, "-c", "import time; time.sleep(4)"], capsys)
    rc, _ = _run(["launch", "--id", "dup", "--", PY, "-c", "print(1)"], capsys)
    assert rc == 3
    _run(["kill", "dup"], capsys)
    with pytest.raises(SystemExit):
        jobs.job_dir("bad id/with slash")


def test_lost_wrapper_is_detected(lab, capsys):
    _run(["launch", "--id", "lost", "--", PY, "-c", "import time; time.sleep(4)"], capsys)
    job = lab / "jobs" / "lost"
    st = jobs.read_status(job)
    jobs.kill_tree(st["wrapper_pid"])            # wrapper dies without writing an exit code
    deadline = time.time() + 10
    while time.time() < deadline and jobs.pid_alive(st["child_pid"]):
        time.sleep(0.05)
    assert jobs.live_status(job)["state"] == "lost"


def test_status_list_and_cli_entrypoint(lab, capsys):
    _run(["launch", "--id", "one", "--", PY, "-c", "print('done')"], capsys)
    _wait_state(lab / "jobs" / "one", {"finished"})
    _, listing = _run(["status"], capsys)
    assert [j["id"] for j in listing["jobs"]] == ["one"]
    env = {**os.environ, "ATCGAN_LAB_DIR": str(lab)}
    out = subprocess.run([PY, "scripts/lab/jobs.py", "tail", "one", "-n", "1"],
                         capture_output=True, text=True, env=env, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "done"


# --- relocate ------------------------------------------------------------------

def test_relocate_rewrites_all_spellings_and_checks(tmp_path, capsys):
    old = "/Users/kevin/repos/ai/atc-gan"
    new = tmp_path / "atc-gan"
    wav = new / "clips" / "a.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")
    csv_f = tmp_path / "m.csv"
    csv_f.write_text(f"audio,text\n{old}/clips/a.wav,hello\n{old}/clips/missing.wav,x\n",
                     encoding="utf-8")
    jsonl_f = tmp_path / "m.jsonl"
    jsonl_f.write_text(json.dumps({"path": old.replace("/", "\\") + "\\clips\\a.wav"}) + "\n",
                       encoding="utf-8")
    json_f = tmp_path / "manifest.json"
    json_f.write_text(json.dumps({"clips_dir": f"{old}/clips", "nested": [{"audio": f"{old}/clips/a.wav"}]}),
                      encoding="utf-8")
    rc = relocate.main(["--from", old, "--to", str(new), str(tmp_path)])
    assert rc == 0
    assert old in csv_f.read_text(encoding="utf-8")                # dry run leaves files alone
    rc = relocate.main(["--from", old, "--to", str(new), "--apply", "--check", str(tmp_path)])
    assert rc == 1                                                  # one missing target reported
    new_fwd = str(new).replace("\\", "/")
    assert csv_f.read_text(encoding="utf-8").startswith(f"audio,text\n{new_fwd}/clips/a.wav,hello")
    assert json.loads(jsonl_f.read_text(encoding="utf-8"))["path"] == f"{new_fwd}/clips/a.wav"
    assert json.loads(json_f.read_text(encoding="utf-8"))["clips_dir"] == f"{new_fwd}/clips"
    text = capsys.readouterr().out
    assert "1 referenced files missing" in text


def test_relocate_variants():
    assert relocate.variants("C:/x/y/") == ["C:\\\\x\\\\y", "C:\\x\\y", "C:/x/y"]
    text, n = relocate.rewrite_text('{"path": "C:\\\\x\\\\y\\\\clips\\\\a.wav"}', "C:/x/y", "/new")
    assert (text, n) == ('{"path": "/new/clips/a.wav"}', 1)
    assert relocate.rewrite_text("no prefix here", "/a/b", "/c") == ("no prefix here", 0)
