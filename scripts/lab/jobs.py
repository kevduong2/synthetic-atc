#!/usr/bin/env python
"""Detached job runner for the lab: launch, status, watch, kill, and a GPU lock.

Why this exists: an agent's shell tool cannot be trusted to keep a
multi-hour process alive across turns (or across a closed terminal), and a
cheap monitoring agent should not spend LLM turns polling. So every long job
goes through here:

    uv run python scripts/lab/jobs.py launch --gpu --id win2_gate -- \
        uv run python scripts/rl_power_check.py --out runs/win2_gate ...
    uv run python scripts/lab/jobs.py status win2_gate --tail 20
    uv run python scripts/lab/jobs.py watch win2_gate --interval 300 --max-wait 1500
    uv run python scripts/lab/jobs.py kill win2_gate
    uv run python scripts/lab/jobs.py lock status

`launch` starts a tiny wrapper process, fully detached from the calling
terminal (new session on POSIX, new process group + DETACHED_PROCESS on
Windows). The wrapper runs the real command with stdout+stderr appended to
`lab/jobs/<id>/log.txt`, records pids and the exit code in `status.json`, and
releases the GPU lock when the command ends. `watch` blocks in *this* process
and returns a single JSON event (finished / failed / error_pattern / stalled /
timeout / lost / killed) so the caller spends one turn per event, not per poll.

Layout (gitignored):  lab/jobs/<id>/{cmd.json,cmd.txt,log.txt,status.json,status.md}
GPU lock:             lab/GPU_LOCK  (json: id, pid, since)

Set ATCGAN_LAB_DIR to relocate the `lab/` directory (tests do this).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

WIN = os.name == "nt"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
DEFAULT_ERROR_RE = (
    r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError"
    r"|CUDA error|device-side assert|loss(?:=|: ?)nan\b|\bnan\b loss"
    r"|Segmentation fault|Killed\b|FileNotFoundError|ModuleNotFoundError"
)
TERMINAL_STATES = ("finished", "failed", "killed", "lost")


# --- paths and small helpers --------------------------------------------------

def lab_dir() -> Path:
    override = os.environ.get("ATCGAN_LAB_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "lab"


def jobs_dir() -> Path:
    return lab_dir() / "jobs"


def lock_path() -> Path:
    return lab_dir() / "GPU_LOCK"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _die(msg: str, code: int = 2) -> None:
    print(json.dumps({"error": msg}), file=sys.stderr)
    sys.exit(code)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if WIN:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie answers kill(0) but is dead: reap it if it is ours, else ask ps.
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        pass
    try:
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True
    return not stat.startswith("Z")


def kill_tree(pid: int, grace_s: float = 5.0) -> None:
    """Terminate `pid` and everything it spawned."""
    if WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, check=False)
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def tail_lines(path: Path, n: int, max_bytes: int = 65536) -> list[str]:
    if n <= 0 or not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - max_bytes))
        data = fh.read()
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    # tqdm-style progress bars rewrite the line with '\r'; keep the last state.
    lines = [seg.split("\r")[-1] for seg in text.split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    return lines[-n:]


def gpu_snapshot() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"util_pct": int(parts[0]), "mem_used_mb": int(parts[1]),
                         "mem_total_mb": int(parts[2])})
    return {"gpus": gpus} if gpus else None


# --- lock ----------------------------------------------------------------------

def lock_holder() -> dict | None:
    return _read_json(lock_path())


def lock_is_live() -> tuple[dict | None, bool]:
    holder = lock_holder()
    if not holder:
        return None, False
    pid = holder.get("pid") or 0
    if pid == 0:                       # provisional lock written by `launch`
        return holder, (time.time() - holder.get("since_ts", 0)) < 60
    return holder, pid_alive(pid)


def write_lock(job_id: str, pid: int) -> None:
    lock_path().parent.mkdir(parents=True, exist_ok=True)
    _write_json(lock_path(), {"id": job_id, "pid": pid, "since": _now(),
                              "since_ts": time.time()})


def release_lock(job_id: str | None = None, force: bool = False) -> bool:
    holder = lock_holder()
    if not holder:
        return False
    if not force and job_id is not None and holder.get("id") != job_id:
        return False
    try:
        lock_path().unlink()
    except FileNotFoundError:
        return False
    return True


# --- job state -----------------------------------------------------------------

def job_dir(job_id: str) -> Path:
    if not ID_RE.match(job_id):
        _die(f"bad job id {job_id!r}: use letters, digits, '_', '-', '.' (max 64)")
    return jobs_dir() / job_id


def read_status(job: Path) -> dict | None:
    return _read_json(job / "status.json")


def write_status(job: Path, status: dict) -> None:
    _write_json(job / "status.json", status)


def live_status(job: Path) -> dict:
    """status.json plus liveness; marks a job `lost` if its wrapper died silently."""
    st = read_status(job)
    if st is None:
        _die(f"no such job: {job.name}", 2)
    if st.get("state") in ("starting", "running"):
        child_alive = pid_alive(st.get("child_pid"))
        wrapper_alive = pid_alive(st.get("wrapper_pid"))
        if st["state"] == "running" and not child_alive and not wrapper_alive:
            st = {**st, "state": "lost", "ended": _now(),
                  "note": "wrapper and child both gone without an exit code"}
            write_status(job, st)
            if st.get("gpu"):
                release_lock(job.name)
        elif st["state"] == "starting" and not wrapper_alive:
            age = time.time() - st.get("created_ts", time.time())
            if age > 30:
                st = {**st, "state": "lost", "ended": _now(),
                      "note": "wrapper never reported a child pid"}
                write_status(job, st)
                if st.get("gpu"):
                    release_lock(job.name)
    log = job / "log.txt"
    st["log"] = str(log)
    st["log_bytes"] = log.stat().st_size if log.exists() else 0
    if st.get("started_ts") and st["state"] == "running":
        st["elapsed_s"] = round(time.time() - st["started_ts"])
    return st


# --- commands ------------------------------------------------------------------

def cmd_launch(a: argparse.Namespace) -> int:
    if not a.cmd:
        _die("no command given; put it after `--`")
    job = job_dir(a.id)
    existing = read_status(job)
    if existing and existing.get("state") in ("starting", "running"):
        st = live_status(job)
        if st["state"] in ("starting", "running"):
            _die(f"job {a.id} is still {st['state']} (child pid {st.get('child_pid')})", 3)
    if a.gpu:
        holder, live = lock_is_live()
        if holder and live and holder.get("id") != a.id:
            _die(f"GPU lock held by job {holder['id']} (pid {holder.get('pid')}) since "
                 f"{holder.get('since')}; one GPU stream at a time", 4)
        if holder and not live:
            release_lock(force=True)
    job.mkdir(parents=True, exist_ok=True)
    for name in ("status.md",):
        try:
            (job / name).unlink()
        except FileNotFoundError:
            pass
    (job / "log.txt").write_text("", encoding="utf-8")
    env = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep:
            _die(f"--env expects KEY=VALUE, got {kv!r}")
        env[k] = v
    spec = {"id": a.id, "cmd": a.cmd, "cwd": str(Path(a.cwd).resolve()),
            "gpu": bool(a.gpu), "env": env, "created": _now()}
    _write_json(job / "cmd.json", spec)
    (job / "cmd.txt").write_text(subprocess.list2cmdline(a.cmd) + "\n", encoding="utf-8")
    write_status(job, {"id": a.id, "state": "starting", "gpu": bool(a.gpu),
                       "created": spec["created"], "created_ts": time.time(),
                       "cmd": subprocess.list2cmdline(a.cmd)})
    if a.gpu:
        write_lock(a.id, pid=0)

    wrapper = [sys.executable, str(Path(__file__).resolve()), "_wrap", a.id]
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL, "close_fds": True, "cwd": spec["cwd"]}
    if WIN:
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        try:
            proc = subprocess.Popen(wrapper, creationflags=flags | breakaway, **kw)
        except OSError:
            proc = subprocess.Popen(wrapper, creationflags=flags, **kw)
    else:
        proc = subprocess.Popen(wrapper, start_new_session=True, **kw)

    deadline = time.time() + 10
    st = read_status(job) or {}
    while time.time() < deadline and st.get("state") == "starting":
        time.sleep(0.1)
        st = read_status(job) or st
    out = {"id": a.id, "state": st.get("state"), "wrapper_pid": proc.pid,
           "child_pid": st.get("child_pid"), "gpu": bool(a.gpu),
           "log": str(job / "log.txt"), "status": str(job / "status.json")}
    if st.get("state") == "failed":
        out["error"] = st.get("error")
    print(json.dumps(out))
    return 0 if st.get("state") in ("running", "finished") else 1


def cmd_wrap(a: argparse.Namespace) -> int:
    """Internal: runs inside the detached wrapper process."""
    job = job_dir(a.id)
    spec = _read_json(job / "cmd.json") or {}
    env = {**os.environ, **spec.get("env", {}), "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8"}
    base = read_status(job) or {"id": a.id, "gpu": spec.get("gpu", False)}
    t0 = time.time()
    with open(job / "log.txt", "ab", buffering=0) as log:
        kw: dict = {"cwd": spec["cwd"], "env": env, "stdin": subprocess.DEVNULL,
                    "stdout": log, "stderr": subprocess.STDOUT}
        if WIN:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        try:
            child = subprocess.Popen(spec["cmd"], **kw)
        except Exception as exc:  # noqa: BLE001 - anything here is a launch failure
            write_status(job, {**base, "state": "failed", "exit_code": 127,
                               "error": repr(exc), "ended": _now()})
            if spec.get("gpu"):
                release_lock(a.id)
            return 1
        write_status(job, {**base, "state": "running", "wrapper_pid": os.getpid(),
                           "child_pid": child.pid, "started": _now(),
                           "started_ts": t0})
        if spec.get("gpu"):
            write_lock(a.id, child.pid)
        rc = child.wait()
    current = read_status(job) or base
    state = "killed" if current.get("state") == "killed" else ("finished" if rc == 0 else "failed")
    write_status(job, {**current, "state": state, "exit_code": rc, "ended": _now(),
                       "elapsed_s": round(time.time() - t0)})
    if spec.get("gpu"):
        release_lock(a.id)
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    if not a.id:
        rows = []
        for d in sorted(p for p in jobs_dir().glob("*") if (p / "status.json").exists()):
            st = live_status(d)
            rows.append({k: st.get(k) for k in ("id", "state", "exit_code", "started",
                                                 "ended", "elapsed_s", "gpu")})
        holder, live = lock_is_live()
        print(json.dumps({"jobs": rows, "gpu_lock": holder, "gpu_lock_live": live}, indent=2))
        return 0
    job = job_dir(a.id)
    st = live_status(job)
    st["tail"] = tail_lines(job / "log.txt", a.tail)
    if a.gpu:
        st["gpu_snapshot"] = gpu_snapshot()
    print(json.dumps(st, indent=2))
    return 0


def cmd_tail(a: argparse.Namespace) -> int:
    job = job_dir(a.id)
    for line in tail_lines(job / "log.txt", a.n):
        print(line)
    return 0


def _exit_code_for(event: str) -> int:
    return {"finished": 0, "timeout": 4, "stalled": 3}.get(event, 1)


def cmd_watch(a: argparse.Namespace) -> int:
    job = job_dir(a.id)
    log = job / "log.txt"
    err_re = re.compile(a.error_regex) if a.error_regex else None
    prog_re = re.compile(a.progress_regex) if a.progress_regex else None
    deadline = time.time() + a.max_wait
    stall_s = a.stall_min * 60.0
    last_size = log.stat().st_size if log.exists() else 0
    last_change = log.stat().st_mtime if log.exists() else time.time()
    scanned = 0
    last_progress = None
    matched: list[str] = []
    event = None
    first = True
    while event is None:
        if not first:
            time.sleep(max(0.0, min(a.interval, deadline - time.time())))
        first = False
        st = live_status(job)
        size = log.stat().st_size if log.exists() else 0
        if size != last_size:
            last_size, last_change = size, time.time()
        if size > scanned:
            with log.open("rb") as fh:
                fh.seek(scanned)
                chunk = fh.read(size - scanned).decode("utf-8", errors="replace")
            scanned = size
            if prog_re:
                for m in prog_re.finditer(chunk):
                    last_progress = m.group(0)
            if err_re:
                for line in chunk.replace("\r", "\n").splitlines():
                    if err_re.search(line):
                        matched.append(line.strip())
        if st["state"] in TERMINAL_STATES:
            event = st["state"]
        elif matched and st["state"] != "finished":
            event = "error_pattern"
        elif stall_s > 0 and st["state"] == "running" and (time.time() - last_change) > stall_s:
            event = "stalled"
        elif time.time() >= deadline:
            event = "timeout"
    out = {"event": event, "id": a.id, "state": st["state"], "exit_code": st.get("exit_code"),
           "elapsed_s": st.get("elapsed_s"), "log_bytes": last_size,
           "log_idle_s": round(time.time() - last_change),
           "last_progress": last_progress, "matched": matched[-5:],
           "tail": tail_lines(log, a.tail), "gpu_snapshot": gpu_snapshot(),
           "checked_at": _now()}
    line = (f"{out['checked_at']} {a.id}: {event} (state={st['state']}, "
            f"exit={st.get('exit_code')}, log={last_size}B, idle={out['log_idle_s']}s"
            + (f", progress={last_progress!r}" if last_progress else "")
            + (f", matched={matched[-1]!r}" if matched else "") + ")\n")
    with (job / "status.md").open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(json.dumps(out, indent=2))
    return _exit_code_for(event)


def cmd_kill(a: argparse.Namespace) -> int:
    job = job_dir(a.id)
    st = live_status(job)
    if st["state"] not in ("starting", "running"):
        print(json.dumps({"id": a.id, "state": st["state"], "note": "not running"}))
        return 0
    write_status(job, {**read_status(job), "state": "killed", "killed_at": _now()})
    for pid in (st.get("child_pid"), st.get("wrapper_pid")):
        if pid and pid_alive(pid):
            kill_tree(pid)
    deadline = time.time() + 10
    while time.time() < deadline and pid_alive(st.get("child_pid")):
        time.sleep(0.1)
    if st.get("gpu"):
        release_lock(a.id)
    final = read_status(job) or {}
    if final.get("state") != "killed":
        final = {**final, "state": "killed"}
    final.setdefault("ended", _now())
    write_status(job, final)
    print(json.dumps({"id": a.id, "state": "killed",
                      "child_alive": pid_alive(st.get("child_pid"))}))
    return 0


def cmd_lock(a: argparse.Namespace) -> int:
    holder, live = lock_is_live()
    if a.action == "status":
        print(json.dumps({"held": bool(holder and live), "holder": holder,
                          "stale": bool(holder and not live)}))
        return 0
    if a.action == "release":
        if holder and live and not a.force:
            _die(f"lock held by live job {holder['id']}; use --force only if you are sure", 4)
        released = release_lock(force=True)
        print(json.dumps({"released": released}))
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("launch", help="start a detached job")
    p.add_argument("--id", required=True, help="job id (letters, digits, _ - .)")
    p.add_argument("--gpu", action="store_true", help="acquire the single GPU lock")
    p.add_argument("--cwd", default=".", help="working directory for the command")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="command after `--`")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("_wrap")
    p.add_argument("id")
    p.set_defaults(func=cmd_wrap)

    p = sub.add_parser("status", help="show one job (or all)")
    p.add_argument("id", nargs="?")
    p.add_argument("--tail", type=int, default=10)
    p.add_argument("--gpu", action="store_true", help="include an nvidia-smi snapshot")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("tail", help="last lines of a job log")
    p.add_argument("id")
    p.add_argument("-n", type=int, default=30)
    p.set_defaults(func=cmd_tail)

    p = sub.add_parser("watch", help="block until the job changes state or stalls")
    p.add_argument("id")
    p.add_argument("--interval", type=float, default=300.0, help="seconds between checks")
    p.add_argument("--max-wait", type=float, default=1500.0,
                   help="return with event=timeout after this many seconds")
    p.add_argument("--stall-min", type=float, default=20.0,
                   help="event=stalled if the log has not grown for this many minutes (0=off)")
    p.add_argument("--error-regex", default=DEFAULT_ERROR_RE,
                   help="lines matching this end the watch with event=error_pattern ('' to disable)")
    p.add_argument("--progress-regex", default=r"(?:step|trial|cell|epoch)\s*\d+\s*/\s*\d+|\d+%\|",
                   help="last match is reported as last_progress")
    p.add_argument("--tail", type=int, default=20)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("kill", help="terminate a job and its process tree")
    p.add_argument("id")
    p.set_defaults(func=cmd_kill)

    p = sub.add_parser("lock", help="GPU lock: status | release")
    p.add_argument("action", choices=["status", "release"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_lock)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "launch" and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
