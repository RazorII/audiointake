"""lock.py: one writer of the data folder, waiting runs, re-entry, and the hand-over to a child audiointake."""
from __future__ import annotations

import fcntl
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from audiointake import lock

CHILD = """
import sys
from pathlib import Path
from audiointake import lock
with lock.run_lock(Path(sys.argv[1])):
    print("in", lock.inherited(), flush=True)
"""


def child(data: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    e = dict(os.environ if env is None else env)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.Popen([sys.executable, "-c", CHILD, str(data)], env=e, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def wait_for_line(stream, within: float = 20.0) -> str:
    box: list[str] = []
    t = threading.Thread(target=lambda: box.append(stream.readline()), daemon=True)
    t.start()
    t.join(within)
    return box[0] if box else ""


def held_by_someone(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def test_the_lock_file_and_its_folder(tmp_path):
    data = tmp_path / "data"
    assert not lock.held()
    with lock.run_lock(data) as path:
        assert path == data / "run.lock" and lock.held() and not lock.inherited()
        assert (path.stat().st_mode & 0o777) == 0o600 and (data.stat().st_mode & 0o777) == 0o700
        assert held_by_someone(path)
    assert not lock.held() and not held_by_someone(data / "run.lock") and (data / "run.lock").exists()


def test_the_default_folder_is_the_data_folder():
    from audiointake import platforms
    assert lock.lock_path() == platforms.data_dir() / "run.lock"


def test_re_entry_in_the_same_thread_is_free(tmp_path):
    err = io.StringIO()
    with lock.run_lock(tmp_path, stream=err):
        with lock.run_lock(tmp_path, stream=err):
            lock.require_held()
        assert lock.held() and held_by_someone(tmp_path / "run.lock")
    assert err.getvalue() == "" and not lock.held()


def test_another_data_folder_while_held_is_a_bug(tmp_path):
    with lock.run_lock(tmp_path / "a"):
        with pytest.raises(RuntimeError):
            with lock.run_lock(tmp_path / "b"):
                pass
    assert not lock.held()


def test_require_held_and_child_env_outside_the_lock():
    with pytest.raises(RuntimeError):
        lock.require_held()
    with pytest.raises(RuntimeError):
        lock.child_env({})


def test_another_process_waits_and_says_so(tmp_path):
    with lock.run_lock(tmp_path):
        p = child(tmp_path)
        assert wait_for_line(p.stderr).strip() == lock.WAIT_MESSAGE
        time.sleep(0.5)
        assert p.poll() is None                       # still waiting
    out, err = p.communicate(timeout=30)
    assert p.returncode == 0 and out == "in False\n"


def test_a_child_given_the_lock_runs_at_once(tmp_path):
    with lock.run_lock(tmp_path):
        env = lock.child_env(dict(os.environ))
        assert env[lock.HOLDER_ENV] == str(os.getpid())
        p = child(tmp_path, env)
        out, err = p.communicate(timeout=30)
        assert p.returncode == 0 and out == "in True\n" and lock.WAIT_MESSAGE not in err
        assert held_by_someone(tmp_path / "run.lock")      # still ours


def test_a_marker_naming_another_process_is_ignored(tmp_path):
    with lock.run_lock(tmp_path):
        env = dict(os.environ)
        env[lock.HOLDER_ENV] = "1"
        p = child(tmp_path, env)
        assert wait_for_line(p.stderr).strip() == lock.WAIT_MESSAGE
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 0 and out == "in False\n"


def test_a_marker_with_nobody_holding_takes_the_lock_for_real(tmp_path, monkeypatch):
    monkeypatch.setenv(lock.HOLDER_ENV, str(os.getppid()))
    with lock.run_lock(tmp_path):
        assert not lock.inherited() and held_by_someone(tmp_path / "run.lock")


def test_another_thread_waits_its_turn(tmp_path):
    entered = threading.Event()

    def other():
        with lock.run_lock(tmp_path, stream=io.StringIO()):
            entered.set()

    with lock.run_lock(tmp_path):
        t = threading.Thread(target=other, daemon=True)
        t.start()
        assert not entered.wait(0.5)
    t.join(10)
    assert entered.is_set() and not lock.held()
