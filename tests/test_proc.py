"""proc.py: the allowlist, the scrubbed environment, the stderr tail, timeouts, stalls and process groups."""
from __future__ import annotations

import ast
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from audiointake import proc
from audiointake.contracts import Tools

PY = Path(sys.executable)
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
SRC = Path(__file__).resolve().parents[1] / "src" / "audiointake"


def py(code: str) -> list[str]:
    return [str(PY), "-I", "-c", code]


def run(code: str, **kw):
    kw.setdefault("env", ENV)
    kw.setdefault("timeout", 60)
    kw.setdefault("allowed", [PY])
    return proc.run(py(code), **kw)


def gone(pid: int, within: float = 8.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- refusals before any process exists
@pytest.fixture
def no_popen(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(proc.subprocess, "Popen", boom)


@pytest.mark.parametrize("argv", [[], ["python3", "-c", "1"], ["./python", "-c", "1"], ["/bin/echo", "x"]])
def test_refused_argv_starts_nothing(no_popen, argv):
    with pytest.raises(proc.NotAllowed):
        proc.run(argv, env=ENV, timeout=5, allowed=[PY])


def test_nul_in_an_argument_is_refused(no_popen):
    with pytest.raises(proc.NotAllowed):
        proc.run([str(PY), "-c", "1\x00"], env=ENV, timeout=5, allowed=[PY])


def test_allowlist_compares_unresolved_paths(no_popen, tmp_path):
    link = tmp_path / "python-link"
    link.symlink_to(PY)
    with pytest.raises(proc.NotAllowed):            # the link isn't on the list, though it points at an allowed file
        proc.run([str(link), "-c", "1"], env=ENV, timeout=5, allowed=[PY])


def test_bad_options_are_refused(no_popen):
    with pytest.raises(ValueError):
        proc.run(py("1"), env=ENV, timeout=0, allowed=[PY])
    with pytest.raises(ValueError):
        proc.run(py("1"), env=ENV, timeout=5, allowed=[PY], stdout="keep")     # type: ignore[arg-type]
    with pytest.raises(TypeError):
        proc.run(py("1"), env={"A": 1}, timeout=5, allowed=[PY])               # type: ignore[dict-item]


# --------------------------------------------------------------------------- normal runs
def test_capture_and_returncode():
    r = run("import sys; sys.stdout.write('héllo\\n'); sys.exit(7)")
    assert r.returncode == 7 and not r.ok and not r.timed_out and not r.stalled
    assert r.stdout == "héllo\n".encode()
    assert r.seconds > 0


def test_discard_keeps_no_stdout():
    r = run("print('x' * 100000)", stdout="discard")
    assert r.ok and r.stdout == b""


def test_large_stdout_is_read_while_the_child_runs():
    r = run("import sys; sys.stdout.write('y' * 3_000_000)")
    assert r.ok and len(r.stdout) == 3_000_000


def test_env_is_exactly_what_was_passed(monkeypatch):
    monkeypatch.setenv("AUDIOINTAKE_PROC_MARKER", "leak")
    r = run("import os, json; print(json.dumps(dict(os.environ)))")
    import json
    child = json.loads(r.stdout)
    assert "AUDIOINTAKE_PROC_MARKER" not in child
    for k, v in ENV.items():
        assert child[k] == v
    assert not ({"HOME", "HF_HUB_CACHE", "AUDIOINTAKE_HOME"} & set(child))


def test_stdin_is_dev_null():
    r = run("import sys; print(repr(sys.stdin.read()))")
    assert r.stdout.strip() == b"''"


def test_cwd():
    r = run("import os; print(os.getcwd())", cwd=Path("/"))
    assert r.stdout.strip() == b"/"


def test_stderr_tail_keeps_the_last_five_lines_cut_to_300():
    code = ("import sys\n"
            "for i in range(8): print('line', i, file=sys.stderr)\n"
            "print('', file=sys.stderr); print('   ', file=sys.stderr)\n"
            "print('a' * 200 + 'b' * 400, file=sys.stderr)\n"
            "sys.stderr.write('10%\\r50%\\r100%\\n')\n")
    r = run(code)
    assert r.stderr_tail == ["line 7", "b" * 300, "10%", "50%", "100%"]


def test_stderr_tail_survives_a_flood():
    r = run("import sys\nfor i in range(20000): sys.stderr.write('x' * 50 + '\\n')\nsys.stderr.write('last\\n')")
    assert r.ok and r.stderr_tail[-1] == "last" and len(r.stderr_tail) == 5


def test_tail_helper():
    assert proc.tail(b"a\n\nb\r\nc\rd\n") == ["a", "b", "c", "d"]
    assert proc.tail(b"\xff\xfe broken\n") == ["\ufffd\ufffd broken"]
    assert proc.tail(b"") == []


# --------------------------------------------------------------------------- stopping children
SPAWN_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
    " stdout=subprocess.DEVNULL, stderr={stderr})\n"
    "print(g.pid, flush=True)\n"
    "{after}\n"
)


def test_timeout_kills_the_whole_group():
    r = run(SPAWN_GRANDCHILD.format(stderr="subprocess.DEVNULL", after="time.sleep(60)"), timeout=2)
    assert r.timed_out and not r.ok and r.returncode < 0
    assert r.seconds < 10
    assert gone(int(r.stdout.split()[0]))


def test_stall_predicate_kills():
    t0 = time.monotonic()
    r = run("import time; time.sleep(60)", stall=lambda: time.monotonic() - t0 > 1.5)
    assert r.stalled and not r.timed_out and not r.ok
    assert r.seconds < 10


def test_a_failing_stall_predicate_kills_and_raises():
    def bad():
        raise RuntimeError("watchdog broke")
    with pytest.raises(RuntimeError):
        run("import time; time.sleep(60)", stall=bad)
    assert proc.live_pids() == []


def test_grandchild_holding_the_pipes_is_stopped(monkeypatch):
    monkeypatch.setattr(proc, "DRAIN_S", 1.0)
    r = run(SPAWN_GRANDCHILD.format(stderr="None", after="sys.exit(0)"))
    assert r.returncode == 0
    assert r.seconds < 8
    assert gone(int(r.stdout.split()[0]))


def test_terminate_all_stops_a_running_child():
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", run("import time; time.sleep(60)")))
    t.start()
    end = time.monotonic() + 10
    while not proc.live_pids() and time.monotonic() < end:
        time.sleep(0.02)
    assert proc.live_pids()
    proc.terminate_all(signal.SIGKILL)
    t.join(15)
    assert box["r"].returncode == -signal.SIGKILL
    assert proc.live_pids() == []


@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is macOS's")
def test_keep_awake_holds_caffeinate_for_the_block():
    with proc.keep_awake() as pid:
        assert pid is not None
        os.kill(pid, 0)                                  # running
        assert pid in proc.live_pids()
    assert gone(pid)
    assert pid not in proc.live_pids()


def test_keep_awake_without_caffeinate(tmp_path):
    with proc.keep_awake(tmp_path / "no-such-caffeinate") as pid:
        assert pid is None


# --------------------------------------------------------------------------- tables
def test_child_timeouts():
    assert proc.child_timeout("parakeet", 100) == 600 and proc.child_timeout("parakeet", 1000) == 3000
    assert proc.child_timeout("whisper", 100) == 900 and proc.child_timeout("thai", 600) == 1800
    assert proc.child_timeout("fluidaudio", 100) == 300 and proc.child_timeout("fluidaudio", 4000) == 4000
    assert proc.child_timeout("speaker_id", 99999) == 600
    assert proc.child_timeout("ffmpeg", 100) == 120 and proc.child_timeout("ffprobe", 4000) == 1000
    with pytest.raises(ValueError):
        proc.child_timeout("swift", 10)


def test_allowed_for_names_every_tool():
    t = Tools(ffmpeg=Path("/a/ffmpeg"), ffprobe=None, python=Path("/v/bin/python"), parakeet_cli=Path("/v/bin/p"),
              whisper_cli=Path("/v/bin/w"), fluidaudiocli=None, fluidaudio_tag="v0.18.0")
    got = proc.allowed_for(t, Path("/usr/bin/git"))
    assert got == {Path("/a/ffmpeg"), Path("/v/bin/python"), Path("/v/bin/p"), Path("/v/bin/w"),
                   Path("/usr/bin/caffeinate"), Path("/usr/bin/git")}


# --------------------------------------------------------------------------- only proc starts processes
FORBIDDEN_MODULES = {"subprocess", "pty", "multiprocessing"}
FORBIDDEN_OS = ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp")


def test_only_proc_starts_processes():
    offenders = []
    for f in sorted(SRC.rglob("*.py")):
        if f.name == "proc.py" and f.parent == SRC:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
                if node.module == "os":
                    names |= {"os." + a.name for a in node.names}
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                names = {"os." + node.attr}
            else:
                continue
            bad = (names & FORBIDDEN_MODULES) | {n for n in names if n.startswith("os.") and (
                n[3:] in FORBIDDEN_OS or n[3:].startswith(("exec", "spawn")))}
            if bad:
                offenders.append(f"{f.relative_to(SRC)}:{node.lineno}: {sorted(bad)}")
    assert offenders == []


# --------------------------------------------------------------------------- output_log (setup's build children)
def test_output_log_keeps_everything_the_child_wrote(tmp_path):
    log = tmp_path / "build.log"
    code = ("import sys\n"
            "for i in range(40):\n"
            "    sys.stdout.write('out %d\\n' % i); sys.stdout.flush()\n"
            "    sys.stderr.write('err %d\\n' % i); sys.stderr.flush()\n"
            "sys.exit(1)")
    r = run(code, stdout="discard", output_log=log)
    assert r.returncode == 1 and r.stdout == b""
    assert r.stderr_tail == [f"err {i}" for i in range(35, 40)]
    text = log.read_text(encoding="utf-8")
    assert all(f"out {i}\n" in text and f"err {i}\n" in text for i in range(40))
    assert (log.stat().st_mode & 0o777) == 0o600


def test_output_log_appends_and_capture_still_works(tmp_path):
    log = tmp_path / "build.log"
    log.write_bytes(b"earlier\n")
    r = run("import sys; print('hello'); print('warn', file=sys.stderr)", output_log=log)
    assert r.ok and r.stdout == b"hello\n" and r.stderr_tail == ["warn"]
    assert log.read_bytes().startswith(b"earlier\n") and b"hello\n" in log.read_bytes()


def test_without_output_log_nothing_is_written(tmp_path):
    before = set(tmp_path.iterdir())
    r = run("print('x')", cwd=tmp_path)
    assert r.ok and set(tmp_path.iterdir()) == before


def test_output_log_is_closed_when_the_program_is_refused(tmp_path, no_popen):
    with pytest.raises(proc.NotAllowed):
        proc.run(["/bin/echo", "x"], env=ENV, timeout=5, allowed=[PY], output_log=tmp_path / "x.log")
    assert not (tmp_path / "x.log").exists()


def test_keep_awake_gives_caffeinate_no_environment_but_path(monkeypatch):
    seen = {}
    real = proc.subprocess.Popen

    class Spy(real):
        def __init__(self, args, **kw):
            seen["env"] = kw.get("env")
            super().__init__(args, **kw)

    monkeypatch.setattr(proc.subprocess, "Popen", Spy)
    monkeypatch.setenv("SOME_TOKEN", "x")
    if sys.platform != "darwin":
        pytest.skip("caffeinate is macOS's")
    with proc.keep_awake() as pid:
        assert pid is not None
    assert seen["env"] == {"PATH": "/usr/bin:/bin"}


# --------------------------------------------------------------------------- program(): a stop stops the children
def test_program_turns_a_signal_into_its_exit_status():
    before = {s: signal.getsignal(s) for s in proc.PROGRAM_SIGNALS}

    def body():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        return 0

    assert proc.program(body) == 128 + signal.SIGTERM
    assert {s: signal.getsignal(s) for s in proc.PROGRAM_SIGNALS} == before


def test_a_signal_while_a_child_starts_is_held_until_the_child_is_registered(monkeypatch):
    """A signal landing between the fork and the child's registration used to leak that child: the stop couldn't
    see it. Now it is held and raised once the child is registered, and the stop kills it."""
    real = proc.subprocess.Popen
    started = {}

    class SignalOnStart(real):
        def __init__(self, args, **kw):
            super().__init__(args, **kw)
            started["pid"] = self.pid
            os.kill(os.getpid(), signal.SIGTERM)          # handled before run() registers the child
            time.sleep(0.2)

    def body():
        monkeypatch.setattr(proc.subprocess, "Popen", SignalOnStart)
        run("import time; time.sleep(30)")
        return 0

    assert proc.program(body) == 128 + signal.SIGTERM
    assert "pid" in started and gone(started["pid"])
    assert proc._STARTING == {"depth": 0, "pending": None}


def test_the_hold_never_outlives_a_program(monkeypatch):
    proc._STARTING["pending"] = signal.SIGHUP            # as if a held signal were left over
    assert proc.program(lambda: 0) == 0
    assert proc._STARTING["pending"] is None
