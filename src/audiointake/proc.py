"""The one place audiointake starts a process.

``run`` starts a child from an argv list whose first item is an absolute path on the caller's allowlist (built
from ``Tools`` with ``allowed_for``, plus, in setup only, git, swift and xcode-select from the checked toolchain
folder). Anything else raises ``NotAllowed`` before a process exists. Never a shell; stdin is /dev/null; each child
leads its own process group, so a timeout, a stall or ``terminate_all`` stops everything it started.

A child's stdout is kept in memory only when asked (``stdout="capture"``) and is never logged: a speech model can
print transcript text there. Of stderr only the tail is kept: the last 5 non-blank lines, each cut to its last 300
characters (carriage-return progress lines count as lines).

The one exception is ``output_log``, never for a model or a worker: every byte the child writes to stdout and
stderr is appended to that file as it arrives. Setup's build children (``swift``, ``git``) use it so a failed build
leaves a whole log to point at; the probe uses it for ``ffmpeg -i`` when only the fallback ffmpeg is there (no
ffprobe), because that stderr is the probe's answer, written into the run's work folder, read and deleted at once.
The tail and ``stdout="capture"`` work as without it.

``keep_awake`` holds ``caffeinate -i -w <our pid>`` for the length of a run, so a Mac on battery doesn't sleep in
the middle of one (a closed lid still sleeps). It gets ``PATH=/usr/bin:/bin`` and nothing else of our environment.

``program(body)`` runs a steps program (a worker such as ``python -I -B -m audiointake.steps.thai``, or the ensemble
module's command line) so that stopping it stops its children: SIGTERM, SIGHUP and SIGINT raise ``Killed``, a
BaseException, which unwinds through every ``finally`` (``run`` stops the child it waits on), then
``stop_children`` stops whatever still runs and the program exits 128 + the signal. A signal that lands while
``run`` is starting a child (between the fork and the child's registration) is held until the child is
registered, so no child is ever started that the stop can't see.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .contracts import Tools

TAIL_LINES = 5
TAIL_CHARS = 300
TAIL_BYTES = 64 * 1024        # stderr kept in memory while a child runs, from which the tail is cut
STALL_POLL_S = 1.0            # how often the stall predicate is asked
KILL_GRACE_S = 2.0            # SIGTERM to the group, then SIGKILL after this long
DRAIN_S = 5.0                 # after the child exits, how long its pipes may stay open (a grandchild holding them)


class NotAllowed(ValueError):
    """An argv that ``run`` refuses before starting anything: empty, relative, or not on the allowlist."""


@dataclass(frozen=True)
class Completed:
    returncode: int                              # negative: killed by that signal
    stdout: bytes                                # b"" unless stdout="capture"
    stderr_tail: list[str] = field(default_factory=list)
    timed_out: bool = False
    stalled: bool = False
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.stalled


_LIVE: dict[int, subprocess.Popen] = {}         # pid (= process group id) -> child
_LIVE_LOCK = threading.Lock()
# A stop handler (program()'s, and the CLI's) holds a signal back while the main thread is starting a child
# (``depth`` > 0, see ``hold_signal``) and it is raised again, through whichever handler is installed, once the
# child is registered. Python runs signal handlers in the main thread, so only the main thread's starts are counted
# (a held signal is always raised where the handler would have raised it).
_STARTING: dict = {"depth": 0, "pending": None}


def _register(p: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE[p.pid] = p


def _unregister(p: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE.pop(p.pid, None)


def live_pids() -> list[int]:
    """The process groups of children still running (for tests and the signal path)."""
    with _LIVE_LOCK:
        return [pid for pid, p in _LIVE.items() if p.poll() is None]


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def terminate_all(sig: int) -> None:
    """Send ``sig`` to every live child's process group (SIGTERM or SIGKILL from the CLI's signal handler)."""
    with _LIVE_LOCK:
        pids = list(_LIVE)
    for pid in pids:
        _signal_group(pid, sig)


def allowed_for(tools: Tools, *extra: Path) -> frozenset[Path]:
    """The allowlist for a run: every tool ``Tools`` names (those that exist as paths), plus ``extra``."""
    paths = [tools.ffmpeg, tools.ffprobe, tools.python, tools.parakeet_cli, tools.whisper_cli, tools.fluidaudiocli,
             tools.caffeinate, *extra]
    return frozenset(Path(p) for p in paths if p is not None)


def _check(argv: Sequence[str | os.PathLike[str]], allowed: Iterable[Path]) -> list[str]:
    if not argv:
        raise NotAllowed("an empty argv")
    args: list[str] = []
    for a in argv:
        s = os.fspath(a)
        if not isinstance(s, str):
            raise NotAllowed("argv items must be str or str paths")
        if "\x00" in s:
            raise NotAllowed("an argv item contains a NUL byte")
        args.append(s)
    exe = args[0]
    if not os.path.isabs(exe):
        raise NotAllowed(f"the program must be an absolute path: {exe!r}")
    ok = {os.path.normpath(os.fspath(p)) for p in allowed}
    if os.path.normpath(exe) not in ok:
        raise NotAllowed(f"the program isn't on this run's allowlist: {exe!r}")
    return args


def tail(data: bytes, lines: int = TAIL_LINES, chars: int = TAIL_CHARS) -> list[str]:
    """The last ``lines`` non-blank lines of ``data`` (CR or LF ends a line), each cut to its last ``chars``."""
    text = data.decode("utf-8", errors="replace").replace("\r", "\n")
    kept = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    return [ln[-chars:] for ln in kept[-lines:]]


class _Tee:
    """The file ``output_log`` names, shared by the two reader threads (0600, appended to)."""

    def __init__(self, path: Path) -> None:
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
        self._lock = threading.Lock()

    def write(self, chunk: bytes) -> None:
        with self._lock:
            view = memoryview(chunk)
            while view:
                view = view[os.write(self._fd, view):]

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


def _reader(stream, sink: deque | list | None, bounded: bool, tee: _Tee | None = None) -> None:
    total = 0
    try:
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
            if not chunk:
                break
            if tee is not None:
                tee.write(chunk)
            if sink is None:
                continue
            sink.append(chunk)
            if bounded:
                total += len(chunk)
                while total > TAIL_BYTES and len(sink) > 1:
                    total -= len(sink.popleft())
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _kill_group(p: subprocess.Popen) -> None:
    _signal_group(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(p.pid, signal.SIGKILL)
    try:
        p.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def run(argv: Sequence[str | os.PathLike[str]], *, env: Mapping[str, str], timeout: float,
        allowed: Iterable[Path], cwd: Path | None = None, stdout: Literal["capture", "discard"] = "capture",
        stall: Callable[[], bool] | None = None, output_log: Path | None = None) -> Completed:
    """Run one child to completion, or until ``timeout`` seconds or ``stall()`` returning True kills its group.

    ``env`` is the child's whole environment (nothing else from ours reaches it). A child that leaves a background
    process holding its pipes open has that process group killed ``DRAIN_S`` seconds after it exits.
    ``output_log`` (build children, and the probe's ``ffmpeg -i``) receives all of the child's stdout and stderr.
    """
    args = _check(argv, allowed)
    if stdout not in ("capture", "discard"):
        raise ValueError("stdout must be 'capture' or 'discard'")
    if not timeout or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    child_env = {}
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("env keys and values must be str")
        child_env[k] = v

    tee = _Tee(Path(output_log)) if output_log is not None else None
    start = time.monotonic()
    main = threading.current_thread() is threading.main_thread()
    _STARTING["depth"] += main
    try:
        try:
            p = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE if (stdout == "capture" or tee) else subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, env=child_env,
                                 cwd=os.fspath(cwd) if cwd is not None else None,
                                 start_new_session=True, close_fds=True)
        except BaseException:
            if tee is not None:
                tee.close()
            raise
        _register(p)
    finally:
        _STARTING["depth"] -= main
    if main:
        _raise_held()
    out_chunks: list[bytes] = []
    err_chunks: deque[bytes] = deque()
    threads = [threading.Thread(target=_reader, args=(p.stderr, err_chunks, True, tee), daemon=True)]
    if stdout == "capture" or tee is not None:
        sink = out_chunks if stdout == "capture" else None
        threads.append(threading.Thread(target=_reader, args=(p.stdout, sink, False, tee), daemon=True))
    for t in threads:
        t.start()

    timed_out = stalled = False
    try:
        deadline = start + timeout
        next_poll = start + STALL_POLL_S
        while True:
            now = time.monotonic()
            wait_for = min(deadline, next_poll if stall is not None else deadline) - now
            try:
                p.wait(timeout=max(0.01, wait_for))
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                _kill_group(p)
                break
            if stall is not None and now >= next_poll:
                next_poll = now + STALL_POLL_S
                if stall():
                    stalled = True
                    _kill_group(p)
                    break
    except BaseException:
        _kill_group(p)
        raise
    finally:
        for t in threads:
            t.join(timeout=DRAIN_S)
        if any(t.is_alive() for t in threads):            # a grandchild still holds a pipe: stop the whole group
            _signal_group(p.pid, signal.SIGKILL)
            for t in threads:
                t.join(timeout=DRAIN_S)
        _unregister(p)
        if tee is not None:
            tee.close()

    rc = p.returncode if p.returncode is not None else -signal.SIGKILL
    return Completed(returncode=rc, stdout=b"".join(out_chunks), stderr_tail=tail(b"".join(err_chunks)),
                     timed_out=timed_out, stalled=stalled, seconds=time.monotonic() - start)


# Children's timeouts in seconds (a hang stops here; the models run about 20 times faster than real time).
_TIMEOUTS: dict[str, Callable[[float], float]] = {
    "parakeet": lambda d: max(600.0, 3 * d),
    "whisper": lambda d: max(900.0, 3 * d),
    "thai": lambda d: max(900.0, 3 * d),
    "fluidaudio": lambda d: max(300.0, d),
    "speaker_id": lambda d: 600.0,
    "ffmpeg": lambda d: max(120.0, d / 4),
    "ffprobe": lambda d: max(120.0, d / 4),
}
CHILD_KINDS = tuple(_TIMEOUTS)


def child_timeout(kind: str, duration_s: float) -> float:
    """The timeout for a child of ``kind`` (one of CHILD_KINDS) on audio ``duration_s`` seconds long."""
    if kind not in _TIMEOUTS:
        raise ValueError(f"unknown child kind {kind!r}; one of {CHILD_KINDS}")
    return float(_TIMEOUTS[kind](max(0.0, float(duration_s or 0.0))))


@contextmanager
def keep_awake(caffeinate: Path = Path("/usr/bin/caffeinate")) -> Iterator[int | None]:
    """Hold ``caffeinate -i -w <this pid>`` while the block runs; yields its pid, or None where there is none."""
    if sys.platform != "darwin" or not os.access(caffeinate, os.X_OK):
        yield None
        return
    p = subprocess.Popen([os.fspath(caffeinate), "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"},
                         start_new_session=True, close_fds=True)
    _register(p)
    try:
        yield p.pid
    finally:
        _signal_group(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            _signal_group(p.pid, signal.SIGKILL)
            p.wait(timeout=KILL_GRACE_S)
        _unregister(p)


# --------------------------------------------------------------------------- programs: stopping one stops its children
REAP_GRACE_S = 1.0          # SIGTERM to each child's group, SIGKILL after this long: under KILL_GRACE_S, so it lands
                            # before the SIGKILL the parent's run() sends this program
PROGRAM_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class Killed(BaseException):
    """A program got SIGTERM, SIGHUP or SIGINT. A BaseException, so no ``except Exception`` stops it on its way to
    ``program``; every ``finally`` on the way runs (``run`` stops the child it is waiting for)."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(self.signum)


def hold_signal(signum: int) -> bool:
    """Every stop handler asks this first: True while the main thread is starting a child, and the signal then
    waits until ``run`` has registered that child (so the stop sees it and kills it), when it is raised again."""
    if _STARTING["depth"] > 0:
        if _STARTING["pending"] is None:
            _STARTING["pending"] = int(signum)
        return True
    return False


def ignore_stops(signals: Iterable[int] = PROGRAM_SIGNALS) -> None:
    """After the first stop signal: ignore the rest, so the cleanup on the way out isn't cut short."""
    for s in signals:
        try:
            signal.signal(s, signal.SIG_IGN)
        except ValueError:                       # not the main thread
            pass


def _on_signal(signum: int, _frame) -> None:
    if hold_signal(signum):                      # a child is being started: hold it until the child is registered
        return
    ignore_stops()
    raise Killed(signum)


def _raise_held() -> None:
    """Raise the signal a stop handler held back while a child was being started (the child is registered now),
    through whichever handler is installed: program()'s or the CLI's."""
    if _STARTING["depth"] == 0 and _STARTING["pending"] is not None:
        signum, _STARTING["pending"] = _STARTING["pending"], None
        signal.raise_signal(signum)


def stop_children(grace: float = REAP_GRACE_S) -> None:
    """Stop every child this process started and is still running: SIGTERM to each child's process group, then
    SIGKILL after ``grace`` seconds to any group still there; then reap them, so none is left as a zombie (the
    CLI's stop and program()'s both end here)."""
    terminate_all(signal.SIGTERM)
    end = time.monotonic() + grace
    while live_pids() and time.monotonic() < end:
        time.sleep(0.05)
    terminate_all(signal.SIGKILL)
    with _LIVE_LOCK:
        children = list(_LIVE.values())
    for p in children:
        try:
            p.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            continue
        _unregister(p)


def program(body: Callable[[], int]) -> int:
    """Run a program's ``body`` (its ``main``), stopping its children when the program is stopped: SIGTERM, SIGHUP
    and SIGINT raise ``Killed`` in the main thread; on the way out the ``finally`` blocks remove the work folder and
    temporary files, then ``stop_children`` stops whatever still runs, and the exit status is 128 + the signal.
    The previous handlers are put back afterwards (``main`` can be called in process, as the tests do)."""
    old = {}
    for s in PROGRAM_SIGNALS:
        try:
            old[s] = signal.signal(s, _on_signal)
        except ValueError:                       # not the main thread: nothing to install
            pass
    try:
        return body()
    except Killed as k:
        stop_children()
        return 128 + k.signum
    finally:
        _STARTING["pending"] = None
        for s, h in old.items():
            signal.signal(s, h)
