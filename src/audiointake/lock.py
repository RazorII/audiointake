"""One writer of the data folder at a time: ``<data>/run.lock`` (``fcntl.flock``).

The rule every module follows:

* A command that writes the data folder (``transcribe``, ``apply``, ``relabel``, ``notes``, ``enroll``,
  ``voices remove|rename``, ``setup``, ``packs remove``, ``uninstall``, and the ensemble module's ``transcribe``
  for callers that run it directly) holds ``run_lock()`` for its whole run, taken in the CLI before anything else
  is read or written.
* Library functions it calls (installing, removing or pruning packs, a FluidAudio build, the record writer) assume
  the lock is held and never take it; ``require_held()`` lets them, and their tests, check. Taking it again in the
  same thread only counts up, so a library call that does take it can't deadlock its caller.
* A child audiointake the holder starts on purpose (setup's canary runs ``audiointake transcribe``) is given the
  lock with ``child_env(env)``: that names the holder's pid in ``AUDIOINTAKE_RUN_LOCK_HOLDER``, and the child's
  ``run_lock()`` takes nothing when that pid is its own parent's and the lock file is held. Any other value is
  ignored, so the variable can't let an unrelated process in. The run children's scrubbed environment never
  carries it (it's built fresh, not copied from ours).

A second run waits, and says so once on stderr: "Waiting for another audiointake run to finish". That's waiting
on a run, never on a person. The lock file is never deleted (deleting a lock file races with the next taker).
"""
from __future__ import annotations

import fcntl
import os
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from . import fsutil, platforms

LOCK_NAME = "run.lock"
HOLDER_ENV = "AUDIOINTAKE_RUN_LOCK_HOLDER"
WAIT_MESSAGE = "Waiting for another audiointake run to finish"

_RLOCK = threading.RLock()          # one thread of this process at a time; the same thread may re-enter
_STATE: dict = {"depth": 0, "fd": -1, "path": None, "inherited": False}


def lock_path(data: Path | None = None) -> Path:
    """``<data>/run.lock`` (``data`` defaults to ``platforms.data_dir()``)."""
    return (Path(data) if data is not None else platforms.data_dir()) / LOCK_NAME


def held() -> bool:
    """Whether this process holds the lock (or was given it by its parent)."""
    return _STATE["depth"] > 0


def inherited() -> bool:
    """Whether the lock this process holds was given by its parent (``child_env``)."""
    return held() and bool(_STATE["inherited"])


def require_held() -> None:
    """RuntimeError unless this process holds the lock: for functions that write the data folder."""
    if not held():
        raise RuntimeError("this writes audiointake's data folder and must run under lock.run_lock()")


def child_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` plus the marker that hands the held lock to a child audiointake this process starts."""
    require_held()
    return {**env, HOLDER_ENV: str(os.getpid())}


def _try(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _acquire(path: Path, stream: TextIO) -> None:
    fsutil.private_dir(path.parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        got = _try(fd)
        holder = os.environ.get(HOLDER_ENV, "")
        if not got and holder and holder == str(os.getppid()):
            os.close(fd)                                    # our parent holds it and handed it to us
            _STATE.update(fd=-1, path=path, inherited=True)
            return
        if not got:
            print(WAIT_MESSAGE, file=stream, flush=True)
            fcntl.flock(fd, fcntl.LOCK_EX)                  # PEP 475: retried after a signal
    except BaseException:
        os.close(fd)
        raise
    _STATE.update(fd=fd, path=path, inherited=False)


def _release() -> None:
    fd = _STATE["fd"]
    _STATE.update(fd=-1, path=None, inherited=False)
    if fd >= 0:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def run_lock(data: Path | None = None, *, stream: TextIO | None = None) -> Iterator[Path]:
    """Hold ``<data>/run.lock`` for the block, waiting (and saying so on ``stream``, stderr by default) while
    another run holds it. Re-entrant in the same thread; another thread of this process waits its turn."""
    path = lock_path(data)
    with _RLOCK:
        if _STATE["depth"] == 0:
            _acquire(path, stream if stream is not None else sys.stderr)
        elif Path(_STATE["path"]) != path:
            raise RuntimeError("the run lock is already held for another data folder")
        _STATE["depth"] += 1
        try:
            yield path
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                _release()
