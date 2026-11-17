"""Files: the one atomic writer, hashes, private folders, iCloud placeholders, UTF-8 stdio, a recording's stem.

``atomic_write`` writes ``.partial-<random>`` in the destination's folder, flushes it to disk, then either replaces
the destination (``overwrite=True``) or links it into place, which fails rather than replace a file that is
already there, so an existing file is never lost unless the caller asked. Where the filesystem has no hard links
(exFAT, FAT, some SMB shares), the fallback re-checks that the name is free and creates it with
``O_CREAT|O_EXCL``, so the no-clobber guarantee holds there too; a crash in that window can leave a partly written
file, which the next run's seal or hash check catches. Only the commit steps and the voiceprint writer call it
for a folder the user owns.
"""
from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
import sys
import threading
from pathlib import Path
from typing import Union

from .errors import UsageError

PathLike = Union[str, "os.PathLike[str]"]

PARTIAL_PREFIX = ".partial-"
SF_DATALESS = 0x40000000              # st_flags bit of an iCloud file that isn't downloaded (reading it blocks)
_CHUNK = 1 << 20
# link()'s answers for "this filesystem has no hard links".
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS, errno.EMLINK})

_LIVE_PARTIALS: set[Path] = set()
_LIVE_LOCK = threading.Lock()


def _fsync_fd(fd: int) -> None:
    """Flush a file to stable storage. On macOS plain fsync stops at the drive's cache."""
    if sys.platform == "darwin":
        import fcntl

        full = getattr(fcntl, "F_FULLFSYNC", None)
        if full is not None:
            try:
                fcntl.fcntl(fd, full)
                return
            except OSError:
                pass
    os.fsync(fd)


def _fsync_dir(folder: Path) -> None:
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _unlink_quietly(p: Path) -> None:
    try:
        os.unlink(p)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _exists_error(dest: Path) -> UsageError:
    return UsageError("output_exists", out=dest.name, path=str(dest))


def atomic_write(path: PathLike, data: bytes, *, overwrite: bool, mode: int | None = None) -> None:
    """Write ``data`` to ``path`` all at once: readers see the old file or the whole new one, never part of one.

    ``overwrite=False`` never replaces an existing file (UsageError ``output_exists``, exit 2, and nothing is
    changed). ``mode`` (e.g. 0o600 for a voiceprint) is set exactly, whatever the umask; without it a new file
    gets the usual 0o666 less the umask.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("atomic_write takes bytes")
    dest = Path(path)
    folder = dest.parent
    partial = folder / f"{PARTIAL_PREFIX}{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(partial, flags, 0o666 if mode is None else mode)
    with _LIVE_LOCK:
        _LIVE_PARTIALS.add(partial)
    try:
        try:
            if mode is not None:
                os.fchmod(fd, mode)
            _write_all(fd, bytes(data))
            _fsync_fd(fd)
        finally:
            os.close(fd)
        if overwrite:
            os.replace(partial, dest)
        else:
            try:
                os.link(partial, dest)
            except FileExistsError:
                raise _exists_error(dest) from None
            except OSError as exc:
                if exc.errno not in _NO_HARD_LINKS:
                    raise
                _create_exclusive(dest, bytes(data), mode)
            _unlink_quietly(partial)
        _fsync_dir(folder)
    finally:
        if os.path.lexists(partial):
            _unlink_quietly(partial)
        with _LIVE_LOCK:
            _LIVE_PARTIALS.discard(partial)


def _create_exclusive(dest: Path, data: bytes, mode: int | None) -> None:
    """The no-hard-links fallback: the name must still be free, and O_EXCL makes the creation itself exclusive."""
    if os.path.lexists(dest):
        raise _exists_error(dest)
    try:
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                     0o666 if mode is None else mode)
    except FileExistsError:
        raise _exists_error(dest) from None
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        _write_all(fd, data)
        _fsync_fd(fd)
    finally:
        os.close(fd)


def remove_live_partials() -> int:
    """Delete every partial file a write in this process has open (the SIGTERM/SIGHUP path); returns how many."""
    with _LIVE_LOCK:
        partials = list(_LIVE_PARTIALS)
    n = 0
    for p in partials:
        if os.path.lexists(p):
            _unlink_quietly(p)
            n += 1
    return n


def probe_writable(folder: PathLike) -> bool:
    """Whether a file can be created in ``folder``: create and remove a ``.partial-<random>`` file there."""
    p = Path(folder) / f"{PARTIAL_PREFIX}{secrets.token_hex(8)}"
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    except OSError:
        return False
    os.close(fd)
    _unlink_quietly(p)
    return True


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def private_dir(path: PathLike) -> Path:
    """Create ``path`` (and missing parents) mode 0700, or set an existing folder to 0700; return it.

    Parents that already exist keep their mode. A path that exists and isn't a folder is NotADirectoryError.
    """
    p = Path(path)
    missing = []
    q = p
    while not q.exists() and q != q.parent:
        missing.append(q)
        q = q.parent
    for d in reversed(missing):
        try:
            os.mkdir(d, 0o700)
        except FileExistsError:
            pass
        if d.is_dir():
            os.chmod(d, 0o700)
    if not p.is_dir():
        raise NotADirectoryError(f"{p} exists and isn't a folder")
    if stat.S_IMODE(os.stat(p).st_mode) != 0o700:
        os.chmod(p, 0o700)
    return p


def is_dataless(path: PathLike) -> bool:
    """True for an iCloud file that isn't downloaded to this Mac (checked without reading it)."""
    return bool(getattr(os.stat(path), "st_flags", 0) & SF_DATALESS)


def utf8_stdio() -> None:
    """Write UTF-8 to stdout and stderr whatever the locale, so a file name in any script can't crash a run."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def stem(name: str) -> str:
    """A recording's stem: the name up to its last dot ("Team sync.m4a" -> "Team sync"), or the whole name when
    it has no dot. The same as the shell's ``${name%.*}``, so ".m4a" gives "" and "a.b.c" gives "a.b"."""
    i = name.rfind(".")
    return name[:i] if i != -1 else name
