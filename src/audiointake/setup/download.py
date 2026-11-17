"""Setup's downloader: one pinned file over HTTPS, resumable, checked, placed where it belongs.

Run by setup as a child, one file per child, so a stall or a crash costs one process::

    python -I -m audiointake.setup.download --url URL --dest FINAL --size N (--sha256 HEX | --git-blob HEX)
        [--lock FILE] [--link PATH] [--range START END] [--allow-host HOST]...

* The file is written to ``FINAL.audiointake-partial`` and every attempt resumes from that file's length with
  ``Range: bytes=N-`` (the parent's watchdog kills a child whose partial stops growing, and starts another). A
  server that answers ``200`` instead of ``206`` restarts the file from the beginning.
* ``--lock`` names huggingface_hub's own lock file for the blob (``<hf cache>/.locks/models--…/<blob>.lock``),
  held with ``flock`` while the file is written, so a huggingface_hub download of the same blob waits for us and
  we for it.
* The finished file is checked against ``--size`` and its SHA-256 (``--sha256``) or git blob id (``--git-blob``,
  for small non-LFS files), then renamed to ``FINAL``; ``--link`` adds a relative symlink to it (the
  ``snapshots/<revision>/<path>`` link the Hugging Face cache layout needs). A file that doesn't match is deleted
  at once and nothing is placed (exit 10).
* While it waits for the lock or hashes a file (minutes for a large file on a slow disk), the child touches
  ``FINAL.audiointake-alive`` every few seconds, so the parent's watchdog, which otherwise watches the partial
  file grow, doesn't take a long wait or a long hash for a stall. A stalled network read touches nothing.
* ``--range START END`` fetches bytes [START, END) of the remote file (the canary clip).
* No token or account is ever sent: no ``Authorization`` or ``Cookie`` header (an audit hook refuses one before it
  leaves), ``HF_TOKEN`` and stored logins are never read, ``HF_ENDPOINT`` is ignored, and Xet's protocol is never
  used. Redirects are followed only to the hosts allowed for the starting host (``--allow-host``, default
  ``manifest.REDIRECT_HOSTS``), over HTTPS. A user's ``https_proxy`` is honoured: it is their route out, and the
  pinned hashes make tampering moot.

Prints one JSON line on stdout: ``{"status": "downloaded"|"present", "fetched", "resumed_from", "restarts",
"hosts", "headers"}`` (``headers``: the names of every request header actually sent, never their values).
Exit 0 on success; 10 the file doesn't match its pin; 11 a network error; 12 a redirect off the allowed hosts;
13 an HTTP error status; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import manifest

PARTIAL_SUFFIX = ".audiointake-partial"
ALIVE_SUFFIX = ".audiointake-alive"
BAD_SUFFIX = ".audiointake-bad"            # what earlier versions left; setup sweeps it
BEAT_EVERY_S = 2.0
CHUNK = 1 << 20
READ_TIMEOUT_S = 60.0
USER_AGENT = "audiointake-setup"
EXIT_MISMATCH, EXIT_NETWORK, EXIT_REDIRECT, EXIT_HTTP, EXIT_RANGE = 10, 11, 12, 13, 14
FORBIDDEN_HEADERS = frozenset({"authorization", "cookie"})
_SENT: set[str] = set()
_ALIVE: dict = {"path": None, "t": 0.0}


def alive_path(dest: Path) -> Path:
    """The heartbeat file the child touches while it waits or hashes (the parent's watchdog reads its mtime)."""
    return Path(dest).with_name(Path(dest).name + ALIVE_SUFFIX)


def _beat() -> None:
    """Touch the heartbeat file, at most every ``BEAT_EVERY_S`` (a no-op outside the child)."""
    p = _ALIVE["path"]
    if p is None:
        return
    import time

    now = time.monotonic()
    if now - _ALIVE["t"] < BEAT_EVERY_S:
        return
    _ALIVE["t"] = now
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
        os.utime(p)
    except OSError:
        pass


class Mismatch(Exception):
    """The file (or what the server says about it) doesn't match its pin."""


class RedirectRefused(Exception):
    pass


class WrongRange(Exception):
    """The server answered a resume with a different byte range than asked for. That says nothing about the file
    (a proxy or CDN that mishandles ranges), so the partial is dropped and the parent tries again from the start;
    it is never ``model_changed``."""


def _audit(event: str, args: tuple) -> None:
    """Record the name of every request header http.client sends, and refuse a credential header outright."""
    if event != "http.client.send" or len(args) < 2:
        return
    data = args[1]
    if not isinstance(data, (bytes, bytearray)):
        return
    head = bytes(data).split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    if not lines or b" HTTP/1." not in lines[0]:
        return
    for line in lines[1:]:
        name = line.split(b":", 1)[0].strip().decode("latin-1").lower()
        if not name:
            continue
        if name in FORBIDDEN_HEADERS:
            raise PermissionError(f"refusing to send a {name} header")
        _SENT.add(name)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _allowed(host: str, allow: tuple[str, ...]) -> bool:
    return manifest.host_allowed(host, allow)


def _loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow: tuple[str, ...], hosts: list[str], insecure_loopback: bool) -> None:
        self.allow, self.hosts, self.insecure_loopback = allow, hosts, insecure_loopback

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        host = (parts.hostname or "").lower()
        scheme_ok = parts.scheme == "https" or (self.insecure_loopback and parts.scheme == "http" and _loopback(host))
        if not scheme_ok or not _allowed(host, self.allow):
            raise RedirectRefused(host or newurl[:60])
        if host not in self.hosts:
            self.hosts.append(host)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            for h in ("Authorization", "Cookie"):
                new.headers.pop(h, None)
                new.unredirected_hdrs.pop(h, None)
        return new


def _opener(allow: tuple[str, ...], hosts: list[str], insecure_loopback: bool) -> urllib.request.OpenerDirector:
    op = urllib.request.OpenerDirector()
    for h in (urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ssl.create_default_context()),
              urllib.request.HTTPDefaultErrorHandler(), urllib.request.HTTPErrorProcessor(),
              _Redirects(allow, hosts, insecure_loopback)):
        op.add_handler(h)
    if insecure_loopback:
        op.add_handler(urllib.request.HTTPHandler())
    return op


def _hash(path: Path, git_blob: bool) -> str:
    size = path.stat().st_size
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(b"blob %d\x00" % size)
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(CHUNK), b""):
            h.update(c)
            _beat()
    return h.hexdigest()


def matches(path: Path, size: int, digest: str, git_blob: bool) -> bool:
    """Whether ``path`` is a regular file of ``size`` bytes with that SHA-256 (or git blob id)."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_size == size and _hash(Path(path), git_blob) == digest


def _content_range(value: str | None) -> tuple[int, int | None] | None:
    """(first byte, total or None) from a Content-Range header."""
    if not value or not value.startswith("bytes "):
        return None
    span, _, total = value[6:].partition("/")
    first = span.split("-", 1)[0]
    try:
        return int(first), (None if total.strip() == "*" else int(total))
    except ValueError:
        return None


def _place_link(link: Path, dest: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(dest, link.parent)
    if link.is_symlink() and os.readlink(link) == target:
        return
    tmp = link.parent / f".{link.name}.{os.getpid()}.link"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link)


def _wait_lock(fd: int) -> None:
    """flock ``fd`` exclusively, beating while another holder (a huggingface_hub download) has it."""
    import time

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            _beat()
            time.sleep(0.25)


def fetch(url: str, dest: Path, size: int, digest: str, *, git_blob: bool = False, lock: Path | None = None,
          link: Path | None = None, byte_range: tuple[int, int] | None = None,
          allow: tuple[str, ...] | None = None, insecure_loopback: bool = False) -> dict:
    """Fetch one file (see the module docstring); returns the status dict, raises Mismatch, RedirectRefused,
    urllib.error.HTTPError or OSError."""
    dest = Path(dest)
    start_host = _host(url)
    scheme = urllib.parse.urlsplit(url).scheme
    if not (scheme == "https" or (insecure_loopback and scheme == "http" and _loopback(start_host))):
        raise ValueError("only https URLs are fetched")
    if allow is None:
        allow = manifest.REDIRECT_HOSTS.get(start_host, (start_host,))
    if not _allowed(start_host, allow):
        raise RedirectRefused(start_host)
    if byte_range is not None and byte_range[1] - byte_range[0] != size:
        raise ValueError("--range must span exactly --size bytes")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
    hosts: list[str] = [start_host]
    lock_fd = -1
    if lock is not None:
        Path(lock).parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        _wait_lock(lock_fd)
    try:
        if matches(dest, size, digest, git_blob):
            if link is not None:
                _place_link(link, dest)
            return {"status": "present", "fetched": 0, "resumed_from": 0, "restarts": 0, "hosts": [],
                    "headers": sorted(_SENT)}
        opener = _opener(tuple(allow), hosts, insecure_loopback)
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            have = 0
            partial.unlink()
        resumed_from, restarts, fetched = have, 0, 0
        while have < size:
            first = (byte_range[0] if byte_range else 0) + have
            last = byte_range[0] + size - 1 if byte_range else None
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            if have or byte_range:
                req.add_header("Range", f"bytes={first}-" + (str(last) if last is not None else ""))
            try:
                resp = opener.open(req, timeout=READ_TIMEOUT_S)
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and have and not byte_range:
                    if have == size:
                        break
                    partial.unlink()
                    have, restarts = 0, restarts + 1
                    continue
                raise
            with resp:
                status = resp.status
                skip = 0
                if status == 206:
                    cr = _content_range(resp.headers.get("Content-Range"))
                    if cr is None or cr[0] != first:
                        if have and not byte_range:
                            partial.unlink(missing_ok=True)     # the next attempt starts from the beginning
                        raise WrongRange("the server sent a different byte range than asked for")
                    if not byte_range and cr[1] is not None and cr[1] != size:
                        raise Mismatch(f"the server's file is {cr[1]} bytes; the pin is {size}")
                    mode = "ab"
                elif status == 200:
                    length = resp.headers.get("Content-Length")
                    if not byte_range and length is not None and int(length) != size:
                        raise Mismatch(f"the server's file is {length} bytes; the pin is {size}")
                    if have:
                        restarts += 1
                    have, mode = 0, "wb"
                    skip = byte_range[0] if byte_range else 0
                else:
                    raise urllib.error.HTTPError(url, status, "unexpected status", resp.headers, None)
                with open(partial, mode) as out:
                    while have < size:
                        chunk = resp.read1(CHUNK)          # what has arrived, so a stall keeps it
                        if not chunk:
                            break
                        if skip:
                            cut = min(skip, len(chunk))
                            chunk, skip = chunk[cut:], skip - cut
                            if not chunk:
                                continue
                        chunk = chunk[: size - have]
                        out.write(chunk)
                        out.flush()
                        have += len(chunk)
                        fetched += len(chunk)
            if have < size and status in (200, 206):
                # the connection ended early: try again from where it stopped (the parent caps the attempts)
                raise ConnectionError(f"the connection ended after {have} of {size} bytes")
        if not matches(partial, size, digest, git_blob):
            partial.unlink()
            raise Mismatch("the downloaded file doesn't match its pinned fingerprint")
        os.replace(partial, dest)
        if link is not None:
            _place_link(link, dest)
        return {"status": "downloaded", "fetched": fetched, "resumed_from": resumed_from, "restarts": restarts,
                "hosts": hosts, "headers": sorted(_SENT)}
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m audiointake.setup.download", allow_abbrev=False)
    ap.add_argument("--url", required=True)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--size", required=True, type=int)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sha256")
    g.add_argument("--git-blob")
    ap.add_argument("--lock", type=Path)
    ap.add_argument("--link", type=Path)
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--allow-host", action="append")
    ap.add_argument("--insecure-loopback", action="store_true", help=argparse.SUPPRESS)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    sys.addaudithook(_audit)
    socket.setdefaulttimeout(READ_TIMEOUT_S)
    digest = a.sha256 or a.git_blob
    _ALIVE["path"] = alive_path(a.dest)
    try:
        out = fetch(a.url, a.dest, a.size, digest, git_blob=a.git_blob is not None, lock=a.lock, link=a.link,
                    byte_range=tuple(a.range) if a.range else None,
                    allow=tuple(a.allow_host) if a.allow_host else None, insecure_loopback=a.insecure_loopback)
    except Mismatch as exc:
        print(f"download: {exc}", file=sys.stderr)
        return EXIT_MISMATCH
    except WrongRange as exc:
        print(f"download: {exc}", file=sys.stderr)
        return EXIT_RANGE
    except RedirectRefused as exc:
        print(f"download: a redirect to {exc} is not allowed", file=sys.stderr)
        return EXIT_REDIRECT
    except urllib.error.HTTPError as exc:
        print(f"download: HTTP {exc.code}", file=sys.stderr)
        return EXIT_HTTP
    except (OSError, urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as exc:
        print(f"download: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_NETWORK
    finally:
        try:
            alive_path(a.dest).unlink()
        except OSError:
            pass
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
