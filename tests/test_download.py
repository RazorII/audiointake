"""Setup's downloader against a fake server on 127.0.0.1: resume, stalls, restarts, redirects, hashes, locks, and
never a credential header."""
from __future__ import annotations

import fcntl
import hashlib
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from audiointake.setup import download, store

BODY = bytes((i * 7 + 3) % 251 for i in range(300_000))
# the fake sends 4096-byte pieces, so the stall and cut points below are multiples of 4096
SHA = hashlib.sha256(BODY).hexdigest()
GIT_BLOB = hashlib.sha1(b"blob %d\x00" % len(BODY) + BODY).hexdigest()


class Fake:
    """What the fake server does next; every request it saw (method, path, headers)."""

    def __init__(self) -> None:
        self.mode = "normal"          # normal | ignore_range | wrong_range_once | trickle | stall_after | cut_after
        self.after = 0
        self.body = BODY
        self.requests: list[dict] = []
        self.release = threading.Event()


def _handler(fake: Fake):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            fake.requests.append({"path": self.path, "headers": hdrs})
            if "authorization" in hdrs or "cookie" in hdrs:
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/redirect-ok"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/file")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/redirect-https-off"):
                # a scheme the test allows (loopback http) on a host it doesn't: the host rule must refuse it
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/file")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/redirect-off"):
                self.send_response(302)
                self.send_header("Location", "http://localhost.invalid:1/file")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = fake.body
            start = 0
            rng = hdrs.get("range")
            if rng and fake.mode != "ignore_range":
                spec = rng.split("=", 1)[1]
                a, _, b = spec.partition("-")
                start = int(a)
                end = int(b) if b else len(body) - 1
                if start >= len(body):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(body)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if fake.mode == "wrong_range_once":           # a proxy that mishandles ranges, once
                    fake.mode = "normal"
                    start = 0
                chunk = body[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{len(body)}")
            else:
                chunk = body
                self.send_response(200)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            sent = 0
            step = 4096
            try:
                while sent < len(chunk):
                    if fake.mode in ("stall_after", "cut_after") and start + sent >= fake.after:
                        cut = fake.mode == "cut_after"
                        fake.mode = "normal"                  # the next attempt gets the rest
                        if cut:
                            self.close_connection = True
                            return
                        fake.release.wait(30)
                        return
                    piece = chunk[sent:sent + step]
                    self.wfile.write(piece)
                    self.wfile.flush()
                    sent += len(piece)
                    if fake.mode == "trickle":
                        time.sleep(0.002)
            except (BrokenPipeError, ConnectionResetError):
                return

    return H


@pytest.fixture
def server():
    fake = Fake()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    fake.url = f"http://127.0.0.1:{srv.server_port}"
    yield fake
    fake.release.set()
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def credentials_everywhere(monkeypatch, tmp_path):
    """A token in every place huggingface_hub would look: the fetcher must send none of them."""
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_SECRETVALUE")
    tok = Path(os.environ["HOME"]) / ".cache" / "huggingface"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "token").write_text("hf_SECRETVALUE")
    yield
    (tok / "token").unlink()


def _fetch(fake, dest, **kw):
    kw.setdefault("allow", ("127.0.0.1",))
    return store.fetch_file(fake.url + kw.pop("path", "/file"), dest, len(fake.body), kw.pop("digest", SHA),
                            extra_args=("--insecure-loopback",), **kw)


def _no_credentials(fake):
    for r in fake.requests:
        assert "authorization" not in r["headers"]
        assert "cookie" not in r["headers"]


def test_fetches_checks_and_places_a_file(server, tmp_path, credentials_everywhere):
    dest = tmp_path / "blobs" / SHA
    link = tmp_path / "snapshots" / "rev" / "sub" / "model.bin"
    out = _fetch(server, dest, link=link, lock_file=tmp_path / ".locks" / f"{SHA}.lock")
    assert out["status"] == "downloaded" and out["fetched"] == len(BODY)
    assert dest.read_bytes() == BODY
    assert link.is_symlink() and os.readlink(link) == "../../../blobs/" + SHA
    assert link.read_bytes() == BODY
    assert not dest.with_name(dest.name + ".audiointake-partial").exists()
    assert "authorization" not in out["headers"] and "user-agent" in out["headers"]
    _no_credentials(server)


def test_present_file_is_not_fetched_again(server, tmp_path):
    dest = tmp_path / SHA
    _fetch(server, dest)
    n = len(server.requests)
    out = _fetch(server, dest)
    assert out["status"] == "present" and len(server.requests) == n


def test_git_blob_ids_check_small_non_lfs_files(server, tmp_path):
    out = _fetch(server, tmp_path / GIT_BLOB, digest=GIT_BLOB, git=True)
    assert out["status"] == "downloaded"


def test_resumes_from_the_partial_with_a_range_request(server, tmp_path):
    dest = tmp_path / SHA
    partial = dest.with_name(dest.name + ".audiointake-partial")
    partial.write_bytes(BODY[:123_456])
    out = _fetch(server, dest)
    assert out["resumed_from"] == 123_456 and out["fetched"] == len(BODY) - 123_456
    assert server.requests[-1]["headers"]["range"] == "bytes=123456-"
    assert dest.read_bytes() == BODY


def test_a_server_answering_200_to_a_range_restarts_the_file(server, tmp_path):
    server.mode = "ignore_range"
    dest = tmp_path / SHA
    dest.with_name(dest.name + ".audiointake-partial").write_bytes(b"x" * 50_000)
    out = _fetch(server, dest)
    assert out["restarts"] == 1 and dest.read_bytes() == BODY


def test_a_resume_answered_with_the_wrong_range_starts_again_and_isnt_model_changed(server, tmp_path):
    server.mode = "wrong_range_once"
    dest = tmp_path / SHA
    dest.with_name(dest.name + ".audiointake-partial").write_bytes(BODY[:123_456])
    out = _fetch(server, dest)
    assert out["status"] == "downloaded" and dest.read_bytes() == BODY
    ranges = [r["headers"].get("range") for r in server.requests]
    assert ranges == ["bytes=123456-", None]                    # the second attempt started from the beginning


def test_a_trickle_completes(server, tmp_path):
    server.mode = "trickle"
    assert _fetch(server, tmp_path / SHA)["status"] == "downloaded"


def test_a_stalled_download_is_killed_and_resumed(server, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WATCH_EVERY_S", 0.2)
    monkeypatch.setattr(store, "STALL_S", 1.5)
    server.mode, server.after = "stall_after", 102_400
    dest = tmp_path / SHA
    t0 = time.monotonic()
    out = _fetch(server, dest)
    assert time.monotonic() - t0 < 20
    ranges = [r["headers"].get("range") for r in server.requests]
    assert ranges[0] is None and ranges[1] == "bytes=102400-"     # the second attempt resumed from the partial
    assert out["status"] == "downloaded" and dest.read_bytes() == BODY


def test_a_cut_connection_is_resumed(server, tmp_path):
    server.mode, server.after = "cut_after", 200_704
    out = _fetch(server, tmp_path / SHA)
    assert out["status"] == "downloaded"
    assert any(r["headers"].get("range") == "bytes=200704-" for r in server.requests)


def test_a_redirect_on_the_allowed_host_is_followed(server, tmp_path):
    out = _fetch(server, tmp_path / SHA, path="/redirect-ok")
    assert out["status"] == "downloaded" and [r["path"] for r in server.requests] == ["/redirect-ok", "/file"]


def test_a_redirect_off_the_allowed_hosts_is_refused(server, tmp_path, network_attempts):
    with pytest.raises(store.DownloadFailed) as e:
        _fetch(server, tmp_path / SHA, path="/redirect-off")
    assert e.value.code == "download_failed"
    assert len(server.requests) == 1                          # refused before anything was asked of the other host
    assert not (tmp_path / SHA).exists()


def test_a_wrong_hash_is_model_changed_and_nothing_is_placed_or_kept(server, tmp_path):
    dest = tmp_path / ("0" * 64)
    with pytest.raises(store.DownloadFailed) as e:
        _fetch(server, dest, digest="0" * 64)
    assert e.value.code == "model_changed"
    assert sorted(p.name for p in tmp_path.iterdir()) == []      # no file, no partial, no heartbeat, no .bad


def test_a_size_the_server_disagrees_with_fails_early(server, tmp_path):
    with pytest.raises(store.DownloadFailed) as e:
        store.fetch_file(server.url + "/file", tmp_path / SHA, len(BODY) + 1, SHA, allow=("127.0.0.1",),
                         extra_args=("--insecure-loopback",))
    assert e.value.code == "model_changed"


def test_never_sends_a_credential_header_even_if_asked(monkeypatch):
    with pytest.raises(PermissionError):
        download._audit("http.client.send", (None, b"GET /x HTTP/1.1\r\nHost: h\r\nAuthorization: Bearer t\r\n\r\n"))
    download._audit("http.client.send", (None, b"GET /x HTTP/1.1\r\nHost: h\r\nRange: bytes=1-\r\n\r\n"))
    assert {"host", "range"} <= download._SENT


def test_a_redirect_to_another_host_is_refused_by_the_host_rule(server, tmp_path):
    """The scheme passes (loopback http is allowed here), so it is the one host rule that refuses."""
    with pytest.raises(store.DownloadFailed) as e:
        _fetch(server, tmp_path / SHA, path="/redirect-https-off")
    assert e.value.code == "download_failed"
    assert [r["path"] for r in server.requests] == ["/redirect-https-off"]  # never contacted under the other name


def test_one_host_rule():
    from audiointake.setup import manifest

    assert download._allowed("cdn-lfs.hf.co", (".hf.co",)) and manifest.redirect_allowed("huggingface.co",
                                                                                          "cas-bridge.xethub.hf.co")
    assert not download._allowed("hf.co.evil.example", (".hf.co",))
    assert not manifest.redirect_allowed("huggingface.co", "github.com")


def test_a_long_wait_for_the_lock_isnt_taken_for_a_stall(server, tmp_path, monkeypatch):
    """The child beats while it waits (or hashes), so the watchdog leaves it alone: one attempt, no restart."""
    monkeypatch.setattr(store, "WATCH_EVERY_S", 0.2)
    monkeypatch.setattr(store, "STALL_S", 3.0)                 # longer than the child's 2 s between beats
    lockf = tmp_path / ".locks" / f"{SHA}.lock"
    lockf.parent.mkdir(parents=True)
    fd = os.open(lockf, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    dest = tmp_path / SHA
    result: dict = {}
    t = threading.Thread(target=lambda: result.update(_fetch(server, dest, lock_file=lockf)))
    t.start()
    time.sleep(7.0)                                             # more than twice STALL_S
    assert t.is_alive() and not server.requests
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    t.join(30)
    assert result.get("status") == "downloaded" and len(server.requests) == 1
    assert not download.alive_path(dest).exists()


def test_a_hung_read_is_still_a_stall_with_the_heartbeat(server, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "WATCH_EVERY_S", 0.2)
    monkeypatch.setattr(store, "STALL_S", 1.5)
    server.mode, server.after = "stall_after", 4096
    out = _fetch(server, tmp_path / SHA)
    assert out["status"] == "downloaded" and len(server.requests) == 2


def test_only_https_is_fetched(tmp_path):
    with pytest.raises(ValueError):
        download.fetch("http://example.invalid/x", tmp_path / "x", 1, SHA)


def test_holds_hugging_faces_lock_file_while_writing(server, tmp_path):
    lockf = tmp_path / ".locks" / f"{SHA}.lock"
    lockf.parent.mkdir(parents=True)
    fd = os.open(lockf, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    dest = tmp_path / SHA
    result: dict = {}
    t = threading.Thread(target=lambda: result.update(_fetch(server, dest, lock_file=lockf)))
    t.start()
    time.sleep(1.5)
    assert not dest.exists() and not server.requests            # waiting for the other holder
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    t.join(30)
    assert result.get("status") == "downloaded"


def test_a_sigkilled_child_leaves_only_a_partial_that_the_next_run_resumes(server, tmp_path):
    server.mode, server.after = "stall_after", 151_552
    dest = tmp_path / SHA
    partial = dest.with_name(dest.name + ".audiointake-partial")
    cmd = [sys.executable, "-I", "-B", "-m", "audiointake.setup.download", "--url", server.url + "/file", "--dest",
           str(dest), "--size", str(len(BODY)), "--sha256", SHA, "--allow-host", "127.0.0.1", "--insecure-loopback"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and (not partial.exists() or partial.stat().st_size < 151_552):
        time.sleep(0.05)
    p.send_signal(signal.SIGKILL)
    p.wait(10)
    assert partial.stat().st_size == 151_552 and not dest.exists()
    server.mode = "normal"
    server.release.set()
    out = _fetch(server, dest)
    assert out["resumed_from"] == 151_552 and dest.read_bytes() == BODY


def test_stale_partials_are_swept(tmp_path):
    hub = Path(os.environ["HF_HUB_CACHE"]) / "models--x--y" / "blobs"
    hub.mkdir(parents=True, exist_ok=True)
    old, new = hub / "a.audiointake-partial", hub / "b.audiointake-partial"
    old.write_bytes(b"1")
    new.write_bytes(b"2")
    t = time.time() - 2 * 86400
    os.utime(old, (t, t))
    assert store.sweep_partials() == 1
    assert not old.exists() and new.exists()
    new.unlink()


def test_the_child_reports_json_and_exit_codes(server, tmp_path):
    base = [sys.executable, "-I", "-B", "-m", "audiointake.setup.download", "--allow-host", "127.0.0.1",
            "--insecure-loopback", "--size", str(len(BODY))]
    ok = subprocess.run(base + ["--url", server.url + "/file", "--dest", str(tmp_path / "a"), "--sha256", SHA],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)["status"] == "downloaded"
    bad = subprocess.run(base + ["--url", server.url + "/file", "--dest", str(tmp_path / "b"), "--sha256", "1" * 64],
                         capture_output=True, text=True)
    assert bad.returncode == download.EXIT_MISMATCH
    off = subprocess.run(base + ["--url", server.url + "/redirect-off", "--dest", str(tmp_path / "c"),
                                 "--sha256", SHA], capture_output=True, text=True)
    assert off.returncode == download.EXIT_REDIRECT
    assert subprocess.run(base, capture_output=True).returncode == 2
