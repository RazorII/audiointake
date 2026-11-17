"""tests/conftest.py itself: scratch homes, the network guard, the pinned-audio lookup, the real HF cache."""
from __future__ import annotations

import hashlib
import os
import socket
import sys

import pytest

import conftest


def test_session_points_every_home_at_scratch():
    root = conftest.session_root()
    for name in conftest.SCRATCH_VARS:
        assert os.environ[name].startswith(str(root)), name
    for name in os.environ:
        assert name not in conftest.REMOVED_VARS
        assert not name.startswith(conftest.REMOVED_PREFIXES), name
    assert conftest.REAL_HOME != conftest.session_root()


def test_network_is_blocked(network_attempts):
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 443)
    assert network_attempts and network_attempts[-1][0] == "socket.getaddrinfo"
    for v in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "ALL_PROXY"):
        assert os.environ[v] == conftest.DEAD_PROXY_URL
    assert os.environ["NO_PROXY"] == os.environ["no_proxy"] == conftest.LOOPBACK_ONLY


def test_a_child_cannot_reach_the_internet_through_the_proxy_variables(tmp_path):
    """The guard is per process; a child inherits the proxy variables, which point at a closed loopback port."""
    import subprocess
    import sys
    code = ("import urllib.request, sys\n"
            "try:\n    urllib.request.urlopen('https://example.com/', timeout=5)\n"
            "except OSError as e:\n    print(type(e).__name__); sys.exit(0)\n"
            "sys.exit(1)")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], env=dict(os.environ), capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_dead_proxy_counts_as_an_attempt_in_this_process(network_attempts):
    s = socket.socket()
    try:
        with pytest.raises(OSError):
            s.connect(conftest.DEAD_PROXY)
    finally:
        s.close()
    assert network_attempts and network_attempts[-1][0] == "socket.connect"


def test_loopback_stays_open():
    a, b = socket.socketpair()
    a.close(), b.close()
    assert socket.getaddrinfo("localhost", 80)


def _fake_tool(tmp_path, pins):
    tool = tmp_path / "fetch_test_audio.py"
    tool.write_text(f"AUDIO = {pins!r}\n")
    return tool


@pytest.fixture
def isolated_tool(monkeypatch):
    monkeypatch.delitem(sys.modules, "_audiointake_fetch_test_audio", raising=False)
    yield
    sys.modules.pop("_audiointake_fetch_test_audio", None)


def test_pinned_audio_without_the_tool(monkeypatch, tmp_path, isolated_tool):
    monkeypatch.setattr(conftest, "FETCH_TOOL", tmp_path / "missing.py")
    path, why = conftest.locate_audio("anything")
    assert path is None and "isn't in this checkout" in why


def test_pinned_audio_found_wrong_and_unknown(monkeypatch, tmp_path, isolated_tool):
    root = tmp_path / "audio"
    root.mkdir()
    data = b"RIFF fake clip"
    (root / "clip.wav").write_bytes(data)
    pins = {"clip": {"file": "clip.wav", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
            "short": {"file": "clip.wav", "bytes": len(data) + 1, "sha256": "0" * 64},
            "absent": {"file": "nope.wav", "bytes": 1, "sha256": "0" * 64}}
    monkeypatch.setattr(conftest, "FETCH_TOOL", _fake_tool(tmp_path, pins))
    monkeypatch.setitem(conftest.REAL_ENV, conftest.AUDIO_ENV, str(root))
    assert conftest.locate_audio("clip") == (root / "clip.wav", "")
    assert "bytes" in conftest.locate_audio("short")[1]
    assert "missing" in conftest.locate_audio("absent")[1]
    with pytest.raises(LookupError):
        conftest.locate_audio("not-pinned")


def test_audio_root_defaults_to_the_real_home(monkeypatch):
    monkeypatch.delitem(conftest.REAL_ENV, conftest.AUDIO_ENV, raising=False)
    assert conftest.audio_root() == conftest.REAL_HOME / ".audiointake-dev" / "audio"


@pytest.mark.parametrize("env,want", [
    ({"HF_HUB_CACHE": "/a/hub"}, "/a/hub"), ({"HUGGINGFACE_HUB_CACHE": "/l"}, "/l"),
    ({"HF_HOME": "/h"}, "/h/hub"), ({"XDG_CACHE_HOME": "/x"}, "/x/huggingface/hub"), ({}, None),
])
def test_real_hf_cache(monkeypatch, env, want):
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delitem(conftest.REAL_ENV, k, raising=False)
    for k, v in env.items():
        monkeypatch.setitem(conftest.REAL_ENV, k, v)
    got = conftest.real_hf_cache()
    assert str(got) == (want or str(conftest.REAL_HOME / ".cache" / "huggingface" / "hub"))


def test_a_killed_sessions_folder_is_swept_once_a_day_old(tmp_path, monkeypatch):
    import tempfile
    import time
    import conftest
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    old, new, other = (tmp_path / "audiointake-test-old"), (tmp_path / "audiointake-test-new"), (tmp_path / "keep-me")
    for d in (old, new, other):
        (d / "data").mkdir(parents=True)
    (tmp_path / "audiointake-test-link").symlink_to(other)
    day_ago = time.time() - 2 * 24 * 3600
    for d in (old, other):
        os.utime(d, (day_ago, day_ago))
    assert conftest.sweep_stale_sessions() == 1
    assert not old.exists() and new.is_dir() and other.is_dir() and (tmp_path / "audiointake-test-link").is_symlink()

