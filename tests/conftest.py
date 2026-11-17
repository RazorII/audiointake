"""Shared test setup.

* **Scratch homes.** Before collection, ``HOME``, ``AUDIOINTAKE_HOME``, ``AUDIOINTAKE_CACHE`` and ``HF_HUB_CACHE``
  point into one temporary folder for the whole session (``HF_HOME`` and ``HUGGINGFACE_HUB_CACHE`` are removed),
  so no test can touch the real Application Support, Caches, Logs or Hugging Face cache. Moving only one of them
  would let a test reach the others. Variables the engine's old command-line switches used (``PARAKEET_*``,
  ``THAI*``, ``WHISPER_*``, ``SPEAKER_ID_*``), Hugging Face tokens and the run lock's hand-over marker are removed
  too. The values from before are kept in ``REAL_HOME`` and ``REAL_ENV`` and restored at the end.
* **No network.** Every test not marked ``network`` runs with an audit hook that refuses socket connects, binds
  and sends outside loopback and every DNS lookup except ``localhost`` (``NetworkBlocked``, an OSError). The guard
  covers this process only, so the proxy variables point at a closed loopback port (``DEAD_PROXY``) with only
  loopback exempt: a child process (setup's downloader, git) that tries the internet fails fast instead of going
  out directly. A connect to that port from this process counts as a refused attempt too. Each refused attempt is
  recorded: the ``network_attempts`` fixture lists them, so a test can prove nothing tried even when the code
  under test swallowed the error.
* **Measured memory.** Every test sees a Mac with ``TEST_MEMORY_GB`` of memory, more than
  ``platforms.MEASURED_MEMORY_GB``, so no assertion depends on the machine running the suite: GitHub's hosted Macs
  have 14 GB, which adds the ``memory_unmeasured`` warning to every doctor and setup report. A test of the warning
  sets ``platforms.memory_gb`` itself; a test of the real reading is marked ``host_memory``, and tests marked
  ``models`` see the real machine.
* **Public audio** is reached only through the ``pinned_audio`` fixture: ``pinned_audio(key)`` returns the
  verified path of the clip ``tools/fetch_test_audio.py`` pins under ``key`` (its module-level ``AUDIO`` mapping:
  ``key -> {"file": relative path, "bytes": int, "sha256": hex}``), under ``AUDIOINTAKE_TEST_AUDIO`` or
  ``~/.audiointake-dev/audio`` (both read from the real environment, not the scratch HOME). A missing or
  different file skips the test, naming the fetch command; an unknown key is an error in the test.
* **Real models** (tests marked ``models`` only): the ``models_env`` fixture is the environment for running the
  installed CLI as a child against a set-up data folder, ``AUDIOINTAKE_TEST_HOME`` (from the real environment),
  with the real Hugging Face cache and a scratch HOME per test. Without that variable the test skips. A module-level skip would count as a
  skip under ``-m models``, so models tests use per-test ``skipif`` and these fixtures only.

This file never imports audiointake at module level, so it also works where the package can't be imported.
"""
from __future__ import annotations

import errno
import hashlib
import importlib.util
import ipaddress
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_ENV: dict[str, str] = dict(os.environ)
REAL_HOME = Path(os.path.expanduser("~"))

SCRATCH_VARS = ("HOME", "AUDIOINTAKE_HOME", "AUDIOINTAKE_CACHE", "HF_HUB_CACHE")
REMOVED_VARS = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT",
                "AUDIOINTAKE_TEST_HOME", "AUDIOINTAKE_RUN_LOCK_HOLDER")
REMOVED_PREFIXES = ("PARAKEET_", "THAI", "WHISPER_", "SPEAKER_ID_")
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
DEAD_PROXY = ("127.0.0.1", 9)                  # the discard port; nothing listens there, so a connect is refused
DEAD_PROXY_URL = f"http://{DEAD_PROXY[0]}:{DEAD_PROXY[1]}"
LOOPBACK_ONLY = "127.0.0.1,localhost,::1"


# --------------------------------------------------------------------------- the network guard
class NetworkBlocked(OSError):
    """Raised for a socket operation outside loopback in a test not marked ``network``."""


_GUARD: dict[str, Any] = {"on": False, "attempts": []}
_ADDRESS_EVENTS = {"socket.connect": 1, "socket.bind": 1, "socket.sendto": 1, "socket.sendmsg": 1}
_LOOKUP_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                  "socket.gethostbyaddr", "socket.getnameinfo"}


def _loopback_host(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    h = host.strip().lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _allowed_address(address: Any) -> bool:
    if address is None or isinstance(address, (str, bytes, bytearray)):
        return True                                   # AF_UNIX path, or a send on a connected socket
    if isinstance(address, tuple) and address:
        if len(address) > 1 and (str(address[0]), address[1]) == DEAD_PROXY:
            return False                              # going through the dead proxy is reaching for the internet
        return _loopback_host(address[0])
    return False


def _network_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _GUARD["on"] or not event.startswith("socket."):
        return
    if event in _ADDRESS_EVENTS:
        address = args[_ADDRESS_EVENTS[event]] if len(args) > 1 else None
        if _allowed_address(address):
            return
    elif event in _LOOKUP_EVENTS:
        if event == "socket.getnameinfo":
            target = args[0][0] if args and isinstance(args[0], tuple) and args[0] else None
        else:
            target = args[0] if args else None
        if _loopback_host(target):
            return
    else:
        return
    _GUARD["attempts"].append((event, repr(args)[:200]))
    raise NetworkBlocked(errno.ENETUNREACH, f"network blocked in tests ({event}); mark the test 'network' if it needs it")


sys.addaudithook(_network_hook)


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("network") is not None:
        yield
        return
    for name in PROXY_VARS:
        monkeypatch.setenv(name, DEAD_PROXY_URL)
    monkeypatch.setenv("NO_PROXY", LOOPBACK_ONLY)
    monkeypatch.setenv("no_proxy", LOOPBACK_ONLY)
    _GUARD["attempts"] = []
    _GUARD["on"] = True
    try:
        yield
    finally:
        _GUARD["on"] = False


@pytest.fixture
def network_attempts() -> list[tuple[str, str]]:
    """The socket operations refused so far in this test: (audit event, its arguments)."""
    return _GUARD["attempts"]


# --------------------------------------------------------------------------- measured memory
TEST_MEMORY_GB = 64


@pytest.fixture(autouse=True)
def _measured_memory(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("host_memory") or request.node.get_closest_marker("models"):
        return
    try:
        from audiointake import platforms
    except ImportError:                     # a test that runs where the package can't be imported
        return
    monkeypatch.setattr(platforms, "memory_gb", lambda: TEST_MEMORY_GB)


# --------------------------------------------------------------------------- scratch homes
_SESSION: dict[str, Any] = {}


def session_root() -> Path:
    """The session's scratch folder (HOME and the three caches live under it)."""
    return Path(_SESSION["root"])


SESSION_PREFIX = "audiointake-test-"
STALE_SESSION_S = 24 * 3600


def sweep_stale_sessions(now: float | None = None) -> int:
    """Delete session folders a killed test run left behind (its own cleanup never ran), once a day old: no live
    session is that old. Only this suite's own folders, directly in the temp folder, never through a link."""
    import time
    cutoff = (time.time() if now is None else now) - STALE_SESSION_S
    n = 0
    for p in Path(tempfile.gettempdir()).glob(SESSION_PREFIX + "*"):
        try:
            if p.is_dir() and not p.is_symlink() and p.stat().st_mtime < cutoff and p.stat().st_uid == os.getuid():
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n


def pytest_configure(config: pytest.Config) -> None:
    # Before collection, so module-level code in test files sees the scratch values too.
    if "root" in _SESSION:
        return
    sweep_stale_sessions()
    root = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX)).resolve()
    _SESSION["root"] = str(root)
    _SESSION["saved"] = {k: os.environ.get(k) for k in os.environ}
    paths = {"HOME": root / "home", "AUDIOINTAKE_HOME": root / "data", "AUDIOINTAKE_CACHE": root / "cache",
             "HF_HUB_CACHE": root / "hf" / "hub"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    for name in list(os.environ):
        if name in REMOVED_VARS or name.startswith(REMOVED_PREFIXES):
            del os.environ[name]
    for name, p in paths.items():
        os.environ[name] = str(p)


def pytest_unconfigure(config: pytest.Config) -> None:
    root = _SESSION.pop("root", None)
    saved = _SESSION.pop("saved", None)
    if saved is not None:
        for name in list(os.environ):
            if name not in saved:
                del os.environ[name]
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    if root:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- pinned public audio
AUDIO_ENV = "AUDIOINTAKE_TEST_AUDIO"
FETCH_TOOL = REPO / "tools" / "fetch_test_audio.py"
_VERIFIED: dict[tuple[str, int, int, int], str] = {}


def audio_root() -> Path:
    """``AUDIOINTAKE_TEST_AUDIO`` from the real environment, else ``~/.audiointake-dev/audio`` in the real home."""
    env = REAL_ENV.get(AUDIO_ENV)
    return Path(env).expanduser() if env else REAL_HOME / ".audiointake-dev" / "audio"


def _fetch_tool() -> ModuleType | None:
    name = "_audiointake_fetch_test_audio"
    if name in sys.modules:
        return sys.modules[name]
    if not FETCH_TOOL.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, FETCH_TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def _sha256(path: Path, st: os.stat_result) -> str:
    memo = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if memo not in _VERIFIED:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _VERIFIED[memo] = h.hexdigest()
    return _VERIFIED[memo]


def locate_audio(key: str) -> tuple[Path | None, str]:
    """(verified path, "") for a pinned clip, or (None, why not). LookupError for a key the tool doesn't pin."""
    tool = _fetch_tool()
    if tool is None:
        return None, "tools/fetch_test_audio.py isn't in this checkout"
    pins = getattr(tool, "AUDIO", None)
    if not isinstance(pins, dict):
        return None, "tools/fetch_test_audio.py has no AUDIO mapping"
    if key not in pins:
        raise LookupError(f"tools/fetch_test_audio.py pins no clip {key!r}")
    item = pins[key]
    path = audio_root() / str(item["file"])
    label = f"{key} ({path})"
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, f"{label} is missing"
    except OSError as exc:
        return None, f"{label} can't be checked: {exc.strerror or exc}"
    if not stat.S_ISREG(st.st_mode):
        return None, f"{label} isn't a regular file"
    if st.st_size != int(item["bytes"]):
        return None, f"{label} is {st.st_size} bytes; the pin is {item['bytes']}"
    got = _sha256(path, st)
    if got != item["sha256"]:
        return None, f"{label} has SHA-256 {got[:12]}...; the pin is {str(item['sha256'])[:12]}..."
    return path, ""


@pytest.fixture
def pinned_audio() -> Callable[[str], Path]:
    """``pinned_audio(key)``: the verified pinned clip, or a skip naming it and the command that fetches it."""
    def get(key: str) -> Path:
        path, why = locate_audio(key)
        if path is None:
            pytest.skip(f"pinned test audio not available: {why}. Fetch it with: python3 tools/fetch_test_audio.py "
                        f"(root: {AUDIO_ENV} or ~/.audiointake-dev/audio)")
        return path
    return get


# --------------------------------------------------------------------------- real models (marker "models")
MODELS_HOME_ENV = "AUDIOINTAKE_TEST_HOME"


def real_hf_cache() -> Path:
    """The Hugging Face hub cache of the real environment, resolved as huggingface_hub resolves it."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if REAL_ENV.get(name):
            return Path(os.path.abspath(os.path.expanduser(REAL_ENV[name])))
    if REAL_ENV.get("HF_HOME"):
        return Path(os.path.abspath(os.path.expanduser(REAL_ENV["HF_HOME"]))) / "hub"
    xdg = REAL_ENV.get("XDG_CACHE_HOME")
    base = Path(os.path.abspath(os.path.expanduser(xdg))) if xdg else REAL_HOME / ".cache"
    return base / "huggingface" / "hub"


@pytest.fixture
def models_env(request: pytest.FixtureRequest, tmp_path: Path) -> dict[str, str]:
    """The environment for running the installed CLI as a child in a ``models`` test: the real PATH, the set-up
    data folder ``AUDIOINTAKE_TEST_HOME``, the real Hugging Face cache, and per test a fresh HOME (so the run's log
    lands in ``<HOME>/Library/Logs/audiointake`` where the test can read it), cache folder and TMPDIR."""
    if request.node.get_closest_marker("models") is None:
        pytest.fail("models_env is for tests marked models", pytrace=False)
    home = REAL_ENV.get(MODELS_HOME_ENV)
    if not home:
        pytest.skip(f"{MODELS_HOME_ENV} isn't set: point it at a data folder where `audiointake setup` has run")
    cache, home_dir = tmp_path / "cache", tmp_path / "home"
    cache.mkdir()
    home_dir.mkdir()
    return {
        "PATH": REAL_ENV.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home_dir),
        "USER": REAL_ENV.get("USER", ""),
        "TMPDIR": str(tmp_path),
        "LANG": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AUDIOINTAKE_HOME": str(Path(home).expanduser()),
        "AUDIOINTAKE_CACHE": str(cache),
        "HF_HUB_CACHE": str(real_hf_cache()),
    }
