"""Where audiointake runs and where it keeps things.

The checks, in order, each its own exit-3 code: macOS (``platform_not_macos``), Apple silicon
(``platform_not_apple_silicon``), not an Intel Python under Rosetta (``platform_rosetta``), macOS 14 or later
(``macos_too_old``). 16 GB of memory or less is the warning ``memory_unmeasured``, never a refusal. Free disk is
checked against a plan by setup; ffmpeg and the Swift toolchain are found by setup, with the pure helpers here.

Nothing here starts a process: the machine facts come from ``sysctlbyname`` through ctypes, ``os.uname`` and
``platform.mac_ver`` (which reads a plist).

Folders: ``data_dir()`` (``AUDIOINTAKE_HOME`` or ``~/Library/Application Support/audiointake``), ``cache_dir()``
(``AUDIOINTAKE_CACHE`` or ``~/Library/Caches/audiointake``), ``logs_dir()`` (``~/Library/Logs/audiointake``) and
``hf_cache()``, the Hugging Face hub cache resolved the way huggingface_hub resolves it. None is created here;
``fsutil.private_dir`` creates them 0700. A test run points ``HOME``, ``AUDIOINTAKE_HOME``, ``AUDIOINTAKE_CACHE``
and ``HF_HUB_CACHE`` at scratch folders: moving only one of them would let a test touch the real ones.
"""
from __future__ import annotations

import ctypes
import os
import platform as _platform
import re
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .contracts import Problem
from .errors import SetupError, problem

MIN_MACOS = (14, 0)                 # FluidAudio's and MLX's floor
MEASURED_MEMORY_GB = 16             # at or below this, memory_unmeasured (no full run measured on such a Mac yet)
SWIFT_MIN = (6, 0)                  # FluidAudio's Package.swift declares swift-tools 6.0
FFMPEG_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))   # searched after PATH, in this order
_SF_TRANSLATED = "sysctl.proc_translated"
_OS_NAMES = {"linux": "Linux", "win32": "Windows", "cygwin": "Windows", "freebsd": "FreeBSD"}


# --------------------------------------------------------------------------- machine facts (no processes)
def _sysctl(name: str, kind: str) -> Any:
    """sysctlbyname(name) as int ("int"/"int64") or str ("str"); None off macOS or when the name doesn't exist."""
    if sys.platform != "darwin":
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.sysctlbyname
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                   ctypes.c_size_t]
    fn.restype = ctypes.c_int
    key = name.encode("ascii")
    if kind == "str":
        size = ctypes.c_size_t(0)
        if fn(key, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if fn(key, buf, ctypes.byref(size), None, 0) != 0:
            return None
        return buf.value.decode("utf-8", errors="replace")
    val = ctypes.c_int64(0) if kind == "int64" else ctypes.c_int32(0)
    size = ctypes.c_size_t(ctypes.sizeof(val))
    if fn(key, ctypes.byref(val), ctypes.byref(size), None, 0) != 0:
        return None
    return int(val.value)


def machine() -> str:
    """The machine this process sees: "arm64" natively on Apple silicon, "x86_64" on Intel and under Rosetta."""
    try:
        return os.uname().machine
    except (AttributeError, OSError):
        return ""


def translated() -> bool:
    """True when this process runs under Rosetta (sysctl.proc_translated is 1; it doesn't exist on Intel Macs)."""
    return _sysctl(_SF_TRANSLATED, "int") == 1


def macos_version() -> tuple[int, int] | None:
    """(major, minor) of the running macOS, or None anywhere else or when it can't be read."""
    if sys.platform != "darwin":
        return None
    return parse_version(_platform.mac_ver()[0])


def parse_version(text: str) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?", text or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def memory_bytes() -> int | None:
    return _sysctl("hw.memsize", "int64")


def memory_gb() -> int | None:
    b = memory_bytes()
    return None if b is None else int(round(b / 2**30))


def chip() -> str:
    """The processor's name ("Apple M1"), or "" when it can't be read."""
    return _sysctl("machdep.cpu.brand_string", "str") or ""


def os_name() -> str:
    return "macOS" if sys.platform == "darwin" else _OS_NAMES.get(sys.platform, sys.platform)


def info() -> dict[str, Any]:
    """``{os, macos, chip, memory_gb, translated}`` as setup and doctor report them."""
    mac = _platform.mac_ver()[0] if sys.platform == "darwin" else ""
    return {"os": os_name(), "macos": mac, "chip": chip(), "memory_gb": memory_gb(), "translated": translated()}


# --------------------------------------------------------------------------- the checks
def require_supported() -> dict[str, Any]:
    """``info()`` for a supported Mac, else SetupError with the first failing check's code (exit 3).

    An unreadable macOS version doesn't refuse: it can't be shown to be too old.
    """
    if sys.platform != "darwin":
        raise SetupError("platform_not_macos", os=os_name())
    if machine() != "arm64":
        if translated():
            raise SetupError("platform_rosetta", version=__version__)
        raise SetupError("platform_not_apple_silicon", machine=machine())
    v = macos_version()
    if v is not None and v < MIN_MACOS:
        raise SetupError("macos_too_old", macos=_platform.mac_ver()[0])
    return info()


def warnings() -> list[Problem]:
    """``memory_unmeasured`` at 16 GB or less; nothing else is a platform warning."""
    gb = memory_gb()
    if gb is not None and gb <= MEASURED_MEMORY_GB:
        return [problem("memory_unmeasured", gb=gb)]
    return []


def free_bytes(path: Path) -> int:
    """Free bytes for an unprivileged user on the volume holding ``path`` (its nearest existing folder)."""
    p = Path(path).expanduser()
    while not p.exists() and p != p.parent:
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


# --------------------------------------------------------------------------- the Swift toolchain (pure helpers)
def swift_binary(developer_dir: Path) -> Path | None:
    """The ``swift`` a developer folder provides: ``usr/bin/swift`` (the Command Line Tools) or the default
    toolchain's (Xcode, whose own ``usr/bin`` has no swift). None when neither exists.

    An existence check only, made before any shim is run: never run the path it returns for the version and licence
    check. Only ``/usr/bin/swift`` (the xcode-select shim, which links libxcselect) enforces Xcode's licence, so
    ``swift_problem`` reads ``/usr/bin/swift --version``, run once this check has passed (then the shim no longer
    opens Apple's install window), and setup's builds go through ``/usr/bin/swift`` too."""
    d = Path(developer_dir)
    for cand in (d / "usr" / "bin" / "swift",
                 d / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin" / "swift"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def developer_dir(xcode_select_stdout: str, returncode: int) -> Path | None:
    """The folder ``xcode-select -p`` named, if it exited 0 and the folder exists; else None (``clt_missing``)."""
    if returncode != 0:
        return None
    line = (xcode_select_stdout or "").strip().splitlines()
    if not line:
        return None
    d = Path(line[0].strip())
    return d if d.is_absolute() and d.is_dir() else None


def swift_version(text: str) -> tuple[int, int] | None:
    """The version in ``swift --version``'s output ("Apple Swift version 6.4.2 (...)", "Swift version 6.0")."""
    m = re.search(r"Swift version (\d+)(?:\.(\d+))?", text or "")
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def swift_problem(returncode: int, output: str) -> tuple[str | None, str]:
    """(code or None, version text) from ``/usr/bin/swift --version``'s exit and combined output.

    Exit 69, or output asking to agree to a licence, is ``xcode_license``; a version below 6.0 is
    ``swift_too_old``; any other failure, or output with no version in it, is ``clt_missing``.
    """
    low = (output or "").lower()
    if returncode == 69 or ("license" in low and "agree" in low):
        return "xcode_license", ""
    v = swift_version(output)
    if returncode != 0 or v is None:
        return "clt_missing", ""
    text = "%d.%d" % v
    return ("swift_too_old" if v < SWIFT_MIN else None), text


# --------------------------------------------------------------------------- folders (never created here)
def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(os.path.abspath(os.path.expanduser(v))) if v else None


def data_dir() -> Path:
    """Models audiointake converts or assembles, tools, voices, records, packs' stamps, the canary clip."""
    return _env_path("AUDIOINTAKE_HOME") or Path.home() / "Library" / "Application Support" / "audiointake"


def cache_dir() -> Path:
    """Work folders (one per run, deleted at its end) and the FluidAudio build folder."""
    return _env_path("AUDIOINTAKE_CACHE") or Path.home() / "Library" / "Caches" / "audiointake"


def logs_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "audiointake"


def hf_cache() -> Path:
    """The Hugging Face hub cache, as huggingface_hub resolves it: ``HF_HUB_CACHE``, else the legacy
    ``HUGGINGFACE_HUB_CACHE``, else ``$HF_HOME/hub``, else ``$XDG_CACHE_HOME/huggingface/hub``, else
    ``~/.cache/huggingface/hub``."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        p = _env_path(name)
        if p is not None:
            return p
    home = _env_path("HF_HOME")
    if home is None:
        xdg = _env_path("XDG_CACHE_HOME")
        home = (xdg or Path.home() / ".cache") / "huggingface"
    return home / "hub"
