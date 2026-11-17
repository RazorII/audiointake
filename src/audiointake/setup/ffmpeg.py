"""Finding ffmpeg and ffprobe, and the fallback ffmpeg for a Mac without Homebrew.

``find()`` looks on ``PATH``, then in ``/opt/homebrew/bin``, then in ``/usr/local/bin``; ffprobe is taken from
ffmpeg's own folder when it is there, else searched the same way. Only when no ffmpeg is found anywhere does the
fallback apply: the ffmpeg binary inside the pinned ``imageio-ffmpeg`` wheel (ffmpeg only, no ffprobe), checked
against its pinned SHA-256 and linked as ``<data>/tools/ffmpeg/ffmpeg``, because parakeet-mlx and mlx-whisper look
``ffmpeg`` up on ``PATH`` by that name. ``imageio-ffmpeg`` is a pinned dependency, so every install has the
fallback; ``setup`` links it, and a Mac with Homebrew's ffmpeg never uses it, so transcripts made there don't change.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from .. import platforms, proc

FALLBACK_PACKAGE = "imageio_ffmpeg"
FALLBACK_VERSION = "0.8.1"
FALLBACK_NAME = "ffmpeg-macos-aarch64-v9.0"
FALLBACK_SIZE = 52_070_376
FALLBACK_SHA256 = "591260c945d0eef150e3bf82b0ef988bd36a9cecc18ff05d6679617159f0a95e"


def _exe(p: Path) -> bool:
    return p.is_file() and os.access(p, os.X_OK)


def _search(name: str) -> Path | None:
    hit = shutil.which(name, path=os.environ.get("PATH", ""))
    if hit:
        return Path(os.path.abspath(hit))
    for d in platforms.FFMPEG_DIRS:
        if _exe(d / name):
            return d / name
    return None


def fallback_link() -> Path:
    return platforms.data_dir() / "tools" / "ffmpeg" / "ffmpeg"


def fallback_binary() -> Path | None:
    """The pinned fallback binary inside the installed imageio-ffmpeg package (not imported), or None."""
    try:
        spec = importlib.util.find_spec(FALLBACK_PACKAGE)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    p = Path(list(spec.submodule_search_locations)[0]) / "binaries" / FALLBACK_NAME
    return p if _exe(p) else None


def _fallback_ok(binary: Path) -> bool:
    from .download import _hash

    try:
        return binary.stat().st_size == FALLBACK_SIZE and _hash(binary, False) == FALLBACK_SHA256
    except OSError:
        return False


def link_fallback() -> Path | None:
    """Link the verified fallback binary as ``<data>/tools/ffmpeg/ffmpeg``; None when it isn't available."""
    binary = fallback_binary()
    if binary is None or not _fallback_ok(binary):
        return None
    link = fallback_link()
    from .. import fsutil

    fsutil.private_dir(link.parent)
    if link.is_symlink() and os.readlink(link) == str(binary):
        return link
    tmp = link.with_name(f".ffmpeg.{os.getpid()}")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(str(binary), tmp)
    os.replace(tmp, link)
    return link


def find(*, ensure_fallback: bool = False) -> tuple[Path | None, Path | None]:
    """(ffmpeg, ffprobe). ffprobe is None with the fallback; both None when there is no ffmpeg at all.

    ``ensure_fallback`` (setup, and a run under the run lock) links the verified fallback when no link that works
    is there yet: the first run after a fresh install, or after the tool moved and the old link dangles."""
    ff = _search("ffmpeg")
    if ff is not None:
        probe = ff.parent / "ffprobe"
        return ff, probe if _exe(probe) else _search("ffprobe")
    link = fallback_link()
    if link.is_symlink() and _exe(link) and Path(os.readlink(link)).name == FALLBACK_NAME:
        return link, None
    if ensure_fallback:
        made = link_fallback()
        if made is not None:
            return made, None
    return None, None


def find_without_writing() -> tuple[Path | None, Path | None]:
    """``find()`` for a report that writes nothing (doctor): when no ffmpeg is found and the fallback isn't linked
    yet, the verified fallback binary itself, which setup (or the next run) links. Both None only when there is no
    usable ffmpeg at all, which is the one case ``ffmpeg_missing`` means."""
    ff, fp = find()
    if ff is None:
        binary = fallback_binary()
        if binary is not None and _fallback_ok(binary):
            return binary, None
    return ff, fp


def version(ffmpeg: Path) -> str:
    """ffmpeg's version word ("9.0.2"), or "" when it can't be read."""
    from . import store

    try:
        r = proc.run([ffmpeg, "-hide_banner", "-version"], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                     timeout=30, allowed=store.programs(ffmpeg))
    except (OSError, proc.NotAllowed):
        return ""
    first = r.stdout.decode("utf-8", "replace").splitlines()[:1]
    parts = first[0].split() if first else []
    return parts[2] if len(parts) > 2 and parts[0] == "ffmpeg" and parts[1] == "version" else ""


def prerequisites(ff: Path | None, fp: Path | None) -> list[dict]:
    """``[{name, ok, version, path, code?}]`` for ffmpeg and ffprobe, as setup and doctor report them."""
    out = []
    if ff is None:
        out.append({"name": "ffmpeg", "ok": False, "version": "", "path": None, "code": "ffmpeg_missing"})
    else:
        out.append({"name": "ffmpeg", "ok": True, "version": version(ff), "path": str(ff),
                    "fallback": ff == fallback_link() or ff.name == FALLBACK_NAME})
    out.append({"name": "ffprobe", "ok": fp is not None, "version": "", "path": str(fp) if fp else None})
    return out
