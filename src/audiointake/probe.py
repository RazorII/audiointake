"""What a file holds before anything is transcribed: its audio streams, their channels, its length, its date.

Two ways, giving the same answer:

* **With ffprobe** (the usual case): ``argv.channels_probe`` (one line per audio stream with its channel count)
  and ``argv.duration_probe``, plus the container's ``creation_time`` tag.
* **With ffmpeg only** (the bundled fallback has no ffprobe): ``ffmpeg -hide_banner -nostdin -i FILE`` prints the
  ``Duration:`` line and one ``Stream #…: Audio:`` line per audio stream on stderr. The channel layout becomes a
  count by a closed table: every standard layout ``ffmpeg -layouts`` lists (mono 1, stereo 2, 2.1 3, 3.1, 4.0 and
  quad 4, 5.0 5, 5.1 6, 7.1 8, ...), and ``N channels`` (ffmpeg writes a custom layout as ``4 channels
  (FL+FR+LFE+BC)``) N; anything else is unknown, which never counts as 4. ``Duration:`` has 10 ms steps, so this
  path's length can differ from ffprobe's by up to 5 ms.

The spatial rule: ``spatial`` is "yes" when any audio stream has exactly 4 channels, else "no" (a failed probe
included). ``recorded_at`` comes from the file name when it starts ``YYYYMMDD HHMMSS`` (the way Voice Memos names
recordings), else the container's ``creation_time`` (converted to local time), else the file's modification date.
"""
from __future__ import annotations

import os
import re
import secrets
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import argv, platforms, proc
from .contracts import Tools

# ffmpeg 9.0.2's standard channel layouts (`ffmpeg -layouts`), each with its channel count
LAYOUT_CHANNELS = {
    "mono": 1, "stereo": 2, "binaural": 2, "downmix": 2, "2.1": 3, "3.0": 3, "3.0(back)": 3, "4.0": 4, "quad": 4,
    "quad(side)": 4, "3.1": 4, "5.0": 5, "5.0(side)": 5, "4.1": 5, "5.1": 6, "5.1(side)": 6, "6.0": 6,
    "6.0(front)": 6, "3.1.2": 6, "hexagonal": 6, "6.1": 7, "6.1(back)": 7, "6.1(front)": 7, "7.0": 7, "7.0(front)": 7,
    "7.1": 8, "7.1(wide)": 8, "7.1(wide-side)": 8, "5.1.2": 8, "5.1.2(back)": 8, "octagonal": 8, "cube": 8,
    "5.1.4": 10, "7.1.2": 10, "7.1.4": 12, "7.2.3": 12, "9.1.4": 14, "9.1.6": 16, "hexadecagonal": 16,
}
_DURATION = re.compile(r"^\s*Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_STREAM = re.compile(r"^\s*Stream #\d+:\d+.*?: Audio: (.*)$")
_CREATION = re.compile(r"^\s*creation_time\s*:\s*(\S+)")
_N_CHANNELS = re.compile(r"^(\d+) channels\b")
_NAME_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2}) (\d{2})(\d{2})(\d{2})")
_MIN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8"}


def probe(path: Path, tools: Tools, *, env: Mapping[str, str] | None = None,
          workdir: Path | None = None) -> dict[str, Any]:
    """``{has_audio, channels, spatial, duration_s, creation_time, recorded_at, recorded_at_source}`` for ``path``.

    ``channels`` lists each audio stream's channel count (None where the layout is unknown). ``env`` is the
    children's environment (a minimal one by default); ``workdir`` holds the ffmpeg-only path's stderr file.
    """
    path = Path(path)
    e = dict(env) if env is not None else {**_MIN_ENV, "HOME": os.path.expanduser("~")}
    if tools.ffprobe is not None:
        info = _with_ffprobe(path, tools, e)
    else:
        info = _with_ffmpeg(path, tools, e, workdir)
    info.update(recorded_at(path, info.get("creation_time")))
    return info


def _run(cmd: list[str], tools: Tools, env: Mapping[str, str], **kw: Any) -> proc.Completed:
    return proc.run(cmd, env=env, timeout=proc.child_timeout("ffprobe", 0), allowed=proc.allowed_for(tools), **kw)


def _with_ffprobe(path: Path, tools: Tools, env: Mapping[str, str]) -> dict[str, Any]:
    assert tools.ffprobe is not None
    ch = _run(argv.channels_probe(tools.ffprobe, path), tools, env)
    out = ch.stdout.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in out.split("\n") if ln.strip()]
    channels = [int(x) if x.isdigit() else None for x in lines]
    du = _run(argv.duration_probe(tools.ffprobe, path), tools, env)
    duration = _float(du.stdout.decode("utf-8", errors="replace").strip()) if du.ok else None
    tags = _run(tags_probe(tools.ffprobe, path), tools, env)
    created = tags.stdout.decode("utf-8", errors="replace").strip().split("\n")[0].strip() if tags.ok else ""
    return {"has_audio": bool(lines), "channels": channels, "spatial": argv.spatial(out.rstrip("\n")),
            "duration_s": duration, "creation_time": created or None}


def tags_probe(ffprobe: Path, audio: Path) -> list[str]:
    """The container's ``creation_time`` tag, one line (not part of any transcript, so not in argv.py's table)."""
    return [os.fspath(ffprobe), "-v", "error", "-show_entries", "format_tags=creation_time",
            "-of", "default=noprint_wrappers=1:nokey=1", os.fspath(audio)]


def _with_ffmpeg(path: Path, tools: Tools, env: Mapping[str, str], workdir: Path | None) -> dict[str, Any]:
    folder = Path(workdir) if workdir is not None else platforms.cache_dir() / "work"
    folder.mkdir(parents=True, exist_ok=True)
    errfile = folder / f".probe-{secrets.token_hex(6)}.txt"
    try:
        _run(argv.ffmpeg_info(tools.ffmpeg, path), tools, env, stdout="discard", output_log=errfile)
        text = errfile.read_text(encoding="utf-8", errors="replace") if errfile.exists() else ""
    finally:
        try:
            errfile.unlink()
        except OSError:
            pass
    return parse_ffmpeg_info(text)


def parse_ffmpeg_info(text: str) -> dict[str, Any]:
    """The probe's answer from ``ffmpeg -i``'s stderr."""
    duration = created = None
    channels: list[int | None] = []
    for line in text.replace("\r", "\n").split("\n"):
        m = _DURATION.match(line)
        if m and duration is None:
            duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            continue
        m = _STREAM.match(line)
        if m:
            channels.append(layout_channels(m.group(1)))
            continue
        m = _CREATION.match(line)
        if m and created is None:
            created = m.group(1)
    spatial = "yes" if any(c == 4 for c in channels) else "no"
    return {"has_audio": bool(channels), "channels": channels, "spatial": spatial, "duration_s": duration,
            "creation_time": created}


def layout_channels(audio_desc: str) -> int | None:
    """The channel count in an ffmpeg ``Audio:`` description, from its layout field (the third, comma-separated)."""
    fields = [f.strip() for f in audio_desc.split(",")]
    if len(fields) < 3:
        return None
    layout = fields[2]
    if layout in LAYOUT_CHANNELS:
        return LAYOUT_CHANNELS[layout]
    m = _N_CHANNELS.match(layout)
    return int(m.group(1)) if m else None


def _float(text: str) -> float | None:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if v == v and v >= 0 else None


def recorded_at(path: Path, creation_time: str | None) -> dict[str, Any]:
    """``{recorded_at, recorded_at_source}``: from the file name, the creation_time tag, or the file's date."""
    m = _NAME_DATE.match(Path(path).name)
    if m:
        y, mo, d, h, mi, s = m.groups()
        try:
            datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
            return {"recorded_at": f"{y}-{mo}-{d}T{h}:{mi}:{s}", "recorded_at_source": "filename"}
        except ValueError:
            pass
    t = parse_creation_time(creation_time)
    if t is not None:
        return {"recorded_at": t, "recorded_at_source": "metadata"}
    try:
        st = os.stat(path)
    except OSError:
        return {"recorded_at": None, "recorded_at_source": None}
    return {"recorded_at": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%S"),
            "recorded_at_source": "file_date"}


def parse_creation_time(text: str | None) -> str | None:
    """An ISO ``creation_time`` (UTC when it ends in Z) as local ``YYYY-MM-DDTHH:MM:SS``; None if unusable."""
    if not text:
        return None
    s = text.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.year < 2000:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
