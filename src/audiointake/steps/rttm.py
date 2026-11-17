"""The one RTTM reader: every step that reads diarizer output parses it here.

An RTTM line has ten fields::

    SPEAKER <file-id> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>

The diarizer writes the input wav's basename as ``<file-id>``, and recording names often contain spaces
("Team sync.16k"). Reading the fields left to right with ``split()`` would then take ``<chnl>`` as the start time
and ``<NA>`` as the speaker, and every recording would collapse into one speaker without any error. The nine fields
after ``<file-id>`` never contain spaces, so this reader counts them from the right. A line it can't parse raises
ValueError instead of being skipped.

Two entry points, one per kind of input: ``segments_from_text`` for RTTM text already in memory (a transcript's
kept RTTM) and ``segments_from_path`` for a file. ``segments`` is the file form under its older name.
"""
from __future__ import annotations

import io
import os
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]


def parse_line(line: str) -> dict | None:
    """One RTTM line -> ``{"start", "dur", "spk"}``, or None for a line that isn't a SPEAKER line."""
    x = line.split()
    if not x or x[0] != "SPEAKER":
        return None
    if len(x) < 10:
        raise ValueError(f"RTTM line has {len(x)} fields, expected >= 10: {line.strip()!r}")
    # x[-8]=chnl x[-7]=tbeg x[-6]=tdur x[-5]=ortho x[-4]=stype x[-3]=name x[-2]=conf x[-1]=slat
    try:
        start, dur = float(x[-7]), float(x[-6])
    except ValueError:
        raise ValueError(f"RTTM line has non-numeric tbeg/tdur: {line.strip()!r}") from None
    return {"start": start, "dur": dur, "spk": x[-3]}


def segments_from_text(text: str) -> list[dict]:
    """All SPEAKER turns in RTTM text, in order. Lines end at LF, CR or CRLF, as when the text is read from a file."""
    return [s for s in (parse_line(ln) for ln in io.StringIO(text, newline=None)) if s]


def segments_from_path(path: PathLike) -> list[dict]:
    """All SPEAKER turns in an RTTM file, in file order."""
    with open(path, encoding="utf-8") as f:
        return [s for s in (parse_line(ln) for ln in f) if s]


segments = segments_from_path
