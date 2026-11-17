"""How a transcript reads: the ``(transcript).md`` file, and the dialogue ``notes --into`` appends.

The ``.md`` file::

    # <stem>

    <YYYY-MM-DD> · <h:mm:ss> · <n> speakers (<k> named)

    **<display>** [mm:ss]
    <turn text>

    **<display>** [mm:ss]
    <turn text>

Timestamps are ``[h:mm:ss]`` once the recording is longer than an hour. The date line leaves out the date when
the recording date is unknown, and the speaker part when there are no speaker labels. Without speaker labels
the sentences are grouped into paragraphs wherever the speech pauses for 2 s or more, each headed by its start
time and no name.

Pure functions: nothing here reads or writes a file.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contracts import Turn

HOUR_S = 3600.0
PARAGRAPH_GAP_S = 2.0          # a pause this long starts a new paragraph when there are no speaker labels
SEP = " · "


def clock(seconds: float, *, hours: bool) -> str:
    """``mm:ss`` (minutes may pass 59 only when ``hours`` is off) or ``h:mm:ss``; seconds are cut, not rounded."""
    s = max(0, int(math.floor(float(seconds) + 1e-9)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if hours:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{h * 60 + m:02d}:{sec:02d}"


def as_turn(t: Turn | Mapping[str, Any]) -> Turn:
    """A ``Turn`` from a Turn or from its JSON form (``{"label", "display", "start", "end", "text"}``)."""
    if isinstance(t, Turn):
        return t
    return Turn(label=t.get("label"), display=str(t.get("display") or ""), start=float(t.get("start") or 0.0),
                end=float(t.get("end") or t.get("start") or 0.0), text=str(t.get("text") or ""))


def turn_dict(t: Turn) -> dict[str, Any]:
    """A Turn as the transcript JSON stores it."""
    return {"label": t.label, "display": t.display, "start": round(float(t.start), 3), "end": round(float(t.end), 3),
            "text": t.text}


def dialogue(turns: Iterable[Turn | Mapping[str, Any]], *, duration_s: float | None = None) -> str:
    """The speaker-labelled dialogue: one block per turn, ``**<display>** [mm:ss]`` then the text.

    A turn without a display (no speaker labels) is headed by its time alone. ``duration_s`` decides between
    ``mm:ss`` and ``h:mm:ss``; without it, the last turn's end does.
    """
    ts = [as_turn(t) for t in turns]
    ts = [t for t in ts if t.text.strip()]
    end = duration_s if duration_s is not None else max((t.end for t in ts), default=0.0)
    hours = float(end or 0.0) > HOUR_S
    blocks = []
    for t in ts:
        stamp = f"[{clock(t.start, hours=hours)}]"
        head = f"**{t.display}** {stamp}" if t.display else stamp
        blocks.append(f"{head}\n{t.text.strip()}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def paragraphs(sentences: Sequence[Mapping[str, Any]], *, gap: float = PARAGRAPH_GAP_S) -> list[Turn]:
    """Sentences grouped into unlabelled paragraphs at pauses of ``gap`` seconds or more."""
    out: list[Turn] = []
    cur: list[str] = []
    start = end = None
    last = 0.0
    for s in sentences:
        if not isinstance(s, Mapping):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        st = _num(s.get("start"), last)
        en = _num(s.get("end"), st)
        if cur and end is not None and st - end >= gap:
            out.append(Turn(label=None, display="", start=float(start), end=float(end), text=" ".join(cur)))
            cur, start = [], None
        if start is None:
            start = st
        cur.append(text)
        end = en
        last = en
    if cur:
        out.append(Turn(label=None, display="", start=float(start), end=float(end), text=" ".join(cur)))
    return out


def _num(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def speaker_part(n: int, k: int) -> str:
    return f"{n} speaker{'' if n == 1 else 's'} ({k} named)"


def header(*, title: str, recorded_at: str | None, duration_s: float | None,
           speakers: tuple[int, int] | None) -> str:
    """The title and the date line; ``speakers`` is (labels, named) or None for no speaker labels."""
    meta = []
    if recorded_at:
        meta.append(str(recorded_at)[:10])
    if duration_s is not None:
        meta.append(clock(duration_s, hours=True))
    if speakers is not None:
        meta.append(speaker_part(*speakers))
    lines = [f"# {title}"]
    if meta:
        lines += ["", SEP.join(meta)]
    return "\n".join(lines) + "\n"


def md_file(*, title: str, recorded_at: str | None, duration_s: float | None,
                  turns: Iterable[Turn | Mapping[str, Any]], speakers: tuple[int, int] | None) -> str:
    """The whole ``(transcript).md`` text."""
    head = header(title=title, recorded_at=recorded_at, duration_s=duration_s, speakers=speakers)
    body = dialogue(turns, duration_s=duration_s)
    return head + ("\n" + body if body else "")
