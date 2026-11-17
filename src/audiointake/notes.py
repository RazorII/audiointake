"""Notes files: Claude writes the head (title, summary, decisions, action items); audiointake appends the transcript.

``notes REF --into NOTES.md`` reads the head already in NOTES.md, refuses one that has a ``## Transcript`` line
(``head_invalid``), checks the transcript JSON is what audiointake wrote (``transcript_edited``), and writes, all at
once, the head, a blank line, ``## Transcript``, a blank line and the dialogue ``render.dialogue`` makes from the
transcript JSON's turns. So the transcript in the notes is word for word what the JSON holds, and nobody retypes
it. The child record lists the notes file with its SHA-256.

``--into`` takes the notes file only: the recording, the transcript's own files (anything named
``<stem> (transcript).md|json|txt``, or listed by its record or JSON) are refused (``bad_arguments``), so a slip
can't overwrite the transcript JSON every follow-up rebuilds from. The child record carries the transcript's open
offers, so ``apply`` by the recording's path still finds them after the notes are written.

``notes REF --check NOTES.md`` says whether the file is still exactly what the newest ``--into`` for that path
wrote (exit 0) or was changed since (exit 5 ``output_changed``). It writes nothing. The SHA-256 it compares with
lives in the notes record, so once that record is pruned (90 days) every check says ``output_changed``: the safe
answer, since the skill then leaves the file alone.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from . import fsutil, outputs, record, render
from .errors import Stale, UsageError

TRANSCRIPT_HEADING = "## Transcript"
_HEADING = re.compile(r"^##\s+Transcript\s*$", re.MULTILINE)


def _abs(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def compose(head: str, turns: list[Any], *, duration_s: float | None) -> str:
    """The notes file: the head, a blank line, ``## Transcript``, a blank line, the dialogue."""
    body = render.dialogue(turns, duration_s=duration_s)
    return head.rstrip("\n") + "\n\n" + TRANSCRIPT_HEADING + "\n\n" + body


def read_head(path: Path) -> str:
    """The head Claude wrote; ``input_missing`` when there is no such file, ``head_invalid`` when it already holds
    a ``## Transcript`` section."""
    try:
        head = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UsageError("input_missing", file=str(path)) from None
    except UnicodeDecodeError:
        raise UsageError("head_invalid", path=str(path)) from None
    if _HEADING.search(head):
        raise UsageError("head_invalid", path=str(path))
    return head


def into(ref: str, notes_path: str | os.PathLike[str]) -> dict[str, Any]:
    """``notes REF --into NOTES.md``: write the notes file and return the child record (saved; lock held)."""
    target = _abs(notes_path)
    r = record.resolve(ref, need_json=True, command="notes")
    _not_a_transcript_file(target, r)
    record.reseal(r, required=False)
    head = read_head(target)
    rec = record.new("notes", parent=r.id)
    doc = r.doc or {}
    rec["input"] = _input_of(r)
    record.carry_offers(rec, record.open_offers(r))
    text = compose(head, list(doc.get("turns") or []), duration_s=(doc.get("input") or {}).get("duration_s"))
    data = text.encode("utf-8")
    # SIGTERM and SIGHUP wait while the notes file and its record are written, as a transcript's commit does: a
    # notes file without its record would read as changed by the user forever (``check``), and never be rebuilt
    from .pipeline import _commit_window
    with _commit_window():
        fsutil.atomic_write(target, data, overwrite=True)
        rec["outputs"] = [{"kind": "notes", "path": str(target), "sha256": fsutil.sha256_bytes(data)}]
        record.finish(rec, 0)
        record.save(rec)
    return rec


def _not_a_transcript_file(target: Path, r: record.Ref) -> None:
    """``bad_arguments`` when the notes target is the recording or one of the transcript's own files."""
    own = {str(p) for p in (r.json_path, r.audio) if p is not None}
    for o in (r.record or {}).get("outputs") or []:
        if o.get("path"):
            own.add(str(o["path"]))
    if r.json_path is not None:
        own.update(str(r.json_path.parent / o["name"]) for o in (r.doc or {}).get("outputs") or [] if o.get("name"))
    named = target.name.endswith(tuple(f"{outputs.SUFFIX}.{k}" for k in outputs.NAMED_KINDS))
    if str(target) in own or named:
        raise UsageError("bad_arguments", command="notes", detail=f"--into takes the notes file, not {target.name}, "
                                                                  "which is the recording or its transcript")


def check(ref: str, notes_path: str | os.PathLike[str]) -> dict[str, Any]:
    """``notes REF --check NOTES.md``: ``{"unchanged": True}``, or Stale ``output_changed``."""
    target = _abs(notes_path)
    r = record.resolve(ref, need_json=False, command="notes")
    newest = newest_notes(str(target))
    if newest is None or not target.is_file() or fsutil.sha256_file(target) != newest:
        raise Stale("output_changed", out=target.name, path=str(target))
    return {"unchanged": True, "path": str(target), "id": r.id}


def newest_notes(path: str) -> str | None:
    """The SHA-256 the newest ``notes --into`` recorded for ``path``, or None."""
    for rec in record.iter_records():
        if rec.get("command") != "notes":
            continue
        for o in rec.get("outputs") or []:
            if o.get("kind") == "notes" and o.get("path") == path:
                return o.get("sha256")
    return None


def _input_of(r: record.Ref) -> dict[str, Any]:
    if r.record is not None and r.record.get("input"):
        return dict(r.record["input"])
    doc = r.doc or {}
    inp = dict(doc.get("input") or {})
    return {"path": str(r.audio) if r.audio else None, "sha256": inp.get("sha256"),
            "duration_s": inp.get("duration_s"), "recorded_at": inp.get("recorded_at")}
