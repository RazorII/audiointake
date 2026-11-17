"""Records: what each command did, kept 90 days in ``<data>/records/<id>.json`` (``docs/decision-record.md``).

A record never holds transcript text: no sentence, turn, word or model output. Its ``params`` hold numbers,
codes, pack and label ids, file names and names the user gave; speaker names and file paths do appear.

This module is the data folder's writer for records, run logs (``~/Library/Logs/audiointake/<id>.log``) and
``<data>/speed.json``; ``save`` expects the run lock to be held (``lock.require_held``). It also turns a REF (a
record id, a recording's path or its ``(transcript).json`` path) into the transcript it names (``resolve``), and
checks that transcript is what audiointake wrote.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import statistics
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__, errors, fsutil, lock, outputs, platforms
from .contracts import RECORD_ID_RE, Problem
from .errors import Stale, UsageError

RECORD_VERSION = 1
KEEP_DAYS = 90
SPEED_KEEP = 10
_ID = re.compile(RECORD_ID_RE)
STATUS_FOR_EXIT = {0: "done", 1: "failed", 2: "usage", 3: "failed", 4: "refused", 5: "stale"}


# --------------------------------------------------------------------------- folders
def records_dir() -> Path:
    return platforms.data_dir() / "records"


def record_path(record_id: str) -> Path:
    return records_dir() / f"{record_id}.json"


def log_path(record_id: str) -> Path:
    return platforms.logs_dir() / f"{record_id}.log"


def speed_path() -> Path:
    return platforms.data_dir() / "speed.json"


def is_id(text: str) -> bool:
    return bool(_ID.match(text or ""))


def new_id(now: float | None = None) -> str:
    return f"r-{time.strftime('%Y%m%d', time.localtime(now))}-{secrets.token_hex(4)}"


def now_iso() -> str:
    """Local time with its offset, to the microsecond (records made in the same second still sort in order)."""
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def created_at(rec: Mapping[str, Any]) -> float:
    """A record's ``created`` as a timestamp (0 when it can't be read), so offsets and DST sort correctly."""
    try:
        return datetime.fromisoformat(str(rec.get("created"))).timestamp()
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- building a record
def new(command: str, *, parent: str | None = None) -> dict[str, Any]:
    """An empty record for ``command`` with a fresh id; every field of record version 1 is present."""
    return {
        "record_version": RECORD_VERSION, "id": new_id(), "parent": parent, "tool": {"version": __version__},
        "command": command, "status": None, "exit": None, "created": now_iso(),
        "input": None, "options": None, "options_digest": None, "voices_digest": None,
        "outputs": [], "engines": None, "signals": None, "speakers": [], "counts": None, "timing": None,
        "warnings": [], "notes": [], "offers": [], "error": None, "log": None,
    }


def add(rec: dict[str, Any], p: Problem) -> None:
    """File a problem under warnings, notes or offers by its code's kind (an error code goes to ``error``).

    An offer problem's params are the offer's (``code``, ``language``, ``size`` for a language offer); offers of
    one kind and option are kept once, and so are identical warnings and notes.
    """
    kind = errors.CODES[p.code][0]
    if kind == "error":
        if p.code in errors.ALSO_WARNING and str(p.params.get("pack", "core")) != "core":
            kind = "warning"
        else:
            rec["error"] = {"code": p.code, "params": dict(p.params)}
            return
    if kind == "offer":
        add_offer(rec, p.code, [p.params["code"]] if "code" in p.params else [], params=p.params)
        return
    key = "warnings" if kind == "warning" else "notes"
    item = {"code": p.code, "params": dict(p.params)}
    if item not in rec[key]:
        rec[key].append(item)


# Per-option values an offer keeps: a ``language`` offer's download size per pack code, a ``remember`` offer's
# seconds of clean speech per label. Each option renders with its own value, never another option's.
PER_OPTION = {"language": ("sizes", "size"), "remember": ("speech_s", "seconds")}


def _per_option(kind: str, options: list[str], params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The per-option map ``params`` carries for ``kind``: its stored map, plus the single option's own value
    when the params are one option's (a language offer's ``code`` with its ``size``)."""
    if kind not in PER_OPTION:
        return {}
    key, one = PER_OPTION[kind]
    stored = params.get(key)
    out = {str(k): v for k, v in stored.items()} if isinstance(stored, Mapping) else {}
    if one in params and isinstance(params.get(one), (int, float)) and str(params.get("code", "")) in options:
        out.setdefault(str(params["code"]), params[one])
    return {key: out} if out else {}


def add_offer(rec: dict[str, Any], kind: str, options: list[str], *, params: Mapping[str, Any] | None = None) -> None:
    """Add an offer, or merge its options into the record's offer of that kind. The first params given for a key
    stay; per-option values (``PER_OPTION``) are merged option by option."""
    params = dict(params or {})
    params.update(_per_option(kind, [str(x) for x in options], params))
    for o in rec["offers"]:
        if o["kind"] == kind:
            for opt in options:
                if opt not in o["options"]:
                    o["options"].append(opt)
            have = o.setdefault("params", {})
            for k, v in params.items():
                if isinstance(v, Mapping) and isinstance(have.get(k), Mapping):
                    have[k] = {**v, **have[k]}
                elif k not in have:
                    have[k] = v
            return
    rec["offers"].append({"kind": kind, "options": list(options), "apply_flag": errors.OFFER_FLAGS[kind],
                          "params": params})


def has_warning(rec: Mapping[str, Any], code: str, **match: Any) -> bool:
    return any(w["code"] == code and all(w["params"].get(k) == v for k, v in match.items()) for w in rec["warnings"])


def finish(rec: dict[str, Any], exit_code: int, *, status: str | None = None) -> dict[str, Any]:
    rec["exit"] = int(exit_code)
    rec["status"] = status or STATUS_FOR_EXIT.get(int(exit_code), "failed")
    return rec


def fail(rec: dict[str, Any], exc: errors.AudioIntakeError) -> dict[str, Any]:
    """Record an error raised for this record and set exit and status from its class."""
    p = exc.problem()
    rec["error"] = {"code": p.code, "params": dict(p.params)}
    return finish(rec, exc.exit)


def _context(p: Mapping[str, Any], rec: Mapping[str, Any]) -> Problem:
    prob = Problem(p["code"], dict(p.get("params") or {}))
    return errors.with_context(prob, id=rec.get("id"), log=rec.get("log"))


def finalize(rec: dict[str, Any]) -> dict[str, Any]:
    """Add the record id and log path where a message names them, and the words of every message.

    Each warning, note and the error gets ``message``; each offer gets ``summary``, one per option for a
    ``remember`` offer (its label's seconds come from ``speakers``). Core's words only, never model output.
    """
    for key in ("warnings", "notes"):
        items = []
        for w in rec.get(key) or []:
            p = _context(w, rec)
            items.append({"code": p.code, "params": dict(p.params), "message": errors.render(p.code, p.params)})
        rec[key] = items
    if rec.get("error"):
        p = _context(rec["error"], rec)
        rec["error"] = {"code": p.code, "params": dict(p.params), "message": errors.render(p.code, p.params)}
    speech = {s.get("label"): s.get("speech_s") for s in rec.get("speakers") or [] if s.get("speech_s") is not None}
    for o in rec.get("offers") or []:
        params = {**(o.get("params") or {}), "id": rec.get("id")}
        if o["kind"] == "remember":
            kept = params.get("speech_s") if isinstance(params.get("speech_s"), Mapping) else {}
            o["summary"] = [errors.render("remember", {**params, "label": lab,
                                                       "seconds": speech.get(lab, kept.get(lab, 0))})
                            for lab in o["options"]]
        elif o["kind"] == "language":
            sizes = params.get("sizes") if isinstance(params.get("sizes"), Mapping) else {}
            o["summary"] = [errors.render("language", _language_params(params, sizes, c)) for c in o["options"]]
        else:
            o["summary"] = [errors.render(o["kind"], params)]
    return rec


def _language_params(params: Mapping[str, Any], sizes: Mapping[str, Any], code: str) -> dict[str, Any]:
    """One language option's fields: its own code, name and size (never another option's size)."""
    out = {k: v for k, v in params.items() if k not in ("size", "sizes", "code", "language")}
    out.update(code=code, language=_lang_name(code, params if params.get("code") == code else {}))
    size = sizes.get(code, params.get("size") if params.get("code") == code else None)
    if isinstance(size, (int, float)):
        out["size"] = size
    return out


def _lang_name(code: str, params: Mapping[str, Any]) -> str:
    from .contracts import LANGUAGE_NAMES
    return LANGUAGE_NAMES.get(code, str(params.get("language", code)))


# --------------------------------------------------------------------------- saving and loading
def save(rec: dict[str, Any]) -> Path:
    """Write the record atomically, 0600, after ``finalize``; the run lock must be held."""
    lock.require_held()
    finalize(rec)
    folder = fsutil.private_dir(records_dir())
    path = folder / f"{rec['id']}.json"
    data = (json.dumps(rec, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    fsutil.atomic_write(path, data, overwrite=True, mode=0o600)
    return path


def load(record_id: str, *, path_hint: str = "") -> dict[str, Any]:
    """The record ``record_id``; UsageError ``unknown_record`` when there is none (or it isn't a record id)."""
    if not is_id(record_id):
        raise UsageError("unknown_record", id=record_id, path=path_hint or record_id)
    p = record_path(record_id)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise UsageError("unknown_record", id=record_id, path=path_hint or record_id) from None
    if not isinstance(d, dict) or d.get("id") != record_id:
        raise UsageError("unknown_record", id=record_id, path=path_hint or record_id)
    return d


def try_load(record_id: str | None) -> dict[str, Any] | None:
    if not record_id:
        return None
    try:
        return load(record_id)
    except UsageError:
        return None


def iter_records() -> Iterator[dict[str, Any]]:
    """Every readable record, newest first."""
    folder = records_dir()
    if not folder.is_dir():
        return
    items = []
    for p in folder.glob("r-*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and is_id(str(d.get("id"))):
            items.append(d)
    items.sort(key=lambda d: (created_at(d), d["id"]), reverse=True)
    yield from items


def prune(now: float | None = None) -> int:
    """Delete records and run logs older than 90 days; returns how many files went. Needs the lock."""
    lock.require_held()
    cutoff = (time.time() if now is None else now) - KEEP_DAYS * 86400
    n = 0
    for folder, pattern in ((records_dir(), "r-*.json"), (platforms.logs_dir(), "r-*.log")):
        if not folder.is_dir():
            continue
        for p in folder.glob(pattern):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except OSError:
                pass
    return n


# --------------------------------------------------------------------------- the run log
class RunLog:
    """``~/Library/Logs/audiointake/<id>.log``: step timings, tags, codes and children's stderr tails.

    Never transcript text: callers pass only those. Opened lazily, appended to, 0600 in a 0700 folder.
    """

    def __init__(self, record_id: str) -> None:
        self.path = log_path(record_id)
        self._t0 = time.monotonic()

    def __call__(self, line: str) -> None:
        fsutil.private_dir(self.path.parent)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            stamp = f"{time.monotonic() - self._t0:8.2f}s"
            os.write(fd, f"{stamp} {line.rstrip()}\n".encode("utf-8", errors="replace"))
        finally:
            os.close(fd)

    def tail(self, what: str, lines: list[str]) -> None:
        for ln in lines:
            self(f"{what}: {ln}")

    def remove(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- speed.json
def speed_load() -> dict[str, list[float]]:
    try:
        d = json.loads(speed_path().read_text(encoding="utf-8"))
        routes = d.get("routes") if isinstance(d, dict) else None
        return {k: [float(x) for x in v][-SPEED_KEEP:] for k, v in (routes or {}).items() if isinstance(v, list)}
    except (OSError, ValueError, TypeError):
        return {}


def speed_add(route: str, seconds_per_audio_minute: float) -> None:
    """Add one finished run's seconds per audio minute for ``route`` (the last 10 are kept). Needs the lock."""
    lock.require_held()
    routes = speed_load()
    routes.setdefault(route, []).append(round(float(seconds_per_audio_minute), 3))
    routes[route] = routes[route][-SPEED_KEEP:]
    fsutil.private_dir(platforms.data_dir())
    data = (json.dumps({"version": 1, "routes": routes}, indent=1) + "\n").encode("utf-8")
    fsutil.atomic_write(speed_path(), data, overwrite=True, mode=0o600)


def speed_median(route: str) -> float | None:
    v = speed_load().get(route)
    return float(statistics.median(v)) if v else None


# --------------------------------------------------------------------------- REF: id, audio path, JSON path
@dataclass
class Ref:
    """What a REF names. ``record`` is the newest record of the transcript's chain when it is still kept."""

    ref: str
    record: dict[str, Any] | None          # the newest record of the transcript's chain (the one its JSON names)
    json_path: Path | None
    doc: dict[str, Any] | None
    json_sha: str | None
    audio: Path | None
    named: dict[str, Any] | None = None    # the record the REF itself names: the id's, or the newest for the path

    @property
    def id(self) -> str | None:
        if self.record is not None:
            return self.record["id"]
        return (self.doc or {}).get("record_id")

    @property
    def named_id(self) -> str | None:
        return self.named["id"] if self.named is not None else None


def json_output(rec: Mapping[str, Any] | None) -> Path | None:
    """The transcript JSON a record wrote (or found already done), or None."""
    for o in (rec or {}).get("outputs") or []:
        if o.get("kind") == "json":
            return Path(o["path"])
    return None


_json_output = json_output


def _records_for_audio(path: Path) -> Iterator[dict[str, Any]]:
    for d in iter_records():
        if (d.get("input") or {}).get("path") == str(path):
            yield d


def _with_transcript(path: Path) -> dict[str, Any] | None:
    """The newest record for a recording whose transcript JSON is still where it wrote it; a later run that wrote
    none (``--plan``, a failure, a notes file) doesn't hide it."""
    for d in _records_for_audio(path):
        jp = json_output(d)
        if jp is not None and jp.is_file():
            return d
    return None


def _transcript_beside(path: Path) -> Path | None:
    for stem in (fsutil.stem(path.name), path.name):
        cand = path.parent / outputs.name_for(stem, "json")
        if cand.is_file():
            return cand
    return None


def _abs(text: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(text)))


def resolve(ref: str, *, need_json: bool = True, command: str = "") -> Ref:
    """Turn a REF into the transcript it names, checked.

    A record id loads that record; a path ending in ``(transcript).json`` is the transcript; any other path is a
    recording, found through the newest record that names it or the transcript beside it. The transcript JSON
    then names the newest record of its chain, which is used when it's still kept. The JSON must be exactly what
    audiointake wrote (its seal, and the hash its record lists): otherwise Stale ``transcript_edited``.
    ``need_json`` makes a missing transcript UsageError ``no_transcript_json``.
    """
    rec: dict[str, Any] | None = None
    json_path: Path | None = None
    audio: Path | None = None
    given_audio = False
    if is_id(ref):
        rec = load(ref)
        json_path = _json_output(rec)
    else:
        p = _abs(ref)
        if p.name.endswith(outputs.SUFFIX + ".json"):
            json_path = p
            if not p.is_file():
                raise UsageError("unknown_record", id=ref, path=str(p))
            rec = next((d for d in iter_records() if any(o.get("kind") == "json" and o.get("path") == str(p)
                                                         for o in d.get("outputs") or [])), None)
        else:
            audio, given_audio = p, True
            rec = _with_transcript(p) or next(_records_for_audio(p), None)
            json_path = json_output(rec)
            if json_path is None or not json_path.is_file():
                json_path = _transcript_beside(p) or json_path
            if rec is None and json_path is None:
                raise UsageError("unknown_record", id=ref, path=str(p))
    named = rec
    doc = sha = None
    if json_path is not None and json_path.is_file():
        data = json_path.read_bytes()
        doc = outputs.parse_transcript(data)
        sha = fsutil.sha256_bytes(data)
        if doc is None or not outputs.seal_ok(data):
            raise Stale("transcript_edited", path=str(json_path), file=_input_name(doc, rec, audio))
        newest = try_load(doc.get("record_id"))
        if newest is not None:
            listed = {o.get("path"): o.get("sha256") for o in newest.get("outputs") or [] if o.get("kind") == "json"}
            if listed.get(str(json_path)) not in (None, sha):
                raise Stale("transcript_edited", path=str(json_path), file=_input_name(doc, newest, audio))
            rec = newest
        if not given_audio:
            audio = _recording_of(rec, json_path, doc)
    elif need_json:
        raise UsageError("no_transcript_json", command=command or "this command",
                         path=str(json_path) if json_path else ref)
    elif not given_audio and rec is not None and (rec.get("input") or {}).get("path"):
        audio = Path(rec["input"]["path"])
    return Ref(ref=ref, record=rec, json_path=json_path, doc=doc, json_sha=sha, audio=audio, named=named)


def _recording_of(rec: Mapping[str, Any] | None, json_path: Path, doc: Mapping[str, Any]) -> Path:
    """Where a transcript's recording is: where the record says, unless that's gone and the recording sits beside
    the transcript JSON (the folder was moved with both, or the record has expired)."""
    beside = json_path.parent / str((doc.get("input") or {}).get("name") or "")
    recorded = Path(rec["input"]["path"]) if rec is not None and (rec.get("input") or {}).get("path") else None
    if recorded is not None and (recorded.is_file() or not beside.is_file()):
        return recorded
    return beside


def reseal(r: Ref, *, required: bool) -> Path | None:
    """The recording a transcript was made from, checked against the SHA-256 the transcript names.

    A changed recording is Stale ``input_changed``. A missing one is ``input_missing`` when ``required`` (apply,
    enroll read it); relabel and notes don't read the audio, so they carry on without it.
    """
    want = ((r.doc or {}).get("input") or {}).get("sha256") or ((r.record or {}).get("input") or {}).get("sha256")
    p = r.audio
    if p is None or not p.is_file():
        if required:
            raise UsageError("input_missing", file=str(p) if p else r.ref)
        return None
    if want and fsutil.sha256_file(p) != want:
        raise Stale("input_changed", file=p.name, id=r.id or "")
    return p


def _input_name(doc: Mapping[str, Any] | None, rec: Mapping[str, Any] | None, audio: Path | None) -> str:
    if doc and (doc.get("input") or {}).get("name"):
        return str(doc["input"]["name"])
    if rec and (rec.get("input") or {}).get("path"):
        return Path(rec["input"]["path"]).name
    return audio.name if audio else ""


def show(ref: str) -> dict[str, Any]:
    """The record a REF names: the id's own record, or for a path the newest record of its transcript (once the
    record that wrote the JSON has expired, a later one made from it, such as its notes record)."""
    if is_id(ref):
        return load(ref)
    r = resolve(ref, need_json=False, command="record show")
    if r.record is None:
        later = next(_later_children(r), None)
        if later is None:
            raise UsageError("unknown_record", id=ref, path=str(r.json_path or ref))
        return later
    return r.record


def find(path: str) -> dict[str, Any]:
    """For a recording or transcript path: the newest record id, the speaker labels and their displays. No text."""
    r = resolve(path, need_json=True, command="record find")
    doc = r.doc or {}
    labels = sorted({s["spk"] for s in segments(doc.get("rttm"))}) if doc.get("rttm") else []
    return {"id": r.id, "path": str(r.json_path), "labels": labels, "displays": dict(doc.get("displays") or {})}


def segments(text: str | None) -> list[dict[str, Any]]:
    """RTTM turns ({start, dur, spk}) by the engine's one RTTM reader (``steps.rttm``); an RTTM it can't read
    has no turns here either, so the record's speakers agree with the speaker count the diarmeta line gives."""
    if not text:
        return []
    from .steps import rttm
    try:
        return [dict(s) for s in rttm.segments_from_text(text)]
    except ValueError:
        return []


def labels_of(rttm_text: str | None) -> list[str]:
    return sorted({s["spk"] for s in segments(rttm_text)})


# --------------------------------------------------------------------------- a transcript's open offers
def _later(r: Ref) -> bool:
    """Whether the record a REF names came after the one that wrote the transcript JSON (a re-run that found it
    already done, a notes file), rather than before it (a record the JSON has since replaced)."""
    n = r.named
    if n is None:
        return False
    if r.record is None:
        return True
    return n.get("id") != r.record.get("id") and created_at(n) >= created_at(r.record)


def _later_children(r: Ref) -> Iterator[dict[str, Any]]:
    """Records made from this transcript after its JSON was written that finished and wrote no other JSON: a re-run
    that found it already done (its ``names`` offer when the voiceprints changed), a notes file. A follow-up
    that failed doesn't count: what it found describes a transcript that was never written."""
    rid = r.id
    if not rid:
        return
    since = created_at(r.record) if r.record is not None else 0.0
    for d in iter_records():
        if d.get("parent") != rid or d.get("id") == rid or created_at(d) < since or d.get("status") != "done":
            continue
        jp = json_output(d)
        if jp is None or (r.json_path is not None and str(jp) == str(r.json_path)):
            yield d


def open_offers(r: Ref) -> list[dict[str, Any]]:
    """The offers still open on a transcript: those of the record that wrote its JSON (the JSON's own once that
    record has expired), plus those of the later records that wrote no JSON of their own (``_later_children``,
    and the record the REF names when it is one of them). An earlier record of the chain never counts: what it
    offered may have been acted on since."""
    base = r.record.get("offers") if r.record is not None else (r.doc or {}).get("offers")
    out: dict[str, Any] = {"offers": []}
    carry_offers(out, base)
    for d in _later_children(r):
        carry_offers(out, d.get("offers"))
    if _later(r):
        carry_offers(out, (r.named or {}).get("offers"))
    return out["offers"]


def offers_id(r: Ref) -> str:
    """The record whose offers ``open_offers`` reads, for ``no_offer`` and ``not_an_option``: the later record the
    REF names, else the one that wrote the JSON (an earlier record's offers may have been acted on since)."""
    if _later(r) and r.named_id:
        return r.named_id
    return r.id or r.named_id or r.ref


def carry_offers(rec: dict[str, Any], offers: Any, *, drop: tuple[str, ...] = ()) -> None:
    """Add a transcript's open offers to a record of its chain, leaving out the kinds this command settles."""
    for o in offers or []:
        if not isinstance(o, Mapping) or o.get("kind") not in errors.OFFER_FLAGS or o.get("kind") in drop:
            continue
        add_offer(rec, str(o["kind"]), [str(x) for x in o.get("options") or []], params=o.get("params") or None)


def later_options(r: Ref) -> dict[str, Any]:
    """The options of the newest record a follow-up should read: a later re-run's, else the transcript's."""
    for src in ((r.named if _later(r) else None), r.record):
        if src is not None and src.get("options"):
            return dict(src["options"])
    return dict((r.doc or {}).get("options") or {})
