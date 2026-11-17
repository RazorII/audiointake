"""One recording at a time: check, seal, probe, transcribe, tell speakers apart, name them, merge, write.

``run(inputs, options, rt)`` returns one saved record per input. Per input:

1. **Check** the path (``input_missing``, ``input_not_file``, ``input_not_downloaded``, ``input_not_readable``) and
   that the output folder takes a file (``output_not_writable``); each is exit 2 for that input only.
2. **Seal**: the input's SHA-256 before any work, checked again before the commit (``input_changed``, exit 5).
3. **Probe** (``probe.probe``): no audio stream is ``unsupported_input``.
4. **Existing outputs**: a transcript made from this input with these options is ``already_done`` (or, when only the
   voiceprints changed, ``voices_changed`` with a ``names`` offer); anything else already there is
   ``output_exists`` unless ``--overwrite``. ``--plan`` stops here and records what would run.
5. **Steps**, each model in its own child, one after another: the speech ensemble (``steps.ensemble.transcribe``),
   ``<stem>.16k.wav`` and diarization, the speaker-ID worker, then in this process the merge and the slim
   sidecar. Every failure after the first model falls back exactly as the classic pipeline does, and adds a
   warning: the diarmeta rules are in ``outputs.diarmeta`` and ``_steps`` below.
6. **Commit** (``commit_outputs``, the one writer into the user's folder): everything at the end, atomically,
   with SIGTERM and SIGHUP held back until the outputs and the record are both written (``_commit_window``).

Anything a step raises outside the closed codes (a bug, or the file vanishing under the run) is logged by type and
place only and recorded as ``internal_error`` for that input; the other inputs still run.

The engine's functions (``engine()``) and setup's (``setup_api()``) are imported when first needed, so this module
imports without either; tests replace those two functions with fakes.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import signal
import stat
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import argv, errors, fsutil, outputs, proc, record, render, voices
from . import probe as probe_mod
from .contracts import LANGUAGE_NAMES, ROUTE_OF_LANG, ROUTES, SPEAKER_CAP, EnsembleResult, Merged, Problem, Runtime, Turn
from .errors import AudioIntakeError, Refused, SetupError, Stale, UsageError, problem

FORCEABLE = ("hu",)                  # a single --lang code that forces its route (only for a whole recording)
REMEMBER_MIN_S = 8.0                 # a label needs this much clean speech for a voiceprint
UNTRANSCRIBED_MIN_S = 30.0           # speech_not_transcribed: at least this long ...
UNTRANSCRIBED_SHARE = 0.25           # ... and this share of the diarized speech
SPEAKER_WORKER = "audiointake.steps.speaker_id"
GB = 1e9
# setup errors about the speaker program only: an `apply --lang` install that hits one still installed the pack
SPEAKER_PROGRAM_CODES = frozenset({"clt_missing", "xcode_license", "swift_too_old", "speakers_build_failed",
                                   "build_timeout"})


# --------------------------------------------------------------------------- the other units, imported late
def engine() -> SimpleNamespace:
    """The engine's step functions (``steps.*``), imported when a step first runs."""
    from .steps import ensemble, merge_speakers, slim_asr, speaker_id
    return SimpleNamespace(
        transcribe=ensemble.transcribe, merge=merge_speakers.merge, count_speakers=merge_speakers.count_speakers,
        NoSentences=getattr(merge_speakers, "NoSentences", ()), slim=slim_asr.slim,
        enroll_clips=speaker_id.enroll_clips, enroll_label=speaker_id.enroll_label,
        HeldBack=speaker_id.HeldBack, TooLittleSpeech=speaker_id.TooLittleSpeech,
        SpeakerIdError=getattr(speaker_id, "SpeakerIdError", ()))


def setup_api() -> SimpleNamespace:
    """Setup's functions: resolve the runtime, plan and install packs, doctor, about, uninstall."""
    from . import about, doctor
    from .setup import runtime, store
    return SimpleNamespace(resolve=runtime.resolve, plan=store.plan, install=store.install, remove=store.remove,
                           uninstall=store.uninstall, status=store.status, doctor=doctor.report, about=about.about)


# --------------------------------------------------------------------------- options and results
@dataclass
class Options:
    langs: tuple[str, ...] = ()
    speakers: int | None = None
    names: bool = True
    speakers_off: bool = False
    voices: Path | None = None
    outdir: Path | None = None
    formats: tuple[str, ...] = outputs.DEFAULT_FORMATS
    overwrite: bool = False
    plan: bool = False
    setup_problems: tuple[Problem, ...] = ()
    command: str = "transcribe"
    parent: str | None = None
    stems: Mapping[str, str] | None = None          # apply: the stem each input had (by its path)
    replace: Mapping[str, str] | None = None        # apply: path -> the SHA-256 audiointake wrote there
    expect_sha: str | None = None                   # apply: the input must still be this
    progress: Callable[[str], None] | None = None

    def force_lang(self) -> str | None:
        return self.langs[0] if len(self.langs) == 1 and self.langs[0] in FORCEABLE else None

    def digest(self) -> str:
        return outputs.options_digest(langs=self.langs, names=self.names, speakers_off=self.speakers_off)

    def as_record(self, rt: Runtime | None = None) -> dict[str, Any]:
        return {"lang": list(self.langs), "speakers": self.speakers, "names": self.names,
                "speakers_off": self.speakers_off, "format": list(self.formats),
                "voices": "custom" if self.voices else "default",
                "voices_dir": str(voices.folder(self.voices))}


@dataclass
class Result:
    """What the outputs of one recording are built from."""

    stem: str
    name: str
    sha256: str
    duration_s: float | None
    recorded_at: str | None
    spatial: str
    tag: str
    asr: dict[str, Any]
    ens_text: str
    ens_raw: bytes | None = None              # the ensemble's .txt bytes, moved as they are when the merge gave none
    txt_raw: bytes | None = None              # set when the .txt outputs are those bytes (merge failed, no sentence)
    rttm_text: str | None = None              # the RTTM the merge used (a non-empty file)
    rttm_left: bytes | None = None            # sidecar: a non-empty file left at the RTTM path
    spk_count: int | None = None
    runs: int | None = None
    named: str = "-"
    speakers_raw: bytes | None = None         # sidecar: the speaker worker's file, when non-empty
    speakers: dict[str, Any] | None = None    # the speaker worker's object, when it parsed
    names_override: dict[str, str] = field(default_factory=dict)
    displays: dict[str, str] = field(default_factory=dict)
    txt: str = ""
    turns: list[Turn] = field(default_factory=list)
    slim: str | None = None
    ensemble: EnsembleResult | None = None


_ACTIVE: dict[str, set[Path]] = {"dirs": set(), "logs": set()}


def abort() -> None:
    """The signal path: remove every work folder and in-progress run log this process has open."""
    for d in list(_ACTIVE["dirs"]):
        shutil.rmtree(d, ignore_errors=True)
    for p in list(_ACTIVE["logs"]):
        try:
            p.unlink()
        except OSError:
            pass
    _ACTIVE["dirs"].clear()
    _ACTIVE["logs"].clear()


def cleanup(rt: Runtime) -> None:
    """Delete the run's work folder (at the end of every run)."""
    shutil.rmtree(rt.work_root, ignore_errors=True)
    _ACTIVE["dirs"].discard(Path(rt.work_root))


def _work(rt: Runtime, name: str) -> Path:
    _ACTIVE["dirs"].add(Path(rt.work_root))
    d = Path(rt.work_root) / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return fsutil.private_dir(d)


def _write_work(path: Path, text: str) -> None:
    """A file inside the run's work folder (never a user's folder)."""
    path.write_text(text, encoding="utf-8")


def _remove_work(path: Path | None) -> None:
    """Delete a folder inside the run's work folder."""
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _unlink_work(path: Path) -> None:
    """Delete a file inside the run's work folder."""
    try:
        path.unlink()
    except OSError:
        pass


def _quiet(_line: str) -> None:
    pass


_DEFER: dict[str, Any] = {"depth": 0, "pending": None}


def defer_signal(signum: int) -> bool:
    """The CLI's SIGTERM/SIGHUP handler asks this first: True while a commit is being written, and the signal then
    waits for the commit and its record (``_commit_window`` raises it again right after)."""
    if _DEFER["depth"] > 0:
        _DEFER["pending"] = _DEFER["pending"] or signum
        return True
    return False


@contextmanager
def _commit_window() -> Iterator[None]:
    """SIGTERM and SIGHUP wait while a commit and its record are written, so a signal never leaves half a commit
    (some outputs, no record); one that arrived meanwhile is raised again right after, and the run stops then.
    Python runs signal handlers in the main thread whichever thread the signal reached, so this holds with
    proc's reader threads alive too."""
    _DEFER["depth"] += 1
    try:
        yield
    finally:
        _DEFER["depth"] -= 1
        if _DEFER["depth"] == 0 and _DEFER["pending"] is not None:
            signum, _DEFER["pending"] = _DEFER["pending"], None
            signal.raise_signal(signum)


def _timeout(kind: str, duration_s: float | None) -> float:
    """A child's timeout; with no known length, the engine's cap for that case (a long recording whose container
    gives no duration must not lose its speaker labels to a short timeout)."""
    if duration_s and duration_s > 0:
        return proc.child_timeout(kind, duration_s)
    from .steps.ensemble import NO_DURATION_TIMEOUT
    return float(NO_DURATION_TIMEOUT)


def _where(exc: BaseException) -> str:
    """Where an exception came from, as file:line function for its last frames; never its message (which could
    hold text from a model's output)."""
    frames = traceback.extract_tb(exc.__traceback__)[-3:]
    return "; ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames) or "?"


def _unexpected(exc: Exception, log: Callable[[str], None], file: str) -> AudioIntakeError:
    """The closed error an unexpected exception is recorded as: internal_error for that input (exit 1), with its
    type and place in the run log. A real bug still shows in the log; the other inputs of the batch still run."""
    log(f"unexpected {type(exc).__name__} at {_where(exc)}")
    return errors.error_for("internal_error", file=file)


# --------------------------------------------------------------------------- inputs
def _abs(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(p))))


def check_input(path: Path) -> os.stat_result:
    """A regular, downloaded file this process may read; else the exit-2 code that says why."""
    try:
        st = os.stat(path)
    except PermissionError:
        raise UsageError("input_not_readable", file=path.name) from None
    except OSError:
        raise UsageError("input_missing", file=str(path)) from None
    if not stat.S_ISREG(st.st_mode):
        raise UsageError("input_not_file", file=path.name)
    if fsutil.is_dataless(path):
        raise UsageError("input_not_downloaded", file=path.name)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        raise UsageError("input_not_readable", file=path.name) from None
    os.close(fd)
    return st


def stems_for(paths: Sequence[Path]) -> list[str]:
    """Each input's stem; two inputs in one run with the same stem use their full file names instead."""
    raw = [fsutil.stem(p.name) or p.name for p in paths]
    n = Counter(raw)
    return [p.name if n[s] > 1 else s for p, s in zip(paths, raw)]


# --------------------------------------------------------------------------- the one writer
def commit_outputs(items: Iterable[tuple[str, Path, bytes]], *, overwrite: bool,
                   replace: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Write each ``(kind, path, data)`` with ``fsutil.atomic_write``; return ``[{kind, path, sha256}]``.

    Every target is checked before anything is written. An existing file is ``output_exists`` unless
    ``overwrite``; with ``replace`` (apply, relabel), a file audiointake wrote and nobody changed since (its
    SHA-256 is the one listed) is replaced, and one changed since is ``output_changed`` unless ``overwrite``.
    """
    planned = []
    for kind, path, data in items:
        path = Path(path)
        ow = False
        if os.path.lexists(path):
            listed = replace.get(str(path)) if replace is not None else None
            if listed is not None:
                if not overwrite and fsutil.sha256_file(path) != listed:
                    raise Stale("output_changed", out=path.name, path=str(path))
                ow = True
            elif overwrite:
                ow = True
            else:
                raise UsageError("output_exists", out=path.name, path=str(path))
        planned.append((kind, path, data, ow))
    written = []
    created: list[Path] = []
    replaced: list[tuple[Path, _Before]] = []
    try:
        for kind, path, data, ow in planned:
            before = _before(path) if ow else None
            fsutil.atomic_write(path, data, overwrite=ow)
            if before is not None:
                replaced.append((path, before))
            elif not ow:
                created.append(path)
            written.append({"kind": kind, "path": str(path), "sha256": fsutil.sha256_bytes(data)})
    except BaseException:
        # a commit that failed half-way takes back what it did: the files it created go, and the files it replaced
        # get their old bytes back, so the transcript JSON never lists outputs that aren't the ones beside it
        for p in created:
            try:
                p.unlink()
            except OSError:
                pass
        for p, before in reversed(replaced):
            _put_back(p, before)
        raise
    return written


@dataclass
class _Before:
    """A file as it was before a commit replaced it: its bytes and mode, or the target of a symlink."""

    data: bytes | None = None
    mode: int | None = None
    link: str | None = None


def _before(path: Path) -> _Before | None:
    try:
        if path.is_symlink():
            return _Before(link=os.readlink(path))
        st = path.stat()
        if not stat.S_ISREG(st.st_mode):
            return None
        return _Before(data=path.read_bytes(), mode=stat.S_IMODE(st.st_mode))
    except OSError:
        return None


def _put_back(path: Path, before: _Before) -> None:
    """Restore one replaced file (best effort: the commit's own error is what the record reports)."""
    try:
        if before.link is not None:
            tmp = path.parent / f"{fsutil.PARTIAL_PREFIX}{os.urandom(8).hex()}"
            os.symlink(before.link, tmp)
            os.replace(tmp, path)
        elif before.data is not None:
            fsutil.atomic_write(path, before.data, overwrite=True, mode=before.mode)
    except OSError:
        pass


# --------------------------------------------------------------------------- transcribe
def run(inputs: Iterable[str | os.PathLike[str]], options: Options, rt: Runtime) -> list[dict[str, Any]]:
    """Transcribe each input in turn; one saved record per input (the run lock must be held)."""
    paths = [_abs(p) for p in inputs]
    stems = stems_for(paths)
    if options.stems:
        stems = [options.stems.get(str(p), s) for p, s in zip(paths, stems)]
    _ACTIVE["dirs"].add(Path(rt.work_root))
    use_names = options.names and not options.speakers_off
    vdigest = voices.digest(rt.voices_dir) if use_names else ""
    vnotes = voices.scan_notes(voices.folder(options.voices)) if use_names else []
    claimed: set[str] = set()
    return [_one(p, s, i, len(paths), options, rt, vdigest, vnotes, claimed)
            for i, (p, s) in enumerate(zip(paths, stems), 1)]


def _claim(outdir: Path, stem: str, formats: Sequence[str], claimed: set[str]) -> None:
    """Two inputs of one run never write the same file (the same name in two folders, with ``--outdir``): the
    later one is ``output_exists``, whatever ``--overwrite`` says."""
    paths = [str(outdir / nm) for nm in outputs.names(stem, formats)]
    for p in paths:
        if p in claimed:
            raise UsageError("output_exists", out=Path(p).name, path=p)
    claimed.update(paths)


def _seal(path: Path, o: Options) -> str:
    """The input's SHA-256; a file that went away (deleted, moved, evicted to iCloud) is ``input_changed``."""
    try:
        return fsutil.sha256_file(path)
    except OSError:
        raise Stale("input_changed", file=path.name, **({"id": o.parent} if o.parent else {})) from None


def _save(rec: dict[str, Any], runlog: record.RunLog | None) -> None:
    if runlog is not None:
        _ACTIVE["logs"].discard(runlog.path)
    record.save(rec)


def _one(path: Path, stem: str, i: int, n: int, o: Options, rt: Runtime, vdigest: str,
         vnotes: list[tuple[str, int]], claimed: set[str]) -> dict[str, Any]:
    rec = record.new(o.command, parent=o.parent)
    rec["options"] = o.as_record()
    rec["options_digest"] = o.digest()
    rec["voices_digest"] = vdigest
    rec["input"] = {"path": str(path), "sha256": None, "size": None, "duration_s": None, "channels": None,
                    "spatial": None, "recorded_at": None, "recorded_at_source": None}

    def say(step: str) -> None:
        if o.progress:
            o.progress(f"{i} of {n}: {path.name}: {step}")

    log: Callable[[str], None] = _quiet
    runlog: record.RunLog | None = None
    work: Path | None = None
    t0 = time.monotonic()
    try:
        st = check_input(path)
        outdir = _abs(o.outdir) if o.outdir else path.parent
        if not outdir.is_dir() or not fsutil.probe_writable(outdir):
            raise UsageError("output_not_writable", folder=str(outdir))
        _claim(outdir, stem, o.formats, claimed)
        say("checking")
        sha = _seal(path, o)
        if o.expect_sha and sha != o.expect_sha:
            raise Stale("input_changed", file=path.name, **({"id": o.parent} if o.parent else {}))
        rec["input"].update(sha256=sha, size=st.st_size)
        info = probe_mod.probe(path, rt.tools, env=rt.env, workdir=rt.work_root)
        if not info["has_audio"]:
            raise UsageError("unsupported_input", file=path.name)
        rec["input"].update({k: info.get(k) for k in ("duration_s", "channels", "spatial", "recorded_at",
                                                      "recorded_at_source")})
        if info.get("recorded_at_source") == "file_date":
            record.add(rec, problem("recorded_at_from_file_date"))
        for p in o.setup_problems:
            record.add(rec, p)
        for code, count in vnotes:
            record.add(rec, problem(code, count=count))
        if o.replace is None and not o.overwrite:
            if _resume(path, stem, outdir, sha, o, rec, vdigest):
                record.save(rec)
                return rec
            for name in outputs.names(stem, o.formats):
                if os.path.lexists(outdir / name):
                    raise UsageError("output_exists", out=name, path=str(outdir / name))
        if o.plan:
            _plan(rec, info, o, rt)
            record.finish(rec, 0, status="plan")
            record.save(rec)
            return rec
        runlog = record.RunLog(rec["id"])
        _ACTIVE["logs"].add(runlog.path)
        rec["log"] = str(runlog.path)
        log = runlog
        log(f"audiointake {rec['tool']['version']} {o.command}; routes {sorted(rt.routes)}; formats {list(o.formats)}")
        work = _work(rt, f"{i:03d}")
        res = _steps(path, stem, sha, info, o, rt, work, log, say, rec)
        say("writing")
        if _seal(path, o) != sha:
            raise Stale("input_changed", file=path.name, **({"id": o.parent} if o.parent else {}))
        items = build_items(res, rec, o.formats, outdir, fluidaudio_tag=rt.tools.fluidaudio_tag)
        with _commit_window():
            rec["outputs"] = commit_outputs(items, overwrite=o.overwrite, replace=o.replace)
            secs = time.monotonic() - t0
            dur = res.duration_s or 0.0
            rec["timing"] = {"seconds": round(secs, 1), "rtf": round(secs / dur, 4) if dur else None}
            if dur > 0:
                try:
                    record.speed_add(_speed_route(res.ensemble), secs / (dur / 60.0))
                except OSError:                        # the estimate is a courtesy; the transcript is written
                    log("speed.json: not updated")
            log(f"done in {secs:.1f} s")
            record.finish(rec, 0)
            _save(rec, runlog)
        return rec
    except AudioIntakeError as exc:
        log(f"error: {exc.code}")
        record.fail(rec, exc)
    except Exception as exc:                           # outside the closed set: this input fails, the batch goes on
        if runlog is None:
            runlog = record.RunLog(rec["id"])
            _ACTIVE["logs"].add(runlog.path)
            rec["log"] = str(runlog.path)
            log = runlog
        record.fail(rec, _unexpected(exc, log, path.name))
    finally:
        _remove_work(work)
    _save(rec, runlog)
    return rec


def _speed_route(er: EnsembleResult | None) -> str:
    if er is not None and er.hu_route == "vote":
        return "hu_vote"
    if er is not None and er.thai.turns:
        return "thai_turns"
    return "core"


def _plan(rec: dict[str, Any], info: Mapping[str, Any], o: Options, rt: Runtime) -> None:
    routes = ["asr"] + [r for r in ROUTES if r in rt.routes]
    if not o.speakers_off and rt.tools.fluidaudiocli is not None:
        routes.append("diarization")
        if o.names and rt.voices_dir is not None:
            routes.append("speaker_id")
    spm = record.speed_median("core")
    dur = info.get("duration_s")
    rec["plan"] = {"routes": routes, "estimate_s": round(spm * dur / 60.0, 1) if spm and dur else None}


def _resume(path: Path, stem: str, outdir: Path, sha: str, o: Options, rec: dict[str, Any], vdigest: str) -> bool:
    """``already_done`` / ``voices_changed`` when this input, these options and the requested outputs are what an
    earlier run wrote and nobody changed them."""
    if "json" not in o.formats or "sidecars" in o.formats:
        return False
    jp = outdir / outputs.name_for(stem, "json")
    if not jp.is_file():
        return False
    data = jp.read_bytes()
    doc = outputs.parse_transcript(data)
    if doc is None or not outputs.seal_ok(data):
        return False
    if (doc.get("input") or {}).get("sha256") != sha or doc.get("options_digest") != rec["options_digest"]:
        return False
    listed = {x.get("kind"): x for x in doc.get("outputs") or []}
    existing = []
    for kind in ("md", "txt"):
        if kind in o.formats:
            p = outdir / outputs.name_for(stem, kind)
            if kind not in listed or not p.is_file() or fsutil.sha256_file(p) != listed[kind].get("sha256"):
                return False
            existing.append({"kind": kind, "path": str(p), "sha256": listed[kind]["sha256"]})
    existing.append({"kind": "json", "path": str(jp), "sha256": fsutil.sha256_bytes(data)})
    rec["outputs"] = existing
    rec["parent"] = doc.get("record_id")
    rec["engines"] = doc.get("engines")
    record.carry_offers(rec, doc.get("offers"))          # what the transcript still offers stays on offer
    if doc.get("rttm") and o.names and not o.speakers_off and str(doc.get("voices_digest") or "") != vdigest:
        record.add(rec, problem("voices_changed", file=path.name))
        record.add_offer(rec, "names", [])
    else:
        record.add(rec, problem("already_done", file=path.name))
    record.finish(rec, 0)
    return True


def _call_transcribe(eng: SimpleNamespace, path: Path, work: Path, stem: str, rt: Runtime, o: Options,
                     log: Callable[[str], None], duration: float | None) -> EnsembleResult:
    kw: dict[str, Any] = {"force_lang": o.force_lang(), "log": log}
    try:
        if "duration_s" in inspect.signature(eng.transcribe).parameters:
            kw["duration_s"] = duration
    except (TypeError, ValueError):
        pass
    return eng.transcribe(path, work, stem, "all", rt, **kw)


def _steps(path: Path, stem: str, sha: str, info: Mapping[str, Any], o: Options, rt: Runtime, work: Path,
           log: Callable[[str], None], say: Callable[[str], None], rec: dict[str, Any]) -> Result:
    eng = engine()
    say("transcribing")
    dur = info.get("duration_s")
    try:
        er = _call_transcribe(eng, path, work, stem, rt, o, log, dur)
    except AudioIntakeError as exc:
        code = exc.code if exc.code in ("asr_failed", "decode_failed") else "asr_failed"
        log(f"asr: {code}")
        raise errors.error_for(code, file=path.name) from None
    jp = er.json_path or (work / f"{stem}.json")
    try:
        if jp.stat().st_size == 0:
            raise ValueError("empty")
        asr = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(asr, dict):
            raise ValueError("not an object")
    except (OSError, ValueError):
        log("asr: the ensemble wrote no transcript")
        raise errors.error_for("asr_failed", file=path.name) from None
    tp = er.txt_path or (work / f"{stem}.txt")
    ens_raw: bytes | None
    try:
        ens_raw = tp.read_bytes()                   # bytes, as the classic pipeline moves the file: no newline change
        ens_text = ens_raw.decode("utf-8", errors="replace")
    except OSError:
        ens_raw, ens_text = None, outputs.plain_text(asr)
    log(f"asr: {er.tag}")
    res = Result(stem=stem, name=path.name, sha256=sha, duration_s=dur, recorded_at=info.get("recorded_at"),
                 spatial=info.get("spatial") or "no", tag=er.tag, asr=asr, ens_text=ens_text, ens_raw=ens_raw,
                 ensemble=er)
    _ensemble_problems(er, rec, o)

    names_map: dict[str, str] = {}
    if not o.speakers_off and rt.tools.fluidaudiocli is not None and "nemotron" in rt.models:
        say("speakers")
        wav = _diarize(eng, path, stem, rt, work, log, res, rec)
        if wav is not None and res.rttm_text is not None and rt.voices_dir is not None and "campplus" in rt.models:
            say("names")
            names_map = _match(wav, work / f"{stem}.diar.rttm", rt, work, stem, log, res, rec, rt.voices_dir)
        if wav is not None:
            _unlink_work(wav)
    say("merging")
    _merge_into(eng, res, names_map, log, rec)
    try:
        res.slim = eng.slim(asr)
    except Exception as exc:                       # the slim sidecar is best-effort, as it always was
        log(f"slim: skipped ({type(exc).__name__})")
        res.slim = None
    _speaker_problems(res, rec, o)
    _fill_record(rec, res, rt)
    return res


def _ensemble_problems(er: EnsembleResult, rec: dict[str, Any], o: Options) -> None:
    if not er.fill_ok:
        record.add(rec, problem("fill_failed"))
    if er.hu_route == "vote_failed":
        record.add(rec, problem("hungarian_vote_failed"))
    if er.thai.failed:
        record.add(rec, problem("thai_failed"))
    asked = {str(p.params.get("code")) for p in o.setup_problems if p.code == "pack_not_installed"}
    if er.hu_route == "no_pack" and "hu" not in asked:
        detected_pack(rec, "hu", share=er.hu_share * 100.0)
    if er.thai.screen_only and "th" not in asked:
        detected_pack(rec, "th")


def _pack_plan(code: str) -> dict[str, Any] | None:
    try:
        plan = setup_api().plan([code])
    except Exception:                               # sizes are a courtesy: no plan, no size in the message
        return None
    return plan if isinstance(plan, Mapping) else None


def detected_pack(rec: dict[str, Any], code: str, *, share: float | None = None) -> None:
    """``pack_not_installed`` for a detected language, with a ``language`` offer (or the ``no_room`` wording)."""
    plan = _pack_plan(code) or {}
    plan = plan.get("plan", plan) if isinstance(plan.get("plan", plan), Mapping) else plan
    language = LANGUAGE_NAMES[code]
    need, free = plan.get("disk_needed_bytes"), plan.get("disk_free_bytes")
    if isinstance(need, (int, float)) and isinstance(free, (int, float)) and free < need:
        record.add(rec, problem("pack_not_installed", reason="no_room", code=code, language=language,
                                need=need / GB, free=free / GB))
        return
    params: dict[str, Any] = {"reason": "detected", "code": code, "language": language}
    per = plan.get("packs")
    size = per.get(code) if isinstance(per, Mapping) else None      # the pack's own download, not what it requires
    if not isinstance(size, (int, float)):
        size = plan.get("fetch_bytes")
    if isinstance(size, (int, float)):
        params["size"] = size / GB
    if share is not None:
        params["share"] = share
        params.update(errors.pack_wer_params(code))
    record.add(rec, problem("pack_not_installed", **params))
    record.add(rec, problem("language", **{k: v for k, v in params.items() if k in ("code", "language", "size")}))


def _allowed(rt: Runtime) -> frozenset[Path]:
    return proc.allowed_for(rt.tools)


def _diarize(eng: SimpleNamespace, path: Path, stem: str, rt: Runtime, work: Path, log: Callable[[str], None],
             res: Result, rec: dict[str, Any]) -> Path | None:
    """``<stem>.16k.wav``, then Nemotron. Returns the wav when it was made (speaker ID reads it).

    The classic rules: a failed conversion or a failed diarizer leaves ``speakers=?`` and ``runs=-``; a diarizer
    that succeeds gives ``runs=1`` and the label count (0 for an empty RTTM, and 0 when counting fails); the RTTM is
    used only when the file isn't empty, and the sidecar is kept whenever a non-empty file was left, even by a
    diarizer that failed.
    """
    t = rt.tools
    dur = res.duration_s
    wav = argv.diar_wav_path(work, stem)
    c = proc.run(argv.diar_wav(t.ffmpeg, path, wav), env=rt.env, timeout=_timeout("ffmpeg", dur),
                 allowed=_allowed(rt), stdout="discard")
    if not c.ok:
        log(f"diarization: the 16 kHz conversion failed (exit {c.returncode})")
        record.add(rec, problem("speakers_failed", step="audio conversion", fallback="no speaker labels"))
        return None
    assert t.fluidaudiocli is not None
    rttm = work / f"{stem}.diar.rttm"
    c = proc.run(argv.diarize(t.fluidaudiocli, wav, rttm, models=rt.models["nemotron"].path), env=rt.env,
                 timeout=_timeout("fluidaudio", dur), allowed=_allowed(rt), stdout="discard")
    nonempty = rttm.is_file() and rttm.stat().st_size > 0
    if c.ok:
        text = rttm.read_text(encoding="utf-8", errors="replace") if rttm.is_file() else ""
        try:
            res.spk_count = int(eng.count_speakers(text))
        except Exception as exc:
            log(f"diarization: counting speakers failed ({type(exc).__name__})")
            res.spk_count = 0
        res.runs = 1
        if nonempty:
            res.rttm_text = text
        log(f"diarization: {res.spk_count} speakers")
    else:
        log(f"diarization: failed (exit {c.returncode}{', timed out' if c.timed_out else ''})")
        for ln in c.stderr_tail:
            log(f"fluidaudiocli: {ln}")
        record.add(rec, problem("speakers_failed", step="diarization", fallback="no speaker labels"))
    if nonempty:
        res.rttm_left = rttm.read_bytes()
    return wav


def named_line(path: Path) -> str:
    """diarmeta's ``named=``: the sorted names the speaker file gives, comma-joined; "-" for none; "?" when the
    file can't be read."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        n = sorted({v["name"] for v in d["labels"].values() if v.get("name")})
        return ",".join(n) or "-"
    except Exception:
        return "?"


def names_for_merge(path: Path) -> dict[str, str]:
    """The merge's names from the speaker file; an unreadable file names nobody."""
    try:
        with open(path, encoding="utf-8") as f:
            return {k: v["name"] for k, v in json.load(f).get("labels", {}).items() if v.get("name")}
    except Exception:
        return {}


def _match(wav: Path, rttm: Path, rt: Runtime, work: Path, stem: str, log: Callable[[str], None], res: Result,
           rec: dict[str, Any], voices_dir: Path) -> dict[str, str]:
    """The speaker-ID worker (a child). Returns the names for the merge; fills ``named`` and the sidecar."""
    out = work / f"{stem}.speakers.json"
    cmd = [rt.tools.python, "-I", "-B", "-m", SPEAKER_WORKER, "match", wav, rttm, "--out", out, "--voices",
           voices_dir, "--model", rt.models["campplus"].path]
    c = proc.run(cmd, env=rt.env, timeout=_timeout("speaker_id", res.duration_s),
                 allowed=_allowed(rt), stdout="discard")
    names_map: dict[str, str] = {}
    if c.ok:
        res.named = named_line(out)
        names_map = names_for_merge(out)
        try:
            d = json.loads(out.read_text(encoding="utf-8"))
            res.speakers = d if isinstance(d, dict) else None
        except (OSError, ValueError):
            res.speakers = None
        log(f"speaker ID: {len(names_map)} named")
    else:
        log(f"speaker ID: failed (exit {c.returncode}{', timed out' if c.timed_out else ''})")
        for ln in c.stderr_tail:
            log(f"speaker ID: {ln}")
        record.add(rec, problem("speakers_failed", step="speaker ID", fallback="numbered speakers"))
    if out.is_file() and out.stat().st_size > 0:
        res.speakers_raw = out.read_bytes()
    return names_map


def _merge_into(eng: SimpleNamespace, res: Result, names_map: Mapping[str, str], log: Callable[[str], None],
                rec: dict[str, Any]) -> None:
    """Merge words and speakers; on failure the ensemble's one-line text stands (``merge_failed``)."""
    merged: Merged | None = None
    try:
        merged = eng.merge(res.asr, res.rttm_text, dict(names_map) or None, True)
    except eng.NoSentences:
        merged = None
    except Exception as exc:
        log(f"merge: failed ({type(exc).__name__})")
        record.add(rec, problem("merge_failed"))
        merged = None
    res.txt = merged.text if merged is not None else res.ens_text
    res.txt_raw = res.ens_raw if merged is None else None
    if merged is not None and res.rttm_text is not None and any(t.label for t in merged.turns):
        res.turns = list(merged.turns)
        shown: dict[str, str] = {}
        for t in res.turns:
            if t.label and t.label not in shown:
                shown[t.label] = t.display
        res.displays = {**{k: v for k, v in names_map.items() if k in labels_in(res.rttm_text)}, **shown}
    else:
        res.turns = render.paragraphs(_sentences(res.asr))
        res.displays = {}


def _sentences(asr: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    s = (asr or {}).get("sentences") if isinstance(asr, Mapping) else None
    return [x for x in (s or []) if isinstance(x, Mapping)]


def labels_in(rttm_text: str | None) -> list[str]:
    return record.labels_of(rttm_text)


# --------------------------------------------------------------------------- speech arithmetic (numbers only)
def clean_seconds(segs: Sequence[Mapping[str, Any]], label: str) -> float:
    """Seconds of ``label``'s speech nobody else talks over, by the speaker-ID engine's own cut
    (``steps.speaker_id.clean_pieces``: 0.25 s trimmed at each edge and cut, pieces of 1.5 s or more)."""
    from .steps.speaker_id import clean_pieces
    rows = [(float(s["start"]), float(s["start"]) + float(s["dur"]), s["spk"]) for s in segs]
    return round(sum(b - a for a, b in clean_pieces(rows, label)), 1)


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def untranscribed(segs: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """(seconds of diarized speech no transcribed sentence covers, seconds of diarized speech)."""
    speech = _union((s["start"], s["start"] + s["dur"]) for s in segs)
    covered = _union((float(s.get("start") or 0.0), float(s.get("end") or 0.0)) for s in sentences
                     if str(s.get("text") or "").strip() and s.get("start") is not None)
    total = sum(b - a for a, b in speech)
    overlap = 0.0
    for a, b in speech:
        for c, d in covered:
            overlap += max(0.0, min(b, d) - max(a, c))
    return max(0.0, total - overlap), total


def _speaker_problems(res: Result, rec: dict[str, Any], o: Options) -> None:
    n_sent, n_words = outputs.word_count(res.asr)
    if n_words == 0:
        record.add(rec, problem("no_speech", file=res.name))
    if res.spk_count is not None:
        if res.spk_count >= SPEAKER_CAP:
            record.add(rec, problem("speaker_cap"))
        if o.speakers is not None and res.spk_count != o.speakers:
            record.add(rec, problem("speaker_count_differs", expected=o.speakers, found=res.spk_count))
    if res.rttm_text:
        segs = record.segments(res.rttm_text)
        gone, total = untranscribed(segs, _sentences(res.asr))
        if gone >= UNTRANSCRIBED_MIN_S and total > 0 and gone / total >= UNTRANSCRIBED_SHARE:
            record.add(rec, problem("speech_not_transcribed", minutes=int(gone // 60), seconds=int(gone % 60)))


def speakers_list(res: Result) -> list[dict[str, Any]]:
    """The record's speakers: each label with its display and seconds; ``name`` only when named."""
    segs = record.segments(res.rttm_text)
    from_id = (res.speakers or {}).get("labels") or {}
    out = []
    for label in sorted({s["spk"] for s in segs}):
        v = from_id.get(label) if isinstance(from_id.get(label), Mapping) else None
        item: dict[str, Any] = {"label": label, "display": res.displays.get(label),
                                "speech_s": (v or {}).get("speech_s", clean_seconds(segs, label))}
        if v is not None:
            if v.get("name"):
                item["name"] = v["name"]
            item["score"] = v.get("score")
            item["why"] = v.get("why")
        if label in res.names_override:
            item["name"] = res.names_override[label]
            item["why"] = "named by the user"
        out.append(item)
    return out


def _remember(rec: dict[str, Any], res: Result, labels: Iterable[str]) -> None:
    speech = {s["label"]: s.get("speech_s") or 0 for s in rec["speakers"]}
    keep = [lab for lab in sorted(set(labels)) if float(speech.get(lab) or 0) >= REMEMBER_MIN_S]
    if keep:
        # each label's seconds travel with the offer, so a record that carries it (a re-run found already done, a
        # notes record) words it right without a speakers list of its own
        record.add_offer(rec, "remember", keep, params={"speech_s": {lab: float(speech[lab]) for lab in keep}})


def _named_labels(res: Result) -> set[str]:
    return set(voices.labels_named(res.speakers or {})) | set(res.names_override)


def _fill_record(rec: dict[str, Any], res: Result, rt: Runtime | None) -> None:
    er = res.ensemble
    rec["engines"] = engines(res, rt)
    if er is not None:
        rec["signals"] = {"hu_word_share": round(er.hu_share, 4), "hu_detected": er.hu_detected,
                          "hu_route": er.hu_route, "thai_screened": er.thai.screened,
                          "thai_windows": er.thai.thai_windows, "thai_turns": er.thai.turns,
                          "thai_seconds": round(er.thai.seconds, 1), "speaker_count": res.spk_count}
    rec["speakers"] = speakers_list(res)
    n_sent, n_words = outputs.word_count(res.asr)
    rec["counts"] = {"sentences": n_sent, "words": n_words, "turns": len(res.turns)}
    if res.rttm_text and rec["command"] in ("transcribe", "apply"):
        named = _named_labels(res)
        _remember(rec, res, [s["label"] for s in rec["speakers"] if s["label"] not in named])


def installed_packs() -> list[dict[str, Any]]:
    """``[{id, version}]`` for each pack setup reports installed (``store.status()``)."""
    try:
        rows = setup_api().status()
    except Exception:                               # the record's pack list is provenance, never a reason to fail
        return []
    return [{"id": r.get("id"), "version": r.get("version")} for r in rows or []
            if isinstance(r, Mapping) and r.get("installed")]


def engines(res: Result, rt: Runtime | None) -> dict[str, Any]:
    speaker_engine = (res.speakers or {}).get("engine") if res.speakers else None
    diarizer = outputs.diarizer_name(rt.tools.fluidaudio_tag) if res.runs and rt is not None else None
    return {"asr_tag": res.tag, "diarizer": diarizer, "speaker_id": speaker_engine, "packs": installed_packs()}


# --------------------------------------------------------------------------- building the files
def build_items(res: Result, rec: Mapping[str, Any], formats: Sequence[str], outdir: Path,
                fluidaudio_tag: str | None = None) -> list[tuple[str, Path, bytes]]:
    """Every file the formats ask for, as ``(kind, path, bytes)``; the transcript JSON last, listing the others."""
    stem = res.stem
    items: list[tuple[str, Path, bytes]] = []
    txt = res.txt_raw if res.txt_raw is not None else res.txt.encode("utf-8")
    if "sidecars" in formats:
        tag = fluidaudio_tag or _fa_tag(rec)
        items.append(("sidecar", outdir / outputs.sidecar_name(stem, "txt"), txt))
        if res.slim:
            items.append(("sidecar", outdir / outputs.sidecar_name(stem, "asr"), res.slim.encode("utf-8")))
        if res.rttm_left:
            items.append(("sidecar", outdir / outputs.sidecar_name(stem, "rttm"), res.rttm_left))
        if res.speakers_raw:
            items.append(("sidecar", outdir / outputs.sidecar_name(stem, "speakers"), res.speakers_raw))
        line = outputs.diarmeta(speakers=res.spk_count, runs=res.runs, fluidaudio_tag=tag, spatial=res.spatial,
                                asr_tag=res.tag, named=res.named)
        items.append(("sidecar", outdir / outputs.sidecar_name(stem, "diarmeta"), line.encode("utf-8")))
    if "txt" in formats:
        items.append(("txt", outdir / outputs.name_for(stem, "txt"), txt))
    if "md" in formats:
        items.append(("md", outdir / outputs.name_for(stem, "md"), md_bytes(res)))
    if "json" in formats:
        siblings = [{"kind": k, "name": p.name, "sha256": fsutil.sha256_bytes(d)} for k, p, d in items]
        items.append(("json", outdir / outputs.name_for(stem, "json"), json_bytes(res, rec, siblings)))
    return items


def _fa_tag(rec: Mapping[str, Any]) -> str:
    """The FluidAudio tag for diarmeta when the caller gave none: the record's diarizer, else setup's pin."""
    d = ((rec.get("engines") or {}).get("diarizer") or "")
    if d.startswith("fluidaudio-"):
        return d.split("-")[1]
    from .setup.fluidaudio import TAG
    return TAG


def md_bytes(res: Result) -> bytes:
    speakers = None
    if res.spk_count is not None:
        labels = set(labels_in(res.rttm_text))
        named = {lab for lab in labels if lab in _named_labels(res)}
        speakers = (res.spk_count, len(named))
    text = render.md_file(title=res.stem, recorded_at=res.recorded_at, duration_s=res.duration_s,
                                turns=res.turns, speakers=speakers)
    return text.encode("utf-8")


def json_bytes(res: Result, rec: Mapping[str, Any], siblings: list[dict[str, Any]]) -> bytes:
    opts = dict(rec.get("options") or {})
    doc = {
        "record_id": rec["id"],
        "input": {"name": res.name, "sha256": res.sha256, "duration_s": res.duration_s,
                  "recorded_at": res.recorded_at},
        "options_digest": rec.get("options_digest"), "voices_digest": rec.get("voices_digest") or "",
        # never a folder path: users share transcripts, and a path can name the account or a client's folder.
        # A custom --voices folder is kept on the record only (after it expires, apply can't name it).
        "options": {k: opts.get(k) for k in ("lang", "names", "speakers_off", "format", "voices")},
        "engines": rec.get("engines"),
        "outputs": siblings,
        "asr": res.asr,
        "rttm": res.rttm_text,
        "speakers": voices.reduce_speakers(res.speakers),
        "names_override": dict(res.names_override),
        "displays": dict(res.displays),
        "offers": [{k: o.get(k) for k in ("kind", "options", "apply_flag", "params")} for o in rec.get("offers") or []],
        "turns": [render.turn_dict(t) for t in res.turns],
    }
    return outputs.transcript_bytes(doc)


# --------------------------------------------------------------------------- apply, relabel (from the kept JSON)
def _offer(r: record.Ref, kind: str, flag: str) -> dict[str, Any]:
    for o in record.open_offers(r):
        if o.get("kind") == kind:
            return o
    raise UsageError("no_offer", id=record.offers_id(r), kind=kind, flag=flag)


def _replace_map(r: record.Ref) -> dict[str, str]:
    """path -> SHA-256 for every output the transcript's run wrote (what apply and relabel may replace).

    From the record while it's kept and its JSON is still where it wrote it; otherwise (expired, or the folder was
    moved with its transcript) from the names the JSON lists beside itself."""
    m: dict[str, str] = {}
    jp = r.json_path
    moved = r.record is not None and jp is not None and str(record.json_output(r.record)) != str(jp)
    if r.record is not None and not moved:
        for o in r.record.get("outputs") or []:
            if o.get("path") and o.get("sha256"):
                m[str(o["path"])] = str(o["sha256"])
    elif r.doc is not None and jp is not None:
        for o in r.doc.get("outputs") or []:
            if o.get("name") and o.get("sha256"):
                m[str(jp.parent / o["name"])] = str(o["sha256"])
    if jp is not None and r.json_sha:
        m[str(jp)] = r.json_sha
    return m


def _parent_options(r: record.Ref) -> dict[str, Any]:
    src = r.record if r.record is not None else None
    opts = (src or {}).get("options") or (r.doc or {}).get("options") or {}
    return dict(opts)


def _voices_for(r: record.Ref) -> tuple[bool, Path | None]:
    """(names on, the voiceprint folder) a follow-up matches against: the newest run's choice (a re-run that found
    the voiceprints changed may have named another ``--voices`` folder), else the transcript's. A custom folder that
    only an expired record knew is None: nobody is named rather than matched against the wrong set."""
    opts = record.later_options(r)
    names = bool(opts.get("names", True)) and not bool(opts.get("speakers_off", False))
    if opts.get("voices") == "custom":
        return names, (Path(opts["voices_dir"]) if opts.get("voices_dir") else None)
    return names, voices.default_dir()


def _stem_of(r: record.Ref) -> str:
    assert r.json_path is not None
    return r.json_path.name[: -len(outputs.SUFFIX + ".json")]


def result_from_doc(r: record.Ref) -> Result:
    """A Result rebuilt from the kept transcript JSON (the ASR and RTTM the first merge used)."""
    doc = r.doc or {}
    inp = doc.get("input") or {}
    spatial = ((r.record or {}).get("input") or {}).get("spatial") or "no"
    res = Result(stem=_stem_of(r), name=str(inp.get("name") or ""), sha256=str(inp.get("sha256") or ""),
                 duration_s=inp.get("duration_s"), recorded_at=inp.get("recorded_at"), spatial=spatial,
                 tag=str((doc.get("engines") or {}).get("asr_tag") or ""), asr=dict(doc.get("asr") or {}),
                 ens_text=outputs.plain_text(doc.get("asr")))
    res.rttm_text = doc.get("rttm") or None
    res.names_override = dict(doc.get("names_override") or {})
    res.displays = dict(doc.get("displays") or {})
    kept = doc.get("speakers") or {}
    if kept:
        res.speakers = {"engine": (doc.get("engines") or {}).get("speaker_id"),
                        "labels": {k: dict(v) for k, v in kept.items() if isinstance(v, Mapping)}}
    if res.rttm_text:
        res.spk_count = len(labels_in(res.rttm_text))
        res.runs = 1
    return res


def _child(command: str, r: record.Ref, *, settles: tuple[str, ...] = ()) -> dict[str, Any]:
    """A follow-up's record: the transcript's input, options and digests, and its open offers but the ones this
    command settles, so the next follow-up finds them on the newest record, the JSON and any REF."""
    rec = record.new(command, parent=r.id)
    src = r.record or {}
    rec["input"] = dict(src.get("input") or {}) or {"path": str(r.audio) if r.audio else None,
                                                   "sha256": ((r.doc or {}).get("input") or {}).get("sha256")}
    if r.audio is not None and r.audio.is_file():
        rec["input"]["path"] = str(r.audio)          # where the recording is now (a folder may have moved)
    rec["options"] = _parent_options(r)
    rec["options_digest"] = (r.doc or {}).get("options_digest")
    rec["voices_digest"] = (r.doc or {}).get("voices_digest")
    record.carry_offers(rec, record.open_offers(r), drop=settles)
    return rec


def _open_log(rec: dict[str, Any], what: str) -> record.RunLog:
    runlog = record.RunLog(rec["id"])
    _ACTIVE["logs"].add(runlog.path)
    rec["log"] = str(runlog.path)
    runlog(f"audiointake {rec['tool']['version']} {what}")
    return runlog


def _drop_log(runlog: record.RunLog | None) -> None:
    if runlog is not None:
        _ACTIVE["logs"].discard(runlog.path)
        runlog.remove()


def _prune_remember(rec: dict[str, Any], res: Result) -> None:
    """A carried ``remember`` offer keeps only labels still in the transcript and not named by a voiceprint."""
    labels = set(labels_in(res.rttm_text))
    by_print = set(voices.labels_named(res.speakers or {}))
    for o in list(rec["offers"]):
        if o["kind"] == "remember":
            o["options"] = [lab for lab in o["options"] if lab in labels and lab not in by_print]
            kept = (o.get("params") or {}).get("speech_s")
            if isinstance(kept, dict):
                o["params"]["speech_s"] = {lab: v for lab, v in kept.items() if lab in o["options"]}
            if not o["options"]:
                rec["offers"].remove(o)


def _rewrite(r: record.Ref, res: Result, rec: dict[str, Any], *, overwrite: bool, rt: Runtime | None,
             runlog: record.RunLog | None, remember: Iterable[str] = ()) -> None:
    """Write the re-merged transcript over the outputs the transcript's run wrote (md, json, txt), then save the
    record, with SIGTERM and SIGHUP held back for both.

    ``remember``: labels that get a ``remember`` offer when they have enough clean speech; added before the
    transcript JSON is built, so the JSON carries the offer too."""
    assert r.json_path is not None
    formats = tuple(f for f in (_parent_options(r).get("format") or outputs.DEFAULT_FORMATS)
                    if f in outputs.NAMED_KINDS) or ("json",)
    if "json" not in formats:
        formats = formats + ("json",)
    rec["engines"] = {**((r.doc or {}).get("engines") or {}), **({"speaker_id": res.speakers.get("engine")}
                                                                 if res.speakers and res.speakers.get("engine")
                                                                 else {})}
    if res.runs and rt is not None:
        rec["engines"]["diarizer"] = outputs.diarizer_name(rt.tools.fluidaudio_tag)
    rec["speakers"] = speakers_list(res)
    n_sent, n_words = outputs.word_count(res.asr)
    rec["counts"] = {"sentences": n_sent, "words": n_words, "turns": len(res.turns)}
    _prune_remember(rec, res)
    _remember(rec, res, remember)
    items = build_items(res, rec, formats, r.json_path.parent)
    with _commit_window():
        rec["outputs"] = commit_outputs(items, overwrite=overwrite, replace=_replace_map(r))
        record.finish(rec, 0)
        _save(rec, runlog)


def relabel(ref: str, pairs: Sequence[tuple[str, str]], *, overwrite: bool = False) -> dict[str, Any]:
    """Name labels in this transcript only, keeping every other label's display. Saves the child record."""
    r = record.resolve(ref, need_json=True, command="relabel")
    labels = labels_in((r.doc or {}).get("rttm"))
    for label, name in pairs:
        if label not in labels:
            raise UsageError("unknown_label", id=r.id or ref, label=label, labels=labels or "none")
        voices.check_name(name)
    record.reseal(r, required=False)
    rec = _child("relabel", r)
    runlog = _open_log(rec, "relabel")
    try:
        eng = engine()
        res = result_from_doc(r)
        new = dict(pairs)
        res.names_override = {**res.names_override, **new}
        displays = {lab: d for lab, d in res.displays.items() if lab in labels}
        displays.update(res.names_override)
        _merge_into(eng, res, displays, runlog, rec)
        res.displays = {**displays, **res.displays}
        _rewrite(r, res, rec, overwrite=overwrite, rt=None, runlog=runlog, remember=new)
    except BaseException:
        if rec["exit"] is None:                    # nothing was relabelled: no record, so no log either
            _drop_log(runlog)
        raise
    return rec


def _new_numbers(labels: Sequence[str], keep: Mapping[str, str], need: Iterable[str]) -> dict[str, str]:
    """"Speaker N" for labels that need a number, after the highest number already shown."""
    used = [int(d.split()[-1]) for d in keep.values() if d.startswith("Speaker ") and d.split()[-1].isdigit()]
    n = max(used, default=0)
    out = {}
    for lab in labels:
        if lab in need:
            n += 1
            out[lab] = f"Speaker {n}"
    return out


def apply_choice(ref: str, choice: str, *, overwrite: bool = False,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """``apply REF --choice speakers|names``: diarize and/or match again and re-merge from the kept ASR JSON."""
    r = record.resolve(ref, need_json=True, command="apply")
    flag = f"--choice {choice}"
    _offer(r, choice, flag)
    path = record.reseal(r, required=True)
    assert path is not None
    names, vdir = _voices_for(r)
    rt, probs = setup_api().resolve(need=frozenset(), voices_dir=vdir, names=names and vdir is not None,
                                    speakers=True)
    # labels from a new diarization replace the old ones, so every offer about labels is settled too
    settles = ("speakers", "names", "remember") if choice == "speakers" else ("names",)
    rec = _child("apply", r, settles=settles)
    rec["choice"] = choice
    runlog = _open_log(rec, f"apply --choice {choice}")
    try:
        for p in probs:
            record.add(rec, p)
        eng = engine()
        res = result_from_doc(r)
        work = _work(rt, "apply")
        if progress:
            progress(f"1 of 1: {path.name}: {choice}")
        if choice == "speakers":
            if rt.tools.fluidaudiocli is None or "nemotron" not in rt.models:
                record.carry_offers(rec, [{"kind": "speakers", "options": []}])   # still open: nothing changed
                record.finish(rec, 0)
                _save(rec, runlog)
                return rec
            res.rttm_text, res.spk_count, res.runs, res.displays, res.speakers = None, None, None, {}, None
            wav = _diarize(eng, path, res.stem, rt, work, runlog, res, rec)
            names_map: dict[str, str] = {}
            if wav is not None and res.rttm_text and rt.voices_dir is not None and "campplus" in rt.models:
                names_map = _match(wav, work / f"{res.stem}.diar.rttm", rt, work, res.stem, runlog, res, rec,
                                   rt.voices_dir)
            names_map = {**names_map, **res.names_override}
            _merge_into(eng, res, names_map, runlog, rec)
        else:
            labels = labels_in(res.rttm_text)
            before = set(voices.labels_named(res.speakers or {}))
            wav = argv.diar_wav_path(work, res.stem)
            c = proc.run(argv.diar_wav(rt.tools.ffmpeg, path, wav), env=rt.env,
                         timeout=_timeout("ffmpeg", res.duration_s), allowed=_allowed(rt), stdout="discard")
            voice_names: dict[str, str] = {}
            res.speakers = None
            if not c.ok:
                runlog(f"names: the 16 kHz conversion failed (exit {c.returncode})")
                record.add(rec, problem("speakers_failed", step="audio conversion", fallback="numbered speakers"))
            elif rt.voices_dir is not None and "campplus" in rt.models:
                rttm = work / f"{res.stem}.diar.rttm"
                _write_work(rttm, res.rttm_text or "")
                voice_names = _match(wav, rttm, rt, work, res.stem, runlog, res, rec, rt.voices_dir)
            keep = {lab: d for lab, d in res.displays.items()
                    if lab not in before and lab not in res.names_override and lab not in voice_names}
            need = [lab for lab in labels if lab not in keep and lab not in voice_names
                    and lab not in res.names_override]
            displays = {**keep, **_new_numbers(labels, keep, need), **voice_names, **res.names_override}
            _merge_into(eng, res, displays, runlog, rec)
            res.displays = {**displays, **res.displays}
        # the voiceprint set this transcript now matches: the next re-run is already_done, not a names offer
        rec["voices_digest"] = voices.digest(rt.voices_dir) if names else ""
        _speaker_problems(res, rec, Options())
        unnamed = [lab for lab in labels_in(res.rttm_text) if lab not in _named_labels(res)]
        _rewrite(r, res, rec, overwrite=overwrite, rt=rt, runlog=runlog, remember=unnamed)
        return rec
    except AudioIntakeError as exc:
        runlog(f"error: {exc.code}")
        record.fail(rec, exc)
    except Exception as exc:                       # outside the closed set: recorded, never a traceback
        record.fail(rec, _unexpected(exc, runlog, path.name))
    finally:
        cleanup(rt)
    _save(rec, runlog)
    return rec


def _pack_ready(api: SimpleNamespace, code: str) -> bool:
    """Whether setup reports the pack installed and passing its check (then ``apply --lang`` installs nothing)."""
    try:
        rows = api.status()
    except Exception:
        return False
    for row in rows or []:
        if isinstance(row, Mapping) and row.get("id") == code:
            return bool(row.get("installed")) and bool(row.get("verified", row.get("installed")))
    return False


def _install_error(got: Mapping[str, Any], code: str) -> AudioIntakeError:
    """The error that kept a pack from installing: one naming that pack, else the first that isn't only about the
    speaker program (which never stops a pack), else the first."""
    errs = [e for e in got.get("errors") or [] if isinstance(e, Mapping) and e.get("code") in errors.CODES]
    pick = (next((e for e in errs if (e.get("params") or {}).get("pack") == code), None)
            or next((e for e in errs if e.get("code") not in SPEAKER_PROGRAM_CODES), None)
            or (errs[0] if errs else {"code": "download_failed", "params": {}}))
    try:
        return errors.error_for(str(pick["code"]), **dict(pick.get("params") or {}))
    except ValueError:                             # a warning code in the error list: report the download
        return errors.error_for("download_failed")


def redo_langs(parent: Iterable[Any] | None, code: str) -> tuple[str, ...]:
    """The ``--lang`` codes ``apply --lang CODE`` transcribes again with: the transcript's own codes plus CODE.

    So CODE forces its route only when it is the one code (the transcript had no ``--lang``, or had only CODE): a
    recording transcribed with ``--lang en,hu`` stays one where nothing is forced, since Whisper told the whole
    file is Hungarian writes garbled Hungarian over the English. Every other code the transcript asked for stays
    asked for, so a pack still missing keeps its warning and its ``language`` offer."""
    return tuple(dict.fromkeys([*(str(c) for c in parent or ()), code]))


def apply_lang(ref: str, code: str, *, overwrite: bool = False,
               progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """``apply REF --lang CODE``: install that pack if it's missing (the person asked), then transcribe again with
    the transcript's ``--lang`` codes plus CODE (``redo_langs``).

    The install counts as done when setup's row for this pack says installed and verified; an error about the
    speaker program alone (no Command Line Tools) still installed the pack, and the new transcript carries
    ``no_speaker_labels`` and its ``speakers`` offer from the runtime. A pack that didn't install leaves a child
    record with setup's error.

    The new transcript's offers come from its own run, as any transcription's do: a language still asked for and
    missing (through ``need``), a language detected again, the speaker program's state and the current
    voiceprints. The old transcript's offers aren't carried over, since the run settles them afresh."""
    r = record.resolve(ref, need_json=True, command="apply")
    offer = _offer(r, "language", "--lang")
    if code not in (offer.get("options") or []):
        raise UsageError("not_an_option", option=code, id=record.offers_id(r), options=offer.get("options"))
    path = record.reseal(r, required=True)
    assert path is not None
    api = setup_api()
    extra: list[Problem] = []
    if not _pack_ready(api, code):
        # only the pack the person asked for: no speaker-program build and no setup test recording on the way
        got = api.install([code], progress=progress or _quiet, **_with_kw(api.install, speakers=False, canary=False))
        got = got if isinstance(got, Mapping) else {}
        row = next((p for p in got.get("packs") or [] if isinstance(p, Mapping) and p.get("id") == code), None)
        if row is None or not (row.get("installed") and row.get("verified", row.get("installed"))):
            rec = _child("apply", r)
            rec["lang"] = code
            record.fail(rec, _install_error(got, code))
            record.save(rec)
            return rec
        fetched = int(row.get("fetched_bytes") or 0)
        if fetched > 0:
            extra.append(problem("pack_fetched", pack=code, size=fetched / GB))
    opts = _parent_options(r)
    speakers_off = bool(opts.get("speakers_off", False))
    names, vdir = _voices_for(r)
    custom = record.later_options(r).get("voices") == "custom"
    langs = redo_langs(opts.get("lang"), code)
    need = frozenset(ROUTE_OF_LANG[c] for c in langs if c in ROUTE_OF_LANG)
    rt, probs = api.resolve(need=need, voices_dir=vdir, names=names and vdir is not None, speakers=not speakers_off)
    assert r.json_path is not None
    # a custom folder only an expired record knew: recorded honestly as a run that named nobody
    o = Options(langs=langs, names=bool(opts.get("names", True)) and not (custom and vdir is None),
                speakers_off=speakers_off, voices=vdir if custom else None,
                outdir=r.json_path.parent, formats=tuple(opts.get("format") or outputs.DEFAULT_FORMATS),
                overwrite=overwrite, setup_problems=tuple(extra) + tuple(probs), command="apply", parent=r.id,
                stems={str(path): _stem_of(r)}, replace=_replace_map(r),
                expect_sha=((r.doc or {}).get("input") or {}).get("sha256"), progress=progress)
    try:
        recs = run([path], o, rt)
    finally:
        cleanup(rt)
    return recs[0]


# --------------------------------------------------------------------------- enroll
def _enroll_record(name: str, sources: Sequence[str], parent: str | None = None) -> dict[str, Any]:
    rec = record.new("enroll", parent=parent)
    rec["voice"] = {"name": name, "sources": list(sources)}
    rec["refusals"] = []
    return rec


def _refused(rec: dict[str, Any], problems: Sequence[Problem]) -> None:
    ordered = voices.problems_in_order(list(problems))
    rec["refusals"] = [{"code": p.code, "params": dict(p.params), "message": errors.render(p.code, p.params)}
                       for p in ordered]
    first = ordered[0]
    record.fail(rec, Refused(first.code, **dict(first.params)))


def _campplus(rt: Runtime) -> Path:
    m = rt.models.get("campplus")
    if m is None:
        raise SetupError("not_set_up")
    return m.path


def _enroll_guard(folder: Path, name: str, rt: Runtime) -> None:
    """What the engine would stop on with an uncoded error, refused here in core's words first: a file under
    this name that isn't a voiceprint of this speaker model, and an ffmpeg that can't be run."""
    if not voices.addable(folder, name):
        f = voices.file_for(folder, name)
        raise UsageError("voice_other_model", file=f.name, name=name)
    if not os.access(rt.tools.ffmpeg, os.X_OK):
        raise SetupError("ffmpeg_missing")


def _engine_failure(exc: Exception, rt: Runtime) -> AudioIntakeError:
    """The speaker-ID engine's own failure (its model or sherpa-onnx missing or not the pinned file): the core
    pack is broken, and ``setup`` repairs it."""
    m = rt.models.get("campplus")
    return errors.error_for("pack_broken", pack="core", file=m.tag if m is not None else "campplus")


def _with_kw(fn: Callable[..., Any], **maybe: Any) -> dict[str, Any]:
    """The keyword arguments of ``maybe`` that ``fn`` takes (newer engine signatures add some)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in maybe.items() if k in params}


def _enroll(rec: dict[str, Any], folder: Path, name: str, rt: Runtime, call: Callable[[], Any],
            refusal: Callable[[Exception], list[Problem]]) -> None:
    """Run one engine enroll call. Any failure restores the voiceprint file as it was before the call, so a
    refusal's "Nothing was saved" holds even when earlier clips of the same call had already been added."""
    eng = engine()
    before = voices.snapshot(folder, name)
    try:
        result = call()
    except (eng.HeldBack, eng.TooLittleSpeech) as exc:
        voices.restore(folder, name, before)
        _refused(rec, refusal(exc))
        return
    except AudioIntakeError:
        voices.restore(folder, name, before)
        raise
    except eng.SpeakerIdError as exc:
        voices.restore(folder, name, before)
        raise _engine_failure(exc, rt) from None
    except BaseException:
        voices.restore(folder, name, before)
        raise
    rec["result"] = enroll_result(result)
    record.finish(rec, 0)


def enroll_clips(name: str, clips: Sequence[str | os.PathLike[str]], *, lang: str | None, voices_dir: Path,
                 force: bool, rt: Runtime) -> dict[str, Any]:
    """``enroll NAME CLIP...``: one voiceprint source per clip, checked before it's kept. Saves the record."""
    voices.check_name(name)
    paths = [_abs(c) for c in clips]
    rec = _enroll_record(name, [p.name for p in paths])
    work: Path | None = None
    try:
        for p in paths:
            check_input(p)
            if not probe_mod.probe(p, rt.tools, env=rt.env, workdir=rt.work_root)["has_audio"]:
                raise UsageError("unsupported_input", file=p.name)
        folder = voices.ensure_folder(voices_dir)
        _enroll_guard(folder, name, rt)
        eng = engine()
        model = _campplus(rt)
        work = _work(rt, "enroll")
        kw = _with_kw(eng.enroll_clips, work_dir=work)

        def call() -> Any:
            return eng.enroll_clips(name, paths, lang.upper() if lang else None, folder, model, rt.tools.ffmpeg,
                                    rt.env, force, **kw)

        def refusal(exc: Exception) -> list[Problem]:
            if isinstance(exc, eng.HeldBack):
                return ([_with(p, name=name) for p in getattr(exc, "problems", [])]
                        or [problem("two_voices", file=paths[0].name, name=name)])
            secs = getattr(exc, "seconds", exc.args[0] if exc.args else 0)
            return [problem("too_little_speech", source="clip", file=getattr(exc, "file", None) or paths[0].name,
                            seconds=float(secs or 0), name=name)]
        _enroll(rec, folder, name, rt, call, refusal)
    except AudioIntakeError as exc:
        record.fail(rec, exc)
    finally:
        _remove_work(work)
    record.save(rec)
    return rec


def enroll_from(name: str, ref: str, label: str, *, lang: str | None, voices_dir: Path, force: bool,
                rt: Runtime) -> dict[str, Any]:
    """``enroll NAME --from REF --label speaker_K``: that label's clean speech in a transcript, up to 30 s."""
    voices.check_name(name)
    r = record.resolve(ref, need_json=True, command="enroll")
    rttm_text = (r.doc or {}).get("rttm") or ""
    labels = labels_in(rttm_text)
    if label not in labels:
        raise UsageError("unknown_label", id=r.id or ref, label=label, labels=labels or "none")
    rec = _enroll_record(name, [], parent=r.id)
    work: Path | None = None
    try:
        path = record.reseal(r, required=True)
        assert path is not None
        folder = voices.ensure_folder(voices_dir)
        _enroll_guard(folder, name, rt)
        segs = [s for s in record.segments(rttm_text) if s["spk"] == label]
        start = min((s["start"] for s in segs), default=None)
        end = max((s["start"] + s["dur"] for s in segs), default=None)
        source = voices.source_for_label(path.name, label, start, end)
        rec["voice"]["sources"] = [source]
        work = _work(rt, "enroll")
        stem = _stem_of(r)
        wav = argv.diar_wav_path(work, stem)
        dur = ((r.doc or {}).get("input") or {}).get("duration_s")
        c = proc.run(argv.diar_wav(rt.tools.ffmpeg, path, wav), env=rt.env, timeout=_timeout("ffmpeg", dur),
                     allowed=_allowed(rt), stdout="discard")
        if not c.ok:
            raise errors.error_for("decode_failed", file=path.name)
        eng = engine()
        model = _campplus(rt)

        def call() -> Any:
            return eng.enroll_label(name, wav, rttm_text, label, source, lang.upper() if lang else None, folder,
                                    model, force)

        def refusal(exc: Exception) -> list[Problem]:
            if isinstance(exc, eng.HeldBack):
                return ([_with(p, name=name, label=label) for p in getattr(exc, "problems", [])]
                        or [problem("two_voices", file=path.name, name=name)])
            secs = getattr(exc, "seconds", exc.args[0] if exc.args else 0)
            return [problem("too_little_speech", source="label", label=label, seconds=float(secs or 0), name=name)]
        _enroll(rec, folder, name, rt, call, refusal)
    except AudioIntakeError as exc:
        record.fail(rec, exc)
    finally:
        _remove_work(work)
    record.save(rec)
    return rec


def _with(p: Any, **defaults: Any) -> Problem:
    """A refusal problem from the engine, with the params its message needs filled in where missing."""
    code = getattr(p, "code", None) or (p.get("code") if isinstance(p, Mapping) else None) or "two_voices"
    params = dict(getattr(p, "params", None) or (p.get("params") if isinstance(p, Mapping) else None) or {})
    for k, v in defaults.items():
        params.setdefault(k, v)
    return problem(code, **params)


def enroll_result(d: Any) -> dict[str, Any]:
    """The record's ``result`` from what the voiceprint writer returned: one source's result, or ``results`` per
    clip. Numbers and the name only (never vectors): sources added, skipped (already enrolled), the seconds added,
    the print's source count after, and the check codes ``--force`` overrode."""
    if not isinstance(d, Mapping):
        return {}
    per = d.get("results")
    rows = [x for x in per if isinstance(x, Mapping)] if isinstance(per, list) else [d]
    added = [x for x in rows if x.get("added", True) and not x.get("skipped")]
    out: dict[str, Any] = {"name": d.get("name"), "added": len(added),
                           "skipped": sum(1 for x in rows if x.get("skipped")),
                           "seconds": round(sum(float(x.get("seconds") or 0) for x in added), 1)}
    counts = [x.get("sources") for x in rows if isinstance(x.get("sources"), int)]
    if counts:
        out["sources"] = counts[-1]
    forced = sorted({str(getattr(p, "code", None) or (p.get("code") if isinstance(p, Mapping) else p))
                     for x in rows for p in x.get("forced") or []})
    if forced:
        out["forced"] = forced
    return out
