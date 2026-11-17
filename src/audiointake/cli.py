"""The ``audiointake`` command: argument parsing, the JSON a skill reads, and the exit code.

Commands::

    audiointake setup [--pack hu|th]... [--plan] [--json]
    audiointake packs list|remove ID [--json]
    audiointake transcribe AUDIO... [--paths-from FILE] [--lang CODE[,CODE...]] [--speakers N]
                           [--no-names] [--no-speakers] [--voices DIR] [--outdir DIR]
                           [--format md,json,txt,sidecars] [--overwrite] [--plan] [--json]
    audiointake apply REF (--lang CODE | --choice speakers | --choice names) [--overwrite] [--json]
    audiointake relabel REF --label speaker_K --name NAME [--label ... --name ...] [--overwrite] [--json]
    audiointake notes REF --into NOTES.md [--json]
    audiointake notes REF --check NOTES.md [--json]
    audiointake enroll NAME CLIP... [--lang CODE] [--force] [--voices DIR] [--json]
    audiointake enroll NAME --from REF --label speaker_K [--force] [--voices DIR] [--json]
    audiointake voices list | remove NAME | rename OLD NEW [--voices DIR] [--json]
    audiointake record show REF [--json]
    audiointake record find PATH [--json]
    audiointake doctor [--verify] [--selftest] [--json]
    audiointake about [--json]
    audiointake uninstall [--keep-voices] [--plan] [--json]
    audiointake --version

``REF`` is a record id, or the path of the recording or of its ``(transcript).json``.

**Nothing is ever asked.** No command prompts or reads standard input: every choice is a flag or a default, and
what a person could still do is an offer on the record, acted on later with ``apply`` (or ``enroll``).

**Output.** ``transcribe`` and ``apply`` print ``{"exit": N, "records": [...]}`` with ``--json`` (an invocation
that can't run at all adds ``"error": {code, message}`` and no records); every other command prints one object
with ``exit`` and its result, or ``exit`` and ``error``. Without ``--json`` a short summary goes to standard output,
and errors and progress (``2 of 5: Team sync.m4a: speakers``) to standard error. Never transcript text.

**Exit codes**: 0 done, 1 failed, 2 usage, 3 setup, 4 refused, 5 stale; several records exit with the first of
2, 3, 1, 4, 5, 0 present. SIGHUP and SIGTERM stop the children, remove the work folder and any partial file,
write no record and exit 129 or 143.

**One writer at a time**: the commands that write audiointake's data folder hold ``<data>/run.lock``; a second
run waits and says so. ``transcribe``, ``apply`` and ``setup`` also keep the Mac from idle-sleeping.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__, errors, fsutil, lock, outputs, pipeline, platforms, proc, record, render, voices
from . import notes as notes_mod
from .contracts import LABEL_RE, LANGS, PACK_OF_LANG, PACK_TERMS, PACKS, ROUTE_OF_LANG
from .errors import AudioIntakeError, UsageError

PROG = "audiointake"
CAFFEINATE = Path("/usr/bin/caffeinate")
MAX_PATHS_FROM_BYTES = 16 * 1024 * 1024
COMMANDS_WITH_RECORDS = ("transcribe", "apply")
PACK_CHOICES = tuple(p for p in PACKS if p != "core")


class _ArgError(Exception):
    def __init__(self, message: str, prog: str) -> None:
        super().__init__(message)
        self.message = message
        self.prog = prog


class _Parser(argparse.ArgumentParser):
    """Raises on a bad argument (exit 2 as ``bad_arguments``) and never guesses an abbreviation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> Any:  # type: ignore[override]
        raise _ArgError(message, self.prog)


class Interrupted(BaseException):
    """SIGTERM, SIGHUP or SIGINT arrived: stop everything, write nothing more."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


# --------------------------------------------------------------------------- the parser
def build_parser() -> _Parser:
    p = _Parser(prog=PROG, description="Transcribe recordings on a Mac with Apple silicon, with speakers named "
                                       "word by word. Nothing is ever asked: every choice is a flag or a default.")
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND", parser_class=_Parser)

    s = sub.add_parser("setup", help="check this Mac, download the models, build the speaker program")
    s.add_argument("--pack", action="append", default=[], choices=PACK_CHOICES, help="also install this language pack")
    s.add_argument("--plan", action="store_true", help="say what would be downloaded and how long it takes; change nothing")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("packs", help="list or remove language packs")
    ps = s.add_subparsers(dest="packs_command", required=True, metavar="ACTION", parser_class=_Parser)
    x = ps.add_parser("list")
    x.add_argument("--json", action="store_true")
    x = ps.add_parser("remove")
    x.add_argument("pack", choices=PACKS, metavar="ID")
    x.add_argument("--json", action="store_true")

    s = sub.add_parser("transcribe", help="transcribe recordings; writes beside each or into --outdir",
                       description="Transcribe each recording. Nothing is asked: what's left to decide is listed as "
                                   "offers on each record.")
    s.add_argument("inputs", nargs="*", metavar="AUDIO")
    s.add_argument("--paths-from", action="append", default=[], metavar="FILE",
                   help="read more paths from FILE, one UTF-8 path per line (never standard input)")
    s.add_argument("--lang", metavar="CODE[,CODE...]", help="the language(s) spoken; one code only when the whole "
                                                            "recording is in it")
    s.add_argument("--speakers", type=_positive_int, metavar="N", help="how many people spoke (a check only)")
    s.add_argument("--no-names", action="store_true", help="tell speakers apart without naming anyone")
    s.add_argument("--no-speakers", action="store_true", help="a plain transcript without speaker labels")
    s.add_argument("--voices", metavar="DIR", help="read voiceprints from DIR instead of audiointake's own folder")
    s.add_argument("--outdir", metavar="DIR", help="write the outputs into DIR")
    s.add_argument("--format", metavar="md,json,txt,sidecars", help="what to write (default md,json)")
    s.add_argument("--overwrite", action="store_true", help="replace outputs that are already there")
    s.add_argument("--plan", action="store_true", help="check and probe the inputs and say what would run")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("apply", help="act on an offer a record made")
    s.add_argument("ref", metavar="REF")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--lang", metavar="CODE", help="install that language's pack and transcribe again")
    g.add_argument("--choice", choices=("speakers", "names"), help="add speaker labels, or update the names")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("relabel", help="name speakers in one transcript (no voiceprint is made)")
    s.add_argument("ref", metavar="REF")
    s.add_argument("--label", action="append", default=[], required=True, metavar="speaker_K")
    s.add_argument("--name", action="append", default=[], required=True, metavar="NAME")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("notes", help="append the transcript to a notes file, or check one")
    s.add_argument("ref", metavar="REF")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--into", metavar="NOTES.md")
    g.add_argument("--check", metavar="NOTES.md")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("enroll", help="keep a voiceprint so later recordings name this person",
                       description=voices.CONSENT, epilog=voices.CONSENT,
                       formatter_class=argparse.RawDescriptionHelpFormatter)   # the consent line, never re-wrapped
    s.add_argument("name", metavar="NAME")
    s.add_argument("clips", nargs="*", metavar="CLIP")
    s.add_argument("--from", dest="from_ref", metavar="REF")
    s.add_argument("--label", metavar="speaker_K")
    s.add_argument("--lang", metavar="CODE")
    s.add_argument("--force", action="store_true", help="keep it although the check held it back (only on the "
                                                         "person's own word)")
    s.add_argument("--voices", metavar="DIR")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("voices", help="list, remove or rename voiceprints")
    vs = s.add_subparsers(dest="voices_command", required=True, metavar="ACTION", parser_class=_Parser)
    x = vs.add_parser("list")
    x.add_argument("--voices", metavar="DIR")
    x.add_argument("--json", action="store_true")
    x = vs.add_parser("remove")
    x.add_argument("name", metavar="NAME")
    x.add_argument("--voices", metavar="DIR")
    x.add_argument("--json", action="store_true")
    x = vs.add_parser("rename")
    x.add_argument("old", metavar="OLD")
    x.add_argument("new", metavar="NEW")
    x.add_argument("--voices", metavar="DIR")
    x.add_argument("--json", action="store_true")

    s = sub.add_parser("record", help="show a record, or find the record for a file")
    rs = s.add_subparsers(dest="record_command", required=True, metavar="ACTION", parser_class=_Parser)
    x = rs.add_parser("show")
    x.add_argument("ref", metavar="REF")
    x.add_argument("--json", action="store_true")
    x = rs.add_parser("find")
    x.add_argument("path", metavar="PATH")
    x.add_argument("--json", action="store_true")

    s = sub.add_parser("doctor", help="check the setup")
    s.add_argument("--verify", action="store_true", help="re-hash every model file")
    s.add_argument("--selftest", action="store_true", help="transcribe the kept test clip again (offline)")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("about", help="versions, models, licences, hosts and folders")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("uninstall", help="remove audiointake's models, data, caches and logs")
    s.add_argument("--keep-voices", action="store_true")
    s.add_argument("--plan", action="store_true")
    s.add_argument("--json", action="store_true")
    return p


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a whole number") from None
    if v < 1:
        raise argparse.ArgumentTypeError("it must be 1 or more")
    return v


# --------------------------------------------------------------------------- output
def _print_json(obj: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
    sys.stdout.flush()


def _err(line: str) -> None:
    sys.stderr.write(line.rstrip("\n") + "\n")
    sys.stderr.flush()


def _out(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def _progress(line: str) -> None:
    _err(line)


def _error_obj(exc: AudioIntakeError) -> dict[str, Any]:
    return {"code": exc.code, "message": exc.message}


def _fail(exc: AudioIntakeError, command: str, as_json: bool) -> int:
    if as_json:
        obj: dict[str, Any] = {"exit": exc.exit}
        if command in COMMANDS_WITH_RECORDS:
            obj["records"] = []
        obj["error"] = _error_obj(exc)
        _print_json(obj)
    else:
        _err(exc.message)
    return exc.exit


def _length(seconds: float | None) -> str:
    return render.clock(seconds, hours=True) if seconds else "?"


def _record_lines(rec: Mapping[str, Any]) -> None:
    """A human summary of one record: the error on stderr, everything else on stdout."""
    inp = rec.get("input") or {}
    name = Path(inp["path"]).name if inp.get("path") else (rec.get("voice") or {}).get("name") or rec["id"]
    if rec.get("error"):
        _err(f"{name}: {rec['error']['message']}")
        for r in rec.get("refusals") or []:
            if r.get("message") != rec["error"]["message"]:
                _err(f"  {r['message']}")
        return
    if rec.get("status") == "plan":
        plan = rec.get("plan") or {}
        est = plan.get("estimate_s")
        _out(f"{name}: {_length(inp.get('duration_s'))}; would run {', '.join(plan.get('routes') or [])}"
             + (f"; about {render.clock(est, hours=True)}" if est else ""))
    else:
        spk = rec.get("speakers") or []
        named = sum(1 for s in spk if s.get("name"))
        parts = [_length(inp.get("duration_s"))]
        if spk:
            parts.append(render.speaker_part(len(spk), named))
        written = [Path(o["path"]).name for o in rec.get("outputs") or []]
        _out(f"{name}: {', '.join(parts)}." + (f" Files: {', '.join(written)}." if written else ""))
    for key in ("warnings", "notes"):
        for w in rec.get(key) or []:
            _out(f"  - {w['message']}")
    for o in rec.get("offers") or []:
        options = o.get("options") or []
        for i, line in enumerate(o.get("summary") or []):
            _out(f"  - {line}")
            if o.get("kind") == "language" and i < len(options):        # a pack's terms, here only, never stored
                terms = PACK_TERMS.get(PACK_OF_LANG.get(str(options[i]), ""))
                if terms:
                    _out(f"    {terms}")


def _records_out(recs: list[dict[str, Any]], as_json: bool) -> int:
    code = errors.process_exit(int(r["exit"]) for r in recs)
    if as_json:
        _print_json({"exit": code, "records": recs})
    else:
        for r in recs:
            _record_lines(r)
    return code


def _one_out(obj: Mapping[str, Any], as_json: bool, human: Callable[[], None]) -> int:
    code = int(obj.get("exit", 0) or 0)
    if as_json:
        _print_json({"exit": code, **{k: v for k, v in obj.items() if k != "exit"}})
    else:
        human()
    return code


# --------------------------------------------------------------------------- helpers
def _langs(value: str | None, *, command: str, single: bool = False) -> tuple[str, ...]:
    """The ``--lang`` codes; a usage error names ``command``, the one whose ``--help`` lists the options."""
    if value is None:
        return ()
    codes = tuple(c.strip().lower() for c in value.split(",") if c.strip())
    if not codes:
        raise UsageError("bad_arguments", detail="--lang takes language codes such as en or de", command=command)
    for c in codes:
        if c not in LANGS:
            raise UsageError("unknown_language", code=c, codes=list(LANGS))
    if single and len(codes) != 1:
        raise UsageError("bad_arguments", detail="--lang takes one language code here", command=command)
    return tuple(dict.fromkeys(codes))


def _paths_from(files: Sequence[str]) -> list[str]:
    out: list[str] = []
    for f in files:
        p = Path(os.path.abspath(os.path.expanduser(f)))
        try:
            st = p.stat()
            if not stat.S_ISREG(st.st_mode):              # never a pipe, a device or /dev/stdin: nothing reads stdin
                raise UsageError("bad_arguments", detail=f"--paths-from takes a file, and {f} isn't a regular file",
                                 command="transcribe")
            if st.st_size > MAX_PATHS_FROM_BYTES:
                raise UsageError("bad_arguments", detail=f"--paths-from {f} is too large", command="transcribe")
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise UsageError("input_missing", file=str(p)) from None
        except (OSError, UnicodeDecodeError):
            raise UsageError("input_not_readable", file=p.name) from None
        out += [ln.rstrip("\r") for ln in text.split("\n") if ln.strip()]
    return out


def _require_mac() -> None:
    platforms.require_supported()


def _resolve(**kw: Any) -> tuple[Any, list[Any]]:
    return pipeline.setup_api().resolve(**kw)


# --------------------------------------------------------------------------- commands
def cmd_transcribe(ns: argparse.Namespace) -> int:
    langs = _langs(ns.lang, command="transcribe")
    formats = outputs.parse_formats(ns.format)
    inputs = list(ns.inputs) + _paths_from(ns.paths_from)
    if not inputs:
        raise UsageError("bad_arguments", detail="transcribe needs at least one recording", command="transcribe")
    _require_mac()
    with lock.run_lock():
        record.prune()
        need = frozenset(ROUTE_OF_LANG[c] for c in langs if c in ROUTE_OF_LANG)
        rt, probs = _resolve(need=need, voices_dir=voices.folder(ns.voices), names=not ns.no_names,
                             speakers=not ns.no_speakers)
        try:
            with proc.keep_awake(rt.tools.caffeinate):
                opts = pipeline.Options(langs=langs, speakers=ns.speakers, names=not ns.no_names,
                                        speakers_off=ns.no_speakers, voices=Path(ns.voices) if ns.voices else None,
                                        outdir=Path(ns.outdir) if ns.outdir else None, formats=formats,
                                        overwrite=ns.overwrite, plan=ns.plan, setup_problems=tuple(probs),
                                        progress=_progress)
                recs = pipeline.run(inputs, opts, rt)
        finally:
            pipeline.cleanup(rt)
    return _records_out(recs, ns.json)


def cmd_apply(ns: argparse.Namespace) -> int:
    code = _langs(ns.lang, single=True, command="apply")[0] if ns.lang is not None else None
    _require_mac()
    with lock.run_lock(), proc.keep_awake(CAFFEINATE):
        record.prune()
        if code is not None:
            rec = pipeline.apply_lang(ns.ref, code, overwrite=ns.overwrite, progress=_progress)
        else:
            rec = pipeline.apply_choice(ns.ref, ns.choice, overwrite=ns.overwrite, progress=_progress)
    return _records_out([rec], ns.json)


def cmd_relabel(ns: argparse.Namespace) -> int:
    if len(ns.label) != len(ns.name):
        raise UsageError("bad_arguments", detail="give one --name for each --label", command="relabel")
    for lab in ns.label:
        if not re.match(LABEL_RE, lab):
            raise UsageError("bad_arguments", detail=f"--label takes speaker_0 to speaker_7, not {lab}",
                             command="relabel")
    with lock.run_lock():
        record.prune()
        rec = pipeline.relabel(ns.ref, list(zip(ns.label, ns.name)), overwrite=ns.overwrite)
    return _single_record(rec, ns.json)


def _single_record(rec: dict[str, Any], as_json: bool) -> int:
    code = int(rec["exit"])
    if as_json:
        _print_json({"exit": code, "record": rec})
    else:
        _record_lines(rec)
    return code


def cmd_notes(ns: argparse.Namespace) -> int:
    if ns.into:
        with lock.run_lock():
            record.prune()
            rec = notes_mod.into(ns.ref, ns.into)
        return _single_record(rec, ns.json)
    result = notes_mod.check(ns.ref, ns.check)
    return _one_out({"exit": 0, **result}, ns.json, lambda: _out(f"{Path(result['path']).name} is unchanged."))


def cmd_enroll(ns: argparse.Namespace) -> int:
    name = voices.check_name(ns.name)
    lang = _langs(ns.lang, single=True, command="enroll")[0] if ns.lang is not None else None
    if ns.from_ref is not None:
        if ns.clips or not ns.label:
            raise UsageError("bad_arguments", detail="enroll --from REF takes --label and no clips", command="enroll")
        if not re.match(LABEL_RE, ns.label):
            raise UsageError("bad_arguments", detail=f"--label takes speaker_0 to speaker_7, not {ns.label}",
                             command="enroll")
    elif not ns.clips or ns.label:
        raise UsageError("bad_arguments", detail="enroll takes clips, or --from REF --label speaker_K",
                         command="enroll")
    _require_mac()
    folder = voices.folder(ns.voices)
    with lock.run_lock():
        record.prune()
        rt, _probs = _resolve(need=frozenset(), voices_dir=folder, names=False, speakers=False)
        try:
            if ns.from_ref is not None:
                rec = pipeline.enroll_from(name, ns.from_ref, ns.label, lang=lang, voices_dir=folder,
                                           force=ns.force, rt=rt)
            else:
                rec = pipeline.enroll_clips(name, ns.clips, lang=lang, voices_dir=folder, force=ns.force, rt=rt)
        finally:
            pipeline.cleanup(rt)
    code = int(rec["exit"])
    added = code == 0 and int((rec.get("result") or {}).get("added", 1) or 0) > 0
    line = voices.saved_line(name) if added else voices.nothing_added_line(name)
    if ns.json:
        _print_json({"exit": code, "record": rec, **({"saved": line} if added else
                                                     {"nothing_added": line} if code == 0 else {})})
    elif code == 0:
        _out(line)
    else:
        _record_lines(rec)
    return code


def cmd_voices(ns: argparse.Namespace) -> int:
    folder = voices.folder(ns.voices)
    if ns.voices_command == "list":
        items = voices.listing(folder)

        def human() -> None:
            if not items:
                _out("No voiceprints.")
            for v in items:
                _out(f"{v['name']}: {len(v['sources'])} sources, {v['seconds']:.0f} s"
                     + ("" if v["model_ok"] else ", made with another speaker model")
                     + (f", updated {v['updated']}" if v.get("updated") else ""))
        return _one_out({"exit": 0, "folder": str(folder), "voices": items}, ns.json, human)
    with lock.run_lock():
        if ns.voices_command == "remove":
            res = voices.remove(folder, ns.name)
            return _one_out({"exit": 0, **res}, ns.json, lambda: _out(f"Removed the voiceprint for {res['removed']}."))
        res = voices.rename(folder, ns.old, ns.new)
        return _one_out({"exit": 0, **res}, ns.json, lambda: _out(f"Renamed {res['renamed']} to {res['to']}."))


def cmd_record(ns: argparse.Namespace) -> int:
    if ns.record_command == "show":
        rec = record.show(ns.ref)
        if ns.json:
            _print_json({"exit": 0, "record": rec})
        else:
            _record_lines(rec)
        return 0
    found = record.find(ns.path)
    return _one_out({"exit": 0, **found}, ns.json,
                    lambda: _out(f"{found['id']}: " + ", ".join(f"{lab} = {found['displays'].get(lab, '?')}"
                                                               for lab in found["labels"])))


def _problems_human(obj: Mapping[str, Any]) -> None:
    for key in ("errors", "warnings"):
        for p in obj.get(key) or []:
            if isinstance(p, Mapping) and p.get("code") in errors.CODES:
                _out(f"- {errors.render(p['code'], p.get('params') or {})}")


def cmd_setup(ns: argparse.Namespace) -> int:
    packs = ["core", *dict.fromkeys(ns.pack)]
    _require_mac()
    api = pipeline.setup_api()
    if ns.plan:
        plan = api.plan(packs)
        return _one_out({"exit": 0, "plan": plan}, ns.json, lambda: _out(json.dumps(plan, ensure_ascii=False)))
    with lock.run_lock(), proc.keep_awake(CAFFEINATE):
        record.prune()
        res = dict(api.install(packs, progress=_progress))

    def human() -> None:
        _out("Setup finished." if int(res.get("exit", 0) or 0) == 0 else "Setup didn't finish.")
        _problems_human(res)
    return _one_out(res, ns.json, human)


def cmd_packs(ns: argparse.Namespace) -> int:
    api = pipeline.setup_api()
    if ns.packs_command == "list":
        items = list(api.status())
        return _one_out({"exit": 0, "packs": items}, ns.json,
                        lambda: [_out(f"{p.get('id')}: {'installed' if p.get('installed') else 'not installed'}")
                                 for p in items])
    if ns.pack == "core":
        raise UsageError("core_required")
    with lock.run_lock():
        res = dict(api.remove(ns.pack))
    return _one_out(res, ns.json, lambda: _out(f"Removed the {ns.pack} pack."))


def cmd_doctor(ns: argparse.Namespace) -> int:
    res = dict(pipeline.setup_api().doctor(verify=ns.verify, selftest=ns.selftest))

    def human() -> None:
        _out("audiointake is set up." if int(res.get("exit", 0) or 0) == 0 else "audiointake needs attention:")
        _problems_human(res)
    return _one_out(res, ns.json, human)


def cmd_about(ns: argparse.Namespace) -> int:
    res = dict(pipeline.setup_api().about())
    return _one_out({"exit": 0, **res}, ns.json, lambda: _out(json.dumps(res, ensure_ascii=False, indent=1)))


UNINSTALL_TOOL = "uv tool uninstall audiointake"


def cmd_uninstall(ns: argparse.Namespace) -> int:
    api = pipeline.setup_api()
    if ns.plan:
        res = dict(api.uninstall(keep_voices=ns.keep_voices, plan=True))
    else:
        with lock.run_lock():
            res = dict(api.uninstall(keep_voices=ns.keep_voices, plan=False))
    res.setdefault("command", UNINSTALL_TOOL)

    def human() -> None:
        _out("Would remove audiointake's models, data, caches and logs." if ns.plan
             else "Removed audiointake's models, data, caches and logs.")
        _out(f"To remove the tool itself: {res['command']}")
    return _one_out(res, ns.json, human)


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "setup": cmd_setup, "packs": cmd_packs, "transcribe": cmd_transcribe, "apply": cmd_apply,
    "relabel": cmd_relabel, "notes": cmd_notes, "enroll": cmd_enroll, "voices": cmd_voices, "record": cmd_record,
    "doctor": cmd_doctor, "about": cmd_about, "uninstall": cmd_uninstall,
}


# --------------------------------------------------------------------------- signals and main
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
STOP_EXIT = {signal.SIGTERM: errors.EXIT_SIGTERM, signal.SIGHUP: errors.EXIT_SIGHUP, signal.SIGINT: errors.EXIT_SIGINT}


def _on_signal(signum: int, _frame: Any) -> None:
    # A child being started waits to be registered first, so the stop reaches it; a commit being written finishes
    # first, with its record. Either way the signal lands right after.
    if proc.hold_signal(signum) or pipeline.defer_signal(signum):
        return
    proc.ignore_stops(STOP_SIGNALS)               # one is enough: the cleanup on the way out isn't cut short
    raise Interrupted(signum)


def _install_signals() -> dict[int, Any]:
    old: dict[int, Any] = {}
    for sig in STOP_SIGNALS:
        try:
            old[sig] = signal.signal(sig, _on_signal)
        except ValueError:                       # not the main thread (tests that call main from a thread)
            pass
    return old


def _restore_signals(old: Mapping[int, Any]) -> None:
    for sig, handler in old.items():
        try:
            signal.signal(sig, handler)
        except ValueError:
            pass


def _command_of(args: Sequence[str]) -> str:
    return next((a for a in args if a in COMMANDS), "")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit code."""
    fsutil.utf8_stdio()
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    command = _command_of(args)
    try:
        ns = build_parser().parse_args(args)
    except _ArgError as e:
        detail = e.message[:1].upper() + e.message[1:]
        return _fail(UsageError("bad_arguments", detail=detail, command=command or "COMMAND"), command, as_json)
    except SystemExit as e:                       # --help and --version print and exit 0
        return int(e.code or 0) if isinstance(e.code, int) else 0
    old = _install_signals()
    try:
        return COMMANDS[ns.command](ns)
    except Interrupted as e:
        proc.stop_children()                      # SIGTERM, then SIGKILL to any child still there, reaped
        fsutil.remove_live_partials()
        pipeline.abort()
        return STOP_EXIT.get(e.signum, 128 + int(e.signum))
    except AudioIntakeError as exc:
        return _fail(exc, ns.command, getattr(ns, "json", False))
    finally:
        _restore_signals(old)
