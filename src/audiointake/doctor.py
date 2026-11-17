"""``audiointake doctor``: can this Mac transcribe, and if not, exactly what is missing.

``report`` gathers, opening no connection: the platform and whether it is supported, the memory warning, ffmpeg
and ffprobe, the Command Line Tools and Swift (``xcode-select -p`` first; ``/usr/bin/swift`` only once a folder
holding a swift was named), every pack's state (``store.pack_check``, the very check a run makes: the cheap ledger
check, or with ``verify`` a full re-hash of every file, and whether the stamp still matches the shipped pins), the
speaker program's keyed build (and whether its tag had moved upstream when it was built), the routes a run would
have, and, with ``selftest``, the canary run again from the clip setup kept (``audiointake transcribe --no-names``
as a child, offline).

Without ``selftest`` it writes nothing. The self-test is a real transcription, so like any finished one it saves
one record and one speed figure in ``<data>``; it never reads the voiceprints.

``exit`` is 3 when a run can't start here: an unsupported platform, no ffmpeg, core not set up, or a core pack
that is broken or outdated (the same code ``transcribe`` would exit with). Everything else (a broken or outdated hu
or th pack, no speaker program, a self-test that couldn't run or didn't pass) is reported with exit 0, as a run
would carry on without it. Every message is core's (``errors.render``).
"""
from __future__ import annotations

from typing import Any

from . import __version__, errors, platforms
from .contracts import Problem
from .errors import SetupError, problem

BLOCKING = frozenset({"platform_not_macos", "platform_not_apple_silicon", "platform_rosetta", "macos_too_old",
                      "ffmpeg_missing", "not_set_up"})


def _p(p: Problem) -> dict[str, Any]:
    return {"code": p.code, "params": dict(p.params), "message": errors.render(p.code, p.params)}


def report(*, verify: bool = False, selftest: bool = False) -> dict:
    """Everything ``audiointake doctor`` prints, as JSON values (see the module docstring)."""
    from .setup import fluidaudio, manifest, store
    from .setup import ffmpeg as ffm

    out: dict[str, Any] = {"exit": errors.EXIT_OK, "version": __version__, "platform": platforms.info(),
                           "supported": True, "prerequisites": [], "packs": [], "speakers": {}, "routes": [],
                           "problems": [], "warnings": [], "selftest": None}
    blocking: list[Problem] = []
    warnings: list[Problem] = []
    try:
        platforms.require_supported()
    except SetupError as exc:
        out["supported"] = False
        blocking.append(exc.problem())
        out["problems"] = [_p(p) for p in blocking]
        out["exit"] = errors.EXIT_SETUP
        return out
    warnings += platforms.warnings()
    ff, fp = ffm.find_without_writing()          # the bundled ffmpeg counts before setup has linked it
    out["prerequisites"] = ffm.prerequisites(ff, fp)
    if ff is None:
        blocking.append(problem("ffmpeg_missing"))
    tc = fluidaudio.toolchain()
    out["prerequisites"].append({"name": "swift", "ok": tc["code"] is None, "version": tc.get("swift", ""),
                                 "path": tc.get("developer_dir"), **({"code": tc["code"]} if tc["code"] else {})})
    out["packs"] = store.status()
    core = store.read_stamp("core")
    if core is None:
        blocking.append(problem("not_set_up", size=round(store.plan(["core"])["fetch_bytes"] / store.GB, 1)))
    for p in store.verify(full=verify):
        if p.params.get("pack") == "core":
            blocking.append(p)
        else:
            warnings.append(p)
    info = fluidaudio.check()
    out["speakers"] = {"tag": fluidaudio.TAG, "commit": fluidaudio.COMMIT, "recipe": fluidaudio.RECIPE,
                       "built": info is not None, "path": info["path"] if info else None,
                       "swift": (info or {}).get("swift"), "built_at": (info or {}).get("built_at")}
    if info is None and core is not None:
        warnings.append(problem("no_speaker_labels", reason=_why_no_speakers(tc), scope="mac"))   # no record here
    if info is not None and info.get("tag_moved"):
        warnings.append(problem("fluidaudio_tag_moved", tag=fluidaudio.TAG, commit=fluidaudio.COMMIT[:12]))
    routes: set[str] = set()
    for row in out["packs"]:
        if row["id"] != "core" and row["verified"]:
            routes.update(row["routes"])
    out["routes"] = sorted(routes)
    if selftest and core is not None and not blocking:
        c = manifest.load("core").canary
        out["selftest"] = store.run_canary(c, speakers=info is not None, progress=lambda _m: None, fetch=False,
                                           ffmpeg=ff)
        cp = store.canary_problem(out["selftest"])
        if cp is not None:
            warnings.append(cp)
    out["problems"] = [_p(p) for p in blocking]
    out["warnings"] = [_p(p) for p in warnings]
    out["exit"] = errors.EXIT_SETUP if blocking else errors.EXIT_OK
    return out


def _why_no_speakers(tc: dict) -> str:
    code = tc.get("code")
    if code == "clt_missing":
        return "Apple's Command Line Tools aren't installed"
    if code == "xcode_license":
        return "Xcode's licence hasn't been accepted"
    if code == "swift_too_old":
        return "the Swift on this Mac is too old"
    return "it hasn't been built; run audiointake setup"


def exit_code(rep: dict) -> int:
    return int(rep.get("exit", errors.EXIT_OK))
