"""``resolve()``: the verified ledger turned into this run's ``Runtime`` (contracts.py defines what its arguments
mean; this is the one implementation).

It opens no connection and downloads nothing: models are local paths, and a pack a run asks for that
isn't installed becomes the warning ``pack_not_installed`` and a ``language`` offer. A missing or broken core is
``SetupError`` (exit 3) before any audio is read; a broken hu or th pack switches that pack's routes off with the
warning ``pack_broken``. Files are checked cheaply (size, inode, modification time and blob name against the
ledger); a file whose stat changed is re-hashed once, and a match refreshes its ledger entry (saved while the run
lock is held), so a restore that moves inodes costs one re-hash, not one per run. A stamp whose pins no longer
match the shipped manifest is outdated (``store.pack_check``): core outdated is exit 3, an outdated hu or th pack
is switched off with a warning, exactly as ``doctor`` reports them.
"""
from __future__ import annotations

import datetime as _dt
import os
import secrets
import shutil
import sys
import sysconfig
import time
from pathlib import Path

from .. import platforms
from ..contracts import LANGUAGE_NAMES, PACK_OF_LANG, ROUTE_OF_LANG, ROUTES, Model, Problem, Runtime, Tools
from ..errors import SetupError, problem
from . import manifest, store

WORK_MAX_AGE_S = 24 * 3600
LANG_OF_ROUTE = {route: lang for lang, route in ROUTE_OF_LANG.items()}


def scripts_dir() -> Path:
    """The tool environment's scripts folder (parakeet-mlx, mlx_whisper and python live there)."""
    return Path(sysconfig.get_path("scripts"))


def run_env(work: Path, ffmpeg: Path) -> dict[str, str]:
    """The run children's environment: built fresh, nothing else from the user's environment reaches them."""
    return {"PATH": f"{scripts_dir()}:{ffmpeg.parent}:/usr/bin:/bin", "HOME": str(Path.home()),
            "TMPDIR": str(work), "USER": os.environ.get("USER", ""), "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8",
            "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_CACHE": str(platforms.hf_cache()),
            "ORT_DISABLE_TELEMETRY": "1"}


def prune_work(root: Path, *, max_age_s: float = WORK_MAX_AGE_S) -> int:
    """Delete ``root/*`` folders untouched for longer than a day (a run killed before its own cleanup)."""
    if not root.is_dir():
        return 0
    n = 0
    cutoff = time.time() - max_age_s
    for p in root.iterdir():
        try:
            if p.is_dir() and not p.is_symlink() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n


def new_work_root() -> Path:
    from .. import fsutil

    root = platforms.cache_dir() / "work"
    prune_work(root)
    run_id = _dt.datetime.now().strftime("w-%Y%m%d-%H%M%S-") + secrets.token_hex(4)
    return fsutil.private_dir(root / run_id)


def has_voiceprints(folder: Path) -> bool:
    """A voiceprint folder counts when it holds a usable voiceprint: a top-level ``*.json`` not starting with
    ``.``, made with this speaker model, with vectors (the engine's own reading, ``steps.speaker_id.scan``). A
    folder of other files, or of prints made with another model, has nobody to match, so speaker ID doesn't run
    on it (the ``voices_model_mismatch`` and ``voices_skipped`` notes say why nobody was named), rather than run
    and fail on every recording."""
    from ..steps.speaker_id import scan
    try:
        return scan(folder)["prints"] > 0
    except OSError:
        return False


def _gb(n: int) -> float:
    return n / store.GB


def _model(m: manifest.ModelSpec, entry: dict) -> Model:
    return Model(key=m.key, tag=m.tag, path=Path(entry["path"]), revision=entry.get("revision") or "")


def _check(pid: str) -> tuple[dict | None, Problem | None]:
    """(stamp, the problem that keeps it from being used) for a pack: ``store.pack_check``, the same check
    ``doctor`` reports, so the two never disagree. A stamp the shipped manifest no longer matches (a pin or the
    pack's version changed) is outdated even when every file is intact: a run uses only the pinned models."""
    st, broken, why = store.pack_check(pid)
    if st is None:
        return None, None
    return st, store.pack_problem(pid, broken, why)


def resolve(*, need: frozenset[str] = frozenset(), voices_dir: Path | None = None, names: bool = True,
            speakers: bool = True) -> tuple[Runtime, list[Problem]]:
    """This run's Runtime and the warnings and offers that go on its records; raises SetupError (exit 3)."""
    unknown = set(need) - set(ROUTES)
    if unknown:
        raise ValueError(f"not a route: {sorted(unknown)}")
    from . import ffmpeg as ffm, fluidaudio

    platforms.require_supported()
    ff, fp = ffm.find(ensure_fallback=True)     # links the bundled ffmpeg if setup's link is missing or dangling
    if ff is None:
        raise SetupError("ffmpeg_missing")
    core, bad_core = _check("core")
    if core is None:
        raise SetupError("not_set_up", size=round(_gb(store.plan(["core"])["fetch_bytes"]), 1))
    if bad_core is not None:
        raise SetupError(bad_core.code, **dict(bad_core.params))
    problems: list[Problem] = []
    models: dict[str, Model] = {}
    for m in manifest.load("core").models:
        models[m.key] = _model(m, core["models"][m.key])
    valid: set[str] = {"core"}
    for pid in ("hu", "th"):
        st, bad = _check(pid)
        if st is None:
            continue
        if bad is not None:
            problems.append(bad)
            continue
        valid.add(pid)
        pk = manifest.load(pid)
        for m in pk.models:
            if m.key not in pk.sources:
                models.setdefault(m.key, _model(m, st["models"][m.key]))
    routes: set[str] = set()
    for pid in valid - {"core"}:
        routes.update(manifest.load(pid).routes)
    for route in sorted(set(need) - routes):
        lang = LANG_OF_ROUTE.get(route)
        if lang is None:
            continue
        pack = PACK_OF_LANG.get(lang, lang)
        size = round(_gb(store.plan([pack])["fetch_bytes"]), 1)
        params = {"code": lang, "language": LANGUAGE_NAMES.get(lang, lang), "size": size}
        problems.append(problem("pack_not_installed", reason="asked", **params))
        problems.append(problem("language", **params))
    fa_path: Path | None = None
    info = fluidaudio.check()           # also under --no-speakers: the Thai worker's turns come from it
    if info is not None:
        fa_path = Path(info["path"])
    elif speakers:
        problems.append(problem("no_speaker_labels", reason="the speaker program hasn't been built"))
        problems.append(problem("speakers"))
    voices: Path | None = None
    if speakers and names:
        folder = Path(voices_dir) if voices_dir is not None else platforms.data_dir() / "voices"
        if folder.is_dir() and has_voiceprints(folder):
            voices = folder
    tools = Tools(ffmpeg=ff, ffprobe=fp, python=Path(sys.executable), parakeet_cli=scripts_dir() / "parakeet-mlx",
                  whisper_cli=scripts_dir() / "mlx_whisper", fluidaudiocli=fa_path, fluidaudio_tag=fluidaudio.TAG)
    work = new_work_root()
    return Runtime(tools=tools, models=models, routes=frozenset(routes), env=run_env(work, ff), work_root=work,
                   voices_dir=voices), problems
