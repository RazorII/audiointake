"""A stand-in for setup (``runtime.resolve``, ``store``, ``doctor``, ``about``), with setup's exact signatures.

``FakeSetup(root, ...)`` installs the fake programs under ``root/bin`` and answers ``resolve`` the way the
contracts docstring defines it: ``need`` never switches a route on, a needed route that's missing gives
``pack_not_installed`` (reason ``asked``) and a ``language`` offer, ``speakers=False`` means no FluidAudio and no
warning, and a missing FluidAudio gives ``no_speaker_labels`` and a ``speakers`` offer. Each ``resolve`` makes a
fresh ``<cache>/work/<run-id>/``. Every call is kept in ``calls``.
"""
from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audiointake import errors, fsutil, platforms, voices
from audiointake.contracts import LANGUAGE_NAMES, PACK_TERMS, ROUTE_OF_LANG, Model, Problem, Runtime, Tools

from fakes import install_shims

PACK_ROUTES = {"hu": ("hu_vote", "thai_screen"), "th": ("thai_screen", "thai_turns")}
LANG_OF_ROUTE = {v: k for k, v in ROUTE_OF_LANG.items()}
FLUIDAUDIO_SOURCE = 0.25e9


class FakeSetup:
    def __init__(self, root: Path, *, packs: Iterable[str] = ("core",), fluidaudio: bool = True,
                 campplus: bool = True, ffprobe: bool = True, setup_error: str | None = None,
                 problems: Iterable[Problem] = (), install_result: dict[str, Any] | None = None,
                 sizes: dict[str, float] | None = None, free_bytes: float = 500e9,
                 doctor_result: dict[str, Any] | None = None, fluidaudio_tag: str = "v0.18.0") -> None:
        self.root = Path(root)
        self.bin = install_shims(self.root / "bin")
        self.packs = set(packs)
        self.fluidaudio = fluidaudio
        self.campplus = campplus
        self.ffprobe = ffprobe
        self.setup_error = setup_error
        self.problems = list(problems)
        self.install_result = install_result
        self.sizes = sizes or {"core": 4.3e9, "hu": 3.1e9, "th": 9.3e9}
        self.free_bytes = free_bytes
        self.doctor_result = doctor_result
        self.fluidaudio_tag = fluidaudio_tag
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.work_roots: list[Path] = []
        self._stamp()

    def _stamp(self) -> None:
        d = platforms.data_dir() / "packs"
        d.mkdir(parents=True, exist_ok=True)
        for pid in self.packs:
            (d / f"{pid}.json").write_text(json.dumps({"id": pid, "version": 1}), encoding="utf-8")

    # ------------------------------------------------------------------ runtime.resolve
    @property
    def routes(self) -> frozenset[str]:
        r: set[str] = set()
        for p in self.packs:
            r.update(PACK_ROUTES.get(p, ()))
        return frozenset(r)

    def resolve(self, *, need: frozenset[str] = frozenset(), voices_dir: Path | None = None, names: bool = True,
                speakers: bool = True) -> tuple[Runtime, list[Problem]]:
        self.calls.append(("resolve", {"need": set(need), "voices_dir": voices_dir, "names": names,
                                       "speakers": speakers}))
        if self.setup_error:
            raise errors.error_for(self.setup_error, size=4.3, pack="core", file="config.json")
        b = self.bin
        fa = b / "fluidaudiocli" if (self.fluidaudio and speakers) else None
        tools = Tools(ffmpeg=b / "ffmpeg", ffprobe=(b / "ffprobe") if self.ffprobe else None, python=b / "python",
                      parakeet_cli=b / "parakeet-mlx", whisper_cli=b / "mlx_whisper", fluidaudiocli=fa,
                      fluidaudio_tag=self.fluidaudio_tag, caffeinate=self.root / "no-caffeinate")
        models = {"ultra": Model("ultra", "parakeet-ultra-mlx", self.root / "m" / "ultra", "1" * 40),
                  "v3": Model("v3", "parakeet-tdt-0.6b-v3", self.root / "m" / "v3", "2" * 40)}
        if fa is not None:
            models["nemotron"] = Model("nemotron", "nemotron3", self.root / "m" / "nemotron", "3" * 40)
        if self.campplus:
            models["campplus"] = Model("campplus", "campplus", self.root / "m" / "campplus.onnx", "4" * 64)
        if "thai_screen" in self.routes:
            models["whisper"] = Model("whisper", "whisper-large-v3", self.root / "m" / "whisper", "5" * 40)
        work = fsutil.private_dir(platforms.cache_dir() / "work" / f"run-{secrets.token_hex(4)}")
        self.work_roots.append(work)
        vd = None
        if names and speakers and voices_dir is not None and Path(voices_dir).is_dir():
            if voices.scan(voices_dir).prints:          # as runtime.resolve: only a folder with someone to match
                vd = Path(voices_dir)
        env = {"PATH": f"{b}:/usr/bin:/bin", "HOME": str(self.root), "TMPDIR": str(work), "USER": "test",
               "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "ORT_DISABLE_TELEMETRY": "1"}
        rt = Runtime(tools=tools, models=models, routes=self.routes, env=env, work_root=work, voices_dir=vd)
        probs = list(self.problems)
        for route in sorted(need):
            if route not in self.routes:
                code = LANG_OF_ROUTE[route]
                params = {"reason": "asked", "code": code, "language": LANGUAGE_NAMES[code],
                          "size": round(self.sizes[code] / 1e9, 1)}
                probs.append(errors.problem("pack_not_installed", **params))
                probs.append(errors.problem("language", code=code, language=LANGUAGE_NAMES[code],
                                            size=params["size"]))
        if speakers and not self.fluidaudio:
            probs.append(errors.problem("no_speaker_labels", reason="clt_missing"))
            probs.append(errors.problem("speakers"))
        return rt, probs

    # ------------------------------------------------------------------ store, doctor, about
    def plan(self, pack_ids: list[str]) -> dict[str, Any]:
        self.calls.append(("plan", {"pack_ids": list(pack_ids)}))
        packs = {p: (0 if p in self.packs else self.sizes.get(p, 0)) for p in pack_ids}
        if not self.fluidaudio:                       # the speaker program's source counts under core, as in setup
            packs["core"] = packs.get("core", 0) + FLUIDAUDIO_SOURCE
        fetch = sum(packs.values())
        return {"fetch_bytes": fetch, "disk_needed_bytes": fetch * 1.2, "disk_free_bytes": self.free_bytes,
                "minutes_low": 4, "minutes_high": 40, "hosts": ["huggingface.co"], "packs": packs,
                "terms": [PACK_TERMS[p] for p in pack_ids if p in PACK_TERMS]}

    def install(self, pack_ids: list[str], *, progress: Callable[[str], None], speakers: bool = True,
                canary: bool = True) -> dict[str, Any]:
        self.calls.append(("install", {"pack_ids": list(pack_ids)}))
        self.calls.append(("install_options", {"speakers": speakers, "canary": canary}))
        if self.install_result is not None:
            for row in self.install_result.get("packs") or []:     # what setup stamped, as setup would
                if row.get("installed") and row.get("verified", True):
                    self.packs.add(row["id"])
            self._stamp()
            return dict(self.install_result)
        progress("installing")
        packs = []
        for p in pack_ids:
            fetched = 0 if p in self.packs else int(self.sizes.get(p, 0))
            self.packs.add(p)
            packs.append({"id": p, "installed": True, "verified": True, "fetched_bytes": fetched})
        self._stamp()
        return {"exit": 0, "packs": packs, "warnings": [], "errors": []}

    def remove(self, pack_id: str) -> dict[str, Any]:
        self.calls.append(("remove", {"pack_id": pack_id}))
        self.packs.discard(pack_id)
        return {"exit": 0, "removed": pack_id}

    def uninstall(self, *, keep_voices: bool, plan: bool) -> dict[str, Any]:
        self.calls.append(("uninstall", {"keep_voices": keep_voices, "plan": plan}))
        return {"exit": 0, "plan": plan, "removed": [] if plan else ["models"], "keep_voices": keep_voices}

    def status(self) -> list[dict[str, Any]]:
        self.calls.append(("status", {}))
        return [{"id": p, "installed": p in self.packs, "verified": p in self.packs, "version": 1}
                for p in ("core", "hu", "th")]

    def doctor(self, *, verify: bool, selftest: bool) -> dict[str, Any]:
        self.calls.append(("doctor", {"verify": verify, "selftest": selftest}))
        return dict(self.doctor_result or {"exit": 0, "warnings": [], "errors": []})

    def about(self) -> dict[str, Any]:
        self.calls.append(("about", {}))
        return {"version": "0.1.0", "models": [], "hosts": ["huggingface.co"]}

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(resolve=self.resolve, plan=self.plan, install=self.install, remove=self.remove,
                               uninstall=self.uninstall, status=self.status, doctor=self.doctor, about=self.about)
