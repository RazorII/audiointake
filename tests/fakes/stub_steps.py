"""Stand-ins for the engine's step functions, with its exact signatures, for the CLI and pipeline tests.

``transcribe`` runs the fake recogniser twice as children through ``proc.run`` (primary, then fill), as the real
ensemble runs parakeet-mlx, and writes ``<outdir>/<template>.json`` and ``.txt``. What the description's
``ensemble`` says (``hu_route``, ``hu_share``, ``thai``) comes back on the ``EnsembleResult``, with the tag the
real ensemble would print. ``merge`` is a sentence-level merge with the engine's numbering rule; ``count_speakers``,
``slim``, ``enroll_clips`` and ``enroll_label`` behave as the engine's do, on made-up data: the same return shapes,
``results`` on an exception from a multi-clip enroll, and ``SpeakerIdError`` for the engine's own failures.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audiointake import errors, proc
from audiointake.contracts import EnsembleResult, Log, Merged, Problem, Runtime, ThaiSummary, Turn

from fakes import MODEL_SHA, read_desc

CALLS: list[dict[str, Any]] = []          # every transcribe call's keyword arguments, for assertions
WORK_DIRS: list[Path | None] = []         # enroll_clips' work_dir per call


class NoSentences(ValueError):
    pass


class HeldBack(Exception):
    def __init__(self, problems: list[Problem]) -> None:
        super().__init__("held back")
        self.problems = problems


class TooLittleSpeech(Exception):
    def __init__(self, seconds: float, *, file: str | None = None) -> None:
        super().__init__(seconds)
        self.seconds = seconds
        self.file = file


class SpeakerIdError(Exception):
    """The engine's own failure (its model missing, a print made with another model, ...)."""


def transcribe(audio: Path, outdir: Path, template: str, fmt: str, rt: Runtime, *, force_lang: str | None = None,
               log: Log, duration_s: float | None = None) -> EnsembleResult:
    CALLS.append({"audio": str(audio), "template": template, "fmt": fmt, "force_lang": force_lang,
                  "duration_s": duration_s})
    d = read_desc(audio) or {}
    fail = d.get("fail") or {}
    if fail.get("decode"):
        raise errors.error_for("decode_failed", file=Path(audio).name)
    allowed = proc.allowed_for(rt.tools)

    def asr(model_key: str, name: str) -> Path | None:
        c = proc.run([rt.tools.parakeet_cli, audio, "--model", rt.models[model_key].path, "--output-format", "json",
                      "--output-dir", outdir, "--output-template", name], env=rt.env,
                     timeout=float(d.get("timeout") or proc.child_timeout("parakeet", duration_s or 0)),
                     allowed=allowed, stdout="capture")
        j = Path(outdir) / f"{name}.json"
        if not c.ok:
            for ln in c.stderr_tail:
                log(f"parakeet: {ln}")
        return j if c.ok and j.exists() else None

    prim = asr("ultra", "primary")
    if prim is None:
        raise errors.AsrFailed("asr_failed", file=Path(audio).name)
    tag = rt.models["ultra"].tag
    fill = asr("v3", "fill")
    e = d.get("ensemble") or {}
    hu_route = e.get("hu_route", "none")
    if "hu_vote" in rt.routes and (force_lang == "hu" or hu_route == "no_pack"):
        hu_route = "vote"            # forced, or detected with the pack there: the real ensemble votes either way
    if fill is None:
        fill_ok = False
    else:
        fill_ok = True
        if hu_route == "vote":
            tag = "whisper-large-v3+ultra/v3-vote"
        elif hu_route == "vote_failed":
            tag += "+hu-failed"
        else:
            tag = f"{tag}+{rt.models['v3'].tag}-gapfill"
    th = e.get("thai") or {}
    thai = ThaiSummary(screened=bool(th.get("screened", "thai_screen" in rt.routes)), windows=int(th.get("windows", 0)),
                       thai_windows=int(th.get("thai_windows", 0)), turns=int(th.get("turns", 0)),
                       seconds=float(th.get("seconds", 0.0)), screen_only=bool(th.get("screen_only", False)),
                       failed=bool(th.get("failed", False)))
    if thai.failed:
        tag += "+th-failed"
    elif thai.turns:
        tag += "+typhoon-whisper-large-v3-thai"
    data = json.loads(prim.read_text(encoding="utf-8"))
    final = Path(outdir) / f"{template}.json"
    final.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    txt = Path(outdir) / f"{template}.txt"
    body = " ".join(s["text"].strip() for s in data["sentences"]).strip() + str(d.get("txt_end", "\n"))
    txt.write_bytes(body.encode("utf-8"))
    log(f"ensemble: {tag}")
    if fail.get("vanish"):                               # the recording goes away while it's being worked on
        Path(audio).unlink()
    return EnsembleResult(tag=tag, json_path=final, txt_path=txt, hu_share=float(e.get("hu_share", 0.01)),
                          hu_detected=hu_route in ("vote", "vote_failed", "no_pack"), hu_route=hu_route,
                          fill_ok=fill_ok, thai=thai)


def _segments(text: str | None) -> list[dict[str, Any]]:
    out = []
    for line in (text or "").splitlines():
        x = line.split()
        if not x or x[0] != "SPEAKER":
            continue
        if len(x) < 10:
            raise ValueError("RTTM line has too few fields")
        out.append({"start": float(x[-7]), "end": float(x[-7]) + float(x[-6]), "spk": x[-3]})
    return out


def count_speakers(rttm_text: str | None) -> int:
    return len({s["spk"] for s in _segments(rttm_text)})


def merge(asr: dict, rttm_text: str | None, names: Mapping[str, str] | None, split: bool = True) -> Merged:
    sents = [s for s in asr.get("sentences") or [] if str(s.get("text") or "").strip()]
    if not sents:
        raise NoSentences("no sentence with text")
    diars = _segments(rttm_text) if rttm_text else []
    if not diars:
        lines = [" ".join(s["text"].strip() for s in sents).strip()]
        turns = [Turn(None, "", float(s["start"]), float(s["end"]), s["text"].strip()) for s in sents]
        return Merged(lines=lines, text="\n".join(lines) + "\n", turns=turns)
    names = dict(names or {})
    mapping: dict[str, str] = {}
    lines: list[str] = []
    turns: list[Turn] = []
    cur = None
    buf: list[str] = []
    t0 = t1 = 0.0
    lab0 = None
    for s in sents:
        best, ov = None, 0.0
        for dd in diars:
            o = min(s["end"], dd["end"]) - max(s["start"], dd["start"])
            if o > ov:
                best, ov = dd["spk"], o
        if best is not None and best not in mapping:
            mapping[best] = names.get(best) or f"Speaker {sum(1 for k in mapping if k not in names) + 1}"
        disp = mapping.get(best, "Speaker ?")
        if disp != cur:
            if buf:
                lines.append(f"{cur}: {' '.join(buf)}")
                turns.append(Turn(lab0, cur, t0, t1, " ".join(buf)))
            cur, buf, t0, lab0 = disp, [s["text"].strip()], float(s["start"]), best
        else:
            buf.append(s["text"].strip())
        t1 = float(s["end"])
    if buf:
        lines.append(f"{cur}: {' '.join(buf)}")
        turns.append(Turn(lab0, cur, t0, t1, " ".join(buf)))
    return Merged(lines=lines, text="\n".join(lines) + "\n", turns=turns)


def slim(d: dict) -> str:
    sents = d.get("sentences") or []
    out = {"sentences": [{"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "text": s["text"]}
                         for s in sents if isinstance(s, dict) and s.get("text") and s.get("start") is not None]}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def _write_print(name: str, voices_dir: Path, source: str, seconds: float) -> dict[str, Any]:
    """One source into a print, as the engine's ``_add`` does: a source already there is skipped."""
    p = Path(voices_dir) / f"{name.lower()}.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"name": name, "model_sha256": MODEL_SHA,
                                                                       "vectors": []}
    if d.get("model_sha256") != MODEL_SHA:
        raise SpeakerIdError(f"{p.name} was made with another model")
    if any(v["source"] == source for v in d["vectors"]):
        return {"name": name, "source": source, "added": False, "skipped": True, "seconds": 0.0,
                "sources": len(d["vectors"]), "forced": [], "lines": []}
    d["vectors"].append({"vec": [0.5, 0.5], "source": source, "seconds": seconds, "lang": None,
                         "added": "2026-01-02"})
    d["updated"] = "2026-01-02"
    from audiointake import fsutil
    fsutil.atomic_write(p, json.dumps(d, indent=1).encode("utf-8"), overwrite=p.exists(), mode=0o600)
    return {"name": name, "source": source, "added": True, "skipped": False, "seconds": seconds,
            "sources": len(d["vectors"]), "forced": [], "lines": []}


def _verdict(desc: dict | None) -> tuple[str | None, Any]:
    v = (desc or {}).get("enroll")
    if not v:
        return None, None
    kind, _, arg = str(v).partition(":")
    return kind, arg


def _raise_for(kind: str | None, arg: Any, name: str, file: str) -> None:
    if kind == "held":
        codes = arg.split(",")
        raise HeldBack([errors.problem(c, name=name, file=file, other="Oskel Quorn", percent=35, score=0.31)
                        for c in codes])
    if kind == "little":
        raise TooLittleSpeech(float(arg), file=file)
    if kind == "engine":
        raise SpeakerIdError("the speaker model is missing")
    if kind == "bug":
        raise KeyError(arg)


def enroll_clips(name: str, clips: Sequence[Path], lang: str | None, voices_dir: Path, model: Path, ffmpeg: Path,
                 env: Mapping[str, str], force: bool, *, work_dir: Path | None = None) -> dict[str, Any]:
    """Each clip written as it passes; an exception carries ``results``, the clips added before it (the engine's
    contract)."""
    WORK_DIRS.append(work_dir)
    results: list[dict[str, Any]] = []
    for c in clips:
        try:
            kind, arg = _verdict(read_desc(c))
            if not force or kind != "held":
                _raise_for(kind, arg, name, Path(c).name)
            results.append(_write_print(name, voices_dir, f"sample {Path(c).name}", 31.0))
        except Exception as e:
            e.results = results                      # type: ignore[attr-defined]
            raise
    return {"name": name, "results": results, "lines": []}


def enroll_label(name: str, wav: Path, rttm_text: str, label: str, source: str, lang: str | None, voices_dir: Path,
                 model: Path, force: bool) -> dict[str, Any]:
    kind, arg = _verdict(read_desc(wav))
    if not force or kind != "held":
        _raise_for(kind, arg, name, Path(wav).name)
    return _write_print(name, voices_dir, source, 22.0)


def namespace() -> SimpleNamespace:
    return SimpleNamespace(transcribe=transcribe, merge=merge, count_speakers=count_speakers, NoSentences=NoSentences,
                           slim=slim, enroll_clips=enroll_clips, enroll_label=enroll_label, HeldBack=HeldBack,
                           TooLittleSpeech=TooLittleSpeech, SpeakerIdError=SpeakerIdError)
