"""Fake tool interpreter for the workers: ``-I -B -m audiointake.steps.speaker_id match WAV RTTM --out OUT
--voices DIR --model ONNX`` (``-B`` is required: under ``-I`` only it keeps bytecode out of the source tree).

Writes the speaker file in the engine's shape: ``labels`` for every RTTM label (named as the description's
``match`` says, when that name has a voiceprint in DIR), and ``enrolled`` with every voiceprint's name, heard or
not (the roster that must never reach a transcript or a record). Prints the names to stdout, as the real worker
does. ``fail.match``: ``exit``, ``partial`` (a file, then exit 1), ``garbage`` (not JSON, exit 0), ``nofile``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fakes import STDOUT_MARK, read_desc


def _labels(rttm: Path) -> list[str]:
    out = set()
    for line in rttm.read_text(encoding="utf-8").splitlines():
        x = line.split()
        if len(x) >= 10 and x[0] == "SPEAKER":
            out.add(x[-3])
    return sorted(out)


def _names(folder: Path) -> list[str]:
    names = []
    for p in sorted(folder.glob("*.json")):
        if p.name.startswith("."):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(d, dict) and "name" in d:
            names.append(d["name"])
    return sorted(names)


def main(argv: list[str]) -> int:
    if argv[:2] != ["-I", "-B"]:
        print(f"fake python: started without -I -B: {argv[:2]}", file=sys.stderr)
        return 2
    args = argv[2:]
    if args[:2] != ["-m", "audiointake.steps.speaker_id"] or args[2] != "match":
        print(f"fake python: unexpected {args[:3]}", file=sys.stderr)
        return 2
    wav, rttm = Path(args[3]), Path(args[4])
    out = Path(args[args.index("--out") + 1])
    folder = Path(args[args.index("--voices") + 1])
    if "--model" not in args:
        return 2
    d = read_desc(wav) or {}
    how = (d.get("fail") or {}).get("match")
    if how == "exit":
        print("fake speaker ID: failed", file=sys.stderr)
        return 1
    if how == "nofile":
        return 0
    if how == "garbage":
        out.write_text("not json", encoding="utf-8")
        return 0
    enrolled = _names(folder)
    want = d.get("match") or {}
    labels = {}
    for i, lab in enumerate(_labels(rttm)):
        name = want.get(lab) if want.get(lab) in enrolled else None
        labels[lab] = {"name": name, "score": 0.71 if name else 0.21, "best_guess": enrolled[0] if enrolled else None,
                       "runner_up": [enrolled[-1], 0.12] if enrolled else None, "speech_s": 12.5 + i,
                       "why": "named" if name else "below threshold"}
        print(f"{STDOUT_MARK}\t{lab}\t{name or '-'}")
    obj = {"engine": "sherpa-onnx campplus-zh_en-advanced aa3cfc16", "threshold": 0.44, "margin": 0.08,
           "min_speech_s": 8.0, "enrolled": enrolled, "labels": labels}
    out.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    if how == "partial":
        return 1
    return 0
