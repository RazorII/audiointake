"""Fake parakeet-mlx: writes ``<output-dir>/<output-template>.json`` in parakeet-mlx's shape, made of invented
words that each carry the canary, and prints that text to stdout (as a chatty tool might).

The description's ``fail`` names a template (``primary`` or ``fill``) and how: ``exit`` (exit 1 with a stderr
line), ``hang`` (sleeps; the parent's timeout or a signal stops it), ``partial`` (writes half a file, exit 1).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fakes import STDOUT_MARK, WORDS, read_desc
from fixtures.build import asr_json


def transcript(d: dict) -> dict:
    return asr_json(int(d.get("seed") or 1), sentences=int(d.get("sentences") or 0), words=WORDS)


def main(argv: list[str]) -> int:
    audio = argv[0]
    out_dir = Path(argv[argv.index("--output-dir") + 1])
    name = argv[argv.index("--output-template") + 1]
    d = read_desc(audio)
    if d is None:
        print("could not load audio", file=sys.stderr)
        return 1
    how = (d.get("fail") or {}).get(name)
    if how == "hang":
        marker = d.get("marker")
        if marker:
            Path(marker).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(120)
        return 1
    if how == "exit":
        print("fake parakeet: the model failed to load", file=sys.stderr)
        return 1
    t = transcript(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(t, ensure_ascii=False)
    if how == "partial":
        (out_dir / f"{name}.json").write_text(data[: len(data) // 2], encoding="utf-8")
        print("fake parakeet: interrupted", file=sys.stderr)
        return 1
    print(f"{STDOUT_MARK} {t['text']}")
    (out_dir / f"{name}.json").write_text(data, encoding="utf-8")
    print("fake parakeet: done", file=sys.stderr)
    return 0
