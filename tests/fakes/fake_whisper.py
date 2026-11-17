"""Fake mlx_whisper: writes ``<output-dir>/<output-name>.json`` in mlx_whisper's shape (invented words with the
canary) and prints the text to stdout, as mlx_whisper does."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fakes import STDOUT_MARK, WORDS, read_desc
from fixtures.build import whisper_json


def main(argv: list[str]) -> int:
    d = read_desc(argv[0])
    if d is None or (d.get("fail") or {}).get("whisper"):
        print("fake whisper: failed", file=sys.stderr)
        return 1
    out_dir = Path(argv[argv.index("--output-dir") + 1])
    name = argv[argv.index("--output-name") + 1] if "--output-name" in argv else "whisper"
    w = whisper_json(int(d.get("seed") or 1), words=WORDS)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{STDOUT_MARK} {w['text']}")
    (out_dir / f"{name}.json").write_text(json.dumps(w, ensure_ascii=False), encoding="utf-8")
    return 0
