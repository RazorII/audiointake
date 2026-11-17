"""Fake ffprobe: the channel lines, the duration and the creation_time tag of a fake recording."""
from __future__ import annotations

import sys

from fakes import read_desc


def main(argv: list[str]) -> int:
    d = read_desc(argv[-1])
    if d is None:
        print(f"{argv[-1]}: Invalid data found when processing input", file=sys.stderr)
        return 1
    entries = argv[argv.index("-show_entries") + 1] if "-show_entries" in argv else ""
    if entries == "stream=channels":
        for c in d.get("channels") or []:
            print(c)
    elif entries == "format=duration":
        print(f"{float(d.get('duration') or 0):.6f}")
    elif entries == "format_tags=creation_time":
        if d.get("creation_time"):
            print(d["creation_time"])
    return 0
