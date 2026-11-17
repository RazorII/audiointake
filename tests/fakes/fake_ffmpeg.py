"""Fake ffmpeg: "converts" a fake recording by copying its description into the output wav; with
``-hide_banner ... -i FILE`` it prints the Duration and Stream lines a real ffmpeg prints on stderr.

Fails as the description's ``fail`` says: ``decode`` (every conversion), ``diar_wav`` (the diarizer's
``<stem>.16k.wav`` only).
"""
from __future__ import annotations

import sys
from pathlib import Path

from fakes import read_desc

LAYOUT = {1: "mono", 2: "stereo", 3: "2.1", 4: "4.0", 6: "5.1", 8: "7.1"}


def _info(src: str) -> int:
    d = read_desc(src)
    if d is None:
        print(f"{src}: Invalid data found when processing input", file=sys.stderr)
        return 1
    print(f"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '{src}':", file=sys.stderr)
    print("  Metadata:", file=sys.stderr)
    if d.get("creation_time"):
        print(f"    creation_time   : {d['creation_time']}", file=sys.stderr)
    for i in range(12):
        print(f"    tag{i}           : filler", file=sys.stderr)
    dur = float(d.get("duration") or 0)
    h, rem = divmod(dur, 3600)
    m, s = divmod(rem, 60)
    print(f"  Duration: {int(h):02d}:{int(m):02d}:{s:05.2f}, start: 0.000000, bitrate: 64 kb/s", file=sys.stderr)
    layouts = d.get("layouts") or [LAYOUT.get(int(c), f"{int(c)} channels") for c in d.get("channels") or []]
    for i, lay in enumerate(layouts):
        print(f"  Stream #0:{i}[0x{i + 1}](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, {lay}, fltp, "
              "64 kb/s (default)", file=sys.stderr)
        print("      Metadata:", file=sys.stderr)
        print("        handler_name    : Core Media Audio", file=sys.stderr)
    print("At least one output file must be specified", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    src = argv[argv.index("-i") + 1] if "-i" in argv else ""
    if "-hide_banner" in argv:
        return _info(src)
    if "-f" in argv and argv[argv.index("-f") + 1] == "lavfi":
        Path(argv[-1]).write_text('{"fake_audio": 1, "duration": 3.0, "channels": [1]}', encoding="utf-8")
        return 0
    d = read_desc(src)
    out = argv[-1]
    fail = (d or {}).get("fail") or {}
    if d is None or fail.get("decode"):
        print("Error while decoding stream", file=sys.stderr)
        return 1
    if fail.get("diar_wav") and out.endswith(".16k.wav"):
        print("Conversion failed", file=sys.stderr)
        return 1
    Path(out).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
    return 0
