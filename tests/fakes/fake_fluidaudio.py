"""Fake fluidaudiocli ``nemotron3-diarize WAV --variant offline --threshold 0.4 --output RTTM --models DIR``.

Writes turns among ``speakers`` labels covering the fake transcript's speech (or ``diar_duration`` seconds), with
the wav's stem as the file id. The description's ``fail.diarize``: ``exit`` (no file, exit 1), ``partial`` (a
file, then exit 1), ``empty`` (an empty file, exit 0), ``malformed`` (a broken line, exit 0), ``nofile`` (exit 0,
no file), ``hang``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from fakes import read_desc
from fakes.fake_parakeet import transcript
from fixtures.build import rttm_text


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "nemotron3-diarize":
        print("unknown command", file=sys.stderr)
        return 64
    wav = Path(argv[1])
    out = Path(argv[argv.index("--output") + 1])
    if "--models" not in argv:
        print("refusing to download models", file=sys.stderr)
        return 2
    d = read_desc(wav)
    if d is None:
        return 1
    how = (d.get("fail") or {}).get("diarize")
    if how == "hang":
        time.sleep(120)
        return 1
    if how == "exit":
        print("fake fluidaudio: CoreML failed", file=sys.stderr)
        return 1
    if how == "nofile":
        return 0
    if how == "empty" or int(d.get("speakers") or 0) == 0:
        out.write_text("", encoding="utf-8")
        return 0
    sents = transcript(d)["sentences"]
    end = float(d.get("diar_duration") or (sents[-1]["end"] + 0.5 if sents else 5.0))
    file_id = wav.name[: -len(".wav")] if wav.name.endswith(".wav") else wav.name
    text = rttm_text(int(d.get("seed") or 1), file_id=file_id, speakers=int(d["speakers"]), duration=end,
                     turn=tuple(d.get("turn") or (2.0, 9.0)))
    if how == "malformed":
        text += "SPEAKER broken line\n"
    out.write_text(text, encoding="utf-8")
    if how == "partial":
        print("fake fluidaudio: crashed after writing", file=sys.stderr)
        return 1
    return 0
