#!/usr/bin/env python3
"""How much diarized speech produced no words: the measure behind the `speech_not_transcribed` warning.

  uncovered.py ASR.json RTTM [ASR.json RTTM ...] [--min-seconds 30] [--min-share 0.25]

ASR.json is a transcript's sentence list (the ensemble's JSON or a `.asr.json` sidecar: sentences with start, end
and text); RTTM is the diarizer's output for the same audio. Speech is the union of the RTTM's turns; covered speech
is the part of it inside any sentence with text. Prints, per pair, the diarized speech, the uncovered seconds and
their share, and whether the warning's rule fires (uncovered >= --min-seconds and share >= --min-share). Standard
library only.
"""
from __future__ import annotations

import argparse
import json
import sys


def union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        tot += max(0.0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def measure(asr_path: str, rttm_path: str) -> dict:
    d = json.load(open(asr_path, encoding="utf-8"))
    sents = d.get("sentences") or []
    covered = union([(float(s["start"]), float(s["end"])) for s in sents if str(s.get("text", "")).strip()])
    turns = []
    for line in open(rttm_path, encoding="utf-8"):
        x = line.split()
        if x and x[0] == "SPEAKER":
            turns.append((float(x[-7]), float(x[-7]) + float(x[-6])))
    speech = union(turns)
    total = sum(b - a for a, b in speech)
    unc = total - overlap(speech, covered)
    return {"speech_s": round(total, 1), "uncovered_s": round(unc, 1), "share": round(unc / total, 3) if total else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uncovered.py")
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--min-seconds", type=float, default=30.0)
    ap.add_argument("--min-share", type=float, default=0.25)
    a = ap.parse_args(argv)
    if len(a.pairs) % 2:
        ap.error("give ASR.json and RTTM in pairs")
    for asr, rttm in zip(a.pairs[::2], a.pairs[1::2]):
        m = measure(asr, rttm)
        fires = m["uncovered_s"] >= a.min_seconds and m["share"] >= a.min_share
        print(f"{asr}: speech {m['speech_s']} s, no words over {m['uncovered_s']} s ({100 * m['share']:.1f} %): "
              f"{'fires' if fires else 'quiet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
