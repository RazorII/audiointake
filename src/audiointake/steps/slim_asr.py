"""Slim the speech recogniser's JSON down to sentence and word timestamps: the ``.asr.json`` sidecar.

Usage::

    python -m audiointake.steps.slim_asr <parakeet.json> <out.asr.json>

Keeping the timestamps lets a later diarizer be merged straight onto an existing transcript with merge_speakers,
which reads this format as it is, without transcribing the audio again.

Each sentence keeps its words as ``"w": [[start, end, word], ...]``, so merge_speakers can label word by word on a
later re-merge too (sentence timestamps alone credit a run-on sentence that spans two speakers to one of them). The
per-token arrays are dropped: the file is about four times the sentence-only size but a fifth of the full JSON.
``"w"`` is written only where ``merge_speakers.seg_words`` can rejoin the words to the sentence text exactly;
Whisper's sentences and the Thai turns carry no tokens, so none.

``slim(d)`` returns the sidecar's exact text (compact JSON, non-ASCII kept, no trailing newline).
"""
from __future__ import annotations

import json
import sys

from .merge_speakers import seg_words  # one definition of a word, shared with the merge


def slim_sentence(s: dict) -> dict:
    """One sentence: start and end to 3 decimals, its text, and its words to 2 decimals where they rejoin."""
    out = {"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "text": s["text"]}
    ws = seg_words(s)
    if ws:
        out["w"] = [[round(a, 2), round(b, 2), t] for a, b, t in ws]
    return out


def slim(d: dict) -> str:
    """The ``.asr.json`` sidecar's text for a recogniser JSON ``d``."""
    sents = d.get("sentences") or d.get("segments") or []
    out = {"sentences": [
        slim_sentence(s) for s in sents if isinstance(s, dict) and s.get("text") and s.get("start") is not None
    ]}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: slim_asr <parakeet.json> <out.asr.json>", file=sys.stderr)
        return 1
    src, dst = argv[0], argv[1]
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    text = slim(d)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
