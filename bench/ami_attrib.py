#!/usr/bin/env python3
"""Word-level speaker attribution against a reference RTTM with true speakers (AMI `only_words`).

  ami_attrib.py ASR.json HYP.rttm REF.rttm [--steps DIR] [--min-turn S ...]

  ASR.json   the transcript's sentences with word timings: a full ensemble JSON (tokens), or a sidecar whose
             sentences carry "w" (`--format sidecars` writes <stem>.asr.json)
  HYP.rttm   the diarizer's RTTM for the same audio (<stem>.diar.rttm)
  REF.rttm   BUTSpeechFIT/AMI-diarization-setup, only_words/rttms/test/<meeting>.rttm (Apache-2.0)

Truth per ASR word: the one reference speaker whose segment covers the word's midpoint (none or several: not
judged). Diarizer labels map to reference speakers by the one-to-one assignment with the most shared time, computed
once from the two RTTMs, so every rule shares it. Prints the accuracy of the word rule (the merge's default) and of
the one-speaker-per-sentence rule, with turn counts and how many judged words in 1-2 word turns are right; the last
line (RESULT) is JSON for pooling over meetings. --min-turn S also scores the word rule splitting only at diarizer
turns of at least S seconds. --steps names a folder holding merge_speakers.py and rttm.py to measure instead of the
package's own. Needs numpy and scipy.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path


def load_steps(steps: Path | None):
    if steps is None:
        return importlib.import_module("audiointake.steps.merge_speakers"), importlib.import_module("audiointake.steps.rttm")
    sys.path.insert(0, str(steps))
    mods = []
    for name in ("merge_speakers", "rttm"):
        spec = importlib.util.spec_from_file_location(f"{name}_under_test", steps / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods[0], mods[1]


def rttm_segments(rttm, path: str) -> list[dict]:
    reader = getattr(rttm, "segments_from_path", None) or getattr(rttm, "segments")
    return reader(path)


def score(asr_p: str, hyp_p: str, ref_p: str, steps: Path | None = None, min_turns: tuple[float, ...] = ()) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    M, rttm = load_steps(steps)
    asr = M.asr_segments(json.load(open(asr_p, encoding="utf-8")))
    hyp = [{"start": s["start"], "end": s["start"] + s["dur"], "spk": s["spk"]} for s in rttm_segments(rttm, hyp_p)]
    ref = [(s["start"], s["start"] + s["dur"], s["spk"]) for s in rttm_segments(rttm, ref_p)]
    H = sorted({d["spk"] for d in hyp}); R = sorted({r[2] for r in ref})
    ov = np.zeros((len(H), len(R)))
    for d in hyp:
        for s, e, k in ref:
            o = min(d["end"], e) - max(d["start"], s)
            if o > 0:
                ov[H.index(d["spk"]), R.index(k)] += o
    ri, ci = linear_sum_assignment(-ov)
    mp = {H[i]: R[j] for i, j in zip(ri, ci)}
    ref_sorted = sorted(ref)

    def truth(t):
        ks = {k for s, e, k in ref_sorted if s <= t <= e}
        return ks.pop() if len(ks) == 1 else None

    words = []
    for si, s in enumerate(asr):
        for a, b, w in (s["words"] or []):
            words.append((si, a, b, w, truth((a + b) / 2)))

    def judge(name, labs):
        judged = [(mp.get(l), w[4]) for l, w in zip(labs, words) if w[4]]
        ok = sum(1 for x, y in judged if x == y)
        turns, cur = [], object()
        for l in labs:
            if l != cur:
                turns.append(0); cur = l
            turns[-1] += 1
        idx = sj = sok = 0
        for t in turns:
            for l, w in zip(labs[idx:idx + t], words[idx:idx + t]):
                if t <= 2 and w[4]:
                    sj += 1; sok += mp.get(l) == w[4]
            idx += t
        print(f"  {name:22s} {100 * ok / max(1, len(judged)):5.1f}% of {len(judged)} judged words ({len(judged) - ok} wrong); "
              f"turns {len(turns)}, 1-2 word turns {sum(1 for t in turns if t <= 2)} ({sok}/{sj} judged words in them right)")
        return [ok, len(judged)]

    sentence = {si: M.speaker_for(s, hyp) for si, s in enumerate(asr)}
    word_rule = lambda diars: [lab for lab, _ in M.units([s for s in asr if s["words"]], diars, True)]
    print(f"{Path(asr_p).name}: {len(words)} words, {sum(1 for w in words if w[4])} judged; map {mp}")
    res = {"sentence": judge("sentence rule", [sentence[w[0]] for w in words]), "word": judge("word rule", word_rule(hyp))}
    for mt in min_turns:
        res[f"word+minturn{mt}"] = judge(f"word, turns >= {mt}s", word_rule([d for d in hyp if d["end"] - d["start"] >= mt]))
    print("RESULT " + json.dumps(res))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ami_attrib.py")
    ap.add_argument("asr"); ap.add_argument("hyp"); ap.add_argument("ref")
    ap.add_argument("--steps", type=Path)
    ap.add_argument("--min-turn", type=float, nargs="*", default=[])
    a = ap.parse_args(argv)
    score(a.asr, a.hyp, a.ref, a.steps, tuple(a.min_turn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
