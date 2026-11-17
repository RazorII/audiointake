#!/usr/bin/env python3
"""Speaker attribution of a labelled timeline against a reference RTTM, by time.

  timeline_attrib.py REF.rttm ITEMS.json [--name LABEL]

ITEMS.json: [[start, end, label, text], ...] in time order (words, or segments for a system that labels whole
segments). Each item's label covers [start, end]. Reference: single-speaker speech only (overlap is not judged).
Labels map one-to-one to reference speakers by the most shared time. Prints the share of judged reference speech time
whose covering item carries the right speaker, coverage (judged time with any item over it) and, for items with
text, a word-weighted figure: each item's words count for the reference speaker at the item's midpoint, if there is
exactly one (meaningful for word items only). The RESULT line is JSON for pooling. Needs numpy and scipy.
"""
from __future__ import annotations

import argparse
import json
import sys


def score(ref_p: str, items_p: str, name: str | None = None) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    name = name or items_p
    ref = []
    for l in open(ref_p, encoding="utf-8"):
        x = l.split()
        if x and x[0] == "SPEAKER":
            ref.append((float(x[-7]), float(x[-7]) + float(x[-6]), x[-3]))
    items = [(float(a), float(b), str(c), d) for a, b, c, d in json.load(open(items_p, encoding="utf-8"))]
    step = 0.02
    T = max(max(e for _, e, _ in ref), max(b for _, b, _, _ in items)) + 1
    n = int(T / step) + 1
    R = sorted({k for *_, k in ref}); ri = {k: i for i, k in enumerate(R)}
    cnt = np.zeros(n, dtype=np.int16); who = np.full(n, -1, dtype=np.int16)
    for s, e, k in ref:
        a, b = int(s / step), int(e / step)
        cnt[a:b] += 1; who[a:b] = ri[k]
    single = cnt == 1
    H = sorted({c for _, _, c, _ in items}); hi = {k: i for i, k in enumerate(H)}
    hyp = np.full(n, -1, dtype=np.int32)
    for s, e, c, _ in items:
        hyp[int(s / step):max(int(e / step), int(s / step) + 1)] = hi[c]
    ov = np.zeros((len(H), len(R)))
    m = single & (hyp >= 0)
    np.add.at(ov, (hyp[m], who[m]), 1)
    rr, cc = linear_sum_assignment(-ov)
    mp = {H[r]: R[c] for r, c in zip(rr, cc)}
    mapped = np.array([ri.get(mp.get(h), -2) for h in H] + [-3])
    ok = single & (hyp >= 0) & (mapped[hyp] == who)
    judged = single.sum(); covered = (single & (hyp >= 0)).sum()
    print(f"{name}: time {100 * ok.sum() / max(covered, 1):.1f}% right of covered single-speaker speech; "
          f"coverage {100 * covered / judged:.1f}% of {judged * step / 60:.1f} min; labels {len(H)} -> map {mp}")
    res = {"name": name, "ok_s": round(float(ok.sum() * step), 1), "covered_s": round(float(covered * step), 1),
           "judged_s": round(float(judged * step), 1), "labels": len(H)}
    print("RESULT " + json.dumps(res))
    wj = wok = 0
    for s, e, c, d in items:
        nw = len(str(d).split())
        if not nw:
            continue
        t = int(((s + e) / 2) / step)
        if 0 <= t < n and single[t]:
            wj += nw; wok += nw * (mp.get(c) == R[who[t]])
    if wj:
        print(f"{name}: words {100 * wok / wj:.1f}% right of {wj} words judged at their item's midpoint")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="timeline_attrib.py")
    ap.add_argument("ref"); ap.add_argument("items"); ap.add_argument("--name")
    a = ap.parse_args(argv)
    score(a.ref, a.items, a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
