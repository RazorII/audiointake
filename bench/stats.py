#!/usr/bin/env python3
"""Paired bootstrap intervals for differences in corpus error rate, from `fleurs.py score --json` output.

  stats.py diff SCORE.json LANG_A:SYSTEM_A LANG_B:SYSTEM_B [--subset no_digit] [--resamples N] [--seed S]

Prints the corpus error rate of each side, their difference (A minus B, in points), and a 95 % percentile interval
of that difference from a paired bootstrap: N resamples (default 10,000) of the clip ids, with replacement, drawn with
random.Random(S) (default 42); each resample recomputes both corpus rates over the same drawn ids. Pairing is by
clip id, so it works across languages when the two sets are translations of the same sentences (fleurs.py's de_de
and es_419 ids are the same 50 sentence ids) and across systems on one language. --subset no_digit restricts both
sides to the clips whose reference has no digit in either language. --json prints the result as one JSON object.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def corpus(clips: dict[str, list[int]], ids: list[str]) -> float:
    ref = sum(clips[i][0] for i in ids)
    return 100.0 * sum(clips[i][1] for i in ids) / ref if ref else 0.0


def paired_bootstrap(a: dict[str, list[int]], b: dict[str, list[int]], ids: list[str],
                     resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """(difference, low, high): corpus rate of a minus b over ids, and the 95 % percentile interval."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        draw = [rng.choice(ids) for _ in ids]
        diffs.append(corpus(a, draw) - corpus(b, draw))
    diffs.sort()
    lo = diffs[int(0.025 * resamples)]
    hi = diffs[min(resamples - 1, int(0.975 * resamples))]
    return corpus(a, ids) - corpus(b, ids), lo, hi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stats.py")
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("diff")
    d.add_argument("score", type=Path)
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--subset", choices=("all", "no_digit"), default="all")
    d.add_argument("--resamples", type=int, default=10000)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    score = json.loads(a.score.read_text(encoding="utf-8"))
    sides = []
    for spec in (a.a, a.b):
        lang, _, system = spec.partition(":")
        if lang not in score or system not in score[lang]:
            ap.error(f"{spec}: no such language and system in {a.score}")
        sides.append((lang, system, score[lang][system]))
    (la, sa, ea), (lb, sb, eb) = sides
    ids = sorted(set(ea["clips"]) & set(eb["clips"]), key=lambda x: (len(x), x))
    if a.subset == "no_digit":
        ids = [i for i in ids if i in set(ea["no_digit_ids"]) & set(eb["no_digit_ids"])]
    if not ids:
        ap.error("the two sides share no clip id")
    diff, lo, hi = paired_bootstrap(ea["clips"], eb["clips"], ids, a.resamples, a.seed)
    res = {"a": f"{la}:{sa}", "b": f"{lb}:{sb}", "subset": a.subset, "clips": len(ids),
           "rate_a": round(corpus(ea["clips"], ids), 3), "rate_b": round(corpus(eb["clips"], ids), 3),
           "diff": round(diff, 3), "ci95": [round(lo, 3), round(hi, 3)], "excludes_zero": lo > 0 or hi < 0,
           "resamples": a.resamples, "seed": a.seed}
    if a.json:
        print(json.dumps(res))
    else:
        print(f"{res['a']} {corpus(ea['clips'], ids):.2f}% - {res['b']} {corpus(eb['clips'], ids):.2f}% = {diff:+.2f} points "
              f"[95% {lo:+.2f}, {hi:+.2f}] over {len(ids)} clips ({a.subset}); "
              f"interval {'excludes' if res['excludes_zero'] else 'includes'} 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
