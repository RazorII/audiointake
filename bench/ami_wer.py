#!/usr/bin/env python3
"""Word error rate of transcripts of the four AMI meetings against the AMI manual word transcripts (time-ordered).

  ami_wer.py ANNOTATIONS.zip HYP_DIR [--meetings ES2004a IS1009a TS3003a EN2002a]

ANNOTATIONS.zip is the AMI manual annotations release (ami_public_manual_1.6.2.zip, CC BY 4.0; bench/ami.py fetch
--annotations gets it). HYP_DIR holds <meeting>.items.json for each meeting: [[start, end, label, text], ...], the
words or segments of the system being scored (bench/ami.py run writes ours). The reference is every word of every
speaker's words file, punctuation removed, ordered by start time. Both sides: lower case, everything but a-z, 0-9 and
apostrophes to spaces. Prints per meeting and pooled WER. Standard library only (pure-Python Levenshtein; a minute
or so with 8 processes).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MEETINGS = ("ES2004a", "IS1009a", "TS3003a", "EN2002a")


def norm(ws) -> list[str]:
    out = []
    for w in ws:
        out += re.sub(r"[^a-z0-9']+", " ", str(w).lower()).split()
    return out


def reference(zip_path: str, meeting: str) -> list[str]:
    items = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith(f"words/{meeting}.") and name.endswith(".words.xml"):
                for attrs, txt in re.findall(r'<w\b([^>]*)>([^<]*)</w>', z.read(name).decode("utf-8")):
                    if 'punc="true"' in attrs or not txt.strip():
                        continue
                    st = re.search(r'starttime="([\d.]+)"', attrs)
                    items.append((float(st.group(1)) if st else 1e9, txt))
    return norm(t for _, t in sorted(items))


def lev(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def job(args):
    zip_path, hyp_dir, m = args
    r = reference(zip_path, m)
    h = norm(x[3] for x in json.load(open(Path(hyp_dir) / f"{m}.items.json", encoding="utf-8")))
    return m, len(r), len(h), lev(r, h)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ami_wer.py")
    ap.add_argument("annotations"); ap.add_argument("hyp_dir")
    ap.add_argument("--meetings", nargs="+", default=list(MEETINGS))
    a = ap.parse_args(argv)
    tot = [0, 0, 0]
    with ProcessPoolExecutor(8) as ex:
        for m, nr, nh, d in ex.map(job, [(a.annotations, a.hyp_dir, m) for m in a.meetings]):
            print(f"{m}: ref {nr} hyp {nh} WER {100 * d / nr:.1f}%", flush=True)
            tot[0] += nr; tot[1] += nh; tot[2] += d
    print(f"POOLED ref {tot[0]} hyp {tot[1]} WER {100 * tot[2] / tot[0]:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
