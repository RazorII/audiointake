#!/usr/bin/env python3
"""FLEURS test clips with human transcripts, fixed by id: prepare them and score transcripts against them.

  fleurs.py prep WORKDIR [LANG...] [--parquet-dir DIR] [--keep]
      Fetch each language's FLEURS test parquet (pinned below by dataset commit and SHA-256), write the listed clips to
      WORKDIR/fl/<lang>/<id>.wav with the human transcript in <id>.ref, and delete the parquet again unless --keep.
      A parquet already in --parquet-dir (or WORKDIR) with the pinned size and SHA-256 is used instead of fetching it.
      Default: every language below. Needs pyarrow (pip install "audiointake[bench]").
  fleurs.py score WORKDIR SYSTEM... [--json OUT]
      Corpus word error rate per language for WORKDIR/fo/<system>/<lang>/<id>.txt (character error rate for th_th),
      plus the rate on the clips whose reference has no digit, and an accent-folded rate as a diagnostic. --json writes
      every clip's (reference units, edits) for each system, the input bench/stats.py bootstraps.
  fleurs.py ids LANG PARQUET
      Print the distinct ids of a parquet, reading only its id column.

Normalisation (both sides): NFC, lower case, every character that is neither a word character nor an apostrophe
becomes a space, then split on spaces. Unicode word characters keep á é í ó ú ü ñ ő ű ä ö ß; ¿ and ¡ are dropped.
Stricter than leaderboard normalisers (no number or spelling normalisation), so absolute numbers run a few points
above published FLEURS figures: compare systems, not model cards. Thai has no spaces between words, so th_th is
scored by character, in spoken form on both sides (numbers become Thai words and the repeat mark is spelled out), on
letters, marks and digits only; that needs pythainlp.

Clip choice. FLEURS records each sentence id once per speaker. The ids below are fixed: hu_hu and de_de are 50 ids
chosen in an earlier bench, th_th is random.Random(42).sample(sorted(test ids), 50), and es_419 is the de_de list,
because all 50 de_de ids are in the es_419 test set (checked 2026-11-17), so both languages are scored on
translations of the same sentences. Of each id's recordings, prep takes the first in the parquet's rows after
random.Random(42).shuffle(rows), which reproduces the earlier bench's choice exactly.

FLEURS (google/fleurs) is CC BY 4.0: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech", 2022. Audio is fetched on demand and never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import unicodedata
import urllib.request
from pathlib import Path

DATASET_COMMIT = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"   # google/fleurs refs/convert/parquet, read 2026-11-17
PARQUET_URL = "https://huggingface.co/datasets/google/fleurs/resolve/{commit}/{lang}/test/0000.parquet"
PARQUETS = {  # lang -> (bytes, sha256) of test/0000.parquet at DATASET_COMMIT (the only test shard of each)
    "de_de": (726207372, "3b40c214cabd603f1ddbf3da0499f7780975d6c56b75240f07a8990dbb7a6c07"),
    "es_419": (703040007, "f1e8b7f6eecb91d16b79d91bc1838d8b0e5ebc0b37b317e2b58eb8694c64d3c0"),
    "hu_hu": (695869203, "6917a16373dcddd900aebe82e93a49ad3d4548b170d9c714f6f6914fcad34723"),
    "th_th": (784707801, "9a7223b0678bf9ce9bf0a716a0e703cceecb0bbb7fe6093c84a1a046b723a802"),
}
_DE = ("1662 1665 1670 1687 1688 1697 1704 1710 1712 1721 1722 1730 1742 1748 1767 1770 1772 1778 1786 1787 1811 "
       "1812 1816 1818 1821 1829 1835 1836 1853 1855 1857 1859 1863 1864 1868 1885 1901 1908 1930 1935 1939 1949 "
       "1958 1966 1979 1990 1991 2001 2002 2006")
IDS: dict[str, list[str]] = {
    "hu_hu": ("1660 1670 1676 1677 1680 1683 1684 1694 1713 1721 1726 1732 1735 1736 1737 1738 1742 1750 1754 1755 "
              "1758 1761 1777 1798 1801 1807 1808 1809 1818 1829 1835 1839 1850 1869 1894 1903 1904 1913 1920 1936 "
              "1948 1951 1960 1964 1977 1984 1986 1989 1999 2009").split(),
    "de_de": _DE.split(),
    "es_419": _DE.split(),   # the de_de ids: all 50 are in the es_419 test set, so no Random(42) top-up was needed
    "th_th": ("1663 1672 1673 1675 1676 1682 1700 1704 1707 1709 1712 1717 1723 1731 1739 1741 1761 1770 1771 1772 "
              "1774 1779 1785 1795 1800 1802 1810 1832 1834 1836 1843 1853 1854 1874 1876 1889 1895 1918 1934 1939 "
              "1942 1947 1961 1962 1968 1969 1981 1987 1992 2007").split(),
}
CER_LANGS = {"th_th"}
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
CHUNK = 1 << 20


# --------------------------------------------------------------------------- normalisation and scoring
def norm(text: str) -> list[str]:
    """Words for scoring: lower case, NFC, non-word characters except apostrophes to spaces."""
    return re.sub(r"[^\w' ]+", " ", unicodedata.normalize("NFC", text.lower())).split()


def fold(words: list[str]) -> list[str]:
    """Accent-folded words (diagnostic only): canonical decomposition with combining marks removed."""
    return ["".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn") for w in words]


def thai_spoken(text: str) -> str:
    """Numbers to Thai words and the repeat mark spelled out, so digit and spoken-form transcripts compare equal."""
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import num_to_thaiword
    t = re.sub(r"(?<=\d),(?=\d{3})", "", text.translate(THAI_DIGITS))
    t = re.sub(r"(\d+)(?:\.(\d+))?", lambda m: num_to_thaiword(int(m[1])) + (
        "จุด" + "".join(num_to_thaiword(int(d)) for d in m[2]) if m[2] else ""), t)
    out: list[str] = []
    for w in word_tokenize(t, engine="newmm", keep_whitespace=False):
        out.append(out[-1] if w == "ๆ" and out else w)
    return "".join(out)


def units(text: str, lang: str) -> list[str]:
    """Scoring units: words, or for a character-scored language its letters, marks and digits."""
    if lang not in CER_LANGS:
        return norm(text)
    t = thai_spoken(unicodedata.normalize("NFC", text.lower()))
    return [c for c in t if unicodedata.category(c)[0] in "LMN"]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- prep
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def fetch_parquet(lang: str, dest: Path) -> None:
    size, sha = PARQUETS[lang]
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(PARQUET_URL.format(commit=DATASET_COMMIT, lang=lang),
                                 headers={"User-Agent": "audiointake-bench"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        for b in iter(lambda: r.read(CHUNK), b""):
            f.write(b)
            h.update(b)
    if part.stat().st_size != size or h.hexdigest() != sha:
        part.unlink()
        raise SystemExit(f"fleurs: {lang} parquet doesn't match its pin (size or SHA-256); nothing kept")
    os.replace(part, dest)


def find_parquet(lang: str, dirs: list[Path]) -> Path | None:
    size, sha = PARQUETS[lang]
    for d in dirs:
        p = d / f"{lang}.parquet"
        if p.is_file() and p.stat().st_size == size and sha256_file(p) == sha:
            return p
    return None


def chosen_rows(parquet: Path, lang: str, columns: tuple[str, ...] = ("id", "audio", "raw_transcription")) -> dict:
    """{id: row} for the listed ids: the first recording of each id after Random(42).shuffle of all rows."""
    import pyarrow.parquet as pq
    rows = pq.read_table(str(parquet), columns=list(columns)).to_pylist()
    random.Random(42).shuffle(rows)
    want, out = set(IDS[lang]), {}
    for r in rows:
        i = str(r["id"])
        if i in want and i not in out:
            out[i] = r
    return out


def prep(workdir: Path, langs: list[str], parquet_dirs: list[Path], keep: bool) -> int:
    workdir.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        pqf = find_parquet(lang, parquet_dirs + [workdir])
        fetched = pqf is None
        if fetched:
            pqf = workdir / f"{lang}.parquet"
            fetch_parquet(lang, pqf)
        out = workdir / "fl" / lang
        out.mkdir(parents=True, exist_ok=True)
        rows = chosen_rows(pqf, lang)
        for i, r in rows.items():
            (out / f"{i}.wav").write_bytes(r["audio"]["bytes"])
            (out / f"{i}.ref").write_text(r["raw_transcription"], encoding="utf-8")
        if fetched and not keep:
            pqf.unlink()
        print(f"{lang}: {len(rows)}/{len(IDS[lang])} clips")
        if len(rows) != len(IDS[lang]):
            return 1
    return 0


# --------------------------------------------------------------------------- score
def score(workdir: Path, systems: list[str], json_out: Path | None) -> int:
    report: dict = {}
    for lang in IDS:
        fl = workdir / "fl" / lang
        if not fl.is_dir():
            continue
        ids = IDS[lang]
        raw = {i: (fl / f"{i}.ref").read_text(encoding="utf-8") for i in ids}
        refs = {i: units(raw[i], lang) for i in ids}
        nodigit = [i for i in ids if not re.search(r"\d", raw[i])]
        kind = "CER" if lang in CER_LANGS else "WER"
        line = [f"{lang} {kind} ref_units={sum(map(len, refs.values()))} no_digit_clips={len(nodigit)}"]
        for s in systems:
            d = workdir / "fo" / s / lang
            if not d.is_dir():
                continue
            hyps = {i: (d / f"{i}.txt").read_text(encoding="utf-8") for i in ids}
            clips = {i: [len(refs[i]), edit_distance(refs[i], units(hyps[i], lang))] for i in ids}
            rate = lambda sel: 100 * sum(clips[i][1] for i in sel) / max(1, sum(clips[i][0] for i in sel))
            entry = {"kind": kind, "rate": round(rate(ids), 4), "no_digit_rate": round(rate(nodigit), 4),
                     "clips": clips, "no_digit_ids": nodigit}
            if lang not in CER_LANGS:
                fc = {i: [len(refs[i]), edit_distance(fold(refs[i]), fold(norm(hyps[i])))] for i in ids}
                entry["folded_rate"] = round(100 * sum(v[1] for v in fc.values()) / max(1, sum(v[0] for v in fc.values())), 4)
            report.setdefault(lang, {})[s] = entry
            line.append(f"{s}={entry['rate']:.2f}% (no digit {entry['no_digit_rate']:.2f}%"
                        + (f", folded {entry['folded_rate']:.2f}%" if "folded_rate" in entry else "") + ")")
        print("  ".join(line))
    if json_out:
        json_out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fleurs.py", description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("workdir", type=Path)
    p.add_argument("langs", nargs="*", default=[])
    p.add_argument("--parquet-dir", type=Path, action="append", default=[])
    p.add_argument("--keep", action="store_true")
    s = sp.add_parser("score")
    s.add_argument("workdir", type=Path)
    s.add_argument("systems", nargs="+")
    s.add_argument("--json", type=Path)
    i = sp.add_parser("ids")
    i.add_argument("lang")
    i.add_argument("parquet", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        unknown = [x for x in a.langs if x not in IDS]
        if unknown:
            ap.error(f"unknown language {unknown[0]}; known: {', '.join(sorted(IDS))}")
        return prep(a.workdir, a.langs or sorted(IDS), a.parquet_dir, a.keep)
    if a.cmd == "score":
        return score(a.workdir, a.systems, a.json)
    import pyarrow.parquet as pq
    print(" ".join(sorted({str(x) for x in pq.read_table(str(a.parquet), columns=["id"]).column("id").to_pylist()},
                          key=int)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
