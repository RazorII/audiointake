#!/usr/bin/env python3
"""The AMI room-audio bench: four far-field meetings with reference speakers and words.

  ami.py fetch DIR [--b-sessions] [--annotations]
      The a-sessions of ES2004, IS1009, TS3003 and EN2002 (single distant microphone Array1-01, 16 kHz mono; 92 min
      in all) and their BUT only_words reference RTTMs, each pinned by size and SHA-256. --b-sessions adds the
      b-sessions and their RTTMs (voiceprint enrolment); --annotations the AMI manual word transcripts (ami_wer.py).
      Files already present with the pinned hash are kept.
  ami.py enroll DIR VOICES --model CAMPP.onnx
      Public AMI voiceprints: every participant enrolled from their b-session's clean speech, with the reference RTTM
      as the diarization, so each voiceprint is named by its AMI speaker id (FEE013, ...). Uses the package's
      speaker_id.enroll_label; a source the enrolment check holds back is enrolled anyway and reported (far-field
      voices recorded through one room microphone sound alike to the speaker model).
  ami.py run DIR OUT [--voices VOICES] [--audiointake CMD...]
      Transcribe each a-session with `audiointake transcribe --format sidecars,json`, then score the word rule's
      speaker attribution (ami_attrib.py), speaker by time (timeline_attrib.py), and, with --voices, how many
      diarizer labels were named and named right. With --annotations fetched, also WER (ami_wer.py).

AMI Meeting Corpus: CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/ (Carletta et al., 2005). Reference RTTMs:
BUTSpeechFIT/AMI-diarization-setup, Apache-2.0, at a pinned commit. Audio is fetched on demand, never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True   # no __pycache__ beside the bench scripts, wherever the clone lives
sys.path.insert(0, str(Path(__file__).resolve().parent))

AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Array1-01.wav"
BUT_COMMIT = "2509d8933721023fab4def2618aabd5c28eb82e9"
RTTM_URL = ("https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/" + BUT_COMMIT
            + "/only_words/rttms/test/{m}.rttm")
ANNOTATIONS = ("https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip", 22887865,
               "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d")
A_SESSIONS = ("ES2004a", "IS1009a", "TS3003a", "EN2002a")
B_SESSIONS = ("ES2004b", "IS1009b", "TS3003b", "EN2002b")
WAVS = {  # meeting -> (bytes, sha256) of <m>.Array1-01.wav on the AMI mirror (checked 2026-11-17)
    "ES2004a": (33579394, "6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150"),
    "IS1009a": (26849383, "23fd54fd6dbe23870f10317b256f7c648fe0fd587e56d2f940840309f1578b60"),
    "TS3003a": (48183678, "46f81fb403e40a98c1b694c842404c0db24f3a6c67397b247faf0d3f42cf9163"),
    "EN2002a": (68566744, "6c0644431679e59adc70129f2ba6c2fb737fa731cf1ffe8a033cd7f0c441170e"),
    "ES2004b": (75055832, "8a58a53413f7a47a29f8bd4d1b49e42cd4874eecc0c3f023c28a4702ecb50d88"),
    "IS1009b": (65679463, "d4f4556adb90eac335d3a50e5cfcd36b3186e8de0ba1b3f0b07a3d21088e7ba3"),
    "TS3003b": (70730794, "dea4276863ba77b49d304e28638ae354bcd815e4ebeecf98902ce30782f8ba5a"),
    "EN2002b": (57179180, "e3650cc84250caa0992ac0398f4a3127a8c14ed2c93edc10535245335cc41077"),
}
RTTMS = {  # meeting -> sha256 of the reference RTTM at BUT_COMMIT
    "ES2004a": "9869c6146c2fd9595403edb36c2caeda65c12ffa2c0af4ce48d6814b673fd5a9",
    "IS1009a": "ba38d35ca567f3f1e061d90fdc33579ce60d2060fc8664eaeb77d2e0fcd88b01",
    "TS3003a": "57137f098ee3be9b8d77c9faa4b16e9155e821bdbf21b974dd8771d7e646d185",
    "EN2002a": "a93919ae5f18e476614b7e5a2713c4dc6bba23e95149fba18dbe68602b13ee40",
    "ES2004b": "97a04fa14a37249b09581969939e5024324bbe3fc9c7a77e19976af41147ad7d",
    "IS1009b": "5cb6ae3948c09ce8356991cc79bb593abc1f0d23f1dbadeeb88e17e055283e67",
    "TS3003b": "c728c1b1305a977a5adc32830c8299f1ee6084dee113a7e56f6d9cce540e5b3a",
    "EN2002b": "c4e6ef3f9f493d82d14664a9669479aca378de05639e4640f02f8398e2ef714a",
}
UA = {"User-Agent": "audiointake-bench"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(url: str, dest: Path, size: int | None, sha: str) -> str:
    if dest.is_file() and (size is None or dest.stat().st_size == size) and sha256_file(dest) == sha:
        return "kept"
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if (size is not None and part.stat().st_size != size) or sha256_file(part) != sha:
        part.unlink()
        raise SystemExit(f"ami: {dest.name} doesn't match its pin; nothing kept")
    os.replace(part, dest)
    return "fetched"


def cmd_fetch(d: Path, b: bool, ann: bool) -> int:
    (d / "ref").mkdir(parents=True, exist_ok=True)
    for m in A_SESSIONS + (B_SESSIONS if b else ()):
        w = get(AUDIO_URL.format(m=m), d / f"{m}.Array1-01.wav", *WAVS[m])
        r = get(RTTM_URL.format(m=m), d / "ref" / f"{m}.rttm", None, RTTMS[m])
        print(f"{m}: audio {w}, reference {r}")
    if ann:
        print(f"annotations: {get(ANNOTATIONS[0], d / 'ami_public_manual_1.6.2.zip', *ANNOTATIONS[1:])}")
    return 0


def read_rttm(p: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        x = line.split()
        if x and x[0] == "SPEAKER":
            out.append((float(x[-7]), float(x[-7]) + float(x[-6]), x[-3]))
    return out


def cmd_enroll(d: Path, voices: Path, model: Path) -> int:
    from audiointake.steps import speaker_id
    voices.mkdir(parents=True, exist_ok=True)
    held = 0
    for m in B_SESSIONS:
        rttm_text = (d / "ref" / f"{m}.rttm").read_text(encoding="utf-8")
        for spk in sorted({s for *_, s in read_rttm(d / "ref" / f"{m}.rttm")}):
            src = f"AMI {m} Array1-01 {spk} (BUT only_words reference)"
            args = dict(name=spk, wav=d / f"{m}.Array1-01.wav", rttm_text=rttm_text, label=spk, source=src, lang="EN",
                        voices_dir=voices, model=model)
            try:
                speaker_id.enroll_label(**args, force=False)
                print(f"enrolled {spk} from {m}")
            except speaker_id.HeldBack as e:
                held += 1
                speaker_id.enroll_label(**args, force=True)
                print(f"enrolled {spk} from {m} although the enrolment check held it back: {[p.code for p in e.problems]}")
    print(f"{held} of 16 sources held back by the enrolment check")
    return 0


def cmd_run(d: Path, out: Path, voices: Path | None, cli: list[str]) -> int:
    import ami_attrib
    import timeline_attrib
    out.mkdir(parents=True, exist_ok=True)
    pooled = {"word": [0, 0], "sentence": [0, 0]}
    named = right = labels = 0
    for m in A_SESSIONS:
        wav = d / f"{m}.Array1-01.wav"
        stem = f"{m}.Array1-01"
        vflag = ["--voices", str(voices)] if voices else ["--no-names"]
        r = subprocess.run([*cli, "transcribe", str(wav), "--format", "sidecars,json", "--outdir", str(out),
                            *vflag, "--overwrite", "--json"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            print(f"{m}: audiointake exited {r.returncode}"); return 1
        res = ami_attrib.score(str(out / f"{stem}.asr.json"), str(out / f"{stem}.diar.rttm"), str(d / "ref" / f"{m}.rttm"))
        for k in pooled:
            pooled[k][0] += res[k][0]; pooled[k][1] += res[k][1]
        tj = json.loads((out / f"{stem} (transcript).json").read_text(encoding="utf-8"))
        items = [[t["start"], t["end"], t["label"] or "none", t["text"]] for t in tj.get("turns", [])]
        (out / f"{m}.items.json").write_text(json.dumps(items, ensure_ascii=False))
        timeline_attrib.score(str(d / "ref" / f"{m}.rttm"), str(out / f"{m}.items.json"), m)
        if voices:
            ref = read_rttm(d / "ref" / f"{m}.rttm")
            hyp = read_rttm(out / f"{stem}.diar.rttm")
            mp = ami_attrib_map(hyp, ref)
            for lab, v in tj.get("speakers", {}).items():
                named += 1; right += v.get("name") == mp.get(lab)
            labels += len({s for *_, s in hyp})
    for k, (ok, n) in pooled.items():
        print(f"POOLED {k} rule: {100 * ok / max(1, n):.1f}% of {n} judged words")
    if voices:
        print(f"NAMES: {named} of {labels} diarizer labels named, {right} named right")
    ann = d / "ami_public_manual_1.6.2.zip"
    if ann.is_file():
        import ami_wer
        ami_wer.main([str(ann), str(out)])
    return 0


def ami_attrib_map(hyp, ref) -> dict[str, str]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    H = sorted({s for *_, s in hyp}); R = sorted({s for *_, s in ref})
    ov = np.zeros((len(H), len(R)))
    for a, b, h in hyp:
        for c, e, k in ref:
            o = min(b, e) - max(a, c)
            if o > 0:
                ov[H.index(h), R.index(k)] += o
    ri, ci = linear_sum_assignment(-ov)
    return {H[i]: R[j] for i, j in zip(ri, ci)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ami.py", description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("dir", type=Path)
    f.add_argument("--b-sessions", action="store_true"); f.add_argument("--annotations", action="store_true")
    e = sp.add_parser("enroll"); e.add_argument("dir", type=Path); e.add_argument("voices", type=Path)
    e.add_argument("--model", type=Path, required=True)
    r = sp.add_parser("run"); r.add_argument("dir", type=Path); r.add_argument("out", type=Path)
    r.add_argument("--voices", type=Path)
    r.add_argument("--audiointake", nargs="+", default=[sys.executable, "-m", "audiointake"])
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        return cmd_fetch(a.dir, a.b_sessions, a.annotations)
    if a.cmd == "enroll":
        return cmd_enroll(a.dir, a.voices, a.model)
    return cmd_run(a.dir, a.out, a.voices, a.audiointake)


if __name__ == "__main__":
    sys.exit(main())
