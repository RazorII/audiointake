#!/usr/bin/env python3
"""How many seconds of someone's speech a voiceprint needs: AMI calibration of audiointake's speaker naming.

  enroll_seconds.py --ami DIR --ref DIR --hyp DIR --model CAMPP.onnx [--steps DIR] [--seconds 3,5,8,15,30,60]
                    [--trials 3] [--json OUT]

Each participant of the four AMI test meetings ES2004, IS1009, TS3003 and EN2002 gets a voiceprint from N seconds of
their clean speech in the meeting's b-session (reference RTTM, other speakers' overlap removed, as enrolment cuts it),
for N in --seconds and --trials disjoint stretches of their speech. The a-sessions are then named exactly as a
transcription names them: each diarizer label of --hyp's RTTM is embedded from its longest clean pieces (up to the
match cap) and named when it has enough clean speech, scores at least the threshold against the best voiceprint, and
leads the runner-up by the margin (the module's own constants). All 16 participants are enrolled at once, so for
each label the other 15 are impostors.

Truth: each diarizer label maps to the reference speaker it shares the most time with (the best one-to-one
assignment). Per N it prints, over every label with enough clean speech and every trial: named right, named wrong,
left unnamed; and, with the true speaker's voiceprint removed, how often someone else is named (a stranger named).

DIR layout: --ami holds <meeting>.Array1-01.wav (16 kHz mono) for the a- and b-sessions; --ref holds the BUT
only_words reference RTTMs <meeting>.rttm (BUTSpeechFIT/AMI-diarization-setup, Apache-2.0); --hyp holds the
diarizer's <meeting>.rttm for the a-sessions. --steps names a folder holding speaker_id.py and rttm.py to measure
instead of audiointake.steps.speaker_id. Needs numpy, soundfile and sherpa-onnx.

AMI Meeting Corpus: CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

MEETINGS = ("ES2004", "IS1009", "TS3003", "EN2002")


def load_module(steps: Path | None):
    # The module reads its voiceprint folder from the environment at import; point it at an empty folder.
    os.environ["SPEAKER_ID_VOICEPRINTS"] = tempfile.mkdtemp(prefix="enroll-seconds-")
    if steps is None:
        return importlib.import_module("audiointake.steps.speaker_id")
    sys.path.insert(0, str(steps))
    spec = importlib.util.spec_from_file_location("speaker_id_under_test", steps / "speaker_id.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        x = line.split()
        if x and x[0] == "SPEAKER":
            s, d = float(x[-7]), float(x[-6])
            out.append((s, s + d, x[-3]))
    return out


def chunks(pieces: list[tuple[float, float]], need: float, start_at: float) -> list[tuple[float, float]]:
    """need seconds of the pieces (in time order), starting start_at seconds into their clean speech."""
    out, pos, got = [], 0.0, 0.0
    for a, b in sorted(pieces):
        if got >= need - 1e-9:
            break
        if pos + (b - a) <= start_at:
            pos += b - a
            continue
        a2 = a + max(0.0, start_at - pos)
        pos += b - a
        take = min(b - a2, need - got)
        if take >= 0.5:
            out.append((a2, a2 + take)); got += take
    return out


def best_map(hyp, ref) -> dict[str, str]:
    """Diarizer label -> reference speaker, the one-to-one assignment with the most shared time."""
    H = sorted({s for *_, s in hyp}); R = sorted({s for *_, s in ref})
    ov = {(h, r): 0.0 for h in H for r in R}
    for a, b, h in hyp:
        for c, d, r in ref:
            o = min(b, d) - max(a, c)
            if o > 0:
                ov[(h, r)] += o
    best, pick = -1.0, {}
    if len(H) <= len(R):
        for perm in itertools.permutations(R, len(H)):
            tot = sum(ov[(h, r)] for h, r in zip(H, perm))
            if tot > best:
                best, pick = tot, dict(zip(H, perm))
    else:
        for perm in itertools.permutations(H, len(R)):
            tot = sum(ov[(h, r)] for h, r in zip(perm, R))
            if tot > best:
                best, pick = tot, dict(zip(perm, R))
    return pick


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="enroll_seconds.py")
    ap.add_argument("--ami", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--steps", type=Path)
    ap.add_argument("--seconds", default="3,5,8,15,30,60")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    import numpy as np
    import sherpa_onnx
    import soundfile as sf
    sid = load_module(a.steps)
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(a.model), num_threads=4))
    unit = lambda v: (lambda x: x / (np.linalg.norm(x) + 1e-9))(np.asarray(v, dtype="float32"))

    def audio(m: str):
        x, sr = sf.read(str(a.ami / f"{m}.Array1-01.wav"), dtype="float32", always_2d=False)
        if sr != 16000 or x.ndim != 1:
            sys.exit(f"enroll_seconds: {m} isn't 16 kHz mono")
        return x

    def embed(x, span):
        st = ex.create_stream(); st.accept_waveform(16000, x[int(span[0] * 16000):int(span[1] * 16000)]); st.input_finished()
        return unit(ex.compute(st))

    def pooled(x, spans):
        if not spans:
            return None
        vs = [embed(x, s) for s in spans]
        return unit(np.average(np.stack(vs), axis=0, weights=[b - c for c, b in spans]))

    secs = [float(s) for s in a.seconds.split(",")]
    # enrolment material: clean pieces of each participant in the b-session
    material, have = {}, {}
    for m in MEETINGS:
        segs = read_rttm(a.ref / f"{m}b.rttm")
        x = audio(f"{m}b")
        for spk in sorted({s for *_, s in segs}):
            p = sid.clean_pieces(segs, spk)
            material[spk] = (x, p)
            have[spk] = sum(b - c for c, b in p)
    # the a-sessions' labels, embedded as a transcription embeds them
    labels = []   # (meeting, label, truth, speech seconds, vector)
    for m in MEETINGS:
        hyp, ref = read_rttm(a.hyp / f"{m}a.rttm"), read_rttm(a.ref / f"{m}a.rttm")
        mp, x = best_map(hyp, ref), audio(f"{m}a")
        for lab in sorted({s for *_, s in hyp}):
            p = sid.clean_pieces(hyp, lab)
            speech = sum(b - c for c, b in p)
            take, tot = [], 0.0
            for c, b in sorted(p, key=lambda q: -(q[1] - q[0])):
                if tot >= sid.MATCH_CAP:
                    break
                take.append((c, b)); tot += b - c
            labels.append((m, lab, mp.get(lab), speech, pooled(x, take)))
    judged = [l for l in labels if l[3] >= sid.MIN_SPEECH and l[4] is not None and l[2]]
    print(f"participants {len(material)}, clean enrolment speech per participant "
          f"{min(have.values()):.0f}-{max(have.values()):.0f} s; a-session labels {len(labels)}, "
          f"judged (>= {sid.MIN_SPEECH:.0f} s clean speech, mapped to a participant) {len(judged)}")

    def decide(vec, gallery):
        sc = sorted(((float(vec @ g), n) for n, g in gallery.items()), reverse=True)
        top, second = sc[0], sc[1] if len(sc) > 1 else (-1.0, None)
        return top[1] if top[0] >= sid.THRESHOLD and top[0] - second[0] >= sid.MARGIN else None

    table = []
    for n in secs:
        right = wrong = unnamed = stranger = trials_run = short = 0
        for k in range(a.trials):
            gallery = {}
            for spk, (x, p) in material.items():
                if have[spk] < n * a.trials:
                    short += 1     # not enough speech for disjoint stretches: the trials' stretches overlap
                spans = chunks(p, n, k * max(0.0, have[spk] - n) / max(1, a.trials - 1))
                v = pooled(x, spans)
                if v is not None:
                    gallery[spk] = v
            trials_run += 1
            for m, lab, truth, speech, vec in judged:
                got = decide(vec, gallery)
                right += got == truth; wrong += got is not None and got != truth; unnamed += got is None
                stranger += decide(vec, {s: g for s, g in gallery.items() if s != truth}) is not None
        tot = right + wrong + unnamed
        row = {"seconds": n, "trials": trials_run, "decisions": tot, "right": right, "wrong": wrong,
               "unnamed": unnamed, "stranger_named": stranger, "short_material": short}
        table.append(row)
        print(f"{n:>4.0f} s: named right {right}/{tot} ({100 * right / tot:.0f} %), wrong {wrong}, unnamed {unnamed}; "
              f"true voiceprint removed: someone else named {stranger}/{tot}")
    if a.json:
        a.json.write_text(json.dumps({"labels": len(labels), "judged": len(judged), "table": table,
                                      "threshold": sid.THRESHOLD, "margin": sid.MARGIN, "min_speech": sid.MIN_SPEECH},
                                     indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
