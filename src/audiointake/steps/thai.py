"""Thai for the ASR ensemble: find the Thai turns in a recording and transcribe them with a Thai model.

Parakeet has no Thai. On Thai speech it writes nothing, or phonetic pseudo-English. This worker finds the Thai turns
and transcribes them with a Thai Whisper fine-tune; ``ensemble.thai()`` then puts them in place of Parakeet's
output for those stretches. It runs as a child process::

    python -I -B -m audiointake.steps.thai run WAV ULTRA_JSON V3_JSON|"" OUT --lid-model DIR
        [--fluidaudio BIN --nemotron DIR] [--typhoon DIR --typhoon-tag NAME] [--screen-only] [--rttm R]

WAV is 16 kHz mono; ULTRA_JSON and V3_JSON are the two Parakeet outputs (parakeet-mlx JSON; V3_JSON may be "").
``--lid-model`` is Whisper large-v3 (MLX), used for language ID. ``--fluidaudio`` (an absolute path) and
``--nemotron`` (the diarizer and its model folder, both or neither) give speaker turns; without them the turns come
from pauses in the audio's energy. The diarizer runs with this worker's own environment cut to the run variables
(``RUN_ENV_KEYS``). ``--typhoon`` is the Thai model folder and ``--typhoon-tag`` the name the output gives it (both
or neither: the name never comes from the folder). ``--screen-only`` runs the screen alone: when it finds Thai it
writes ``"screen_only": true`` and exits 0, and nothing is transcribed. Exit 0 with ``{"thai": false, ...}`` when
there is no Thai. Stopped by SIGTERM, SIGHUP or SIGINT, it stops the diarizer it started (``ensemble.run_program``).

1. Screen. Whisper large-v3 language ID on every voiced 30-s window. No window at P(th) >= 0.5 -> ``{"thai":
   false}``, and nothing else runs.
2. Turns. Speaker turns (``--rttm``, else the diarizer is run here, else energy pauses), flattened (an overlap
   stays with the turn that started first), same-speaker runs < 0.3 s apart joined, cut to <= 15 s at the quietest
   20 ms frame. Turns, not the Thai model's own segments: forced to Thai, a fine-tune returns one segment per 30-s
   window and translates the English in it into Thai, so it can't say where the Thai is.
3. Language per turn (large-v3 again), then a veto from Parakeet, in two passes. Pass 1 scores a turn by P(th).
   Pass 2 re-scores the turns within 10 s of a pass-1 Thai turn by P(th) plus the other East and Southeast Asian
   languages (``THAI_LIKE``): on short or distant Thai, Whisper often says vi, zh, ko or ja instead. Only near
   confirmed Thai, because accented English drifts there too. A turn is a candidate at a score >= 0.5, or between
   0.15 and 0.5 when the score beats P(en) and the turn lasts >= 1 s. It is vetoed as English when Parakeet heard
   English there: Ultra has >= 4 words (>= 2 in the low band) at mean confidence >= 0.9 and v3 agrees on >= 60 % of
   them (no v3: confidence >= 0.95). Why both: accented English can reach P(th) 0.55-0.82, but there the two
   Parakeets hear the same words; on Thai each invents its own pseudo-English, or nothing. A known limit: a
   speaker of one of the ``THAI_LIKE`` languages next to Thai speech can be read as Thai.
4. A speaker's Thai turns less than 1 s apart are joined (up to 15 s), so a sentence the diarizer cut at a pause
   reaches the Thai model whole. Each Thai turn -> the Thai model, forced th, without timestamps (the fine-tunes
   emit none) and with a 440-token budget: the default 224 cuts Thai off mid-sentence, because Whisper's tokenizer
   spends about one token per Thai character. A turn the Thai model writes mostly in Latin letters was English
   after all and is dropped (Parakeet's text stays), as are empty turns and Whisper's stock outros; repetition
   loops are collapsed.

Writes ``{"thai": true, "model", "turns_from", "windows", "thai_windows", "turns": [{start, end, spk, p_thai, p_en,
text}], "stats"}``. The mlx and numpy imports are inside the functions that use them, so the module imports on a
machine without a Metal device.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from . import rttm  # the one RTTM reader
from .ensemble import run_program, words  # parakeet-mlx JSON -> words with confidence (stdlib only at import)

SCREEN, TH, MID, MID_SECS = 0.5, 0.5, 0.15, 1.0
THAI_LIKE = ("th", "lo", "km", "my", "vi", "zh", "yue", "ko", "ja", "ms", "id", "jw", "su", "tl")  # pass 2 only
NEAR, LATIN, JOIN_THAI = 10.0, 0.5, 1.0
MAX_TURN, JOIN_GAP = 15.0, 0.3
EN_WORDS, EN_WORDS_MID, EN_CONF, EN_AGREE, EN_CONF_ALONE = 4, 2, 0.9, 0.6, 0.95
N_SAMPLES = 480000          # mlx_whisper.audio.N_SAMPLES: 30 s at 16 kHz, one Whisper window
# Whisper's stock caption outros (learnt from subtitled video), in Thai and English; a turn that is only one of these is dropped.
# The variables a run child gets (DESIGN 7.1): the diarizer gets these from this worker's environment and no others,
# so a caller's tokens or switches never reach it. Under the pipeline the worker's environment is exactly this set.
RUN_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "USER", "LANG", "LC_CTYPE", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE",
                "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_CACHE", "ORT_DISABLE_TELEMETRY")
OUTRO = re.compile(r"^\W*(ขอบคุณ(ที่|สำหรับการ)รับชม|ซับไทย|ติดตาม(ช่อง)?|กดไลค์|กด\s*subscribe|subscribe|"
                   r"thanks? (you )?for watching)\W*$", re.I)


class Audio:
    def __init__(self, wav):
        import numpy as np
        from mlx_whisper.audio import load_audio
        self.a = load_audio(wav)
        x = np.array(self.a)
        self.dur = len(x) / 16000
        n = len(x) // 320  # 20 ms frames
        self.db = 20 * np.log10(np.sqrt((x[: n * 320].reshape(n, 320) ** 2).mean(1) + 1e-12))
        self.thr = np.percentile(self.db, 10) + 10

    def clip(self, s, e):
        return self.a[int(s * 16000):int(e * 16000)]

    def voiced(self, s, e):
        seg = self.db[int(s / .02):max(int(e / .02), int(s / .02) + 1)]
        return float((seg > self.thr).mean()) if len(seg) else 0.0


class LID:
    def __init__(self, model_dir):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
        from mlx_whisper.tokenizer import get_tokenizer
        self.m = load_model(os.fspath(model_dir), dtype=mx.float16)
        self.tok = get_tokenizer(self.m.is_multilingual, num_languages=self.m.num_languages)

    def __call__(self, au, spans):
        import mlx.core as mx
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        out = []
        for i in range(0, len(spans), 8):
            mels = mx.stack([pad_or_trim(log_mel_spectrogram(au.clip(s, e), n_mels=128), 3000, axis=-2)
                             for s, e in spans[i:i + 8]]).astype(mx.float16)
            out += detect_language(self.m, mels, self.tok)[1]
        return out


def screen(au, lid):
    import numpy as np
    wins = [(s, min(s + N_SAMPLES / 16000, au.dur)) for s in np.arange(0, au.dur, N_SAMPLES / 16000)]
    wins = [(float(s), float(e)) for s, e in wins if e - s > 1 and au.voiced(s, e) >= 0.15]
    probs = lid(au, wins) if wins else []
    return [round(p.get("th", 0.0), 3) for p in probs]


def turns_from_segments(segs_in):
    segs = sorted(((x["start"], x["start"] + x["dur"], x["spk"]) for x in segs_in), key=lambda x: x[0])
    flat = []
    for s, e, spk in segs:
        if flat and s < flat[-1][1]:
            if e <= flat[-1][1]:
                continue                      # inside the turn already running: crosstalk
            s = flat[-1][1]                   # overlap stays with the earlier turn
        if e - s > 0.05:
            flat.append([s, e, spk])
    return flat


def turns_from_rttm(path):
    return turns_from_segments(rttm.segments_from_path(path))


def turns_from_energy(au):
    """Without a diarizer: voiced runs, split at pauses >= 0.3 s."""
    v = au.db > au.thr
    out, start, quiet = [], None, 0
    for i, on in enumerate(v):
        if on:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= 15:
                out.append([start * .02, (i - quiet + 1) * .02, "vad"])
                start, quiet = None, 0
    if start is not None:
        out.append([start * .02, len(v) * .02, "vad"])
    return [t for t in out if t[1] - t[0] >= 0.3]


def diarize(au, tmp, fluidaudio, nemotron):
    """Run the diarizer on the worker's wav; the RTTM path, or None when it failed or isn't available.

    The allowlist is the one program this worker was given (its parent checked it); the environment is this
    worker's own, cut to ``RUN_ENV_KEYS``."""
    if not (fluidaudio and nemotron and os.path.isabs(fluidaudio) and os.access(fluidaudio, os.X_OK)):
        return None
    from .. import argv as A
    from .. import proc
    fa, models = os.path.normpath(fluidaudio), os.path.abspath(nemotron)
    r = os.path.join(os.path.abspath(tmp), "diar.rttm")
    env = {k: os.environ[k] for k in RUN_ENV_KEYS if k in os.environ}
    res = proc.run(A.diarize(fa, au.wav, r, models=models), env=env,
                   timeout=proc.child_timeout("fluidaudio", au.dur), allowed={Path(fa)}, stdout="discard")
    return r if res.returncode == 0 and not res.timed_out and os.path.exists(r) else None


def turns(au, rttm_path, tmp, fluidaudio=None, nemotron=None):
    import numpy as np
    if not (rttm_path and os.path.exists(rttm_path) and os.path.getsize(rttm_path)):
        rttm_path = diarize(au, tmp, fluidaudio, nemotron)
    t = turns_from_rttm(rttm_path) if rttm_path else turns_from_energy(au)
    src = "nemotron" if rttm_path else "energy"
    joined = []
    for s, e, spk in t:
        if joined and joined[-1][2] == spk and s - joined[-1][1] < JOIN_GAP:
            joined[-1][1] = e
        else:
            joined.append([s, e, spk])
    out = []
    for s, e, spk in joined:
        while e - s > MAX_TURN:
            lo, hi = int((s + 5) / .02), int((s + MAX_TURN) / .02)
            cut = (lo + int(np.argmin(au.db[lo:hi]))) * .02
            out.append([s, cut, spk]); s = cut
        out.append([s, e, spk])
    return out, src


def in_turn(ws, s, e):
    return [w for w in ws if s <= (w["start"] + w["end"]) / 2 < e]


def english(u, v, nmin):
    """Parakeet heard English here: enough confident words, and v3 hears the same ones."""
    if len(u) < nmin or sum(w["conf"] for w in u) / len(u) < EN_CONF:
        return False
    if v is None:
        return sum(w["conf"] for w in u) / len(u) >= EN_CONF_ALONE
    a, b = [w["norm"] for w in u], [w["norm"] for w in v]
    m = sum(x.size for x in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return 2 * m / (len(a) + len(b)) >= EN_AGREE


def is_thai(p_th, p_en, secs, u, v):
    if p_th >= TH:
        return not english(u, v, EN_WORDS)
    if p_th < MID or p_th <= p_en or secs < MID_SECS:
        return False
    return not english(u, v, EN_WORDS_MID)


LOOP_LONG = re.compile(r"(?<!\S)(\S.{4,79}?)(?:\s*\1){2,}")
LOOP_SHORT = re.compile(r"(?<!\S)(\S{1,4})(?:\s+\1(?!\S)){3,}")


def collapse(text):
    """Repetition loops -> one copy + [×N]: a short word 4+ times in a row (three times stays), or a phrase of 5+
    characters 3+ times. A cut-off start of one more copy at the very end is dropped. Character-based on purpose:
    regex \\w doesn't match Thai vowel and tone marks."""
    def sub(rx, t):
        out, i = [], 0
        for m in rx.finditer(t):
            g = m.group(1).strip()
            out.append(t[i:m.start()] + f"{g} [×{len(re.findall(re.escape(g), m.group(0)))}]")
            i = m.end()
            tail = re.match(r"\s*(\S+)", t[i:])
            if tail and g.startswith(tail.group(1)) and not t[i + tail.end():].strip():
                i += tail.end()
        return "".join(out) + t[i:]
    return sub(LOOP_LONG, sub(LOOP_SHORT, text)).strip()


class NoThaiModel(Exception):
    """Thai was found but no Thai model was given (and the run wasn't --screen-only)."""


def run(wav, ultra, v3, out, rttm_path=None, *, lid_model, fluidaudio=None, nemotron=None, typhoon=None,
        typhoon_tag=None, screen_only=False, log=None):
    if bool(typhoon) != bool(typhoon_tag):
        raise ValueError("typhoon and typhoon_tag go together: the Thai model's name never comes from its folder")
    log = log or (lambda line: print(line, file=sys.stderr))
    t0 = time.time()
    au = Audio(wav); au.wav = os.path.abspath(wav)
    lid = LID(lid_model)
    p = screen(au, lid)
    if not p or max(p) < SCREEN:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dict(thai=False, windows=len(p), max_p_th=max(p or [0])), f)
        log(f"thai: none ({len(p)} windows, max P(th) {max(p or [0]):.2f}, {time.time() - t0:.0f}s)")
        return
    n_thai = sum(x >= SCREEN for x in p)
    if screen_only:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dict(thai=True, screen_only=True, windows=len(p), thai_windows=n_thai, max_p_th=max(p)), f)
        log(f"thai: found ({n_thai} of {len(p)} windows, max P(th) {max(p):.2f}), screen only, "
            f"{time.time() - t0:.0f}s")
        return
    if not (typhoon and os.path.exists(os.path.join(typhoon, "weights.safetensors"))):
        raise NoThaiModel(f"thai: Thai found ({n_thai} windows) but no Thai model was given (--typhoon)")
    with tempfile.TemporaryDirectory(prefix="thai.") as tmp:
        ts, src = turns(au, rttm_path, tmp, fluidaudio, nemotron)
    probs = lid(au, [(s, e) for s, e, _ in ts])
    with open(ultra, encoding="utf-8") as f:
        U = words(json.load(f))
    V = None
    if v3 and os.path.exists(v3):
        with open(v3, encoding="utf-8") as f:
            V = words(json.load(f))
    del lid
    ev = lambda s, e: (in_turn(U, s, e), in_turn(V, s, e) if V is not None else None)
    anchors = [i for i, ((s, e, _), pr) in enumerate(zip(ts, probs))
               if is_thai(pr.get("th", 0.0), pr.get("en", 0.0), e - s, *ev(s, e))]
    anchor_set = set(anchors)
    picked, near = [], 0
    for i, ((s, e, spk), pr) in enumerate(zip(ts, probs)):
        pth, pen = pr.get("th", 0.0), pr.get("en", 0.0)
        ok = i in anchor_set
        if not ok and any(ts[j][0] - NEAR <= e and s <= ts[j][1] + NEAR for j in anchors):
            pth = sum(pr.get(k, 0.0) for k in THAI_LIKE)
            ok = is_thai(pth, pen, e - s, *ev(s, e)); near += ok
        if ok:
            picked.append(dict(start=round(s, 3), end=round(e, 3), spk=spk, p_thai=round(pth, 3), p_en=round(pen, 3)))
    joined = []
    for t in picked:
        j = joined[-1] if joined else None
        if j and j["spk"] == t["spk"] and t["start"] - j["end"] < JOIN_THAI and t["end"] - j["start"] <= MAX_TURN:
            j.update(end=t["end"], p_thai=max(j["p_thai"], t["p_thai"]), p_en=min(j["p_en"], t["p_en"]))
        else:
            joined.append(dict(t))
    kept, st = [], dict(turns=len(ts), thai=len(picked), joined=len(picked) - len(joined), near=near,
                        latin=0, empty=0, outro=0, looped=0)
    picked = joined
    import mlx_whisper
    for t in picked:
        r = mlx_whisper.transcribe(au.clip(t["start"], t["end"]), path_or_hf_repo=os.fspath(typhoon), language="th",
                                   temperature=0.0, without_timestamps=True, sample_len=440,
                                   condition_on_previous_text=False, verbose=None)
        text = re.sub(r"\s+", " ", r["text"]).strip()
        if not text:
            st["empty"] += 1; continue
        if OUTRO.match(text):
            st["outro"] += 1; continue
        letters = [ch for ch in text if ch.isalpha()]
        if letters and sum(ch.isascii() for ch in letters) / len(letters) >= LATIN:
            st["latin"] += 1; continue
        c = collapse(text)
        st["looped"] += c != text
        kept.append(dict(t, text=c))
    secs = sum(t["end"] - t["start"] for t in kept)
    model = typhoon_tag
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(thai=True, model=model, turns_from=src, windows=len(p), thai_windows=n_thai,
                       turns=kept, stats=st), f, ensure_ascii=False)
    log(f"thai: {len(kept)} Thai turns ({secs / 60:.1f} min) of {len(ts)} {src} turns; "
        + " ".join(f"{k}={v}" for k, v in st.items()) + f"; {model}, {time.time() - t0:.0f}s")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -I -B -m audiointake.steps.thai",
                                 description="Find and transcribe the Thai turns in a recording.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("wav"); r.add_argument("ultra"); r.add_argument("v3"); r.add_argument("out")
    r.add_argument("--lid-model", required=True)
    r.add_argument("--fluidaudio"); r.add_argument("--nemotron")
    r.add_argument("--typhoon"); r.add_argument("--typhoon-tag")
    r.add_argument("--screen-only", action="store_true")
    r.add_argument("--rttm")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    o = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if bool(o.fluidaudio) != bool(o.nemotron):
        ap.error("--fluidaudio and --nemotron go together")
    if bool(o.typhoon) != bool(o.typhoon_tag):
        ap.error("--typhoon and --typhoon-tag go together")
    if o.fluidaudio and not os.path.isabs(o.fluidaudio):
        ap.error("--fluidaudio must be an absolute path")

    def body() -> int:
        try:
            run(o.wav, o.ultra, o.v3, o.out, o.rttm, lid_model=o.lid_model, fluidaudio=o.fluidaudio,
                nemotron=o.nemotron, typhoon=o.typhoon, typhoon_tag=o.typhoon_tag, screen_only=o.screen_only)
        except NoThaiModel as e:
            print(str(e), file=sys.stderr)
            return 1
        return 0
    return run_program(body)


if __name__ == "__main__":
    sys.exit(main())
