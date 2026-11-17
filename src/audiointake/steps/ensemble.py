"""Combine several speech recognisers into one transcript: the ASR ensemble.

Parakeet Ultra is the primary model and Parakeet TDT 0.6B v3 fills its gaps. Each model's extra words are almost
always real (they fail by dropping words, in different places), so Ultra's text is kept and v3 only fills the gaps
where Ultra has nothing; choosing between swapped words by confidence made word errors worse. A filled run becomes
its own sentence, so the diarizer labels it by who spoke at that moment (a backchannel v3 caught is usually the
other person's).

The vote route, for a language Parakeet reads less well (one table entry, ``VOTE_LANGS``): Whisper large-v3 is more
accurate on clean speech in that language, and letting the two Parakeets overrule it where they agree helps further.
On room audio, though, Whisper loops, invents caption outros and drops whole ~28 s windows. So the vote keeps
Whisper's segments, votes inside them, and lets Ultra replace or fill what Whisper lost or invented. The route runs
when ``lang()`` finds the language's function words (at least ``min_words`` words and a share of ``min_share``), or
when the caller forces it (``force_lang``), and only with the route's pack installed.

Thai: Parakeet has none, and writes nothing or pseudo-English on Thai speech. The Thai worker (``steps.thai``, a
child process) screens every recording with Whisper's language ID; when it finds Thai it transcribes the Thai
speaker turns with a Thai Whisper fine-tune, and ``thai()`` here drops Parakeet's words inside those turns and puts
the Thai in their place.

Command line (each mode prints one stats line on stderr; no transcript text)::

    python -m audiointake.steps.ensemble lang    <primary.json>                                  -> "hu" or "other"
    python -m audiointake.steps.ensemble gapfill <primary.json> <fill.json> <out.json> [--tau 0.5]
    python -m audiointake.steps.ensemble hu      <whisper.json> <ultra.json> <v3.json> <16k.wav> <out.json>
    python -m audiointake.steps.ensemble thai    <base.json> <thai.json> <out.json>
    python -m audiointake.steps.ensemble transcribe <audio> --output-dir D --output-template T [--output-format txt|json|all]

``transcribe`` runs the models through the set-up runtime (``setup.runtime.resolve``): it writes D/T.json
(parakeet-mlx's shape) and/or D/T.txt, prints the ASR tag on stdout and stats on stderr, and exits 3 when the
primary model fails or a merge step fails (nothing is written then); a setup problem exits with its own code (3
when audiointake isn't set up). A failed fill model, vote or Thai screen falls back to the text it has. Two
environment switches are honoured here and nowhere else: ``THAI=off`` skips the Thai screen, and
``PARAKEET_FILL_MODEL=""`` (set and empty) runs without the fill model. Children never see either.

Run as a program (this module, the Thai worker, the speaker worker), a steps module stops its children when it is
stopped: every child leads its own process group (``proc.run``), so a signal to the program's group doesn't reach
them by itself. ``run_program`` turns SIGTERM, SIGHUP and SIGINT into ``Killed``, which unwinds through every
``finally`` (the work folder and temporary files go), stops every child still running and exits 128 + the signal.

The output keeps parakeet-mlx's shape (``{"text", "sentences": [{start, end, text, ...}]}``), which is all
merge_speakers and slim_asr read.
"""
from __future__ import annotations

import copy
import dataclasses
import difflib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import unicodedata
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .. import argv as A
from .. import proc
from ..contracts import EnsembleResult, Log, Runtime, ThaiSummary
from ..errors import AsrFailed

HU_WORDS = set("hogy nem és akkor van egy ez csak már még mert volt igen nincs jó itt ott kell hát "
               "azt ezt tehát vagy nagyon lesz mit miért aztán szóval persze köszönöm szia".split())
OUTRO = re.compile(r"megnézt|feliratoz|iratkozz|köszönöm a figyelm", re.I)

VOTE_TAG = "whisper-large-v3+ultra/v3-vote"     # the ASR tag when the vote route produced the transcript
NO_DURATION_TIMEOUT = 6 * 3600.0                # a child's cap when the recording's length isn't known


@dataclass(frozen=True)
class VoteLang:
    """One language the vote route serves: how ``lang()`` recognises it and how Whisper is told to read it."""
    code: str                    # the language code (--lang, the Whisper --language value)
    words: frozenset[str]        # function words, normalised as norm() normalises
    min_words: int               # the recogniser must have heard at least this many words
    min_share: float             # and at least this share of them must be function words
    outro: re.Pattern[str]       # Whisper's caption outros in this language: a segment holding one is dropped
    route: str                   # the route (contracts.ROUTES) that must be installed for the vote to run


VOTE_LANGS: tuple[VoteLang, ...] = (
    VoteLang(code="hu", words=frozenset(HU_WORDS), min_words=30, min_share=0.08, outro=OUTRO, route="hu_vote"),
)
VOTE_LANG = {e.code: e for e in VOTE_LANGS}


def _stderr(line: str) -> None:
    print(line, file=sys.stderr)


def norm(w):
    return re.sub(r"[^\w']+", "", unicodedata.normalize("NFC", w.lower())).strip("'")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def words(d):
    """parakeet-mlx JSON -> words: dict(norm, surface, start, end, conf, sent, toks=[token idx])."""
    out = []
    for si, s in enumerate(d["sentences"]):
        for ti, t in enumerate(s.get("tokens") or []):
            if not out or out[-1]["sent"] != si or t["text"].startswith(" "):
                out.append(dict(surface=t["text"].strip(), start=t["start"], end=t["end"],
                                conf=t.get("confidence", 1.0), sent=si, toks=[ti]))
            else:
                w = out[-1]
                w["surface"] += t["text"]; w["end"] = t["end"]
                w["conf"] = min(w["conf"], t.get("confidence", 1.0)); w["toks"].append(ti)
    for w in out:
        w["norm"] = norm(w["surface"])
    return [w for w in out if w["norm"]]


def finish(sents, keep_order=False):
    sents = [s for s in sents if s["text"].strip()]
    if not keep_order:
        sents.sort(key=lambda s: s["start"])
    for s in sents:
        s["duration"] = round(s["end"] - s["start"], 3)
    return {"text": "".join(s["text"] for s in sents).strip(), "sentences": sents}


# --- lang --------------------------------------------------------------------------------

def shares(primary, table=VOTE_LANGS):
    """(words heard, {code: function-word share}) for the recogniser's JSON at ``primary``."""
    ws = [w["norm"] for w in words(load(primary))]
    return len(ws), {e.code: sum(w in e.words for w in ws) / max(1, len(ws)) for e in table}


def lang(primary, table=VOTE_LANGS):
    """The first table language whose function words the primary transcript shows, or "other"; and its share
    (for "other", the first entry's share)."""
    n, sh = shares(primary, table)
    for e in table:
        if n >= e.min_words and sh[e.code] >= e.min_share:
            return e.code, sh[e.code]
    return "other", (sh[table[0].code] if table else 0.0)


def piece(toks, **extra):
    """A sentence made of some of another sentence's tokens."""
    toks = [dict(t) for t in toks]
    if not toks[0]["text"].startswith(" "):
        toks[0]["text"] = " " + toks[0]["text"]
    return dict(start=toks[0]["start"], end=toks[-1]["end"], text="".join(t["text"] for t in toks),
                tokens=toks, **extra)


# --- gapfill -----------------------------------------------------------------------------

def gapfill(primary, fill, out, tau=0.5, *, log: Log = _stderr):
    A_, B = load(primary), load(fill)
    wa, wb = words(A_), words(B)
    sents = copy.deepcopy(A_["sentences"])
    ins = {}       # (sent, token index to insert BEFORE) -> [tokens]
    new = []       # standalone sentences
    n_words = n_places = 0
    sm = difflib.SequenceMatcher(None, [w["norm"] for w in wa], [w["norm"] for w in wb], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "insert":
            continue
        run = wb[j1:j2]
        if sum(w["conf"] for w in run) / len(run) < tau:
            continue
        toks = [copy.deepcopy(B["sentences"][w["sent"]]["tokens"][ti]) for w in run for ti in w["toks"]]
        for t in toks:
            t["fill"] = True
        if not toks[0]["text"].startswith(" "):
            toks[0]["text"] = " " + toks[0]["text"]
        rs, re_ = run[0]["start"], run[-1]["end"]
        prev = wa[i1 - 1] if i1 > 0 else None
        nxt = wa[i1] if i1 < len(wa) else None
        if prev and (nxt and prev["sent"] == nxt["sent"] or rs <= sents[prev["sent"]]["end"] + 0.3):
            key = (prev["sent"], prev["toks"][-1] + 1)          # right after prev, same sentence
        elif nxt and re_ >= sents[nxt["sent"]]["start"] - 0.3:
            key = (nxt["sent"], nxt["toks"][0])                 # right before nxt, same sentence
        else:
            key = None
        if key:
            ins.setdefault(key, []).extend(toks)
        else:
            new.append(dict(start=rs, end=re_, text="".join(t["text"] for t in toks), tokens=toks, fill=True))
        n_words += len(run); n_places += 1
    # Each filled run becomes its own segment, splitting the sentence it lands in, so the diarizer labels it by
    # who spoke at that moment: a backchannel v3 caught is usually the other person. merge_speakers rejoins
    # same-speaker neighbours.
    final, new = [], sorted(new, key=lambda x: x["start"])
    for si, s in enumerate(sents):
        while new and new[0]["start"] < s["start"]:
            final.append(new.pop(0))
        toks = s.get("tokens") or []
        cuts = sorted(ti for (sj, ti) in ins if sj == si)
        if not cuts:
            final.append(s); continue
        prev = 0
        for ti in cuts:
            if toks[prev:ti]:
                final.append(piece(toks[prev:ti]))
            final.append(piece(ins[(si, ti)], fill=True))
            prev = ti
        if toks[prev:]:
            final.append(piece(toks[prev:]))
    final += new
    with open(out, "w", encoding="utf-8") as f:
        json.dump(finish(final, keep_order=True), f, ensure_ascii=False)
    log(f"gapfill: +{n_words} words in {n_places} places ({len(new)} new sentences); primary {len(wa)} words")


# --- the vote ----------------------------------------------------------------------------

class Energy:
    def __init__(self, wav_path):
        import numpy as np
        with wave.open(os.fspath(wav_path)) as f:
            x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        n = len(x) // 320  # 20 ms frames at 16 kHz
        rms = np.sqrt((x[: n * 320].reshape(n, 320) ** 2).mean(1) + 1e-12)
        self.db = 20 * np.log10(rms)
        self.thr = np.percentile(self.db, 10) + 10

    def voiced(self, t0, t1):
        if t1 - t0 < 0.5:
            mid = (t0 + t1) / 2; t0, t1 = max(0, mid - 0.25), mid + 0.25
        seg = self.db[int(t0 / 0.02): max(int(t1 / 0.02), int(t0 / 0.02) + 1)]
        return float((seg > self.thr).mean()) if len(seg) else 0.0


def in_span(ws, s, e):
    return [w for w in ws if s <= (w["start"] + w["end"]) / 2 < e]


def collapse_repeats(text):
    """'a b c a b c a b c a b c ...' -> 'a b c [×N]' (a loop inside one segment)."""
    pat = re.compile(r"\b(\w+(?:\W+\w+){0,3}?)(?:\W+\1\b){3,}", re.I)
    return pat.sub(lambda m: f"{m.group(1)} [×{len(re.findall(re.escape(m.group(1)), m.group(0), re.I))}]", text)


def vote(wtext, U, V):
    """Whisper text vs Ultra/v3 words of the same span. A Whisper run is replaced only where Ultra has something
    else AND v3, at the same aligned position, says exactly what Ultra says (2 against 1). Substitutions only: a
    word one side has and the other lacks may belong to the neighbouring segment, and on FLEURS the whole gain came
    from substitutions anyway. Positional agreement matters: accepting Ultra's words when they occur anywhere in
    v3's span gave no gain at all."""
    W = [(norm(x), x) for x in wtext.split()]
    W = [x for x in W if x[0]]
    wn, un, vn = [x[0] for x in W], [w["norm"] for w in U], [w["norm"] for w in V]
    wv = difflib.SequenceMatcher(None, wn, vn, autojunk=False).get_opcodes()

    def to_v(i):  # Whisper word index -> aligned v3 index
        for tg, i1, i2, j1, j2 in wv:
            if i1 <= i < i2:
                return j1 + (i - i1 if tg == "equal" else 0)
        return len(vn)
    out, changed = [], 0
    for tg, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wn, un, autojunk=False).get_opcodes():
        if tg == "replace" and vn[to_v(i1):to_v(i2)] == un[j1:j2]:
            out += [w["surface"] for w in U[j1:j2]]; changed += i2 - i1
        else:
            out += [x[1] for x in W[i1:i2]]
    return " ".join(out), changed


def hu(whisper, ultra, v3, wav, out, *, outro: re.Pattern[str] = OUTRO, log: Log = _stderr):
    """The vote route: Whisper's segments, voted on by Ultra and v3, with Ultra filling what Whisper lost.
    ``outro`` is the language's caption-outro pattern (the table entry's)."""
    segs = [dict(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in load(whisper)["segments"] if s["text"].strip()]
    U, V = words(load(ultra)), words(load(v3))
    usents = load(ultra)["sentences"]
    en = Energy(wav)
    texts = [s["text"] for s in segs]
    kept, st = [], dict(voted=0, replaced=0, dropped=0, collapsed=0, filled=0, fill_words=0)
    for i, s in enumerate(segs):
        uw = in_span(U, s["start"], s["end"])
        dur = s["end"] - s["start"]
        distinct = {norm(x) for x in s["text"].split()} - {""}
        near_dups = sum(texts[j] == s["text"] for j in range(max(0, i - 2), min(len(segs), i + 3)))
        bad = (outro.search(s["text"]) or (dur >= 15 and len(distinct) <= 4)
               or (near_dups >= 3 and len(uw) < 2))
        if bad or (not uw and en.voiced(s["start"], s["end"]) < 0.2):
            if bad and uw:
                kept.append(dict(start=s["start"], end=s["end"], source="ultra",
                                 text=" " + " ".join(w["surface"] for w in uw))); st["replaced"] += 1
            else:
                st["dropped"] += 1
            continue
        text = collapse_repeats(s["text"])
        st["collapsed"] += text != s["text"]
        text, ch = vote(text, uw, in_span(V, s["start"] - 0.3, s["end"] + 0.3))
        st["voted"] += ch
        kept.append(dict(start=s["start"], end=s["end"], text=" " + text, source="whisper"))
    kept.sort(key=lambda s: s["start"])
    bounds = [(0.0, kept[0]["start"] if kept else 1e9)] + \
             [(a["end"], b["start"]) for a, b in zip(kept, kept[1:])] + \
             ([(kept[-1]["end"], 1e9)] if kept else [])
    for a, b in bounds:
        if b - a < 6:
            continue
        fill = [s for s in usents if s["start"] >= a - 0.5 and s["end"] <= b + 0.5 and s["text"].strip()]
        if sum(len(s["text"].split()) for s in fill) >= 6:
            for s in fill:
                kept.append(dict(start=s["start"], end=s["end"], text=s["text"], source="ultra-fill"))
                st["filled"] += 1; st["fill_words"] += len(s["text"].split())
    with open(out, "w", encoding="utf-8") as f:
        json.dump(finish(kept), f, ensure_ascii=False)
    log("hu: " + " ".join(f"{k}={v}" for k, v in st.items()) + f"; whisper {len(segs)} segments")


# --- thai --------------------------------------------------------------------------------

THAI_EDGE, THAI_GAP, THAI_KEEP_CONF = 0.4, 1.0, 0.95


def thai(base, thai_json, out, *, log: Log = _stderr):
    """Parakeet's words (or, for the vote route, Whisper's sentences) whose midpoint falls inside a Thai turn are
    dropped, and each Thai turn becomes one sentence (lang "th"). Inside a Thai turn Parakeet only ever has nothing
    or phonetic noise; a turn is one speaker, so the diarizer labels the Thai sentence the same way it labelled the
    turn. Parakeet's noise also spills just past a turn's edges and into short pauses between Thai turns (stray
    syllables and full stops at confidence 0.7-0.94), so words below 0.95 within 0.4 s of a Thai turn, or in a gap
    of under 1 s between two, go too; confident English next to Thai stays."""
    B, T = load(base), load(thai_json)
    spans = sorted((t["start"], t["end"]) for t in T["turns"])
    zones = [(s - THAI_EDGE, e + THAI_EDGE) for s, e in spans] + \
            [(a[1], b[0]) for a, b in zip(spans, spans[1:]) if b[0] - a[1] < THAI_GAP]
    in_span_ = lambda m: any(s <= m < e for s, e in spans)
    inside = lambda m, c=0.0: in_span_(m) or (c < THAI_KEEP_CONF and any(s <= m < e for s, e in zones))
    sents, gone = [], 0
    for s in B["sentences"]:
        toks = s.get("tokens") or []
        if not toks:
            if in_span_((s["start"] + s["end"]) / 2):
                gone += len(s["text"].split())
            else:
                sents.append(s)
            continue
        run, cut = [], False
        for t in toks:
            if inside((t["start"] + t["end"]) / 2, t.get("confidence", 1.0)):
                gone += bool(t["text"].strip()); cut = True
                if run:
                    sents.append(piece(run)); run = []
                continue
            # a Thai turn between two kept words splits the sentence even when Parakeet heard nothing inside the
            # turn, so no sentence spans Thai speech (the diarizer labels each sentence by the speaker who covers
            # most of it)
            if run and any(run[-1]["end"] <= a < t["start"] for a, _ in spans):
                sents.append(piece(run)); run = []; cut = True
            run.append(t)
        if run:
            sents.append(piece(run) if cut else s)
    sents = [x for x in sents if any(ch.isalnum() for ch in x["text"])]  # a piece left with only "." is noise
    for t in T["turns"]:
        sents.append(dict(start=t["start"], end=t["end"], text=" " + t["text"], lang="th"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(finish(sents), f, ensure_ascii=False)
    secs = sum(e - s for s, e in spans)
    log(f"thai: {len(spans)} Thai turns ({secs / 60:.1f} min) in place of {gone} Parakeet words")


# --- transcribe (the single entry point) ---------------------------------------------------

def parakeet_argv(cli, audio, model_path, outdir, name) -> list[str]:
    """parakeet-mlx on one file, JSON out, a fixed output name."""
    return [os.fspath(cli), os.fspath(audio), "--model", os.fspath(model_path), "--output-format", "json",
            "--output-dir", os.fspath(outdir), "--output-template", name]


def whisper_vote_argv(cli, wav, model_path, language, outdir) -> list[str]:
    """mlx_whisper as the vote route runs it: the language forced, no conditioning on earlier text, greedy, a
    compression-ratio guard; its own folder and a fixed name (mlx_whisper cuts the output name at its last dot)."""
    return [os.fspath(cli), os.fspath(wav), "--model", os.fspath(model_path),
            "--language", language, "--condition-on-previous-text", "False",
            "--temperature", "0", "--compression-ratio-threshold", "2.0",
            "--output-format", "json", "--output-dir", os.fspath(outdir), "--output-name", "whisper"]


def thai_worker_argv(rt: Runtime, wav, ultra_json, v3_json, out) -> list[str]:
    """The Thai worker child: the screen always; the turns only with the Thai model (else ``--screen-only``)."""
    a = [os.fspath(rt.tools.python), "-I", "-B", "-m", "audiointake.steps.thai", "run", os.fspath(wav),
         os.fspath(ultra_json), os.fspath(v3_json) if v3_json else "", os.fspath(out),
         "--lid-model", os.fspath(rt.models["whisper"].path)]
    if rt.tools.fluidaudiocli is not None and "nemotron" in rt.models:
        a += ["--fluidaudio", os.fspath(rt.tools.fluidaudiocli), "--nemotron", os.fspath(rt.models["nemotron"].path)]
    if "thai_turns" in rt.routes and "typhoon" in rt.models:
        a += ["--typhoon", os.fspath(rt.models["typhoon"].path), "--typhoon-tag", rt.models["typhoon"].tag]
    else:
        a.append("--screen-only")
    return a


def _under(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def transcribe(audio: Path, outdir: Path, template: str, fmt: str, rt: Runtime, *, force_lang: str | None = None,
               log: Log, duration_s: float | None = None) -> EnsembleResult:
    """Run the ensemble on ``audio``; write ``<outdir>/<template>.json`` and/or ``.txt`` (``fmt`` "json", "txt" or
    "all") and return what happened. Raises ``AsrFailed`` when the primary model fails (nothing is written).

    ``outdir`` must be under ``rt.work_root`` (the wavs the run makes, hundreds of MB, are written beside it).
    ``force_lang`` is a single language code the whole recording is in: a code with a vote route forces that route
    even under ``lang()``'s floors; any other code changes nothing. ``duration_s`` (the probe's) sizes the
    children's timeouts; without it each child gets ``NO_DURATION_TIMEOUT``. ``log`` receives stats lines, model
    tags and children's stderr tails, never transcript text.

    Routes: the vote route runs only with its pack (``VoteLang.route`` in ``rt.routes`` and the Whisper model);
    the Thai screen whenever ``thai_screen`` is a route and Whisper is installed, the Thai turns only with
    ``thai_turns`` and the Thai model (else the worker screens only and reports what it found).
    """
    if fmt not in ("txt", "json", "all"):
        raise ValueError("fmt must be 'txt', 'json' or 'all'")
    audio = Path(os.path.abspath(audio))
    out = Path(os.path.abspath(outdir))
    if not _under(out, rt.work_root):
        raise ValueError(f"outdir must be under the run's work folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{template}.json"
    allowed = proc.allowed_for(rt.tools)

    def timeout(kind: str) -> float:
        return proc.child_timeout(kind, duration_s) if duration_s else NO_DURATION_TIMEOUT

    primary = rt.models["ultra"]
    fill = rt.models.get("v3")
    tag = primary.tag
    hu_share, hu_detected, hu_route = 0.0, False, "none"
    fill_ok = True
    summary = ThaiSummary(False)
    with tempfile.TemporaryDirectory(prefix="asr-ensemble.", dir=out) as tmp:
        def asr(m, name):
            r = proc.run(parakeet_argv(rt.tools.parakeet_cli, audio, m.path, tmp, name), env=rt.env,
                         timeout=timeout("parakeet"), allowed=allowed, stdout="discard")
            j = Path(tmp) / f"{name}.json"
            if r.ok and j.exists():
                return str(j)
            log(f"asr: {m.tag} exited {r.returncode}" + (" (timed out)" if r.timed_out else ""))
            for line in r.stderr_tail:
                log(f"  {line}")
            return None

        def ffmpeg(args) -> bool:
            r = proc.run(args, env=rt.env, timeout=timeout("ffmpeg"), allowed=allowed, stdout="discard")
            for line in r.stderr_tail:
                log(f"  ffmpeg: {line}")
            return r.ok

        prim = asr(primary, "primary")
        if not prim:
            log(f"asr: primary model {primary.tag} failed")
            # an unreadable recording is decode_failed, not a failed model: ffmpeg decodes it to nowhere and says
            check = proc.run(A.decode_check(rt.tools.ffmpeg, audio), env=rt.env, timeout=timeout("ffmpeg"),
                             allowed=allowed, stdout="discard")
            if check.returncode != 0 and not check.timed_out:
                for line in check.stderr_tail:
                    log(f"  ffmpeg: {line}")
                log("asr: ffmpeg couldn't decode the recording")
                raise AsrFailed("decode_failed", file=audio.name)
            raise AsrFailed("asr_failed", file=audio.name)
        merged = prim
        wav = f"{tmp}/a.16k.wav"
        have_wav = lambda: os.path.exists(wav) or ffmpeg(A.ensemble_wav(rt.tools.ffmpeg, audio, wav))
        try:
            code, hu_share = lang(prim)
        except Exception as e:
            raise AsrFailed("asr_failed", file=audio.name) from e
        hu_detected = code != "other"
        entry = VOTE_LANG.get(force_lang or "") or VOTE_LANG.get(code)
        fj = None
        written: list[Path] = []
        try:
            if fill:
                fj = asr(fill, "fill")
                if not fj:
                    fill_ok = False
                    log(f"asr: fill model {fill.tag} failed — {tag} only")
                elif entry is not None and entry.route in rt.routes and "whisper" in rt.models:
                    norm_wav = f"{tmp}/a.norm.wav"
                    ok = have_wav()
                    ok = ok and ffmpeg(A.loudnorm_wav(rt.tools.ffmpeg, audio, norm_wav))
                    if ok:
                        r = proc.run(whisper_vote_argv(rt.tools.whisper_cli, norm_wav, rt.models["whisper"].path,
                                                       entry.code, f"{tmp}/whisper"),
                                     env=rt.env, timeout=timeout("whisper"), allowed=allowed, stdout="discard")
                        ok = r.returncode == 0 and not r.timed_out
                        if not ok:
                            for line in r.stderr_tail:
                                log(f"  whisper: {line}")
                    if ok:
                        hu(f"{tmp}/whisper/whisper.json", prim, fj, wav, f"{tmp}/merged.json", outro=entry.outro,
                           log=log)
                        merged, tag = f"{tmp}/merged.json", VOTE_TAG
                        hu_route = "vote"
                    else:
                        tag += f"+{entry.code}-failed"
                        hu_route = "vote_failed"
                        log("asr: the Whisper vote failed — Ultra only")
                else:
                    gapfill(prim, fj, f"{tmp}/merged.json", log=log)
                    merged, tag = f"{tmp}/merged.json", f"{tag}+{fill.tag}-gapfill"
                    hu_route = "no_pack" if entry is not None else "gapfill"
            # Thai: screened on every recording; any failure keeps the transcript as it is and marks the tag.
            if "thai_screen" in rt.routes and "whisper" in rt.models:
                tj = f"{tmp}/thai.json"
                r = proc.run(thai_worker_argv(rt, wav, prim, fj if fill and fj else "", tj), env=rt.env,
                             timeout=timeout("thai"), allowed=allowed, stdout="discard") if have_wav() else None
                if r is not None and r.ok and os.path.exists(tj):
                    log(next((ln for ln in reversed(r.stderr_tail) if ln.startswith("thai:")), "thai: ?"))
                    td = load(tj)
                    summary = ThaiSummary(screened=True, windows=int(td.get("windows") or 0),
                                          thai_windows=int(td.get("thai_windows") or 0),
                                          screen_only=bool(td.get("screen_only")))
                    if td.get("thai") and td.get("turns"):
                        thai(merged, tj, f"{tmp}/thai-merged.json", log=log)
                        merged, tag = f"{tmp}/thai-merged.json", f"{tag}+{td['model']}-thai"
                        summary.turns = len(td["turns"])
                        summary.seconds = round(sum(t["end"] - t["start"] for t in td["turns"]), 3)
                else:
                    tag += "+th-failed"
                    summary = ThaiSummary(False, failed=True)
                    last = list(r.stderr_tail) if r is not None else []
                    log("asr: Thai screen failed — transcript kept without it: "
                        + (last[-1][-300:] if last else "no 16 kHz wav"))
            d = load(merged)
            json_path = txt_path = None
            if fmt in ("json", "all"):
                written.append(final)
                shutil.copyfile(merged, final)
                json_path = final
            if fmt in ("txt", "all"):
                txt_path = out / f"{template}.txt"
                written.append(txt_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(x["text"].strip() for x in d["sentences"]).strip() + "\n")
        except AsrFailed:
            raise
        except Exception as e:
            # Any failure inside the merge steps fails the transcription, as it always has: nothing is written.
            log(f"asr: merging failed: {type(e).__name__}")
            for p in written:
                Path(p).unlink(missing_ok=True)
            raise AsrFailed("asr_failed", file=audio.name) from e
    return EnsembleResult(tag=tag, json_path=json_path, txt_path=txt_path, hu_share=hu_share,
                          hu_detected=hu_detected, hu_route=hu_route, fill_ok=fill_ok, thai=summary)


# --- the command line ---------------------------------------------------------------------

def runtime_from_env(rt: Runtime, environ: Mapping[str, str]) -> Runtime:
    """The two engine switches this command line honours: ``THAI=off`` drops the Thai routes, and
    ``PARAKEET_FILL_MODEL`` set and empty drops the fill model. Every other variable is ignored."""
    routes, models = set(rt.routes), dict(rt.models)
    if environ.get("THAI", "on") == "off":
        routes -= {"thai_screen", "thai_turns"}
    if environ.get("PARAKEET_FILL_MODEL") == "":
        models.pop("v3", None)
    return dataclasses.replace(rt, routes=frozenset(routes), models=models)


def _duration(rt: Runtime, audio: Path, log: Log = _stderr) -> float | None:
    """The recording's length from ffprobe, only to size the children's timeouts: None (each child gets
    ``NO_DURATION_TIMEOUT``) when there is no ffprobe or it can't say."""
    if rt.tools.ffprobe is None:
        return None
    try:
        r = proc.run(A.duration_probe(rt.tools.ffprobe, audio), env=rt.env, timeout=120.0,
                     allowed=proc.allowed_for(rt.tools), stdout="capture")
        return float(r.stdout.decode("utf-8", "replace").strip().splitlines()[0]) if r.ok else None
    except (OSError, ValueError, IndexError) as e:            # proc.NotAllowed is a ValueError
        log(f"ensemble: the recording's length is unknown ({type(e).__name__}); each child gets "
            f"{NO_DURATION_TIMEOUT / 3600:.0f} h")
        return None


def cli_transcribe(args: list[str], *, environ: Mapping[str, str] | None = None,
                   resolve: Callable[..., tuple[Runtime, list]] | None = None) -> int:
    import argparse

    from .. import lock
    from ..errors import AudioIntakeError

    ap = argparse.ArgumentParser(prog="python -m audiointake.steps.ensemble transcribe")
    ap.add_argument("audio"); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-template", required=True)
    ap.add_argument("--output-format", default="all", choices=["txt", "json", "all"])
    o = ap.parse_args(args)
    if resolve is None:
        from ..setup.runtime import resolve
    env = os.environ if environ is None else environ
    audio = Path(os.path.abspath(o.audio))
    dest = Path(os.path.abspath(o.output_dir))
    try:
        with lock.run_lock():
            rt, problems = resolve(names=False)
            for p in problems:
                _stderr(f"ensemble: {p.code}")
            rt = runtime_from_env(rt, env)
            try:
                work = rt.work_root / "ensemble"
                res = transcribe(audio, work, o.output_template, o.output_format, rt, log=_stderr,
                                 duration_s=_duration(rt, audio))
                dest.mkdir(parents=True, exist_ok=True)
                for p in (res.json_path, res.txt_path):
                    if p is not None:
                        shutil.copyfile(p, dest / p.name)
            finally:
                shutil.rmtree(rt.work_root, ignore_errors=True)
    except AsrFailed:
        _stderr(f"asr: transcribing {audio.name} failed; nothing was written")
        return 3
    except AudioIntakeError as e:
        _stderr(f"ensemble: {e.message}")
        return e.exit
    print(res.tag)
    return 0


# --- programs: a stopped steps program stops its children -------------------------------------

# The program wrapper lives in proc (the one place that starts and stops processes); these names stay here for the
# steps modules and their callers.
REAP_GRACE_S = proc.REAP_GRACE_S
PROGRAM_SIGNALS = proc.PROGRAM_SIGNALS
Killed = proc.Killed
stop_children = proc.stop_children
run_program = proc.program


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return run_program(lambda: _dispatch(argv))


def _dispatch(argv: list[str]) -> int:
    if not argv:
        _stderr(__doc__ or "")
        return 1
    cmd, a = argv[0], argv[1:]
    if cmd == "lang":
        l, share = lang(a[0]); print(l)
        _stderr(f"lang: hu-word share {share:.3f}")
    elif cmd == "gapfill":
        tau = float(a[a.index("--tau") + 1]) if "--tau" in a else 0.5
        gapfill(a[0], a[1], a[2], tau)
    elif cmd == "hu":
        hu(*a[:5])
    elif cmd == "thai":
        thai(*a[:3])
    elif cmd == "transcribe":
        return cli_transcribe(a)
    else:
        _stderr(__doc__ or "")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
