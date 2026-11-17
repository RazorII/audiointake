"""Speaker identification: the diarizer's labels -> the names of enrolled people, by voiceprint.

``match`` runs right after diarization (as a child process); the sidecar it writes makes merge_speakers print
"Ana: ..." instead of "Speaker 1: ..." for every label it names. Unnamed labels stay "Speaker N".

Model: 3D-Speaker CAM++ (zh+en, "advanced"), ``3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`` from
sherpa-onnx's speaker-recognition models, run by sherpa-onnx. Voiceprints and thresholds belong to that exact file
(``MODEL_SHA256``); a print made with another file is skipped, never mixed in.

Decision per label: clean speech = its RTTM segments minus any overlap with another label, 0.25 s trimmed off each
edge, pieces >= 1.5 s cut to <= 10 s; label embedding = duration-weighted mean of the longest pieces up to 60 s.
Named when >= 8 s of clean speech, cosine to that person's voiceprint >= 0.44, and >= 0.08 ahead of the runner-up.

Command line (``--voices DIR`` and ``--model ONNX`` are required by every subcommand)::

    python -I -B -m audiointake.steps.speaker_id match <16k.wav> <diar.rttm> [--out speakers.json] --voices DIR
        --model ONNX                                        prints label<TAB>name|-<TAB>score<TAB>why per label
    python -I -B -m audiointake.steps.speaker_id enroll-label <Name> <16k.wav> <diar.rttm> <label> --source "<what>"
        [--lang XX] [--force] --voices DIR --model ONNX     add one recording's clean speech of <label> (up to 30 s)
    python -I -B -m audiointake.steps.speaker_id enroll <Name> <LANG> <clip> [...] [--force] [--ffmpeg PATH]
        --voices DIR --model ONNX                           clips in any audio format (up to 60 s each)
    python -I -B -m audiointake.steps.speaker_id list --voices DIR --model ONNX    voiceprints, their sources and model
    python -I -B -m audiointake.steps.speaker_id check --voices DIR --model ONNX   the enrollment check on every source

Every failure prints ``speaker_id: <message>`` on stderr and exits 1. Stopped by SIGTERM, SIGHUP or SIGINT, it
stops the ffmpeg it started (``ensemble.run_program``).

Enrollment check: every enrollment path ends in ``_add``, which refuses a source (``HeldBack``, nothing written)
when
  - the person has a print and audiointake would not name them from this source (< 0.44, or < 0.08 ahead of someone
    else): ``not_matching_print``;
  - the person has no print yet and audiointake would already name this voice as someone enrolled:
    ``matches_someone_else``;
  - >= 20 % of its seconds are pieces scoring < 0.30 against the person's print or higher against another print:
    ``mixed_voices``;
  - or its pieces split into two voices (average-linkage groups with mean cosine < 0.35, the smaller >= 20 % of the
    speech): ``two_voices``.
``force`` enrolls anyway, once the user has confirmed by ear.

Voiceprints: ``<voices>/<name lowercased>.json`` ``{name, model_sha256, vectors: [{vec, source, seconds, lang,
added}], updated}``. Every top-level ``*.json`` not starting with "." is read as a voiceprint, so other state goes
in a subfolder; a stray file is skipped with a warning.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Union

# onnxruntime's official builds send usage telemetry by default; it reads this once, at import, so it must be set
# before anything imports onnxruntime (sherpa-onnx does). Forced, so a stray 0 can't turn it back on.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from ..contracts import Problem  # noqa: E402
from ..errors import AudioIntakeError, problem  # noqa: E402
from . import rttm  # noqa: E402  the one RTTM reader (file-ids contain spaces)

PathLike = Union[str, "os.PathLike[str]"]

MODEL_SHA256 = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
ENGINE = f"sherpa-onnx campplus-zh_en-advanced {MODEL_SHA256[:8]}"
THRESHOLD, MARGIN, MIN_SPEECH = 0.44, 0.08, 8.0
TRIM, MIN_PIECE, MAX_PIECE = 0.25, 1.5, 10.0
MATCH_CAP, ENROLL_CAP = 60.0, 30.0
PIECE_FLOOR, FOREIGN_SHARE, SPLIT_MAX = 0.30, 0.20, 0.35   # the enrollment check, see the module docstring
NAME_MAX = 64
CLIP_FFMPEG_TIMEOUT = 900.0


class SpeakerIdError(Exception):
    """A failure the command line reports as ``speaker_id: <message>`` with exit 1."""


class ClipUnreadable(SpeakerIdError, AudioIntakeError):
    """ffmpeg couldn't read an enrollment clip: ``decode_failed`` (exit 1) with the clip's file name. The command
    line prints it as ``speaker_id: ffmpeg failed on <clip>``, after ffmpeg's own last lines (``stderr_tail``)."""

    stderr_tail: tuple[str, ...] | list[str] = ()


class HeldBack(Exception):
    """A new voiceprint source failed the enrollment check; nothing was written for it. ``problems`` holds the
    reasons as closed codes, in the check's order; the message holds them as sentences."""

    def __init__(self, problems: list[Problem], message: str = "") -> None:
        self.problems = list(problems)
        super().__init__(message or ", ".join(p.code for p in self.problems))


class TooLittleSpeech(Exception):
    """A source had less than ``MIN_SPEECH`` seconds of usable speech; nothing was written for it."""

    def __init__(self, seconds: float, *, file: str | None = None, label: str | None = None, name: str | None = None,
                 message: str = "") -> None:
        self.seconds = float(seconds)
        self.file, self.label, self.name = file, label, name
        super().__init__(message or f"{seconds:.1f} s of usable speech; a voiceprint needs >= {MIN_SPEECH:.0f} s")

    def problem(self) -> Problem:
        params: dict[str, Any] = {"seconds": self.seconds, "source": "label" if self.label else "clip"}
        for k in ("file", "label", "name"):
            if getattr(self, k) is not None:
                params[k] = getattr(self, k)
        return problem("too_little_speech", **params)


@dataclass(frozen=True)
class Held:
    """One reason a source is held back: its closed code with params, and the sentence the command line prints."""
    problem: Problem
    sentence: str


def die(msg):
    raise SpeakerIdError(msg)


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def _today() -> str:
    return date.today().isoformat()


def check_name(name: str) -> str:
    """A person's name as a voiceprint takes it: 1-64 characters, not only spaces, no "/" or NUL, not starting
    with "." (``name_invalid`` otherwise)."""
    from ..errors import error_for
    if (not isinstance(name, str) or not 1 <= len(name) <= NAME_MAX or not name.strip() or "/" in name
            or "\x00" in name or name.startswith(".")):
        raise error_for("name_invalid")
    return name


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _extractor(model):
    try:
        import sherpa_onnx
    except ImportError:
        die("sherpa-onnx isn't installed in this Python")
    model = Path(model)
    if not model.exists():
        die(f"the speaker model is missing at {model}")
    if _sha(model) != MODEL_SHA256:
        die(f"{model} is not the calibrated CAM++ model (sha256 differs); voiceprints and thresholds belong to that one")
    return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=4))


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


def _load(wav):
    import soundfile as sf
    a, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        die(f"{wav}: {sr} Hz, expected a 16 kHz mono wav")
    return a


def clean_pieces(segs, label):
    """The label's speech with every overlap with another label removed, edges trimmed, cut into 1.5-10 s pieces."""
    others = [(a, b) for a, b, s in segs if s != label]
    out = []
    for a, b, s in segs:
        if s != label:
            continue
        cur = [(a + TRIM, b - TRIM)]
        for oa, ob in others:
            nxt = []
            for ca, cb in cur:
                if ob <= ca or oa >= cb:
                    nxt.append((ca, cb)); continue
                if oa > ca: nxt.append((ca, oa - TRIM))
                if ob < cb: nxt.append((ob + TRIM, cb))
            cur = nxt
        for ca, cb in cur:
            while cb - ca >= MIN_PIECE:
                e = min(cb, ca + MAX_PIECE); out.append((ca, e)); ca = e
    return out


def piece_vecs(ex, audio, pieces, cap):
    """Unit embeddings of the longest pieces up to cap seconds -> [(vector, seconds)]."""
    out, tot = [], 0.0
    for a, b in sorted(pieces, key=lambda p: -(p[1] - p[0])):
        if tot >= cap:
            break
        st = ex.create_stream(); st.accept_waveform(16000, audio[int(a * 16000):int(b * 16000)]); st.input_finished()
        out.append((_unit(ex.compute(st)), b - a)); tot += b - a
    return out


def pool(pvs):
    """Duration-weighted mean of piece embeddings -> (unit vector, seconds used)."""
    import numpy as np
    if not pvs:
        return None, 0.0
    return _unit(np.average(np.stack([v for v, _ in pvs]), axis=0, weights=[s for _, s in pvs])), sum(s for _, s in pvs)


def pooled(ex, audio, pieces, cap):
    """Duration-weighted mean embedding of the longest pieces up to cap seconds -> (unit vector, seconds used)."""
    return pool(piece_vecs(ex, audio, pieces, cap))


def two_voices(pvs):
    """Average-linkage grouping of the pieces down to two groups -> (mean cosine between them, smaller group's share
    of the seconds). One voice keeps the groups alike; a label holding two people splits into two unlike groups."""
    import numpy as np
    if len(pvs) < 2:
        return 1.0, 0.0
    X = np.stack([v for v, _ in pvs]); w = np.array([s for _, s in pvs]); C = X @ X.T
    groups = [[i] for i in range(len(pvs))]
    while len(groups) > 2:
        _, i, j = max((float(np.mean(C[np.ix_(groups[i], groups[j])])), i, j)
                      for i in range(len(groups)) for j in range(i + 1, len(groups)))
        groups[i] += groups.pop(j)
    return float(np.mean(C[np.ix_(groups[0], groups[1])])), min(w[groups[0]].sum(), w[groups[1]].sum()) / w.sum()


def enrollment_problems(name, vec, pvs=None, prints=None, *, voices_dir=None, source=None) -> list[Held]:
    """Why this new source shouldn't go into name's print, as codes with their sentences; empty when it's fine.
    vec is the pooled source embedding, pvs its pieces [(vector, seconds)] when the audio is at hand. ``prints``
    defaults to the prints in ``voices_dir``; ``source`` (what the source is) goes into the problems' params."""
    prints = load_prints(voices_dir) if prints is None else prints
    own = prints.get(name)
    others = sorted(((float(vec @ v), n) for n, v in prints.items() if n != name), reverse=True)
    probs: list[Held] = []
    extra = {"file": source} if source is not None else {}
    if own is not None:
        s = float(vec @ own)
        if s < THRESHOLD or (others and s - others[0][0] < MARGIN):
            p = problem("not_matching_print", name=name, score=s, **(
                {"other": others[0][1], "other_score": others[0][0]} if others else {}), **extra)
            probs.append(Held(p, f"it scores {s:.2f} against {name}'s voiceprint"
                                 + (f" and {others[0][0]:.2f} against {others[0][1]}'s" if others else "")
                                 + f", so audiointake wouldn't name {name} from it (needs {THRESHOLD} and a {MARGIN} lead)"))
    elif others and others[0][0] >= THRESHOLD and others[0][0] - (others[1][0] if len(others) > 1 else -1.0) >= MARGIN:
        p = problem("matches_someone_else", name=name, other=others[0][1], score=others[0][0], **extra)
        probs.append(Held(p, f"audiointake would already name this voice {others[0][1]} ({others[0][0]:.2f})"))
    if pvs:
        tot = sum(s for _, s in pvs)
        if own is not None:
            bad = sum(sec for v, sec in pvs if float(v @ own) < PIECE_FLOOR
                      or any(float(v @ o) >= float(v @ own) for n, o in prints.items() if n != name))
            if bad / tot >= FOREIGN_SHARE:
                p = problem("mixed_voices", name=name, percent=100 * bad / tot, seconds=bad, total=tot, **extra)
                probs.append(Held(p, f"{bad:.0f} of its {tot:.0f} s don't sound like {name}"))
        inter, minority = two_voices(pvs)
        if inter < SPLIT_MAX and minority >= FOREIGN_SHARE:
            p = problem("two_voices", name=name, share=100 * minority, similarity=inter, **extra)
            probs.append(Held(p, f"it sounds like two voices: {minority:.0%} of the speech is unlike the rest "
                                 f"(similarity {inter:.2f})"))
    return probs


def _segs_from_text(text):
    return [(s["start"], s["start"] + s["dur"], s["spk"]) for s in rttm.segments_from_text(text)]


def _segs(rttm_path):
    return [(s["start"], s["start"] + s["dur"], s["spk"]) for s in rttm.segments_from_path(rttm_path)]


def _json_files(voices_dir) -> list[Path]:
    """The ``*.json`` files directly in the folder, sorted, without names starting with "." (``._`` metadata files)."""
    if voices_dir is None:
        return []
    d = Path(voices_dir)
    if not d.is_dir():
        return []
    return sorted(f for f in d.glob("*.json") if not f.name.startswith(".") and f.is_file())


def _print_files(voices_dir, *, quiet=False) -> Iterator[tuple[Path, dict]]:
    """(path, data) for every voiceprint in the folder; any other JSON there (a list, a file without vectors) is
    skipped with a warning, so one stray file can't stop anyone being named."""
    for f in _json_files(voices_dir):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            if not quiet:
                warn(f"speaker_id: skipping {f.name}: not valid JSON")
            continue
        except OSError:
            if not quiet:
                warn(f"speaker_id: skipping {f.name}: not readable")
            continue
        if not (isinstance(d, dict) and "vectors" in d and "name" in d):
            if not quiet:
                warn(f"speaker_id: skipping {f.name}: not a voiceprint")
            continue
        yield f, d


def load_prints(voices_dir, *, quiet=False):
    import numpy as np
    prints = {}
    for f, d in _print_files(voices_dir, quiet=quiet):
        if d.get("model_sha256") != MODEL_SHA256:
            if not quiet:
                warn(f"speaker_id: skipping {f.name}: made with another model")
            continue
        if d.get("vectors"):
            prints[d["name"]] = _unit(np.mean(np.stack([_unit(v["vec"]) for v in d["vectors"]]), axis=0))
    return prints


def scan(voices_dir) -> dict:
    """How the folder's files read as voiceprints, as counts: ``prints`` (usable), ``other_model`` (made with
    another speaker model, left out) and ``not_prints`` (other files, skipped)."""
    files = _json_files(voices_dir)
    good = list(_print_files(voices_dir, quiet=True))
    other = sum(1 for _, d in good if d.get("model_sha256") != MODEL_SHA256)
    usable = sum(1 for _, d in good if d.get("model_sha256") == MODEL_SHA256 and d.get("vectors"))
    return {"prints": usable, "other_model": other, "not_prints": len(files) - len(good)}


def _sources(d: dict) -> list:
    """[source, added] per vector of a print, for the digest; never the vector itself."""
    vs = d.get("vectors")
    return [[v.get("source"), v.get("added")] for v in vs if isinstance(v, dict)] if isinstance(vs, list) else []


def voices_digest(voices_dir) -> str:
    """The voiceprint set's digest: per usable print (made with this model), in file-name order, its file
    name, model hash, ``updated`` date and each source's description and date, never its vectors; "" when there is
    none. ``voices.digest`` gives the same value (the records store it). The same set gives the same digest; adding
    or removing a print, or a source (even a second one added the same day), changes it."""
    items = [[f.name, d.get("model_sha256"), d.get("updated"), _sources(d)]
             for f, d in _print_files(voices_dir, quiet=True) if d.get("model_sha256") == MODEL_SHA256]
    if not items:
        return ""
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


# --- match --------------------------------------------------------------------------------

def _match(wav, rttm_path, voices_dir, model):
    prints = load_prints(voices_dir)
    if not prints:
        die(f"no voiceprints in {voices_dir}")
    ex, audio, segs = _extractor(model), _load(wav), _segs(rttm_path)
    res, lines = {}, []
    for label in sorted({s for _, _, s in segs}):
        pieces = clean_pieces(segs, label)
        speech = sum(b - a for a, b in pieces)
        emb, used = pooled(ex, audio, pieces, MATCH_CAP)
        if emb is None:
            res[label] = dict(name=None, score=None, runner_up=None, speech_s=round(speech, 1), why="no clean speech"); continue
        sc = sorted(((float(emb @ v), n) for n, v in prints.items()), reverse=True)
        top_s, top_n = sc[0]; second = sc[1] if len(sc) > 1 else (-1.0, None)
        why = ("too little speech" if speech < MIN_SPEECH else "below threshold" if top_s < THRESHOLD
               else "too close to runner-up" if top_s - second[0] < MARGIN else "named")
        res[label] = dict(name=top_n if why == "named" else None, score=round(top_s, 3), best_guess=top_n,
                          runner_up=[second[1], round(second[0], 3)] if second[1] else None, speech_s=round(speech, 1), why=why)
        lines.append(f"{label}\t{res[label]['name'] or '-'}\t{top_s:.2f}\t{why}" + ("" if why == "named" else f" (best guess {top_n})"))
    sidecar = dict(engine=ENGINE, threshold=THRESHOLD, margin=MARGIN, min_speech_s=MIN_SPEECH, enrolled=sorted(prints),
                   labels=res)
    return sidecar, lines


def match(wav, rttm, voices_dir, model) -> dict:
    """The speakers sidecar for one recording: every diarizer label with its best match, and the name for each
    label that is named. ``rttm`` is the RTTM file's path. Raises ``SpeakerIdError`` (no voiceprints, a missing
    or different model, a wav that isn't 16 kHz)."""
    return _match(wav, rttm, voices_dir, model)[0]


def sidecar_text(sidecar: dict) -> str:
    """The sidecar's file text (indent 1, non-ASCII kept, no trailing newline)."""
    return json.dumps(sidecar, indent=1, ensure_ascii=False)


# --- enroll --------------------------------------------------------------------------------

def _write_print(f: Path, d: dict) -> None:
    f.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = f.parent / f".partial-{secrets.token_hex(6)}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(d, indent=1, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, f)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _add(name, vec, source, seconds, lang, pvs=None, force=False, *, voices_dir) -> dict:
    """The one place a source enters a voiceprint, so the enrollment check guards every path (raises HeldBack)."""
    check_name(name)
    f = Path(voices_dir) / f"{name.lower()}.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else dict(name=name, model_sha256=MODEL_SHA256, vectors=[])
    if d.get("model_sha256") != MODEL_SHA256:
        die(f"{f.name} was made with another model; move it aside before enrolling with this one")
    if any(v["source"] == source for v in d["vectors"]):
        return dict(name=name, source=source, added=False, skipped=True, seconds=0.0, sources=len(d["vectors"]),
                    forced=[], lines=[f"{name}: '{source}' already enrolled, skipped"])
    held = enrollment_problems(name, vec, pvs, voices_dir=voices_dir, source=source)
    if held and not force:
        raise HeldBack([h.problem for h in held], f"{name} from {source}: " + "; ".join(h.sentence for h in held))
    lines = []
    if held:
        lines.append(f"{name}: enrolled despite the check (--force): " + "; ".join(h.sentence for h in held))
    d["vectors"].append(dict(vec=[round(float(x), 6) for x in vec], source=source, seconds=round(seconds, 1), lang=lang,
                             added=_today()))
    d["updated"] = _today()
    _write_print(f, d)
    lines.append(f"{name}: +{seconds:.0f} s from {source} ({len(d['vectors'])} sources)")
    return dict(name=name, source=source, added=True, skipped=False, seconds=round(seconds, 1),
                sources=len(d["vectors"]), forced=[h.problem for h in held], lines=lines)


def enroll_label(name, wav, rttm_text, label, source, lang, voices_dir, model, force) -> dict:
    """Add one recording's clean speech of ``label`` (up to 30 s; at least 8 s) to ``name``'s voiceprint.
    ``rttm_text`` is the recording's RTTM; ``source`` describes what this is. Raises ``TooLittleSpeech`` or
    ``HeldBack`` (nothing written), and ``name_invalid`` before anything is loaded."""
    check_name(name)
    lang = lang.upper() if lang else None
    ex, audio = _extractor(model), _load(wav)
    pvs = piece_vecs(ex, audio, clean_pieces(_segs_from_text(rttm_text), label), ENROLL_CAP)
    emb, used = pool(pvs)
    if emb is None or used < MIN_SPEECH:
        raise TooLittleSpeech(used, label=label, name=name,
                              message=f"{label} has {used:.1f} s of clean speech; enrollment needs >= {MIN_SPEECH:.0f} s")
    return _add(name, emb, source, used, lang, pvs, force=force, voices_dir=voices_dir)


def enroll_clips(name, clips, lang, voices_dir, model, ffmpeg, env, force, *, work_dir=None) -> dict:
    """Add each clip (any audio format ffmpeg reads) as one source of ``name``'s voiceprint: up to 60 s of its
    longest 10 s pieces, at least 8 s. Clips go in one after another, each written as it passes, so a clip that
    fails (``TooLittleSpeech``, ``HeldBack``, ``ClipUnreadable`` when ffmpeg can't read it, or anything else)
    leaves the ones before it enrolled: every exception raised here carries ``results``, the clips already added.

    ``ffmpeg`` is an absolute path to an executable; ``env`` is its whole environment. Each clip's 16 kHz copy goes
    in ``work_dir``, else in ``env["TMPDIR"]`` (the run's work folder when ``env`` is the run's), else the system's
    temporary folder, and is deleted after the clip. ``name_invalid`` is raised before anything is loaded."""
    from .. import argv as A
    from .. import proc
    check_name(name)
    if not (os.path.isabs(os.fspath(ffmpeg)) and os.access(ffmpeg, os.X_OK)):
        die(f"ffmpeg isn't an executable at {ffmpeg}")
    lang = lang.upper() if lang else None
    tmp_dir = work_dir or (env.get("TMPDIR") if os.path.isdir(env.get("TMPDIR") or "") else None)
    ex = _extractor(model)
    results: list[dict] = []
    for s in clips:
        fd, tmpname = tempfile.mkstemp(suffix=".wav", dir=os.fspath(tmp_dir) if tmp_dir else None)
        os.close(fd)
        tmp = Path(tmpname)
        try:
            r = proc.run(A.clip_wav(ffmpeg, os.path.abspath(s), str(tmp)), env=dict(env),
                         timeout=CLIP_FFMPEG_TIMEOUT, allowed={Path(ffmpeg)}, stdout="discard")
            if r.returncode or r.timed_out:
                e = ClipUnreadable("decode_failed", f"ffmpeg failed on {s}", file=Path(s).name)
                e.stderr_tail = list(r.stderr_tail)
                raise e
            audio = _load(tmp)
            n = len(audio) / 16000
            pieces = [(a, min(n, a + MAX_PIECE)) for a in [i * MAX_PIECE for i in range(int(n // MAX_PIECE) + 1)]
                      if min(n, a + MAX_PIECE) - a >= MIN_PIECE]
            pvs = piece_vecs(ex, audio, pieces, ENROLL_CAP * 2)
            emb, used = pool(pvs)
            if emb is None or used < MIN_SPEECH:
                raise TooLittleSpeech(used, file=Path(s).name, name=name,
                                      message=f"{s}: {used:.1f} s usable; a sample needs >= {MIN_SPEECH:.0f} s")
            results.append(_add(name, emb, f"sample {Path(s).name}", used, lang, pvs, force=force,
                                voices_dir=voices_dir))
        except Exception as e:
            e.results = results                      # the clips enrolled before this one
            raise
        finally:
            tmp.unlink(missing_ok=True)
    return dict(name=name, results=results, lines=[ln for r in results for ln in r["lines"]])


# --- list, check, remove, rename -------------------------------------------------------------

def list_prints(voices_dir) -> list[dict]:
    """Every voiceprint: name, file, how many sources and seconds, whether it was made with this model, when it
    was updated, and each source (seconds, language, description)."""
    out = []
    for f, d in _print_files(voices_dir):
        out.append(dict(name=d["name"], file=f.name, sources=len(d["vectors"]),
                        seconds=sum(v["seconds"] for v in d["vectors"]), model_ok=d.get("model_sha256") == MODEL_SHA256,
                        updated=d.get("updated"),
                        vectors=[dict(seconds=v["seconds"], lang=v.get("lang"), source=v["source"]) for v in d["vectors"]]))
    return out


def list_lines(prints: list[dict]) -> list[str]:
    lines = []
    for p in prints:
        ok = "ok" if p["model_ok"] else "OTHER MODEL"
        lines.append(f"{p['name']}: {p['sources']} sources, {p['seconds']:.0f} s, {ok}, updated {p['updated']}")
        for v in p["vectors"]:
            lines.append(f"   {v['seconds']:5.0f} s  {v.get('lang') or '--'}  {v['source']}")
    return lines


def check_prints(voices_dir, model=None) -> list[dict]:
    """Each stored source against its person's other sources and everyone else's print, as if it were enrolled
    now. Stored sources keep only their pooled vector, so the piece tests need the audio and don't run here. Prints
    are checked against the pinned model hash; ``model`` isn't read."""
    import numpy as np
    files = [d for _, d in _print_files(voices_dir) if d.get("model_sha256") == MODEL_SHA256]
    full = {d["name"]: _unit(np.mean(np.stack([_unit(v["vec"]) for v in d["vectors"]]), axis=0)) for d in files if d["vectors"]}
    out = []
    for d in files:
        for k, v in enumerate(d["vectors"]):
            rest = [_unit(w["vec"]) for j, w in enumerate(d["vectors"]) if j != k]
            prints = {n: p for n, p in full.items() if n != d["name"]}
            if rest:
                prints[d["name"]] = _unit(np.mean(np.stack(rest), axis=0))
            held = enrollment_problems(d["name"], _unit(v["vec"]), prints=prints, source=v["source"])
            out.append(dict(name=d["name"], source=v["source"], held=bool(held), problems=[h.problem for h in held],
                            sentences=[h.sentence for h in held]))
    return out


def check_lines(results: list[dict]) -> list[str]:
    lines = []
    for r in results:
        lines.append(f"{r['name']:8s} {'HELD' if r['held'] else 'ok  '} {r['source'][:70]}"
                     + (f"\n         {'; '.join(r['sentences'])}" if r["held"] else ""))
    lines.append(f"{sum(r['held'] for r in results)} of {len(results)} stored sources would be held back today")
    return lines


def _find_print(voices_dir, name) -> tuple[Path, dict]:
    from ..errors import error_for
    for f, d in _print_files(voices_dir, quiet=True):
        if d.get("name") == name or f.name == f"{name.lower()}.json":
            return f, d
    raise error_for("voice_not_found", name=name)


def remove_print(voices_dir, name) -> Path:
    """Delete ``name``'s voiceprint file; its path. ``voice_not_found`` when there is none."""
    f, _ = _find_print(voices_dir, name)
    f.unlink()
    return f


def rename_print(voices_dir, old, new) -> Path:
    """Give ``old``'s voiceprint the name ``new`` (its ``name`` field and its file name); the new path. Refused
    when ``new`` isn't a valid name or another voiceprint already has its file name."""
    from ..errors import error_for
    check_name(new)
    f, d = _find_print(voices_dir, old)
    dest = f.parent / f"{new.lower()}.json"
    if dest.exists() and dest != f:
        raise error_for("bad_arguments", detail=f"A voiceprint called {new} already exists", command="voices")
    d["name"] = new
    _write_print(dest, d)
    if dest != f:
        f.unlink()
    return dest


# --- the command line ----------------------------------------------------------------------

def _take(args: list[str], flag: str) -> str | None:
    """Remove ``flag VALUE`` from args and return VALUE (None when absent)."""
    if flag not in args:
        return None
    i = args.index(flag)
    if i + 1 >= len(args):
        die(f"{flag} needs a value")
    v = args[i + 1]
    del args[i:i + 2]
    return v


def _out(lines):
    for ln in lines:
        print(ln)


def cmd_match(args, voices, model):
    if len(args) < 2:
        die("usage: match <16k.wav> <diar.rttm> [--out speakers.json] --voices DIR --model ONNX")
    wav, diar = Path(args[0]), Path(args[1])
    out = Path(args[args.index("--out") + 1]) if "--out" in args else None
    sidecar, lines = _match(wav, diar, voices, model)
    _out(lines)
    if out:
        out.write_text(sidecar_text(sidecar), encoding="utf-8")


def cmd_enroll_label(args, voices, model):
    if len(args) < 4 or "--source" not in args:
        die('usage: enroll-label <Name> <16k.wav> <diar.rttm> <label> --source "<what>" [--lang EN] --voices DIR --model ONNX')
    name, wav, diar, label = args[0], Path(args[1]), Path(args[2]), args[3]
    lang = args[args.index("--lang") + 1].upper() if "--lang" in args else None
    text = diar.read_text(encoding="utf-8")
    try:
        r = enroll_label(name, wav, text, label, args[args.index("--source") + 1], lang, voices, model,
                         force="--force" in args)
    except TooLittleSpeech as e:
        die(str(e))
    _out(r["lines"])


def cmd_enroll(args, voices, model, ffmpeg):
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if len(args) < 3:
        die("usage: enroll <Name> <LANG> <sample> [...] [--force] --voices DIR --model ONNX")
    name, lang = args[0], args[1].upper()
    if ffmpeg is None:
        import shutil
        from ..platforms import FFMPEG_DIRS
        ffmpeg = shutil.which("ffmpeg") or next((str(d / "ffmpeg") for d in FFMPEG_DIRS
                                                 if os.access(d / "ffmpeg", os.X_OK)), None)
        if ffmpeg is None:
            die("ffmpeg isn't installed (pass --ffmpeg PATH)")
    ffmpeg = os.path.abspath(ffmpeg)
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/"), "TMPDIR": tempfile.gettempdir(),
           "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8"}
    try:
        r = enroll_clips(name, args[2:], lang, voices, model, ffmpeg, env, force)
    except Exception as e:
        _out([ln for x in getattr(e, "results", []) for ln in x["lines"]])     # the clips enrolled before it
        for ln in getattr(e, "stderr_tail", []):                               # ffmpeg's own words, as before
            warn(ln)
        if isinstance(e, TooLittleSpeech):
            die(str(e))
        raise
    _out(r["lines"])


def cmd_list(_, voices, model):
    _out(list_lines(list_prints(voices)))


def cmd_check(_, voices, model):
    _out(check_lines(check_prints(voices, model)))


def main(argv: list[str] | None = None) -> int:
    from .ensemble import run_program
    args = list(sys.argv[1:] if argv is None else argv)
    return run_program(lambda: _main(args))


def _main(args: list[str]) -> int:
    cmds = {"match": cmd_match, "enroll-label": cmd_enroll_label, "enroll": cmd_enroll, "list": cmd_list,
            "check": cmd_check}
    try:
        if not args or args[0] not in cmds:
            die("usage: speaker_id match|enroll-label|enroll|list|check ... --voices DIR --model ONNX  (see --help)")
        cmd, rest = args[0], args[1:]
        voices, model = _take(rest, "--voices"), _take(rest, "--model")
        ffmpeg = _take(rest, "--ffmpeg")
        if voices is None or model is None:
            die(f"{cmd} needs --voices DIR and --model ONNX")
        if cmd == "enroll":
            cmd_enroll(rest, Path(voices), Path(model), ffmpeg)
        else:
            cmds[cmd](rest, Path(voices), Path(model))
    except HeldBack as e:
        print(f"speaker_id: held back, nothing enrolled: {e}. If the user confirms it by ear anyway, rerun with "
              f"--force", file=sys.stderr)
        return 1
    except SpeakerIdError as e:
        print(f"speaker_id: {e}", file=sys.stderr)
        return 1
    except AudioIntakeError as e:                     # name_invalid, from the name check
        print(f"speaker_id: {e.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    sys.exit(main())
