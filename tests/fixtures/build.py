"""Seeded builders for synthetic test data: speech-recogniser JSON, Whisper JSON, RTTM text, wav audio, a Runtime.

Everything is made up and reproducible: the same seed gives the same bytes. Words come from ``WORDS``, a fixed
list of invented words, so no test data can carry real speech. Nothing here touches the network or a real model.

* ``asr_json`` has parakeet-mlx's JSON shape: ``{"text", "sentences": [{"text", "start", "end", "duration",
  "confidence", "tokens": [{"text", "start", "end", "duration", "confidence"}]}]}``. A token that starts with a
  space begins a word; the closing full stop is its own token; times sit on the model's 0.08 s grid, rounded to
  3 decimals; a sentence's text is its tokens joined, and the top-level text the sentences joined and stripped.
* ``whisper_json`` has mlx_whisper's JSON shape: ``{"text", "segments": [{"id", "seek", "start", "end", "text",
  "tokens", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"}], "language"}``.
* ``rttm_text`` writes the diarizer's RTTM lines: ``SPEAKER <file id> 1 <start> <dur> <NA> <NA> speaker_<k>
  <NA> <NA>``, sorted by start, three decimals.
* ``wav_bytes`` / ``write_wav``: 16-bit PCM in pure Python (sine, silence, seeded noise, or tone bursts with
  pauses, which an energy-based turn finder splits).
* ``fake_runtime`` builds a ``contracts.Runtime`` whose tools sit under a folder the test owns (the files
  themselves are the test's to create).
"""
from __future__ import annotations

import io
import json
import math
import random
import statistics
import struct
import sys
import wave
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

GRID = 0.08                       # the recogniser's frame step, seconds
WORDS: tuple[str, ...] = (
    "zorvel", "brimdal", "quessa", "toraph", "velmine", "snarth", "kelvo", "plimbo", "dravik", "fennow",
    "glimor", "hastrel", "jostik", "mirtel", "nobrax", "pelvan", "quorm", "sablo", "tervin", "ulmox",
    "vandrel", "wexo", "yarsk", "zintar", "abrel", "crondi", "dulvo", "eskar", "frimble", "gostra",
    "helvix", "ixtel", "jarnoc", "lumvar", "moxel", "nirvo", "oskel", "prandle", "quibbet", "rostav",
)


def _r3(x: float) -> float:
    return round(x + 0.0, 3)


def _split_word(rng: random.Random, word: str) -> list[str]:
    """One word as 1 or 2 sub-word tokens, the first with the leading space."""
    if len(word) >= 5 and rng.random() < 0.5:
        cut = rng.randint(2, len(word) - 2)
        return [" " + word[:cut], word[cut:]]
    return [" " + word]


def asr_sentence(rng: random.Random, t: float, *, words: Sequence[str] = WORDS, n_words: int = 6,
                 conf: tuple[float, float] = (0.85, 1.0), low_rate: float = 0.0,
                 low: tuple[float, float] = (0.2, 0.6), capital: bool = True) -> tuple[dict[str, Any], float]:
    """One parakeet-mlx sentence starting at time ``t``; returns it and the time its last token ends."""
    toks: list[dict[str, Any]] = []
    picked = [rng.choice(words) for _ in range(n_words)]
    if capital and picked:
        picked[0] = picked[0][:1].upper() + picked[0][1:]
    pieces: list[str] = []
    for w in picked:
        pieces += _split_word(rng, w)
    pieces.append(".")
    for p in pieces:
        frames = 0 if p == "." else rng.randint(1, 4)
        c = rng.uniform(*low) if (p != "." and rng.random() < low_rate) else rng.uniform(*conf)
        start = t
        t = t + frames * GRID
        toks.append({"text": p, "start": _r3(start), "end": _r3(t), "duration": _r3(t - start),
                     "confidence": _r3(c)})
    s = {"text": "".join(x["text"] for x in toks), "start": toks[0]["start"], "end": toks[-1]["end"],
         "duration": _r3(toks[-1]["end"] - toks[0]["start"]),
         "confidence": _r3(statistics.fmean(x["confidence"] for x in toks)), "tokens": toks}
    return s, t


def asr_json(seed: int = 0, *, sentences: int = 6, words: Sequence[str] = WORDS, start: float = 0.0,
             pause: tuple[float, float] = (0.16, 1.2), n_words: tuple[int, int] = (3, 12),
             conf: tuple[float, float] = (0.85, 1.0), low_rate: float = 0.0) -> dict[str, Any]:
    """A parakeet-mlx-shaped transcript of ``sentences`` invented sentences, reproducible from ``seed``.

    ``low_rate`` is the share of word tokens given a low confidence (0.2-0.6), for gap-fill tests.
    """
    rng = random.Random(seed)
    t = start
    out: list[dict[str, Any]] = []
    for _ in range(sentences):
        s, t = asr_sentence(rng, t, words=words, n_words=rng.randint(*n_words), conf=conf, low_rate=low_rate)
        out.append(s)
        t = round((t + rng.uniform(*pause)) / GRID) * GRID
    return {"text": "".join(s["text"] for s in out).strip(), "sentences": out}


def whisper_json(seed: int = 0, *, segments: int = 4, words: Sequence[str] = WORDS, start: float = 0.0,
                 language: str = "en", n_words: tuple[int, int] = (4, 14)) -> dict[str, Any]:
    """An mlx_whisper-shaped transcript (segment level only, as its JSON output is without word timestamps)."""
    rng = random.Random(seed)
    t = start
    segs: list[dict[str, Any]] = []
    for i in range(segments):
        n = rng.randint(*n_words)
        ws = [rng.choice(words) for _ in range(n)]
        ws[0] = ws[0][:1].upper() + ws[0][1:]
        text = " " + " ".join(ws) + "."
        dur = round(n * rng.uniform(0.25, 0.45), 2)
        segs.append({"id": i, "seek": int(t * 100) // 3000 * 3000, "start": round(t, 2), "end": round(t + dur, 2),
                     "text": text, "tokens": [50365 + rng.randint(0, 9000) for _ in range(n + 1)],
                     "temperature": 0.0, "avg_logprob": round(-rng.uniform(0.05, 0.6), 6),
                     "compression_ratio": round(rng.uniform(1.0, 1.8), 6),
                     "no_speech_prob": round(rng.uniform(0.0, 0.1), 6)})
        t = t + dur + rng.uniform(0.2, 1.5)
    return {"text": "".join(s["text"] for s in segs), "segments": segs, "language": language}


def rttm_line(file_id: str, start: float, dur: float, label: str) -> str:
    return f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>\n"


def rttm_text(seed: int = 0, *, file_id: str = "clip.16k", speakers: int = 2, duration: float = 60.0,
              turn: tuple[float, float] = (0.5, 6.0), gap: tuple[float, float] = (0.0, 1.0),
              overlap_rate: float = 0.0) -> str:
    """Diarizer turns covering about ``duration`` seconds among ``speakers`` labels (speaker_0 ...).

    ``overlap_rate`` is the share of turns that start before the previous one ends.
    """
    if not 1 <= speakers <= 8:
        raise ValueError("the diarizer labels at most 8 speakers")
    rng = random.Random(seed)
    t, prev, lines = 0.0, -1, []
    rows: list[tuple[float, float, str]] = []
    while t < duration:
        k = rng.randrange(speakers)
        if speakers > 1:
            while k == prev:
                k = rng.randrange(speakers)
        d = min(rng.uniform(*turn), max(0.1, duration - t))
        s = t
        if rows and rng.random() < overlap_rate:
            s = max(0.0, t - rng.uniform(0.1, 0.8))
        rows.append((s, d, f"speaker_{k}"))
        prev = k
        t = s + d + rng.uniform(*gap)
    for s, d, label in sorted(rows):
        lines.append(rttm_line(file_id, s, d, label))
    return "".join(lines)


def wav_bytes(seconds: float = 1.0, *, rate: int = 16000, kind: str = "sine", freq: float = 440.0,
              amplitude: float = 0.3, seed: int = 0, channels: int = 1,
              burst: tuple[float, float] = (1.5, 0.7)) -> bytes:
    """16-bit PCM wav. kind: sine, silence, noise (seeded white noise) or bursts (tone ``burst[0]`` s on,
    ``burst[1]`` s off)."""
    if kind not in ("sine", "silence", "noise", "bursts"):
        raise ValueError("kind must be sine, silence, noise or bursts")
    rng = random.Random(seed)
    n = int(round(seconds * rate))
    amp = max(0.0, min(1.0, amplitude)) * 32767
    on, off = burst
    period = on + off
    frames = bytearray()
    for i in range(n):
        t = i / rate
        if kind == "silence":
            v = 0.0
        elif kind == "noise":
            v = rng.uniform(-1.0, 1.0)
        elif kind == "bursts" and (t % period) >= on:
            v = 0.0
        else:
            v = math.sin(2 * math.pi * freq * t)
        sample = struct.pack("<h", int(round(v * amp)))
        frames += sample * channels
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def write_wav(path: Path, seconds: float = 1.0, **kw: Any) -> Path:
    path = Path(path)
    path.write_bytes(wav_bytes(seconds, **kw))
    return path


def dumps(d: Any) -> str:
    """JSON text as the recogniser writes it: one line, non-ASCII kept."""
    return json.dumps(d, ensure_ascii=False)


def fake_model(key: str, tag: str, path: Path, revision: str = "0" * 40) -> Any:
    from audiointake.contracts import Model
    return Model(key=key, tag=tag, path=Path(path), revision=revision)


def fake_runtime(root: Path, *, models: Mapping[str, Any] | None = None, routes: Iterable[str] = (),
                 env: Mapping[str, str] | None = None, voices_dir: Path | None = None, fluidaudio: bool = True,
                 ffprobe: bool = True, fluidaudio_tag: str = "v0.18.0") -> Any:
    """A Runtime whose tools are ``<root>/bin/<name>`` (the test writes them) and whose work root is a fresh
    0700 ``<root>/work``. ``python`` is this interpreter. ``env`` defaults to the scrubbed child set."""
    from audiointake.contracts import ROUTES, Runtime, Tools
    from audiointake.fsutil import private_dir

    root = Path(root)
    bin_ = root / "bin"
    bin_.mkdir(parents=True, exist_ok=True)
    work = private_dir(root / "work")
    tools = Tools(ffmpeg=bin_ / "ffmpeg", ffprobe=(bin_ / "ffprobe") if ffprobe else None,
                  python=Path(sys.executable), parakeet_cli=bin_ / "parakeet-mlx", whisper_cli=bin_ / "mlx_whisper",
                  fluidaudiocli=(bin_ / "fluidaudiocli") if fluidaudio else None, fluidaudio_tag=fluidaudio_tag)
    rset = frozenset(routes)
    unknown = rset - set(ROUTES)
    if unknown:
        raise ValueError(f"unknown routes {sorted(unknown)}")
    if env is None:
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(root / "home"), "TMPDIR": str(work), "USER": "test",
               "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_CACHE": str(root / "hf"),
               "ORT_DISABLE_TELEMETRY": "1"}
    return Runtime(tools=tools, models=dict(models or {}), routes=rset, env=dict(env), work_root=work,
                   voices_dir=voices_dir)
