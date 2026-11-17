"""Seeded builders the step goldens were made from, shared by the goldens' generator and the tests.

Everything is invented and reproducible from a seed: words come from ``build.WORDS`` (made up) plus whatever word
list a caller passes, voices are tones at fixed loudness levels, and voiceprints are seeded vectors with placeholder
names. Nothing here stores audio, a voiceprint or RTTM text in the repository: the tests build them in a temporary
folder when they run. Standard library only, except ``FakeExtractor.compute``, which returns what numpy reads.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
import struct
import wave
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fixtures.build import GRID, WORDS

PLACEHOLDER_NAMES = ("Zorvel", "Brimdal", "Quessa", "Toraph", "Velmine")


def _r3(x: float) -> float:
    return round(x + 0.0, 3)


# --------------------------------------------------------------------------- the language-share inputs
RATIOS = (0.0, 0.02, 0.05, 0.07, 0.075, 0.08, 0.085, 0.09, 0.12, 0.2, 0.5)


def lang_input(seed: int, function_words: Sequence[str]) -> dict[str, Any]:
    """A recogniser-shaped transcript whose words mix ``function_words`` into made-up words at a seeded ratio,
    with a seeded length around the 30-word floor, and capitals and punctuation that ``norm`` strips."""
    fw = sorted(function_words)
    rng = random.Random(seed)
    n = rng.randint(5, 80)
    ratio = rng.choice(RATIOS)
    picked = [(rng.choice(fw) if rng.random() < ratio else rng.choice(WORDS)) for _ in range(n)]
    sents: list[dict[str, Any]] = []
    t = 0.0
    i = 0
    while i < n:
        k = min(n - i, rng.randint(3, 12))
        toks: list[dict[str, Any]] = []
        for j, w in enumerate(picked[i:i + k]):
            if j == 0:
                w = w[:1].upper() + w[1:]
            if rng.random() < 0.1:
                w = w + ","
            frames = rng.randint(1, 4)
            toks.append({"text": " " + w, "start": _r3(t), "end": _r3(t + frames * GRID),
                         "duration": _r3(frames * GRID), "confidence": _r3(rng.uniform(0.6, 1.0))})
            t += frames * GRID
        toks.append({"text": ".", "start": _r3(t), "end": _r3(t), "duration": 0.0, "confidence": 0.99})
        sents.append({"text": "".join(x["text"] for x in toks), "start": toks[0]["start"], "end": toks[-1]["end"],
                      "duration": _r3(toks[-1]["end"] - toks[0]["start"]), "confidence": 0.9, "tokens": toks})
        t = round((t + rng.uniform(0.2, 1.0)) / GRID) * GRID
        i += k
    return {"text": "".join(s["text"] for s in sents).strip(), "sentences": sents}


# --------------------------------------------------------------------------- voices as tones
def tone_track_wav(segments: Iterable[tuple[float, float, float]], rate: int = 16000) -> bytes:
    """16-bit mono PCM wav of consecutive ``(seconds, frequency, amplitude)`` stretches (amplitude 0 = silence)."""
    frames = bytearray()
    t0 = 0
    for seconds, freq, amp in segments:
        n = int(round(seconds * rate))
        a = max(0.0, min(1.0, amp)) * 32767
        for i in range(n):
            v = math.sin(2 * math.pi * freq * (t0 + i) / rate) if amp else 0.0
            frames += struct.pack("<h", int(round(v * a)))
        t0 += n
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


# Loudness level of each placeholder voice; a voice's stretch of audio is a tone at this amplitude.
VOICE_AMPLITUDE = {"speaker_0": 0.3, "speaker_1": 0.5, "speaker_2": 0.7, "speaker_3": 0.9}


def voice_layout(seed: int, labels: Sequence[str], seconds: float, *, turn: tuple[float, float] = (2.0, 9.0),
                 overlap_rate: float = 0.15) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, float]]]:
    """Turns for ``labels`` over about ``seconds`` (start, end, label), and the tone stretches that voice them:
    each turn is its label's tone; a stretch two turns overlap is the louder voice."""
    rng = random.Random(seed)
    turns: list[tuple[float, float, str]] = []
    t, prev = 0.5, None
    while t < seconds:
        lab = rng.choice([x for x in labels if x != prev] or list(labels))
        d = rng.uniform(*turn)
        s = t
        if turns and rng.random() < overlap_rate:
            s = max(0.0, t - rng.uniform(0.3, 1.5))
        turns.append((round(s, 3), round(s + d, 3), lab))
        prev = lab
        t = s + d + rng.uniform(0.0, 0.8)
    end = max(e for _, e, _ in turns) + 0.5
    step = 0.02
    stretches: list[tuple[float, float, float]] = []
    x = 0.0
    while x < end - 1e-9:
        here = [lab for s, e, lab in turns if s <= x < e]
        amp = max((VOICE_AMPLITUDE[lab] for lab in here), default=0.0)
        freq = 220.0 + 110.0 * (sorted(VOICE_AMPLITUDE).index(here[0]) if here else 0)
        if stretches and stretches[-1][1] == freq and stretches[-1][2] == amp:
            stretches[-1] = (round(stretches[-1][0] + step, 6), freq, amp)
        else:
            stretches.append((step, freq, amp))
        x = round(x + step, 6)
    return turns, stretches


def rttm_of(turns: Iterable[tuple[float, float, str]], file_id: str) -> str:
    """The diarizer's RTTM text for (start, end, label) turns."""
    return "".join(f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n"
                   for s, e, lab in sorted(turns))


# --------------------------------------------------------------------------- a stand-in speaker model
class _Stream:
    def __init__(self) -> None:
        self.samples: Any = None

    def accept_waveform(self, rate: int, samples: Any) -> None:
        self.samples = samples

    def input_finished(self) -> None:
        pass


class FakeExtractor:
    """Stands in for the speaker-embedding model: a piece's vector is its loudness level's base vector plus noise
    seeded from the piece's own samples, so the same audio always gives the same vector and each placeholder voice
    (a loudness level) gives vectors close to each other."""

    def __init__(self, seed: int = 7, dim: int = 16, noise: float = 0.35) -> None:
        self.dim, self.noise = dim, noise
        rng = random.Random(seed)
        self.bases = {lvl: [rng.gauss(0, 1) for _ in range(dim)] for lvl in range(0, 21)}

    def create_stream(self) -> _Stream:
        return _Stream()

    def compute(self, st: _Stream) -> list[float]:
        import numpy as np
        x = np.asarray(st.samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
        level = int(math.floor(rms * 20 + 0.5))
        rng = random.Random(hashlib.sha256(x.tobytes()).digest())
        base = self.bases.get(level, self.bases[0])
        return [b + self.noise * rng.gauss(0, 1) for b in base]


def voiceprint(name: str, vecs: Sequence[Sequence[float]], model_sha256: str, *, sources: Sequence[str] | None = None,
               day: str = "2026-01-01") -> dict[str, Any]:
    """A voiceprint file's content for a placeholder ``name``."""
    srcs = list(sources or [f"sample {name.lower()}-{i}.wav" for i in range(len(vecs))])
    return {"name": name, "model_sha256": model_sha256,
            "vectors": [{"vec": [round(float(x), 6) for x in v], "source": s, "seconds": 20.0 + i, "lang": "EN",
                         "added": day} for i, (v, s) in enumerate(zip(vecs, srcs))],
            "updated": day}


def voice_vectors(extractor: FakeExtractor, level: int, n: int, seed: int) -> list[list[float]]:
    """``n`` vectors near the base vector of a loudness level (as enrolled sources of one voice would be)."""
    rng = random.Random(seed)
    return [[b + extractor.noise * rng.gauss(0, 1) for b in extractor.bases[level]] for _ in range(n)]


def level_of(label: str) -> int:
    """The FakeExtractor level a placeholder voice's tone falls in."""
    return int(math.floor(VOICE_AMPLITUDE[label] / math.sqrt(2) * 20 + 0.5))


def prints_for(extractor: FakeExtractor, who: Mapping[str, str], model_sha256: str, *, n: int = 2,
               seed: int = 11) -> list[dict[str, Any]]:
    """Voiceprints for placeholder names, each near one voice's level: ``who`` maps name -> label."""
    return [voiceprint(name, voice_vectors(extractor, level_of(label), n, seed + i), model_sha256)
            for i, (name, label) in enumerate(sorted(who.items()))]
