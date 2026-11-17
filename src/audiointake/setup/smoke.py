"""Setup's smoke tests: does each verified model load and run on generated audio, and the canary's check.

A smoke is a 3 s generated sine (``argv.tone_wav``) run through the model's own program in a child, exactly as a
run calls it, offline: ``parakeet-mlx`` for the two Parakeet models, ``mlx_whisper`` for Whisper and the
converted Typhoon, and this module's own ``campplus`` worker for the speaker-embedding model (``python -I -m
audiointake.setup.smoke campplus ONNX WAV``). Nemotron is smoke-tested with the FluidAudio build
(``fluidaudio.smoke``). A smoke passes when the child exits 0 within its cap and leaves its output.

``canary_check`` reads the transcript JSON the canary run wrote and passes when every ``must_find`` word appears
in it and the RTTM it carries names at least ``min_speakers`` speakers. It reports counts, never text.
"""
from __future__ import annotations

import json
import re
import struct
import sys
import tempfile
import time
from pathlib import Path

from .. import argv, proc
from .manifest import Canary, Smoke

CAP_S = {"parakeet": 600.0, "whisper": 900.0, "whisper_lid": 900.0, "typhoon": 900.0, "campplus": 300.0}


def wav_header(n_bytes: int, *, rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """A canonical 44-byte PCM WAV header for ``n_bytes`` of sample data."""
    block = channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * block, block, bits)
            + b"data" + struct.pack("<I", n_bytes))


def _command(kind: str, model: Path, wav: Path, out: Path, scripts: Path) -> tuple[list[str], Path]:
    if kind == "parakeet":
        return ([str(scripts / "parakeet-mlx"), str(wav), "--model", str(model), "--output-format", "json",
                 "--output-dir", str(out), "--output-template", "smoke"], out / "smoke.json")
    if kind in ("whisper", "whisper_lid", "typhoon"):
        lang = "th" if kind == "typhoon" else "hu"
        return ([str(scripts / "mlx_whisper"), str(wav), "--model", str(model), "--language", lang,
                 "--output-format", "json", "--output-dir", str(out), "--output-name", "smoke"], out / "smoke.json")
    if kind == "campplus":
        return ([sys.executable, "-I", "-B", "-m", "audiointake.setup.smoke", "campplus", str(model), str(wav),
                 str(out / "smoke.json")], out / "smoke.json")
    raise ValueError(f"no smoke for {kind!r}")


def model_smoke(smoke: Smoke, model: Path, ffmpeg: Path, scripts: Path) -> dict:
    """``{ok, seconds, code?}`` for one model on generated audio, offline."""
    from . import runtime, store

    allowed = store.programs(ffmpeg, scripts / "parakeet-mlx", scripts / "mlx_whisper")
    with tempfile.TemporaryDirectory(prefix="audiointake-smoke-") as tmp:
        t = Path(tmp)
        wav = t / "tone.wav"
        env = runtime.run_env(t, ffmpeg)
        g = proc.run(argv.tone_wav(ffmpeg, wav, seconds=smoke.seconds, kind="sine"), env=env, timeout=120,
                     allowed=allowed, stdout="discard")
        if not g.ok:
            return {"ok": False, "seconds": round(g.seconds, 2), "code": "ffmpeg_failed"}
        cmd, expect = _command(smoke.kind, model, wav, t, scripts)
        t0 = time.monotonic()
        r = proc.run(cmd, env=env, timeout=CAP_S.get(smoke.kind, 600.0), allowed=allowed, stdout="discard")
        sec = round(time.monotonic() - t0, 2)
        if r.timed_out:
            return {"ok": False, "seconds": sec, "code": "timeout"}
        if r.returncode != 0 or not expect.is_file():
            return {"ok": False, "seconds": sec, "code": f"exit_{r.returncode}"}
        return {"ok": True, "seconds": sec}


# --------------------------------------------------------------------------- the canary
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def transcript_words(doc: dict) -> set[str]:
    """Every word of a transcript JSON's turns (or, without turns, its ASR sentences), lowercased."""
    texts: list[str] = []
    for t in doc.get("turns") or []:
        if isinstance(t, dict) and isinstance(t.get("text"), str):
            texts.append(t["text"])
    if not texts:
        for s in (doc.get("asr") or {}).get("sentences") or []:
            if isinstance(s, dict) and isinstance(s.get("text"), str):
                texts.append(s["text"])
            elif isinstance(s, str):
                texts.append(s)
    return {w.lower() for text in texts for w in _WORD.findall(text)}


def rttm_speakers(text: str | None) -> int:
    labels = set()
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            labels.add(parts[7])
    return len(labels)


def canary_check(outdir: Path, clip: Path, c: Canary, *, speakers: bool) -> dict:
    """``{ok, found, of, speakers, min_speakers}`` from the canary run's transcript JSON (counts, never text)."""
    stem = clip.name.rsplit(".", 1)[0]
    p = Path(outdir) / f"{stem} (transcript).json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": False, "found": 0, "of": len(c.must_find), "speakers": None, "min_speakers": c.min_speakers,
                "code": "no_transcript"}
    words = transcript_words(doc)
    found = sum(1 for w in c.must_find if w.lower() in words)
    n = rttm_speakers(doc.get("rttm")) if speakers else None
    ok = found == len(c.must_find) and (n is None or n >= c.min_speakers)
    return {"ok": ok, "found": found, "of": len(c.must_find), "speakers": n, "min_speakers": c.min_speakers}


# --------------------------------------------------------------------------- the CAM++ worker (a child)
def campplus(onnx: str, wav: str, out: str) -> int:
    import numpy as np
    import sherpa_onnx
    import soundfile as sf

    samples, rate = sf.read(wav, dtype="float32", always_2d=False)
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=onnx,
                                                                                           num_threads=1))
    s = ex.create_stream()
    s.accept_waveform(rate, np.asarray(samples, dtype=np.float32))
    s.input_finished()
    if not ex.is_ready(s):
        return 1
    v = np.array(ex.compute(s))
    Path(out).write_text(json.dumps({"dim": int(v.shape[0]), "finite": bool(np.isfinite(v).all())}))
    return 0 if v.shape[0] > 0 and np.isfinite(v).all() else 1


def main(args: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if args is None else args)
    if len(a) == 4 and a[0] == "campplus":
        return campplus(a[1], a[2], a[3])
    print("usage: python -m audiointake.setup.smoke campplus ONNX WAV OUT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
