#!/usr/bin/env python3
"""Build the path fixtures: short recordings, made from public audio, that each send a transcription down one route.

  fixtures.py build OUTDIR --fleurs WORKDIR --ami PATH [--ami-start SECONDS] [--ffmpeg BIN]
  fixtures.py check OUTDIR                      print each fixture's size and SHA-256

WORKDIR is bench/fleurs.py's prep folder (fl/<lang>/<id>.wav), or tools/fetch_test_audio.py's <root>/fleurs. PATH
is the AMI meeting ES2004a (ES2004a.Array1-01.wav, 16 kHz mono, as bench/ami.py fetch saves it, or a folder holding
it), or an excerpt of it that starts --ami-start seconds into the meeting and covers 825-915 s (the test-audio tool's
ami/ES2004a-825-915.wav with --ami-start 825). Fixtures are named by what they exercise:

  vote.wav          about two minutes of FLEURS clips in a language with a pack whose vote route reads it better
                    than the core models (over 30 words, a function-word share over the route's floor), so the
                    Whisper vote runs
  turns.wav         four FLEURS clips of an unsegmented-script language Parakeet doesn't read, each followed by 10 s
                    of AMI English, so the language screen finds those turns and the turn route transcribes them
  gapfill-de.wav    five German FLEURS clips: the core route (Parakeet with gap-fill)
  gapfill-es.wav    five Spanish FLEURS clips: the core route
  silence.wav       20 s of digital silence: an empty diarization
  aac-m4a.m4a       gapfill-de.wav re-encoded to AAC in an .m4a, as phones and Voice Memos save audio
  spatial-qta.qta   a QuickTime file with two audio streams of AMI ES2004a, the shape of a spatial voice memo: stereo
                    AAC of 825-885 s first, then a 4-channel AAC stream of other speech (885-915 s); exercises the
                    spatial flag and ffmpeg's own choice of stream (no -map anywhere), which shows in the transcript
                    and the diarization because the two streams hold different speech

Clips are joined in a fixed order with 0.5 s of silence between them, as 16 kHz mono 16-bit PCM written by this
script (each clip decoded by ffmpeg first), so the WAV fixtures are byte-reproducible; the AAC ones are made with
ffmpeg's built-in encoder and +bitexact, reproducible with the same ffmpeg. Audio is never committed: rebuild it.

AMI Meeting Corpus: CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/. FLEURS: CC BY 4.0, google/fleurs.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import wave
from pathlib import Path

sys.dont_write_bytecode = True   # no __pycache__ beside the bench scripts, wherever the clone lives
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleurs  # noqa: E402

RATE = 16000
GAP = b"\0\0" * (RATE // 2)                       # 0.5 s of 16-bit silence
VOTE_LANG = "hu_hu"                 # a language whose pack adds the Whisper vote
TURNS_LANG = "th_th"                # an unsegmented script the core models don't read
VOTE_SECONDS = 120.0
AMI_FILE = "ES2004a.Array1-01.wav"
AMI_TURNS = (825.0, 840.0, 855.0, 870.0)          # four 10 s windows of AMI ES2004a, one after each FLEURS clip
AMI_SPATIAL = (825.0, 885.0)                       # the first (stereo) stream
AMI_SPATIAL_SECOND = (885.0, 915.0)                # the second (4-channel) stream: different speech
FIXTURES = ("vote.wav", "turns.wav", "gapfill-de.wav", "gapfill-es.wav", "silence.wav", "aac-m4a.m4a", "spatial-qta.qta")


def pcm(ffmpeg: str, src: Path) -> bytes:
    """A clip decoded to 16 kHz mono signed 16-bit little-endian PCM."""
    r = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(src), "-ar", str(RATE), "-ac", "1",
                        "-f", "s16le", "-acodec", "pcm_s16le", "-"], capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise SystemExit(f"fixtures: ffmpeg couldn't decode {src.name}")
    return r.stdout


def write_wav(path: Path, frames: bytes, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames)


def ami_window(ami: Path, start: float, end: float, offset: float = 0.0) -> bytes:
    """Seconds start-end of the meeting, from a file that begins offset seconds into it."""
    with wave.open(str(ami), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit("fixtures: the AMI file isn't 16 kHz mono 16-bit")
        if start < offset or (end - offset) * RATE > w.getnframes():
            raise SystemExit(f"fixtures: the AMI file doesn't cover {start:g}-{end:g} s of the meeting")
        w.setpos(int(round((start - offset) * RATE)))
        return w.readframes(int((end - start) * RATE))


def interleave(mono: bytes, channels: int) -> bytes:
    out = bytearray(len(mono) * channels)
    for k in range(channels):
        for i in range(0, len(mono), 2):
            j = i * channels + 2 * k
            out[j:j + 2] = mono[i:i + 2]
    return bytes(out)


def clip(wd: Path, lang: str, i: str) -> Path:
    p = wd / "fl" / lang / f"{i}.wav"
    if not p.is_file():
        raise SystemExit(f"fixtures: {lang}/{i}.wav is missing; run fleurs.py prep {wd} {lang}")
    return p


def build(out: Path, wd: Path, ami: Path, ffmpeg: str, offset: float = 0.0) -> int:
    out.mkdir(parents=True, exist_ok=True)
    if ami.is_dir():
        ami = ami / AMI_FILE
    if not ami.is_file():
        raise SystemExit(f"fixtures: {ami} is missing; run ami.py fetch or tools/fetch_test_audio.py")
    # vote: clips in the fixed id order until two minutes
    parts, secs = [], 0.0
    for i in fleurs.IDS[VOTE_LANG]:
        b = pcm(ffmpeg, clip(wd, VOTE_LANG, i))
        parts.append(b)
        secs += len(b) / 2 / RATE
        if secs >= VOTE_SECONDS:
            break
    write_wav(out / "vote.wav", GAP.join(parts))
    # turns: FLEURS clip, then 10 s of AMI English, four times
    parts = []
    for i, start in zip(fleurs.IDS[TURNS_LANG][:4], AMI_TURNS):
        parts += [pcm(ffmpeg, clip(wd, TURNS_LANG, i)), ami_window(ami, start, start + 10.0, offset)]
    write_wav(out / "turns.wav", GAP.join(parts))
    for name, lang in (("gapfill-de.wav", "de_de"), ("gapfill-es.wav", "es_419")):
        write_wav(out / name, GAP.join(pcm(ffmpeg, clip(wd, lang, i)) for i in fleurs.IDS[lang][:5]))
    write_wav(out / "silence.wav", b"\0\0" * (RATE * 20))
    run = lambda *a: subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", *map(str, a)], check=True,
                                    stdin=subprocess.DEVNULL)
    bitexact = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", "-c:a", "aac", "-b:a", "96k"]
    run("-i", out / "gapfill-de.wav", *bitexact, out / "aac-m4a.m4a")
    tmp = out / ".spatial-parts"
    tmp.mkdir(exist_ok=True)
    try:
        write_wav(tmp / "stereo.wav", interleave(ami_window(ami, *AMI_SPATIAL, offset), 2), channels=2)
        write_wav(tmp / "quad.wav", interleave(ami_window(ami, *AMI_SPATIAL_SECOND, offset), 4), channels=4)
        run("-i", tmp / "stereo.wav", "-i", tmp / "quad.wav", "-map", "0:a", "-map", "1:a",
            "-filter:a:1", "aformat=channel_layouts=4.0", *bitexact, "-f", "mov", out / "spatial-qta.qta")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return check(out)


def check(out: Path) -> int:
    missing = 0
    for name in FIXTURES:
        p = out / name
        if not p.is_file():
            print(f"{name}: missing"); missing += 1; continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {p.stat().st_size} bytes, sha256 {h}")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fixtures.py", description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("outdir", type=Path)
    b.add_argument("--fleurs", type=Path, required=True)
    b.add_argument("--ami", type=Path, required=True)
    b.add_argument("--ami-start", type=float, default=0.0)
    b.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    c = sp.add_parser("check")
    c.add_argument("outdir", type=Path)
    a = ap.parse_args(argv)
    return build(a.outdir, a.fleurs, a.ami, a.ffmpeg, a.ami_start) if a.cmd == "build" else check(a.outdir)


if __name__ == "__main__":
    sys.exit(main())
