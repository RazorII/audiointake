"""Every fixed command line audiointake runs, built in one place.

Pure functions returning argv lists for ``proc.run``; nothing here starts a process. Each list is fixed argument
for argument: the transcripts' bytes depend on these exact options (and on there being no ``-map``, so ffmpeg's
own choice of audio stream applies), so change one only together with the outputs it is measured against.

Every path must be absolute. ffmpeg reads a relative name with a colon in it ("http:x.m4a", "concat:a|b") as a
protocol, and a path starting with "/" is always a plain file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]

SAMPLE_RATE = "16000"
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
DIAR_VARIANT = "offline"
DIAR_THRESHOLD = "0.4"
DIAR_WAV_SUFFIX = ".16k.wav"     # the diarizer names each RTTM line after the wav's stem: <stem>.16k


def _abs(p: PathLike, what: str) -> str:
    s = os.fspath(p)
    if not isinstance(s, str):
        raise TypeError(f"{what} must be a str path, not {type(s).__name__}")
    if not os.path.isabs(s):
        raise ValueError(f"{what} must be an absolute path: {s!r}")
    if "\x00" in s:
        raise ValueError(f"{what} contains a NUL byte")
    return s


def ensemble_wav(ffmpeg: PathLike, audio: PathLike, wav: PathLike) -> list[str]:
    """The ASR ensemble's 16 kHz mono wav (the Thai screen and the vote read it)."""
    return [_abs(ffmpeg, "ffmpeg"), "-nostdin", "-y", "-v", "error", "-vn", "-i", _abs(audio, "audio"),
            "-ar", SAMPLE_RATE, "-ac", "1", _abs(wav, "wav")]


def loudnorm_wav(ffmpeg: PathLike, audio: PathLike, wav: PathLike) -> list[str]:
    """The loudness-normalised 16 kHz mono wav the Whisper vote reads."""
    return [_abs(ffmpeg, "ffmpeg"), "-nostdin", "-y", "-v", "error", "-vn", "-i", _abs(audio, "audio"),
            "-ar", SAMPLE_RATE, "-ac", "1", "-af", LOUDNORM, _abs(wav, "wav")]


def diar_wav(ffmpeg: PathLike, audio: PathLike, wav: PathLike) -> list[str]:
    """The diarizer's 16 kHz mono wav, ``<stem>.16k.wav`` (run with stdout and stderr discarded).

    No ``-v error`` here, as the diarization step has always run it; the name matters because the RTTM's
    file id is the wav's stem.
    """
    w = _abs(wav, "wav")
    if not w.endswith(DIAR_WAV_SUFFIX):
        raise ValueError(f"the diarizer's wav must be named <stem>{DIAR_WAV_SUFFIX}: {w!r}")
    return [_abs(ffmpeg, "ffmpeg"), "-nostdin", "-y", "-vn", "-i", _abs(audio, "audio"),
            "-ar", SAMPLE_RATE, "-ac", "1", w]


def clip_wav(ffmpeg: PathLike, clip: PathLike, wav: PathLike) -> list[str]:
    """An enrolment clip as 16 kHz mono wav."""
    return [_abs(ffmpeg, "ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", _abs(clip, "clip"),
            "-ar", SAMPLE_RATE, "-ac", "1", _abs(wav, "wav")]


def channels_probe(ffprobe: PathLike, audio: PathLike) -> list[str]:
    """One line per audio stream with its channel count (the spatial rule reads it)."""
    return [_abs(ffprobe, "ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries", "stream=channels",
            "-of", "csv=p=0", _abs(audio, "audio")]


def duration_probe(ffprobe: PathLike, audio: PathLike) -> list[str]:
    """The container's duration in seconds, one line."""
    return [_abs(ffprobe, "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
            _abs(audio, "audio")]


def ffmpeg_info(ffmpeg: PathLike, audio: PathLike) -> list[str]:
    """Without ffprobe: ffmpeg prints the Duration and Stream lines on stderr (and exits 1, having no output)."""
    return [_abs(ffmpeg, "ffmpeg"), "-hide_banner", "-nostdin", "-i", _abs(audio, "audio")]


def decode_check(ffmpeg: PathLike, audio: PathLike) -> list[str]:
    """Decode the recording's audio to nowhere, stopping at the first error. Run only after the speech model failed,
    to tell an unreadable recording (``decode_failed``) from a failed model (``asr_failed``); no output depends
    on it."""
    return [_abs(ffmpeg, "ffmpeg"), "-nostdin", "-v", "error", "-xerror", "-i", _abs(audio, "audio"),
            "-vn", "-sn", "-dn", "-f", "null", "-"]


def diarize(fluidaudiocli: PathLike, wav: PathLike, rttm: PathLike, *, models: PathLike) -> list[str]:
    """Nemotron diarization, offline profile, threshold 0.4, from the pinned model folder (never a download)."""
    return [_abs(fluidaudiocli, "fluidaudiocli"), "nemotron3-diarize", _abs(wav, "wav"), "--variant", DIAR_VARIANT,
            "--threshold", DIAR_THRESHOLD, "--output", _abs(rttm, "rttm"), "--models", _abs(models, "models")]


def tone_wav(ffmpeg: PathLike, wav: PathLike, *, seconds: float = 3.0, kind: str = "sine") -> list[str]:
    """Generated 16 kHz mono audio for smoke tests: a 440 Hz sine, pink noise or silence. Not tied to any output."""
    sources = {
        "sine": f"sine=frequency=440:sample_rate=16000:duration={seconds:g}",
        "noise": f"anoisesrc=color=pink:sample_rate=16000:duration={seconds:g}:seed=1",
        "silence": f"anullsrc=channel_layout=mono:sample_rate=16000:duration={seconds:g}",
    }
    if kind not in sources:
        raise ValueError(f"kind must be one of {sorted(sources)}")
    if not 0 < seconds <= 600:
        raise ValueError("seconds must be in (0, 600]")
    return [_abs(ffmpeg, "ffmpeg"), "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", sources[kind],
            "-ar", SAMPLE_RATE, "-ac", "1", _abs(wav, "wav")]



def spatial(channels_stdout: str) -> str:
    """"yes" when any line channels_probe printed is exactly "4" (a 4-channel stream), else "no".

    A failed probe is "no": pass "" for it.
    """
    return "yes" if any(line == "4" for line in channels_stdout.split("\n")) else "no"


def diar_wav_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>.16k.wav``."""
    return folder / f"{stem}{DIAR_WAV_SUFFIX}"
