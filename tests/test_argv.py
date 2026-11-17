"""argv.py: every fixed command line, argument for argument, and its refusals."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from audiointake import argv

FF, FP, FA = "/x/bin/ffmpeg", "/x/bin/ffprobe", "/x/bin/fluidaudiocli"
AUDIO, WAV = "/in/Team sync.m4a", "/w/a.16k.wav"


def test_ensemble_wav():
    assert argv.ensemble_wav(FF, AUDIO, WAV) == [
        FF, "-nostdin", "-y", "-v", "error", "-vn", "-i", AUDIO, "-ar", "16000", "-ac", "1", WAV]


def test_loudnorm_wav():
    assert argv.loudnorm_wav(FF, AUDIO, "/w/a.norm.wav") == [
        FF, "-nostdin", "-y", "-v", "error", "-vn", "-i", AUDIO, "-ar", "16000", "-ac", "1",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "/w/a.norm.wav"]


def test_diar_wav():
    assert argv.diar_wav(FF, AUDIO, "/w/Team sync.16k.wav") == [
        FF, "-nostdin", "-y", "-vn", "-i", AUDIO, "-ar", "16000", "-ac", "1", "/w/Team sync.16k.wav"]


def test_diar_wav_needs_the_16k_name():
    with pytest.raises(ValueError, match=r"\.16k\.wav"):
        argv.diar_wav(FF, AUDIO, "/w/Team sync.wav")


def test_clip_wav():
    assert argv.clip_wav(FF, "/c/ana.m4a", "/w/clip.wav") == [
        FF, "-nostdin", "-v", "error", "-y", "-i", "/c/ana.m4a", "-ar", "16000", "-ac", "1", "/w/clip.wav"]


def test_channels_probe():
    assert argv.channels_probe(FP, AUDIO) == [
        FP, "-v", "error", "-select_streams", "a", "-show_entries", "stream=channels", "-of", "csv=p=0", AUDIO]


def test_duration_probe():
    assert argv.duration_probe(FP, AUDIO) == [
        FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", AUDIO]


def test_ffmpeg_info():
    assert argv.ffmpeg_info(FF, AUDIO) == [FF, "-hide_banner", "-nostdin", "-i", AUDIO]


def test_diarize():
    assert argv.diarize(FA, WAV, "/w/a.diar.rttm", models="/m/nemotron") == [
        FA, "nemotron3-diarize", WAV, "--variant", "offline", "--threshold", "0.4", "--output", "/w/a.diar.rttm",
        "--models", "/m/nemotron"]


def test_diarize_needs_the_model_folder():
    with pytest.raises(TypeError):
        argv.diarize(FA, WAV, "/w/a.diar.rttm")          # type: ignore[call-arg]


def test_paths_are_str_and_pathlike_alike():
    assert argv.ensemble_wav(Path(FF), Path(AUDIO), Path(WAV)) == argv.ensemble_wav(FF, AUDIO, WAV)


@pytest.mark.parametrize("bad", ["relative.m4a", "http:x.m4a", "concat:a|b", "~/x.m4a", ""])
def test_relative_paths_are_refused(bad):
    with pytest.raises(ValueError, match="absolute"):
        argv.ensemble_wav(FF, bad, WAV)


def test_relative_program_is_refused():
    with pytest.raises(ValueError, match="absolute"):
        argv.channels_probe("ffprobe", AUDIO)


def test_nul_is_refused():
    with pytest.raises(ValueError, match="NUL"):
        argv.clip_wav(FF, "/c/a\x00b.m4a", "/w/clip.wav")


def test_bytes_paths_are_refused():
    with pytest.raises(TypeError):
        argv.ensemble_wav(FF, b"/in/a.m4a", WAV)      # type: ignore[arg-type]


@pytest.mark.parametrize("out,want", [
    ("2\n4\n", "yes"), ("4", "yes"), ("4\n", "yes"), ("2\n", "no"), ("", "no"), ("14\n", "no"), ("4 \n", "no"),
    ("1\n2\n", "no"), ("40", "no"), ("2\n4\n6\n", "yes"),
])
def test_spatial_rule(out, want):
    assert argv.spatial(out) == want


def test_diar_wav_path():
    assert argv.diar_wav_path(Path("/w"), "Team sync") == Path("/w/Team sync.16k.wav")


def test_tone_wav_args():
    a = argv.tone_wav(FF, "/w/t.wav", seconds=3, kind="sine")
    assert a[:7] == [FF, "-nostdin", "-y", "-v", "error", "-f", "lavfi"]
    assert a[-5:] == ["-ar", "16000", "-ac", "1", "/w/t.wav"]
    with pytest.raises(ValueError):
        argv.tone_wav(FF, "/w/t.wav", kind="speech")
    with pytest.raises(ValueError):
        argv.tone_wav(FF, "/w/t.wav", seconds=0)


@pytest.mark.parametrize("kind", ["sine", "noise", "silence"])
def test_tone_wav_runs_with_real_ffmpeg(tmp_path, kind):
    ff = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
                                         if Path(p).exists()), None)
    if ff is None:
        pytest.skip("no ffmpeg on this machine")
    out = tmp_path / f"{kind}.wav"
    r = subprocess.run(argv.tone_wav(ff, out, seconds=1.5, kind=kind), capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert abs(w.getnframes() - 24000) <= 160
