"""Output names and formats, the digests, the diarmeta line, the sealed transcript JSON, the .md renderer, the probe."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from audiointake import outputs, probe, render
from audiointake.contracts import Turn
from audiointake.errors import UsageError

from fakes import audio, use_fakes


# --------------------------------------------------------------------------- formats and names
def test_formats_default_and_order():
    assert outputs.parse_formats(None) == ("md", "json")
    assert outputs.parse_formats("sidecars,txt, md") == ("md", "txt", "sidecars")


@pytest.mark.parametrize("value", ["", ",", "md,pdf", "srt"])
def test_formats_refuse_unknown(value):
    with pytest.raises(UsageError) as e:
        outputs.parse_formats(value)
    assert e.value.code == "bad_arguments" and e.value.exit == 2


def test_names():
    assert outputs.name_for("Team sync", "md") == "Team sync (transcript).md"
    assert outputs.names("a", ("md", "json", "txt")) == ["a (transcript).md", "a (transcript).json",
                                                         "a (transcript).txt"]
    assert outputs.names("a", ("sidecars",)) == ["a.txt", "a.asr.json", "a.diar.rttm", "a.speakers.json",
                                                 "a.diarmeta.txt"]


def test_options_digest():
    d = outputs.options_digest(langs=["en", "de"], names=True, speakers_off=False)
    assert d == outputs.options_digest(langs=["de", "en", "en"], names=True, speakers_off=False)
    assert d != outputs.options_digest(langs=["de"], names=True, speakers_off=False)
    assert d != outputs.options_digest(langs=["en", "de"], names=False, speakers_off=False)
    assert d != outputs.options_digest(langs=["en", "de"], names=True, speakers_off=True)
    assert len(d) == 64


def test_diarmeta_line_is_the_classic_one():
    line = outputs.diarmeta(speakers=3, runs=1, fluidaudio_tag="v0.18.0", spatial="no",
                            asr_tag="parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill", named="Ana Doe,Bo Roe")
    assert line == ("speakers=3 runs=1 threshold=0.4 engine=fluidaudio-v0.18.0-nemotron3-offline spatial=no "
                    "asr=parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill named=Ana Doe,Bo Roe\n")
    assert outputs.diarmeta(speakers=None, runs=None, fluidaudio_tag="v0.18.0", spatial="yes", asr_tag="x",
                            named="-").startswith("speakers=? runs=- threshold=0.4 ")


def test_plain_text_is_the_ensembles_line():
    asr = {"sentences": [{"text": " Hello there."}, {"text": " Second one. "}, {"text": ""}]}
    assert outputs.plain_text(asr) == "Hello there. Second one.\n"
    assert outputs.plain_text({"sentences": []}) == "\n"
    assert outputs.word_count(asr) == (2, 4)


# --------------------------------------------------------------------------- the seal
def test_transcript_seal_round_trip_and_edits():
    data = outputs.transcript_bytes({"record_id": "r-20260101-00000000", "turns": [{"text": 'x "seal": "y"'}]})
    assert data.endswith(b"\n")
    assert outputs.seal_ok(data)
    doc = outputs.parse_transcript(data)
    assert doc["format"] == "audiointake-transcript" and doc["version"] == 1 and len(doc["seal"]) == 64
    assert list(doc)[:3] == ["format", "version", "seal"]
    edited = data.replace(b"r-20260101", b"r-20260102")
    assert not outputs.seal_ok(edited)
    assert not outputs.seal_ok(b'{"format": "audiointake-transcript"}')


def test_parse_transcript_refuses_other_files():
    assert outputs.parse_transcript(b"not json") is None
    assert outputs.parse_transcript(json.dumps({"format": "other", "version": 1}).encode()) is None
    assert outputs.parse_transcript(json.dumps({"format": "audiointake-transcript", "version": 2}).encode()) is None


# --------------------------------------------------------------------------- the renderer
def test_clock():
    assert render.clock(192.9, hours=False) == "03:12"
    assert render.clock(3725, hours=True) == "1:02:05"
    assert render.clock(0, hours=True) == "0:00:00"


def test_dialogue_switches_to_hours_past_an_hour():
    turns = [Turn("speaker_0", "Speaker 1", 5.0, 9.0, "One."), Turn("speaker_1", "Ana", 3700.2, 3705, "Two.")]
    assert render.dialogue(turns, duration_s=3800) == "**Speaker 1** [0:00:05]\nOne.\n\n**Ana** [1:01:40]\nTwo.\n"
    assert render.dialogue(turns[:1], duration_s=3600) == "**Speaker 1** [00:05]\nOne.\n"
    assert render.dialogue([{"label": None, "display": "", "start": 61, "end": 62, "text": "Plain."}]) == \
        "[01:01]\nPlain.\n"
    assert render.dialogue([]) == ""


def test_paragraphs_break_at_two_second_pauses():
    sents = [{"start": 0, "end": 2, "text": " A."}, {"start": 2.5, "end": 4, "text": " B."},
             {"start": 6.0, "end": 7, "text": " C."}, {"start": 7.1, "end": 8, "text": ""}]
    ps = render.paragraphs(sents)
    assert [(p.start, p.text) for p in ps] == [(0.0, "A. B."), (6.0, "C.")]
    assert all(p.label is None and p.display == "" for p in ps)


def test_header_variants():
    t = [Turn("speaker_0", "Ana", 0, 1, "Hi.")]
    md = render.md_file(title="Team sync", recorded_at="2026-11-16T09:30:00", duration_s=2473.2, turns=t,
                              speakers=(3, 1))
    assert md == "# Team sync\n\n2026-11-16 · 0:41:13 · 3 speakers (1 named)\n\n**Ana** [00:00]\nHi.\n"
    md = render.md_file(title="x", recorded_at=None, duration_s=30, turns=t, speakers=None)
    assert md.splitlines()[2] == "0:00:30"
    assert render.speaker_part(1, 0) == "1 speaker (0 named)"


# --------------------------------------------------------------------------- the probe
def test_layout_table_is_closed():
    assert probe.layout_channels("aac (LC), 48000 Hz, 4.0, fltp") == 4
    assert probe.layout_channels("aac (LC), 48000 Hz, quad, fltp") == 4
    assert probe.layout_channels("pcm_s16le, 16000 Hz, mono, s16") == 1
    assert probe.layout_channels("opus, 48000 Hz, 3 channels, fltp") == 3
    assert probe.layout_channels("aac, 48000 Hz, 5.1(side), fltp") == 6
    assert probe.layout_channels("pcm_s16le, 16000 Hz, 3.1, s16") == 4
    assert probe.layout_channels("pcm_s16le, 16000 Hz, 4 channels (FL+FR+LFE+BC), s16") == 4   # a custom layout
    assert probe.layout_channels("aac, 48000 Hz, ambisonic 1, fltp") is None
    assert probe.layout_channels("aac") is None


def test_parse_ffmpeg_info():
    text = ("Input #0, mov, from 'x':\n  Metadata:\n    creation_time   : 2026-11-16T08:30:00.000000Z\n"
            "  Duration: 00:41:13.20, start: 0.000000, bitrate: 64 kb/s\n"
            "  Stream #0:0[0x1](und): Audio: aac (LC), 48000 Hz, stereo, fltp, 64 kb/s (default)\n"
            "  Stream #0:1[0x2](und): Audio: aac (LC), 48000 Hz, 4.0, fltp\n"
            "  Stream #0:2[0x3](und): Video: h264\n")
    info = probe.parse_ffmpeg_info(text)
    assert info["has_audio"] and info["channels"] == [2, 4] and info["spatial"] == "yes"
    assert info["duration_s"] == pytest.approx(2473.2)
    assert info["creation_time"] == "2026-11-16T08:30:00.000000Z"
    assert probe.parse_ffmpeg_info("nothing")["has_audio"] is False


def test_recorded_at_precedence(tmp_path):
    p = tmp_path / "20261116 093000 Team.m4a"
    p.write_text("x")
    assert probe.recorded_at(p, "2025-01-01T00:00:00Z") == {"recorded_at": "2026-11-16T09:30:00",
                                                            "recorded_at_source": "filename"}
    q = tmp_path / "Team.m4a"
    q.write_text("x")
    r = probe.recorded_at(q, "2026-03-04T05:06:07")
    assert r["recorded_at_source"] == "metadata"
    assert r == {"recorded_at": probe.parse_creation_time("2026-03-04T05:06:07"), "recorded_at_source": "metadata"}
    os.utime(q, (time.time(), 1_700_000_000))
    r = probe.recorded_at(q, "1970-01-01T00:00:00Z")
    assert r["recorded_at_source"] == "file_date"
    assert r["recorded_at"] == time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(1_700_000_000))


def test_creation_time_utc_becomes_local():
    from datetime import datetime, timezone
    want = datetime(2026, 11, 16, 8, 30, tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    assert probe.parse_creation_time("2026-11-16T08:30:00.000000Z") == want
    assert probe.parse_creation_time("garbage") is None


CASES = [
    {"channels": [1], "duration": 75.5},
    {"channels": [2], "duration": 12.0, "creation_time": "2026-11-16T08:30:00.000000Z"},
    {"channels": [2, 4], "duration": 600.25},
    {"channels": [6], "duration": 3700.0},
    {"channels": [4], "duration": 30.0},
]


@pytest.mark.parametrize("desc", CASES)
def test_both_probe_paths_agree(monkeypatch, tmp_path, desc):
    with_probe = use_fakes(monkeypatch, tmp_path)
    rt1, _ = with_probe.resolve()
    from fakes.stub_setup import FakeSetup
    rt2, _ = FakeSetup(tmp_path / "noprobe", ffprobe=False).resolve()
    assert rt2.tools.ffprobe is None
    a = audio(tmp_path / "clip.m4a", **desc)
    one = probe.probe(a, rt1.tools, env=rt1.env, workdir=rt1.work_root)
    two = probe.probe(a, rt2.tools, env=rt2.env, workdir=rt2.work_root)
    assert one == two
    assert one["spatial"] == ("yes" if 4 in desc["channels"] else "no")
    assert one["channels"] == desc["channels"]
    assert list(rt2.work_root.iterdir()) == []                     # the stderr file is gone


def test_probe_without_audio(monkeypatch, tmp_path):
    fs = use_fakes(monkeypatch, tmp_path)
    rt, _ = fs.resolve()
    p = tmp_path / "notes.m4a"
    p.write_text("these are not sounds")
    assert probe.probe(p, rt.tools, env=rt.env)["has_audio"] is False


# the probe against the real ffmpeg 9 and ffprobe: layouts the fake can't show (3.1, a custom one, two streams)
HB = Path("/opt/homebrew/bin")
REAL_LAYOUTS = [("l31.wav", ["3.1"]), ("quad.wav", ["quad"]), ("l40.wav", ["4.0"]), ("custom.wav", ["FL+FR+LFE+BC"]),
                ("side.m4a", ["quad(side)"]), ("l51.wav", ["5.1"]), ("l60.wav", ["6.0"]), ("mono.opus", ["mono"]),
                ("two.mp4", ["stereo", "quad"])]


@pytest.mark.skipif(not ((HB / "ffmpeg").exists() and (HB / "ffprobe").exists()),
                    reason="needs Homebrew's ffmpeg and ffprobe")
@pytest.mark.parametrize("name,layouts", REAL_LAYOUTS, ids=[n for n, _ in REAL_LAYOUTS])
def test_both_probe_paths_agree_on_real_files(tmp_path, name, layouts):
    import subprocess
    import sys

    from audiointake.contracts import Tools
    f = tmp_path / name
    cmd = [str(HB / "ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for lay in layouts:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r=16000:cl={lay}"]
    for i in range(len(layouts)):
        cmd += ["-map", str(i)]
    cmd += ["-t", "3"] + (["-c:a", "aac"] if name.endswith(".mp4") else []) + [str(f)]
    subprocess.run(cmd, check=True, timeout=60, stdin=subprocess.DEVNULL)
    def tools(ffprobe):
        return Tools(ffmpeg=HB / "ffmpeg", ffprobe=(HB / "ffprobe") if ffprobe else None, python=Path(sys.executable),
                     parakeet_cli=tmp_path / "none", whisper_cli=tmp_path / "none", fluidaudiocli=None,
                     fluidaudio_tag="v0.18.0")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8"}
    work = tmp_path / "work"
    work.mkdir()
    one = probe.probe(f, tools(True), env=env, workdir=work)
    two = probe.probe(f, tools(False), env=env, workdir=work)
    assert one["has_audio"] and two["has_audio"]
    assert one["channels"] == two["channels"] and None not in two["channels"]
    assert one["spatial"] == two["spatial"] == ("yes" if 4 in one["channels"] else "no")
    assert abs(one["duration_s"] - two["duration_s"]) <= 0.0051              # ffmpeg -i shows 10 ms steps
    assert list(work.iterdir()) == []
