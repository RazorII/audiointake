"""tests/fixtures/build.py: the seeded builders are reproducible and have the shapes the engine reads."""
from __future__ import annotations

import io
import re
import stat
import wave

import pytest

from fixtures import build


def test_asr_json_shape_and_reproducibility():
    a, b = build.asr_json(7, sentences=5), build.asr_json(7, sentences=5)
    assert a == b and a != build.asr_json(8, sentences=5)
    assert len(a["sentences"]) == 5
    assert a["text"] == "".join(s["text"] for s in a["sentences"]).strip()
    for s in a["sentences"]:
        assert set(s) == {"text", "start", "end", "duration", "confidence", "tokens"}
        assert s["text"] == "".join(t["text"] for t in s["tokens"])
        assert s["tokens"][0]["text"].startswith(" ") and s["tokens"][-1]["text"] == "."
        assert s["start"] == s["tokens"][0]["start"] and s["end"] == s["tokens"][-1]["end"]
        for t in s["tokens"]:
            assert set(t) == {"text", "start", "end", "duration", "confidence"}
            assert round(t["start"] / build.GRID, 6) == round(t["start"] / build.GRID)
            assert t["end"] >= t["start"] and 0 <= t["confidence"] <= 1
    starts = [s["start"] for s in a["sentences"]]
    assert starts == sorted(starts)


def test_asr_words_come_from_the_list():
    a = build.asr_json(3, sentences=4)
    words = set(re.findall(r"[a-z]+", a["text"].lower()))
    assert words <= set(build.WORDS)


def test_low_confidence_share():
    a = build.asr_json(1, sentences=30, low_rate=0.5)
    confs = [t["confidence"] for s in a["sentences"] for t in s["tokens"] if t["text"] != "."]
    low = sum(1 for x in confs if x < 0.6) / len(confs)
    assert 0.3 < low < 0.7


def test_whisper_json_shape():
    w = build.whisper_json(2, segments=3, language="de")
    assert w["language"] == "de" and len(w["segments"]) == 3
    assert w["text"] == "".join(s["text"] for s in w["segments"])
    assert all(s["end"] > s["start"] for s in w["segments"])
    assert w == build.whisper_json(2, segments=3, language="de")


def test_rttm_text():
    r = build.rttm_text(4, file_id="Team sync.16k", speakers=3, duration=120)
    lines = r.splitlines()
    assert lines and r.endswith("\n")
    rx = re.compile(r"^SPEAKER (.+) 1 (\d+\.\d{3}) (\d+\.\d{3}) <NA> <NA> (speaker_[0-7]) <NA> <NA>$")
    seen, starts = set(), []
    for ln in lines:
        m = rx.match(ln)
        assert m and m.group(1) == "Team sync.16k"
        starts.append(float(m.group(2)))
        seen.add(m.group(4))
    assert starts == sorted(starts) and seen <= {"speaker_0", "speaker_1", "speaker_2"} and len(seen) == 3
    assert r == build.rttm_text(4, file_id="Team sync.16k", speakers=3, duration=120)
    with pytest.raises(ValueError):
        build.rttm_text(speakers=9)


@pytest.mark.parametrize("kind", ["sine", "silence", "noise", "bursts"])
def test_wav_bytes(kind):
    data = build.wav_bytes(1.25, kind=kind, seed=3)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 20000)
        frames = w.readframes(20000)
    if kind == "silence":
        assert set(frames) == {0}
    else:
        assert any(frames)
    assert data == build.wav_bytes(1.25, kind=kind, seed=3)


def test_stereo_wav(tmp_path):
    p = build.write_wav(tmp_path / "s.wav", 0.5, channels=2)
    with wave.open(str(p)) as w:
        assert w.getnchannels() == 2 and w.getnframes() == 8000


def test_fake_runtime(tmp_path):
    rt = build.fake_runtime(tmp_path, routes={"thai_screen"}, fluidaudio=False)
    assert rt.tools.fluidaudiocli is None and rt.tools.ffprobe == tmp_path / "bin" / "ffprobe"
    assert rt.routes == frozenset({"thai_screen"})
    assert stat.S_IMODE(rt.work_root.stat().st_mode) == 0o700
    assert rt.env["HF_HUB_OFFLINE"] == "1" and rt.voices_dir is None
    with pytest.raises(ValueError):
        build.fake_runtime(tmp_path, routes={"nope"})
    m = build.fake_model("ultra", "parakeet-ultra-mlx", tmp_path / "ultra")
    assert build.fake_runtime(tmp_path, models={"ultra": m}).models["ultra"].tag == "parakeet-ultra-mlx"
