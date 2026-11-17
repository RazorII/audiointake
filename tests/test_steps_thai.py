"""The Thai worker: --screen-only, the energy turns without a diarizer, the diarizer through argv.diarize, and its
command line. The models are stood in for: the language ID, the audio loader and the Thai model are replaced, so
these run without a Metal device."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from audiointake.steps import thai

np = pytest.importorskip("numpy")
FIX = Path(__file__).parent / "fixtures" / "steps"


class FakeAudio:
    """10 s: voiced runs of 2 s with 0.5 s pauses (an energy turn finder splits them)."""

    def __init__(self, wav):
        self.dur = 10.0
        frames = []
        t = 0.0
        while t < self.dur:
            frames.append(-20.0 if (t % 2.5) < 2.0 else -90.0)
            t = round(t + 0.02, 6)
        self.db = np.asarray(frames)
        self.thr = float(np.percentile(self.db, 10) + 10)
        self.a = np.zeros(int(self.dur * 16000), dtype=np.float32)

    def clip(self, s, e):
        return self.a[int(s * 16000):int(e * 16000)]

    def voiced(self, s, e):
        seg = self.db[int(s / .02):max(int(e / .02), int(s / .02) + 1)]
        return float((seg > self.thr).mean()) if len(seg) else 0.0


class FakeLID:
    """P(th) high for every span: the screen finds Thai and every turn is a candidate."""

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def __call__(self, au, spans):
        return [{"th": 0.9, "en": 0.05} for _ in spans]


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setattr(thai, "Audio", FakeAudio)
    monkeypatch.setattr(thai, "LID", FakeLID)
    calls = []

    def transcribe(clip, **kw):
        calls.append(kw)
        return {"text": "กขค งจฉ"}
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    (tmp_path / "typhoon").mkdir()
    (tmp_path / "typhoon" / "weights.safetensors").write_bytes(b"x")
    (tmp_path / "u.json").write_bytes((FIX / "asr-t1.json").read_bytes())
    return tmp_path, calls


def _run(tmp_path, *extra):
    out = tmp_path / "out.json"
    rc = thai.main(["run", str(tmp_path / "a.16k.wav"), str(tmp_path / "u.json"), "", str(out), "--lid-model",
                    str(tmp_path / "lid"), *extra])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


def test_screen_only_reports_thai_and_transcribes_nothing(worker, capsys):
    tmp, calls = worker
    rc, d = _run(tmp, "--screen-only")
    assert rc == 0
    assert d == {"thai": True, "screen_only": True, "windows": 1, "thai_windows": 1, "max_p_th": 0.9}
    assert calls == []
    assert capsys.readouterr().err.startswith("thai: found (1 of 1 windows")


def test_screen_only_changes_nothing_when_there_is_no_thai(worker, monkeypatch):
    tmp, _ = worker
    monkeypatch.setattr(thai, "LID", lambda m: (lambda au, spans: [{"th": 0.1, "en": 0.8} for _ in spans]))
    a = _run(tmp, "--screen-only")
    (tmp / "out.json").unlink()
    b = _run(tmp)
    assert a == b == (0, {"thai": False, "windows": 1, "max_p_th": 0.1})


def test_without_a_diarizer_the_turns_come_from_energy(worker):
    tmp, calls = worker
    rc, d = _run(tmp, "--typhoon", str(tmp / "typhoon"), "--typhoon-tag", "thai-model-x")
    assert rc == 0
    assert d["thai"] is True and d["turns_from"] == "energy" and d["model"] == "thai-model-x"
    assert [t["spk"] for t in d["turns"]] and {t["spk"] for t in d["turns"]} == {"vad"}
    assert d["stats"]["turns"] == 4                               # 2 s runs at 0, 2.5, 5, 7.5 s
    assert all(c["path_or_hf_repo"] == str(tmp / "typhoon") and c["language"] == "th" and c["sample_len"] == 440
               for c in calls)


def test_thai_found_without_a_thai_model_fails(worker, capsys):
    tmp, _ = worker
    rc, d = _run(tmp)
    assert rc == 1 and d is None
    assert "no Thai model was given" in capsys.readouterr().err


def test_the_diarizer_runs_through_argv_diarize(worker, monkeypatch):
    tmp, _ = worker
    from audiointake import proc
    seen = []
    monkeypatch.setenv("HF_TOKEN", "not-a-token")              # a caller's variables never reach the diarizer
    monkeypatch.setenv("FLUIDAUDIO_BIN", "/elsewhere")
    rttm_text = (FIX / "diar-2.rttm.txt").read_text(encoding="utf-8")

    def fake_run(argv, **kw):
        seen.append((argv, kw))
        Path(argv[argv.index("--output") + 1]).write_text(rttm_text, encoding="utf-8")
        return proc.Completed(0, b"", [])
    monkeypatch.setattr(proc, "run", fake_run)
    fa = tmp / "fluidaudiocli"
    fa.write_text("#!/bin/sh\n", encoding="utf-8")
    fa.chmod(0o755)
    rc, d = _run(tmp, "--fluidaudio", str(fa), "--nemotron", str(tmp / "nemo"), "--typhoon", str(tmp / "typhoon"),
                 "--typhoon-tag", "thai-model-x")
    assert rc == 0 and d["turns_from"] == "nemotron" and d["model"] == "thai-model-x"
    argv, kw = seen[0]
    assert argv[:2] == [str(fa), "nemotron3-diarize"]
    assert argv[3:] == ["--variant", "offline", "--threshold", "0.4", "--output", argv[8], "--models", str(tmp / "nemo")]
    assert kw["allowed"] == {fa} and kw["stdout"] == "discard"
    assert set(kw["env"]) <= set(thai.RUN_ENV_KEYS)
    assert kw["env"] == {k: os.environ[k] for k in thai.RUN_ENV_KEYS if k in os.environ}


def test_the_thai_model_name_never_comes_from_its_folder(worker):
    tmp, _ = worker
    with pytest.raises(ValueError, match="go together"):
        thai.run(str(tmp / "a.16k.wav"), str(tmp / "u.json"), "", str(tmp / "out.json"), lid_model="m",
                 typhoon=str(tmp / "typhoon"))
    assert not (tmp / "out.json").exists()


@pytest.mark.parametrize("extra", [["--fluidaudio", "/x"], ["--nemotron", "/x"], ["--typhoon-tag", "x"],
                                   ["--typhoon", "/x"], ["--fluidaudio", "rel/fluidaudiocli", "--nemotron", "/x"]])
def test_command_line_refuses_half_pairs_and_a_relative_diarizer(extra, tmp_path):
    with pytest.raises(SystemExit) as ei:
        thai.main(["run", "w", "u", "", "o", "--lid-model", "m", *extra])
    assert ei.value.code == 2


def test_lid_model_is_required():
    with pytest.raises(SystemExit) as ei:
        thai.main(["run", "w", "u", "", "o"])
    assert ei.value.code == 2


def test_window_size_is_whispers():
    """Read from mlx_whisper's source, not imported (importing it needs mlx and a Metal device)."""
    import importlib.util
    import re
    spec = importlib.util.find_spec("mlx_whisper")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("mlx-whisper isn't installed")
    src = (Path(list(spec.submodule_search_locations)[0]) / "audio.py").read_text(encoding="utf-8")
    val = {k: int(v) for k, v in re.findall(r"^(SAMPLE_RATE|CHUNK_LENGTH) = (\d+)$", src, re.M)}
    assert re.search(r"^N_SAMPLES = CHUNK_LENGTH \* SAMPLE_RATE\b", src, re.M)
    assert thai.N_SAMPLES == val["CHUNK_LENGTH"] * val["SAMPLE_RATE"] == 480000
