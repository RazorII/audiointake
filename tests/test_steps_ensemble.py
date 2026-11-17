"""steps.ensemble.transcribe and its command line, with stand-in children (no model runs).

The stand-ins are tiny scripts: a recogniser that copies its model folder's result.json to the output name it is
given, a Whisper that does the same as whisper.json, an ffmpeg that writes a fixed wav, and a Python that plays the
Thai worker. Each records the arguments and environment it got, so the tests see exactly what a real child would
have been told. The merges themselves are the real ones, checked against the goldens.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import pytest

from fixtures import build

from audiointake import errors, proc
from audiointake.contracts import EnsembleResult, Model
from audiointake.steps import ensemble

FIX = Path(__file__).parent / "fixtures" / "steps"
SRC = Path(__file__).resolve().parents[1] / "src" / "audiointake"

RECORDER = r'''
import json, os, sys
def record(folder, name):
    with open(os.path.join(folder, name), "a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\n")
'''

FAKE_PARAKEET = RECORDER + r'''
a = sys.argv[1:]
model, d, t = a[a.index("--model") + 1], a[a.index("--output-dir") + 1], a[a.index("--output-template") + 1]
record(model, "calls.jsonl")
if os.path.exists(os.path.join(model, "fail")):
    sys.exit(1)
os.makedirs(d, exist_ok=True)
with open(os.path.join(model, "result.json"), encoding="utf-8") as src, open(os.path.join(d, t + ".json"), "w", encoding="utf-8") as dst:
    dst.write(src.read())
'''

FAKE_WHISPER = RECORDER + r'''
a = sys.argv[1:]
model, d = a[a.index("--model") + 1], a[a.index("--output-dir") + 1]
record(model, "calls.jsonl")
if os.path.exists(os.path.join(model, "fail")):
    sys.exit(2)
os.makedirs(d, exist_ok=True)
with open(os.path.join(model, "result.json"), encoding="utf-8") as src, open(os.path.join(d, "whisper.json"), "w", encoding="utf-8") as dst:
    dst.write(src.read())
'''

FAKE_FFMPEG = RECORDER + r'''
here = os.path.dirname(os.path.abspath(__file__))
record(here, "ffmpeg-calls.jsonl")
if os.path.exists(os.path.join(here, "ffmpeg-fail")):
    sys.exit(1)
if sys.argv[-1] == "-":                                  # the decode check: decodes to nowhere
    if os.path.exists(os.path.join(here, "decode-fail")):
        print("Invalid data found when processing input", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
with open(os.path.join(here, "template.wav"), "rb") as src, open(sys.argv[-1], "wb") as dst:
    dst.write(src.read())
'''

FAKE_PYTHON = RECORDER + r'''
here = os.path.dirname(os.path.abspath(__file__))
record(here, "thai-calls.jsonl")
a = sys.argv[1:]
assert a[:5] == ["-I", "-B", "-m", "audiointake.steps.thai", "run"], a
out = a[8]
res = os.path.join(here, "thai-result.json")
if not os.path.exists(res):
    print("thai: Thai found (2 windows) but no Thai model", file=sys.stderr)
    sys.exit(1)
with open(res, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
    dst.write(src.read())
print("thai: 4 Thai turns (0.2 min) of 6 nemotron turns", file=sys.stderr)
'''


class Rig:
    def __init__(self, root: Path):
        self.root = root
        self.rt = build.fake_runtime(root)
        b = root / "bin"
        for name, body in (("parakeet-mlx", FAKE_PARAKEET), ("mlx_whisper", FAKE_WHISPER), ("ffmpeg", FAKE_FFMPEG),
                           ("ffprobe", FAKE_FFMPEG), ("python", FAKE_PYTHON), ("fluidaudiocli", "")):
            p = b / name
            p.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
            p.chmod(0o755)
        self.wav(30.0)
        self.models: dict[str, Model] = {}
        for key, tag in (("ultra", "parakeet-ultra-mlx"), ("v3", "parakeet-tdt-0.6b-v3"),
                         ("whisper", "whisper-large-v3-mlx"), ("typhoon", "typhoon-whisper-large-v3"),
                         ("nemotron", "nemotron3-offline")):
            d = root / "models" / key
            d.mkdir(parents=True)
            self.models[key] = Model(key, tag, d, "0" * 40)
        tools = dataclasses.replace(self.rt.tools, python=b / "python")
        self.rt = dataclasses.replace(self.rt, tools=tools)
        self.audio = root / "in" / "Team sync.m4a"
        self.audio.parent.mkdir()
        self.audio.write_bytes(b"not really audio")

    def wav(self, seconds: float):
        build.write_wav(self.root / "bin" / "template.wav", seconds, kind="bursts")

    def result(self, key: str, fixture: str):
        (self.models[key].path / "result.json").write_bytes((FIX / fixture).read_bytes())

    def fail(self, key: str):
        (self.models[key].path / "fail").write_text("", encoding="utf-8")

    def runtime(self, keys=("ultra", "v3"), routes=(), fluidaudio=True):
        tools = self.rt.tools if fluidaudio else dataclasses.replace(self.rt.tools, fluidaudiocli=None)
        return dataclasses.replace(self.rt, models={k: self.models[k] for k in keys}, routes=frozenset(routes),
                                   tools=tools)

    def calls(self, where: Path, name: str = "calls.jsonl") -> list[dict]:
        p = where / name
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def thai_calls(self):
        return self.calls(self.root / "bin", "thai-calls.jsonl")

    def run(self, rt, *, fmt="all", force_lang=None, duration_s=None):
        lines: list[str] = []
        out = rt.work_root / "e"
        res = ensemble.transcribe(self.audio, out, "Team sync", fmt, rt, force_lang=force_lang, log=lines.append,
                                  duration_s=duration_s)
        return res, out, lines


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def _txt(d: dict) -> str:
    return " ".join(x["text"].strip() for x in d["sentences"]).strip() + "\n"


def test_gapfill_route(rig):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    res, out, lines = rig.run(rig.runtime())
    assert isinstance(res, EnsembleResult)
    assert res.tag == "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill"
    assert (out / "Team sync.json").read_bytes() == (FIX / "out-gapfill-1.json").read_bytes()
    d = json.loads((FIX / "out-gapfill-1.json").read_text(encoding="utf-8"))
    assert (out / "Team sync.txt").read_text(encoding="utf-8") == _txt(d)
    assert (res.json_path, res.txt_path) == (out / "Team sync.json", out / "Team sync.txt")
    assert (res.hu_route, res.hu_detected, res.fill_ok) == ("gapfill", False, True)
    assert res.thai.screened is False and res.thai.failed is False
    assert any(x.startswith("gapfill: +") for x in lines)
    assert not any(p.name.startswith("asr-ensemble.") for p in out.iterdir())     # its temporary folder is gone


@pytest.mark.parametrize("fmt,files", [("json", {"Team sync.json"}), ("txt", {"Team sync.txt"}),
                                       ("all", {"Team sync.json", "Team sync.txt"})])
def test_output_formats(rig, fmt, files):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    res, out, _ = rig.run(rig.runtime(), fmt=fmt)
    assert {p.name for p in out.iterdir()} == files
    assert (res.json_path is not None) == ("Team sync.json" in files)
    assert (res.txt_path is not None) == ("Team sync.txt" in files)


def test_fill_model_failing_keeps_the_primary(rig):
    rig.result("ultra", "asr-g1a.json"); rig.fail("v3")
    res, out, lines = rig.run(rig.runtime())
    assert res.tag == "parakeet-ultra-mlx"
    assert (res.fill_ok, res.hu_route) == (False, "none")
    assert (out / "Team sync.json").read_bytes() == (FIX / "asr-g1a.json").read_bytes()
    assert any("fill model parakeet-tdt-0.6b-v3 failed" in x for x in lines)


def test_no_fill_model_is_not_a_failure(rig):
    rig.result("ultra", "asr-g1a.json")
    res, out, _ = rig.run(rig.runtime(keys=("ultra",)))
    assert (res.tag, res.fill_ok, res.hu_route) == ("parakeet-ultra-mlx", True, "none")
    assert rig.calls(rig.models["v3"].path) == []


def test_primary_failing_raises_and_writes_nothing(rig):
    rig.fail("ultra")
    with pytest.raises(errors.AsrFailed) as ei:
        rig.run(rig.runtime())
    assert ei.value.code == "asr_failed" and ei.value.exit == errors.EXIT_FAILED
    out = rig.rt.work_root / "e"
    assert not (out / "Team sync.json").exists() and not (out / "Team sync.txt").exists()
    assert rig.calls(rig.models["v3"].path) == []


def test_a_recording_ffmpeg_cant_decode_is_decode_failed(rig):
    rig.fail("ultra")
    (rig.root / "bin" / "decode-fail").write_text("", encoding="utf-8")
    with pytest.raises(errors.AsrFailed) as ei:
        res, out, lines = rig.run(rig.runtime())
    assert ei.value.code == "decode_failed" and ei.value.exit == errors.EXIT_FAILED
    assert ei.value.message == errors.render("decode_failed", {"file": "Team sync.m4a"})
    check = rig.calls(rig.root / "bin", "ffmpeg-calls.jsonl")[-1]["argv"]
    assert check[-4:] == ["-dn", "-f", "null", "-"] and "-xerror" in check
    assert rig.calls(rig.models["v3"].path) == []


def test_a_failed_model_on_a_readable_recording_stays_asr_failed(rig):
    rig.fail("ultra")
    with pytest.raises(errors.AsrFailed) as ei:
        rig.run(rig.runtime())
    assert ei.value.code == "asr_failed"
    assert rig.calls(rig.root / "bin", "ffmpeg-calls.jsonl")[-1]["argv"][-1] == "-"      # the decode check ran


def test_outdir_must_be_under_the_work_root(rig, tmp_path):
    with pytest.raises(ValueError):
        ensemble.transcribe(rig.audio, tmp_path / "elsewhere", "x", "all", rig.runtime(), log=lambda s: None)


def _vote_rig(rig):
    rig.result("ultra", "asr-v1u.json"); rig.result("v3", "asr-v1v.json"); rig.result("whisper", "whisper-1.json")
    case = next(c for c in json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))["cases"]
                if c["kind"] == "ensemble" and c["cli"][0] == "hu" and c["cli"][1] == "whisper-1.json")
    rig.wav(case["wav"]["seconds"])                       # the energy wav the golden was made with
    return case


def test_forced_vote_route(rig):
    case = _vote_rig(rig)
    res, out, lines = rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"hu_vote"}), force_lang="hu")
    assert res.tag == ensemble.VOTE_TAG == "whisper-large-v3+ultra/v3-vote"
    assert res.hu_route == "vote"
    assert (out / "Team sync.json").read_bytes() == (FIX / case["expect"]).read_bytes()
    call = rig.calls(rig.models["whisper"].path)[0]["argv"]
    assert call[call.index("--language") + 1] == "hu"
    assert call[call.index("--model") + 1] == str(rig.models["whisper"].path)
    assert [x for x in call if x.startswith("--")] == ["--model", "--language", "--condition-on-previous-text",
                                                      "--temperature", "--compression-ratio-threshold",
                                                      "--output-format", "--output-dir", "--output-name"]
    ff = [c["argv"] for c in rig.calls(rig.root / "bin", "ffmpeg-calls.jsonl")]
    assert any("loudnorm=I=-16:TP=-1.5:LRA=11" in a for a in ff)


def test_detected_language_takes_the_vote(rig, tmp_path):
    _vote_rig(rig)
    p = tmp_path / "doc.json"
    for seed in range(200):
        p.write_text(json.dumps(_lang_doc(seed), ensure_ascii=False), encoding="utf-8")
        if ensemble.lang(p)[0] == "hu":
            break
    (rig.models["ultra"].path / "result.json").write_bytes(p.read_bytes())
    res, _, _ = rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"hu_vote"}))
    assert (res.hu_detected, res.hu_route, res.tag) == (True, "vote", ensemble.VOTE_TAG)
    assert res.hu_share >= 0.08


def _lang_doc(seed):
    from fixtures.steps import synth
    return synth.lang_input(seed, sorted(ensemble.HU_WORDS))


def test_vote_failing_keeps_the_primary(rig):
    _vote_rig(rig)
    rig.fail("whisper")
    res, out, _ = rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"hu_vote"}), force_lang="hu")
    assert (res.tag, res.hu_route) == ("parakeet-ultra-mlx+hu-failed", "vote_failed")
    assert (out / "Team sync.json").read_bytes() == (FIX / "asr-v1u.json").read_bytes()


def test_vote_without_its_pack_gap_fills(rig):
    _vote_rig(rig)
    res, _, _ = rig.run(rig.runtime(keys=("ultra", "v3", "whisper")), force_lang="hu")
    assert (res.hu_route, res.tag) == ("no_pack", "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill")
    assert rig.calls(rig.models["whisper"].path) == []


def test_another_language_forces_nothing(rig):
    _vote_rig(rig)
    res, _, _ = rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"hu_vote"}), force_lang="de")
    assert res.hu_route == "gapfill"


def test_thai_turns_route(rig):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    case = next(c for c in json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))["cases"]
                if c["kind"] == "ensemble" and c["cli"][:2] == ["thai", "out-gapfill-1.json"])
    (rig.root / "bin" / "thai-result.json").write_bytes((FIX / case["cli"][2]).read_bytes())
    rt = rig.runtime(keys=("ultra", "v3", "whisper", "typhoon", "nemotron"), routes={"thai_screen", "thai_turns"})
    res, out, lines = rig.run(rt)
    assert res.tag == "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill+thai-model-x-thai"
    assert (out / "Team sync.json").read_bytes() == (FIX / case["expect"]).read_bytes()
    a = rig.thai_calls()[0]["argv"]
    assert [Path(x).name for x in a[5:9]] == ["a.16k.wav", "primary.json", "fill.json", "thai.json"]
    assert a[a.index("--lid-model") + 1] == str(rig.models["whisper"].path)
    assert a[a.index("--typhoon") + 1] == str(rig.models["typhoon"].path)
    assert a[a.index("--typhoon-tag") + 1] == "typhoon-whisper-large-v3"
    assert a[a.index("--fluidaudio") + 1] == str(rig.rt.tools.fluidaudiocli)
    assert a[a.index("--nemotron") + 1] == str(rig.models["nemotron"].path)
    assert "--screen-only" not in a
    td = json.loads((FIX / case["cli"][2]).read_text(encoding="utf-8"))
    assert (res.thai.screened, res.thai.turns, res.thai.windows, res.thai.thai_windows) == \
        (True, len(td["turns"]), td["windows"], td["thai_windows"])
    assert res.thai.seconds == round(sum(t["end"] - t["start"] for t in td["turns"]), 3)
    assert "thai: 4 Thai turns (0.2 min) of 6 nemotron turns" in lines


def test_thai_screen_only_without_the_thai_pack(rig):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    (rig.root / "bin" / "thai-result.json").write_text(json.dumps(
        {"thai": True, "screen_only": True, "windows": 3, "thai_windows": 2, "max_p_th": 0.91}), encoding="utf-8")
    res, out, _ = rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"thai_screen"}, fluidaudio=False))
    a = rig.thai_calls()[0]["argv"]
    assert "--screen-only" in a and "--typhoon" not in a and "--fluidaudio" not in a
    assert res.tag == "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill"          # no tag change, nothing failed
    assert (res.thai.screened, res.thai.screen_only, res.thai.failed, res.thai.thai_windows) == (True, True, False, 2)
    assert (out / "Team sync.json").read_bytes() == (FIX / "out-gapfill-1.json").read_bytes()


def test_thai_worker_failing_marks_the_tag(rig):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    res, out, lines = rig.run(rig.runtime(keys=("ultra", "v3", "whisper", "typhoon"),
                                          routes={"thai_screen", "thai_turns"}))
    assert res.tag.endswith("-gapfill+th-failed")
    assert (res.thai.failed, res.thai.screened) == (True, False)
    assert any(x.startswith("asr: Thai screen failed") and "no Thai model" in x for x in lines)
    assert (out / "Team sync.json").read_bytes() == (FIX / "out-gapfill-1.json").read_bytes()


def test_no_16k_wav_fails_the_thai_screen(rig):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    (rig.root / "bin" / "ffmpeg-fail").write_text("", encoding="utf-8")
    res, _, lines = rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"thai_screen"}))
    assert res.tag.endswith("+th-failed") and rig.thai_calls() == []
    assert any(x.endswith("no 16 kHz wav") for x in lines)


def test_children_get_the_runtime_environment_only(rig, monkeypatch):
    for k, v in (("THAI", "on"), ("PARAKEET_MODEL", "x/y"), ("PARAKEET_FILL_MODEL", "x/z"), ("WHISPER_BIN", "/x"),
                 ("SPEAKER_ID_VOICEPRINTS", "/x"), ("HF_TOKEN", "x")):
        monkeypatch.setenv(k, v)
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    (rig.root / "bin" / "thai-result.json").write_text(json.dumps({"thai": False, "windows": 1, "max_p_th": 0.0}),
                                                       encoding="utf-8")
    rig.run(rig.runtime(keys=("ultra", "v3", "whisper"), routes={"thai_screen"}))
    envs = [c["env"] for c in rig.calls(rig.models["ultra"].path) + rig.calls(rig.models["v3"].path)
            + rig.thai_calls() + rig.calls(rig.root / "bin", "ffmpeg-calls.jsonl")]
    assert len(envs) >= 4
    for env in envs:
        assert not [k for k in env if k.startswith(("THAI", "PARAKEET_", "WHISPER_", "SPEAKER_ID_", "HF_TOKEN"))]
        assert {k: env.get(k) for k in rig.rt.env} == dict(rig.rt.env)


def test_children_timeouts_follow_the_duration(rig, monkeypatch):
    seen = []
    real = proc.run

    def spy(argv, **kw):
        seen.append((Path(argv[0]).name, kw["timeout"]))
        return real(argv, **kw)
    monkeypatch.setattr(proc, "run", spy)
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    (rig.root / "bin" / "thai-result.json").write_text(json.dumps({"thai": False, "windows": 1, "max_p_th": 0.0}),
                                                       encoding="utf-8")
    rt = rig.runtime(keys=("ultra", "v3", "whisper"), routes={"thai_screen"})
    rig.run(rt, duration_s=1000.0)
    assert dict(seen) == {"parakeet-mlx": 3000.0, "ffmpeg": 250.0, "python": 3000.0}
    seen.clear()
    rig.run(rt)
    assert {t for _, t in seen} == {ensemble.NO_DURATION_TIMEOUT}


# --------------------------------------------------------------------------- the command line and its switches
def test_runtime_from_env(rig):
    rt = rig.runtime(keys=("ultra", "v3", "whisper"), routes={"thai_screen", "thai_turns", "hu_vote"})
    assert ensemble.runtime_from_env(rt, {}) == rt
    assert ensemble.runtime_from_env(rt, {"THAI": "on", "PARAKEET_FILL_MODEL": "some/model"}) == rt
    off = ensemble.runtime_from_env(rt, {"THAI": "off"})
    assert off.routes == {"hu_vote"} and off.models == rt.models
    nofill = ensemble.runtime_from_env(rt, {"PARAKEET_FILL_MODEL": ""})
    assert "v3" not in nofill.models and nofill.routes == rt.routes
    assert "v3" in rt.models                                   # the runtime given is never changed


def _cli(rig, environ, *, fail=None):
    calls = []

    def resolve(**kw):
        calls.append(kw)
        if fail is not None:
            raise fail
        rt = rig.runtime(keys=("ultra", "v3", "whisper", "typhoon"), routes={"thai_screen", "thai_turns"})
        return rt, [errors.problem("no_speaker_labels", reason="test")]
    dest = rig.root / "dest"
    rc = ensemble.cli_transcribe([str(rig.audio), "--output-dir", str(dest), "--output-template", "T",
                                  "--output-format", "all"], environ=environ, resolve=resolve)
    return rc, dest, calls


def test_cli_maps_the_two_switches_onto_the_runtime(rig, capsys):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    rc, dest, calls = _cli(rig, {"THAI": "off", "PARAKEET_FILL_MODEL": ""})
    out = capsys.readouterr()
    assert rc == 0 and out.out == "parakeet-ultra-mlx\n"
    assert calls == [{"names": False}]
    assert rig.calls(rig.models["v3"].path) == [] and rig.thai_calls() == []
    assert (dest / "T.json").read_bytes() == (FIX / "asr-g1a.json").read_bytes()
    assert (dest / "T.txt").exists()
    assert not rig.rt.work_root.exists()                       # the run's work folder is removed
    assert "ensemble: no_speaker_labels" in out.err


def test_cli_without_switches_runs_every_installed_route(rig, capsys):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")
    (rig.root / "bin" / "thai-result.json").write_text(json.dumps({"thai": False, "windows": 1, "max_p_th": 0.0}),
                                                       encoding="utf-8")
    rc, dest, _ = _cli(rig, {})
    assert rc == 0 and capsys.readouterr().out == "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill\n"
    assert len(rig.thai_calls()) == 1
    assert (dest / "T.json").read_bytes() == (FIX / "out-gapfill-1.json").read_bytes()


def test_cli_exits_3_when_the_primary_fails_or_setup_is_missing(rig, capsys):
    rig.fail("ultra")
    assert _cli(rig, {})[0] == 3
    assert "asr: transcribing Team sync.m4a failed; nothing was written" in capsys.readouterr().err
    rc, _, _ = _cli(rig, {}, fail=errors.SetupError("not_set_up", size=4.3))
    assert rc == 3 and "isn't set up yet" in capsys.readouterr().err


def test_only_the_ensemble_command_line_reads_the_engine_switches():
    """THAI and PARAKEET_FILL_MODEL are read from the environment in one place; the other old switches nowhere."""
    hits = {}
    for p in sorted(SRC.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        for name in ("THAI", "PARAKEET_FILL_MODEL", "PARAKEET_MODEL", "WHISPER_BIN", "THAI_MODEL", "FLUIDAUDIO_BIN",
                     "SPEAKER_ID_VOICEPRINTS", "AUDIO_VOICEPRINTS"):
            if re.search(r'(?:environ\.get\(|environ\[|getenv\()\s*["\']' + name + r'["\']', text):
                hits.setdefault(name, []).append(p.relative_to(SRC).as_posix())
    assert hits == {"THAI": ["steps/ensemble.py"], "PARAKEET_FILL_MODEL": ["steps/ensemble.py"]}


def test_an_unknown_duration_never_stops_the_command_line(rig):
    lines: list[str] = []
    rt = dataclasses.replace(rig.rt, tools=dataclasses.replace(rig.rt.tools, ffprobe=rig.root / "bin" / "missing"))
    assert ensemble._duration(rt, rig.audio, lines.append) is None
    assert lines == ["ensemble: the recording's length is unknown (FileNotFoundError); each child gets 6 h"]
    rt = dataclasses.replace(rig.rt, tools=dataclasses.replace(rig.rt.tools, ffprobe=Path("relative/ffprobe")))
    assert ensemble._duration(rt, rig.audio, lines.append) is None and "(ValueError)" in lines[-1]
    assert ensemble._duration(dataclasses.replace(rig.rt, tools=dataclasses.replace(rig.rt.tools, ffprobe=None)),
                              rig.audio, lines.append) is None and len(lines) == 2


def test_a_merge_step_failing_exits_3_on_the_command_line(rig, capsys, monkeypatch):
    rig.result("ultra", "asr-g1a.json"); rig.result("v3", "asr-g1b.json")

    def broken(*a, **kw):
        raise KeyError("sentences")
    monkeypatch.setattr(ensemble, "gapfill", broken)
    rc, dest, _ = _cli(rig, {"THAI": "off"})
    assert rc == 3 and not dest.exists()
    assert "asr: merging failed: KeyError" in capsys.readouterr().err


def test_the_module_runs_as_a_program():
    import subprocess
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "audiointake.steps.ensemble", "nope"], capture_output=True,
                       text=True)
    assert r.returncode == 1 and "transcribe <audio> --output-dir D" in r.stderr
