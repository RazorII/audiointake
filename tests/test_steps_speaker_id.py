"""steps.speaker_id beyond the goldens: the library calls, the voiceprint folder rules, names, and the digest."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from fixtures.steps import synth

from audiointake import errors
from audiointake.steps import speaker_id as S

np = pytest.importorskip("numpy")
FIX = Path(__file__).parent / "fixtures" / "steps"
SPK = json.loads((FIX / "speaker-id.json").read_text(encoding="utf-8"))


def _write(folder: Path, name: str, data, fname=None):
    p = folder / (fname or f"{name.lower()}.json")
    p.write_text(data if isinstance(data, str) else json.dumps(data, indent=1), encoding="utf-8")
    return p


@pytest.fixture
def voices(tmp_path):
    ex = synth.FakeExtractor()
    d = tmp_path / "voices"
    d.mkdir()
    for p in synth.prints_for(ex, {"Zorvel": "speaker_0", "Quessa": "speaker_2"}, S.MODEL_SHA256):
        _write(d, p["name"], p)
    return d


@pytest.fixture
def model(monkeypatch, tmp_path):
    ex = synth.FakeExtractor()
    monkeypatch.setattr(S, "_extractor", lambda m: ex)
    monkeypatch.setattr(S, "_today", lambda: "2026-01-01")
    return tmp_path / "model.onnx"


@pytest.fixture
def rec(tmp_path):
    turns, stretches = synth.voice_layout(701, SPK["labels"], 70.0)
    wav = tmp_path / "rec.wav"
    wav.write_bytes(synth.tone_track_wav(stretches))
    return wav, synth.rttm_of(turns, "rec 1.16k")


def test_dotfiles_and_strays_are_skipped(voices, capsys):
    _write(voices, "x", {"name": "Zorvel", "vectors": [], "model_sha256": S.MODEL_SHA256}, fname="._zorvel.json")
    _write(voices, "x", "[1, 2]", fname="list.json")
    _write(voices, "x", "{broken", fname="broken.json")
    (voices / "dir.json").mkdir()
    _write(voices, "x", {"name": "Velmine", "model_sha256": "0" * 64,
                         "vectors": [{"vec": [1.0] * 16, "source": "s", "seconds": 9, "lang": None, "added": "x"}]},
           fname="velmine.json")
    prints = S.load_prints(voices)
    assert sorted(prints) == ["Quessa", "Zorvel"]
    err = capsys.readouterr().err
    assert "skipping list.json: not a voiceprint" in err and "skipping broken.json: not valid JSON" in err
    assert "skipping velmine.json: made with another model" in err and "._zorvel" not in err
    assert S.scan(voices) == {"prints": 2, "other_model": 1, "not_prints": 2}
    assert [p["name"] for p in S.list_prints(voices)] == ["Quessa", "Velmine", "Zorvel"]


def test_scan_and_digest_of_a_missing_folder(tmp_path):
    assert S.scan(tmp_path / "none") == {"prints": 0, "other_model": 0, "not_prints": 0}
    assert S.scan(None)["prints"] == 0
    assert S.voices_digest(tmp_path / "none") == S.voices_digest(None) == ""


def test_voices_digest_agrees_with_the_voices_module(voices, tmp_path):
    """One digest for the records: the engine's and the CLI's give the same value on every folder shape."""
    from audiointake import voices as V
    (tmp_path / "empty").mkdir()
    only_other = tmp_path / "only-other"
    only_other.mkdir()
    other = {"name": "Velmine", "model_sha256": "0" * 64,
             "vectors": [{"vec": [1.0] * 16, "source": "s", "seconds": 9, "lang": None, "added": "x"}]}
    _write(only_other, "Velmine", other)
    seen = []
    for d in (None, tmp_path / "none", tmp_path / "empty", only_other, voices):
        seen.append(S.voices_digest(d))
        assert seen[-1] == V.digest(d), d
    assert seen[:4] == ["", "", "", ""] and len(seen[4]) == 64
    _write(voices, "Velmine", other)                                   # another model: left out of both
    _write(voices, "x", {"name": "Zorvel", "vectors": [], "model_sha256": S.MODEL_SHA256}, fname="._z.json")
    _write(voices, "x", "[1, 2]", fname="list.json")
    _write(voices, "x", "{broken", fname="broken.json")
    assert S.voices_digest(voices) == V.digest(voices) == seen[4]
    _write(voices, "Ondrel", {"name": "Ondrel", "model_sha256": S.MODEL_SHA256, "vectors": [], "updated": "x"})
    unreadable = _write(voices, "x", "{}", fname="locked.json")
    unreadable.chmod(0)
    try:
        assert S.voices_digest(voices) == V.digest(voices) != seen[4]
    finally:
        unreadable.chmod(0o600)


def test_voices_digest_follows_the_set_never_the_vectors(voices):
    d0 = S.voices_digest(voices)
    assert d0 == S.voices_digest(voices) and len(d0) == 64
    f = voices / "zorvel.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    data["vectors"][0]["vec"][0] += 0.5                      # vectors alone don't change it
    f.write_text(json.dumps(data), encoding="utf-8")
    assert S.voices_digest(voices) == d0
    data["updated"] = "2026-02-02"
    f.write_text(json.dumps(data), encoding="utf-8")
    d1 = S.voices_digest(voices)
    assert d1 != d0
    same_day = dict(data["vectors"][0], source="another reading (0-30 s)")
    data["vectors"].append(same_day)                         # a second source the same day changes it too
    f.write_text(json.dumps(data), encoding="utf-8")
    d2 = S.voices_digest(voices)
    assert d2 != d1 and data["updated"] == "2026-02-02"
    data["vectors"].pop()
    f.write_text(json.dumps(data), encoding="utf-8")
    assert S.voices_digest(voices) == d1
    _write(voices, "x", "{}", fname="._x.json")              # a dotfile doesn't count
    assert S.voices_digest(voices) == d1
    (voices / "quessa.json").unlink()
    assert S.voices_digest(voices) != d1


def test_match_library_returns_the_sidecar(voices, model, rec, tmp_path):
    wav, rttm_text = rec
    r = tmp_path / "rec.rttm"
    r.write_text(rttm_text, encoding="utf-8")
    sc = S.match(wav, r, voices, model)
    assert sc["engine"] == S.ENGINE and sc["enrolled"] == ["Quessa", "Zorvel"]
    named = {k: v["name"] for k, v in sc["labels"].items() if v["name"]}
    assert named == {"speaker_0": "Zorvel", "speaker_2": "Quessa"}
    assert sc["labels"]["speaker_1"]["why"] == "below threshold"
    assert S.sidecar_text(sc) == json.dumps(sc, indent=1, ensure_ascii=False)


def test_match_without_voiceprints_fails(model, rec, tmp_path):
    wav, rttm_text = rec
    r = tmp_path / "rec.rttm"
    r.write_text(rttm_text, encoding="utf-8")
    with pytest.raises(S.SpeakerIdError, match="no voiceprints"):
        S.match(wav, r, tmp_path / "empty", model)


def test_enroll_label_writes_a_private_file(model, rec, tmp_path):
    wav, rttm_text = rec
    v = tmp_path / "new-voices"
    r = S.enroll_label("Brimdal", wav, rttm_text, "speaker_1", "rec 1 speaker_1", "en", v, model, False)
    assert r["added"] and r["lines"] == ["Brimdal: +18 s from rec 1 speaker_1 (1 sources)"]
    f = v / "brimdal.json"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and stat.S_IMODE(v.stat().st_mode) == 0o700
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["vectors"][0]["lang"] == "EN" and d["updated"] == "2026-01-01"
    assert not [p for p in v.iterdir() if p.name.startswith(".partial-")]
    again = S.enroll_label("Brimdal", wav, rttm_text, "speaker_1", "rec 1 speaker_1", "en", v, model, False)
    assert again["skipped"] and not again["added"]


def test_enroll_label_refusals_carry_codes(voices, model, rec):
    wav, rttm_text = rec
    with pytest.raises(S.HeldBack) as ei:
        S.enroll_label("Brimdal", wav, rttm_text, "speaker_0", "rec 1 speaker_0", None, voices, model, False)
    p = ei.value.problems
    assert [x.code for x in p] == ["matches_someone_else"]
    assert p[0].params["other"] == "Zorvel" and p[0].params["name"] == "Brimdal"
    assert p[0].params["file"] == "rec 1 speaker_0"
    assert "Zorvel" in errors.render(p[0].code, p[0].params)
    assert not (voices / "brimdal.json").exists()
    with pytest.raises(S.TooLittleSpeech) as ei:
        S.enroll_label("Brimdal", wav, rttm_text, "speaker_7", "x", None, voices, model, False)
    tp = ei.value.problem()
    assert tp.code == "too_little_speech" and tp.params["source"] == "label" and tp.params["label"] == "speaker_7"
    assert errors.render(tp.code, tp.params).startswith("speaker_7 has 0 s of speech")


def test_forced_enrolment_records_what_it_overrode(voices, model, rec):
    wav, rttm_text = rec
    r = S.enroll_label("Zorvel", wav, rttm_text, "speaker_1", "rec 1 speaker_1", None, voices, model, True)
    assert r["added"] and [p.code for p in r["forced"]] == ["not_matching_print", "mixed_voices"]
    assert r["lines"][0].startswith("Zorvel: enrolled despite the check (--force): it scores")


@pytest.mark.parametrize("name", ["", " ", "a/b", ".hidden", "x" * 65, "a\x00b"])
def test_invalid_names_are_refused(name, voices, model, rec):
    with pytest.raises(errors.UsageError) as ei:
        S.check_name(name)
    assert ei.value.code == "name_invalid"
    wav, rttm_text = rec
    with pytest.raises(errors.UsageError):
        S.enroll_label(name, wav, rttm_text, "speaker_1", "x", None, voices, model, True)


def test_remove_and_rename(voices):
    with pytest.raises(errors.UsageError) as ei:
        S.remove_print(voices, "Nobody")
    assert ei.value.code == "voice_not_found"
    new = S.rename_print(voices, "Quessa", "Velmine")
    assert new == voices / "velmine.json" and not (voices / "quessa.json").exists()
    assert json.loads(new.read_text(encoding="utf-8"))["name"] == "Velmine"
    assert stat.S_IMODE(new.stat().st_mode) == 0o600
    with pytest.raises(errors.UsageError) as ei:
        S.rename_print(voices, "Velmine", "Zorvel")
    assert ei.value.code == "bad_arguments"
    with pytest.raises(errors.UsageError) as ei:
        S.rename_print(voices, "Velmine", "../x")
    assert ei.value.code == "name_invalid"
    assert S.remove_print(voices, "Velmine") == new and not new.exists()
    assert sorted(S.load_prints(voices)) == ["Zorvel"]


def test_check_prints_matches_list_order(voices):
    res = S.check_prints(voices)
    assert [(r["name"], r["held"]) for r in res] == [("Quessa", False), ("Quessa", False), ("Zorvel", False),
                                                     ("Zorvel", False)]
    assert S.check_lines(res)[-1] == "0 of 4 stored sources would be held back today"


@pytest.mark.parametrize("cmd", ["enroll", "enroll-label"])
@pytest.mark.parametrize("name", ["a/b", ".hidden"])
def test_an_invalid_name_is_refused_on_the_command_line_before_the_model_loads(cmd, name, rec, tmp_path, capsys,
                                                                               monkeypatch):
    def no_model(m):
        raise AssertionError("the model was loaded")
    monkeypatch.setattr(S, "_extractor", no_model)
    wav, rttm_text = rec
    r = tmp_path / "rec.rttm"
    r.write_text(rttm_text, encoding="utf-8")
    args = ([cmd, name, "en", str(wav), "--ffmpeg", "/bin/cat"] if cmd == "enroll"
            else [cmd, name, str(wav), str(r), "speaker_1", "--source", "x"])
    rc = S.main([*args, "--voices", str(tmp_path / "v"), "--model", str(tmp_path / "m.onnx")])
    err = capsys.readouterr().err
    assert rc == 1 and err == f"speaker_id: {errors.render('name_invalid', {})}\n"
    assert not (tmp_path / "v").exists()


FAKE_FFMPEG = r"""
import os, shutil, sys
a = sys.argv[1:]
src, dst = a[a.index("-i") + 1], a[-1]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calls"), "a", encoding="utf-8") as f:
    f.write(dst + "\n")
if "unreadable" in os.path.basename(src):
    sys.stderr.write(os.path.basename(src) + ": Invalid data found when processing input\n")
    sys.exit(1)
shutil.copyfile(src, dst)
"""


@pytest.fixture
def clips(tmp_path):
    """A 20 s clip of one placeholder voice, a clip ffmpeg can't read, and an ffmpeg stand-in."""
    good = tmp_path / "c1.wav"
    good.write_bytes(synth.tone_track_wav([(20.0, 330.0, 0.5)]))
    bad = tmp_path / "unreadable.m4a"
    bad.write_bytes(b"not audio")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    ff = bin_ / "ffmpeg"
    ff.write_text(f"#!{sys.executable}\n" + FAKE_FFMPEG, encoding="utf-8")
    ff.chmod(0o755)
    return good, bad, ff


def test_a_clip_ffmpeg_cant_read_after_one_that_went_in(model, clips, tmp_path, capsys):
    good, bad, ff = clips
    work = tmp_path / "work"
    work.mkdir()
    v = tmp_path / "v"
    with pytest.raises(S.ClipUnreadable) as ei:
        S.enroll_clips("Brimdal", [good, bad], "en", v, model, ff, {"PATH": "/usr/bin:/bin", "TMPDIR": str(work)},
                       False)
    e = ei.value
    assert isinstance(e, errors.AudioIntakeError) and isinstance(e, S.SpeakerIdError)
    assert (e.code, e.exit, e.detail) == ("decode_failed", 1, {"file": "unreadable.m4a"})
    assert [r["source"] for r in e.results] == ["sample c1.wav"]
    assert e.stderr_tail == ["unreadable.m4a: Invalid data found when processing input"]
    assert len(json.loads((v / "brimdal.json").read_text(encoding="utf-8"))["vectors"]) == 1
    dsts = (ff.parent / "calls").read_text(encoding="utf-8").split()
    assert len(dsts) == 2 and all(Path(d).parent == work for d in dsts)      # the copies went in the work folder
    assert list(work.iterdir()) == []                                          # and are gone

    rc = S.main(["enroll", "Quessa", "en", str(good), str(bad), "--ffmpeg", str(ff), "--voices",
                 str(tmp_path / "v2"), "--model", str(model)])
    out = capsys.readouterr()
    assert rc == 1
    assert out.out == "Quessa: +20 s from sample c1.wav (1 sources)\n"         # the clip that went in, as before
    assert out.err == ("unreadable.m4a: Invalid data found when processing input\n"
                       f"speaker_id: ffmpeg failed on {bad}\n")


def test_every_failure_after_a_clip_went_in_carries_the_results(model, clips, tmp_path, monkeypatch):
    good, _, ff = clips
    real = S._add
    calls = []

    def add(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(*a, **kw)
    monkeypatch.setattr(S, "_add", add)
    second = tmp_path / "c2.wav"
    second.write_bytes(good.read_bytes())
    with pytest.raises(OSError) as ei:
        S.enroll_clips("Brimdal", [good, second], None, tmp_path / "v", model, ff, {"PATH": "/usr/bin:/bin"}, False)
    assert [r["source"] for r in ei.value.results] == ["sample c1.wav"]


def test_enroll_clips_needs_an_absolute_executable_ffmpeg(model, clips, tmp_path):
    good, _, _ = clips
    for ff in ("ffmpeg", str(tmp_path / "missing")):
        with pytest.raises(S.SpeakerIdError, match="ffmpeg isn't an executable"):
            S.enroll_clips("Brimdal", [good], None, tmp_path / "v", model, ff, {}, False)


def test_every_subcommand_needs_voices_and_model(capsys):
    for cmd in ("match", "enroll-label", "enroll", "list", "check"):
        assert S.main([cmd]) == 1
        assert "needs --voices DIR and --model ONNX" in capsys.readouterr().err
    assert S.main(["nope", "--voices", "v", "--model", "m"]) == 1
    assert S.main([]) == 1


def test_the_extractor_refuses_another_model(tmp_path):
    pytest.importorskip("sherpa_onnx")
    m = tmp_path / "model.onnx"
    with pytest.raises(S.SpeakerIdError, match="missing"):
        S._extractor(m)
    m.write_bytes(b"something else")
    with pytest.raises(S.SpeakerIdError, match="not the calibrated CAM"):
        S._extractor(m)


def test_held_back_message_speaks_of_the_user(voices, model, rec, tmp_path, capsys):
    wav, rttm_text = rec
    r = tmp_path / "rec.rttm"
    r.write_text(rttm_text, encoding="utf-8")
    rc = S.main(["enroll-label", "Brimdal", str(wav), str(r), "speaker_0", "--source", "x", "--voices", str(voices),
                 "--model", str(model)])
    err = capsys.readouterr().err
    assert rc == 1 and err.endswith("If the user confirms it by ear anyway, rerun with --force\n")
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
