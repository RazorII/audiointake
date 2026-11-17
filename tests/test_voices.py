"""Voiceprints: names, the folder scan, the digest, list/remove/rename, and enroll from clips or a label."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from audiointake import errors, lock, platforms, voices
from audiointake.errors import UsageError

from fakes import audio, cli, cli_json, stub_steps, use_fakes, voiceprint

A, B, C = "Vandrel Oskel", "Nirvo Crondi", "Quibbet Lumvar"


@pytest.fixture
def fx(monkeypatch, tmp_path):
    return use_fakes(monkeypatch, tmp_path)


def vdir() -> Path:
    return platforms.data_dir() / "voices"


@pytest.mark.parametrize("name", ["Ana", "Jane Doe", "x" * 64, "Ünïcødé Näme"])
def test_good_names(name):
    assert voices.check_name(name) == name


@pytest.mark.parametrize("name", ["", "   ", "x" * 65, "a/b", ".hidden", "nul\x00", "Jane\nDoe", "Jane\rDoe",
                                  "Jane\tDoe", "Jane\u200bDoe", "Jane\x7fDoe"])
def test_bad_names(name):
    with pytest.raises(UsageError) as e:
        voices.check_name(name)
    assert e.value.code == "name_invalid" and e.value.exit == 2


def test_scan_and_digest(fx):
    assert voices.digest(None) == "" and voices.digest(vdir()) == ""
    voiceprint(vdir(), A)
    d1 = voices.digest(vdir())
    assert len(d1) == 64
    voiceprint(vdir(), B, model_sha256="e" * 64)
    (vdir() / "list.json").write_text("[]")
    (vdir() / "broken.json").write_text("{")
    (vdir() / "._resource.json").write_text("{}")
    s = voices.scan(vdir())
    assert [p.name for p, _ in s.prints] == [f"{A.lower()}.json"]
    assert len(s.other_model) == 1 and s.skipped == 2
    assert voices.digest(vdir()) == d1                         # other models and stray files don't count
    p = vdir() / f"{A.lower()}.json"
    d = json.loads(p.read_text())
    d["vectors"][0]["vec"] = [9.0, 9.0, 9.0]
    p.write_text(json.dumps(d))
    assert voices.digest(vdir()) == d1                         # vectors never enter the digest
    d["updated"] = "2026-05-05"
    p.write_text(json.dumps(d))
    assert voices.digest(vdir()) != d1
    assert dict(voices.scan_notes(vdir())) == {"voices_model_mismatch": 1, "voices_skipped": 2}


def test_list_remove_rename(fx, capsys):
    voiceprint(vdir(), A, seconds=31.0)
    voiceprint(vdir(), B, model_sha256="e" * 64)
    code, obj, _ = cli_json(capsys, "voices", "list")
    assert code == 0
    assert [(v["name"], v["model_ok"], v["seconds"]) for v in obj["voices"]] == [(B, False, 30.0), (A, True, 31.0)]
    assert "vec" not in json.dumps(obj["voices"])
    code, out, _ = cli(capsys, "voices", "list")
    assert f"{B}: 1 sources, 30 s, made with another speaker model" in out
    code, obj, _ = cli_json(capsys, "voices", "rename", A, "Vandrel Q Oskel")
    assert code == 0 and obj["file"] == "vandrel q oskel.json"
    newp = vdir() / "vandrel q oskel.json"
    assert json.loads(newp.read_text())["name"] == "Vandrel Q Oskel" and not (vdir() / f"{A.lower()}.json").exists()
    assert stat.S_IMODE(newp.stat().st_mode) == 0o600
    code, obj, _ = cli_json(capsys, "voices", "rename", "vandrel q oskel", "VANDREL Q OSKEL")   # case only
    assert code == 0 and json.loads(newp.read_text())["name"] == "VANDREL Q OSKEL"
    code, obj, _ = cli_json(capsys, "voices", "rename", B, "Vandrel Q Oskel")
    assert code == 2 and obj["error"]["code"] == "voice_exists"
    assert obj["error"]["message"] == "There's already a voiceprint called Vandrel Q Oskel."
    code, obj, _ = cli_json(capsys, "voices", "rename", B, "../x")
    assert code == 2 and obj["error"]["code"] == "name_invalid"
    code, obj, _ = cli_json(capsys, "voices", "remove", "nobody")
    assert code == 2 and obj["error"]["code"] == "voice_not_found"
    assert obj["error"]["message"] == "There's no voiceprint called nobody."
    code, obj, _ = cli_json(capsys, "voices", "remove", B)
    assert code == 0 and obj["removed"] == B and not (vdir() / f"{B.lower()}.json").exists()


def test_voices_folder_option(fx, tmp_path, capsys):
    other = tmp_path / "prints"
    voiceprint(other, A)
    code, obj, _ = cli_json(capsys, "voices", "list", "--voices", other)
    assert [v["name"] for v in obj["voices"]] == [A]


def test_changes_need_the_lock(fx):
    voiceprint(vdir(), A)
    with pytest.raises(RuntimeError):
        voices.remove(vdir(), A)
    with lock.run_lock():
        voices.remove(vdir(), A)


# --------------------------------------------------------------------------- enroll
def test_enroll_help_carries_the_consent_line(capsys):
    code, out, _ = cli(capsys, "enroll", "--help")
    assert code == 0
    assert ("\nVoiceprints are biometric data: only keep one for someone who has agreed to having a voiceprint "
            "kept.\n") in out


def test_enroll_clips_saves_and_says_how_to_remove(fx, tmp_path, capsys):
    clip = audio(tmp_path / "reading.m4a")
    code, out, _ = cli(capsys, "enroll", A, clip, "--lang", "en")
    assert code == 0
    assert out.strip() == (f"Saved. `audiointake voices remove '{A}'` deletes the voiceprint; transcripts and records "
                           "already made keep the name they show.")
    p = vdir() / f"{A.lower()}.json"
    assert p.exists() and stat.S_IMODE(vdir().stat().st_mode) == 0o700
    code, obj, _ = cli_json(capsys, "enroll", B, clip)
    assert code == 0 and obj["record"]["command"] == "enroll" and obj["record"]["voice"]["name"] == B
    assert obj["saved"].startswith("Saved.")


REFUSALS = [("two_voices", "two_voices"), ("mixed_voices", "mixed_voices"),
            ("matches_someone_else", "matches_someone_else"), ("not_matching_print", "not_matching_print"),
            ("two_voices,not_matching_print", "not_matching_print"),
            ("mixed_voices,matches_someone_else", "matches_someone_else")]


@pytest.mark.parametrize("held,first", REFUSALS)
def test_enroll_refusals(fx, tmp_path, capsys, held, first):
    clip = audio(tmp_path / "reading.m4a", enroll=f"held:{held}")
    code, obj, _ = cli_json(capsys, "enroll", A, clip)
    assert code == 4
    rec = obj["record"]
    assert rec["error"]["code"] == first and rec["status"] == "refused"
    assert sorted(r["code"] for r in rec["refusals"]) == sorted(held.split(","))
    assert rec["error"]["message"] == errors.render(first, rec["error"]["params"])
    assert not (vdir() / f"{A.lower()}.json").exists()
    code, obj, _ = cli_json(capsys, "enroll", A, clip, "--force")
    assert code == 0


def test_enroll_too_little_speech(fx, tmp_path, capsys):
    clip = audio(tmp_path / "short.m4a", enroll="little:5.2")
    code, obj, _ = cli_json(capsys, "enroll", A, clip)
    assert code == 4 and obj["record"]["error"]["code"] == "too_little_speech"
    assert obj["record"]["error"]["message"].startswith("short.m4a has 5.2 s of clear speech from one voice")
    code, out, err = cli(capsys, "enroll", A, clip)
    assert code == 4 and "a voiceprint needs at least 8 s" in err


def test_enroll_from_a_transcript_label(fx, tmp_path, capsys):
    a = audio(tmp_path / "Team sync.m4a")
    code, obj, _ = cli_json(capsys, "transcribe", a)
    rid = obj["records"][0]["id"]
    code, obj, _ = cli_json(capsys, "enroll", A, "--from", rid, "--label", "speaker_1")
    assert code == 0, obj
    src = obj["record"]["voice"]["sources"][0]
    assert src.startswith("Team sync.m4a speaker_1 ")
    assert obj["record"]["parent"] == rid
    code, obj, _ = cli_json(capsys, "enroll", A, "--from", rid, "--label", "speaker_6")
    assert code == 2 and obj["error"]["code"] == "unknown_label"
    a.write_text(a.read_text() + " ")
    code, obj, _ = cli_json(capsys, "enroll", A, "--from", rid, "--label", "speaker_1")
    assert code == 5 and obj["record"]["error"]["code"] == "input_changed"


def test_enroll_from_a_label_with_too_little_speech(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", enroll="little:3.0")
    rid = cli_json(capsys, "transcribe", a)[1]["records"][0]["id"]
    code, obj, _ = cli_json(capsys, "enroll", A, "--from", rid, "--label", "speaker_0")
    assert code == 4
    assert obj["record"]["error"]["message"].startswith("speaker_0 has 3 s of speech without anyone talking over them")


@pytest.mark.parametrize("args", [[A], [A, "x.m4a", "--label", "speaker_0"], [A, "x.m4a", "--from", "r-20260101-00000000"],
                                  [A, "--from", "r-20260101-00000000"], [".x", "x.m4a"]])
def test_enroll_argument_shapes(fx, tmp_path, capsys, args):
    code, obj, _ = cli_json(capsys, "enroll", *args)
    assert code == 2 and obj["error"]["code"] in ("bad_arguments", "name_invalid")


def test_enroll_missing_clip(fx, tmp_path, capsys):
    code, obj, _ = cli_json(capsys, "enroll", A, tmp_path / "absent.m4a")
    assert code == 2 and obj["record"]["error"]["code"] == "input_missing"


def test_enroll_into_a_missing_custom_folder(fx, tmp_path, capsys):
    clip = audio(tmp_path / "reading.m4a")
    code, obj, _ = cli_json(capsys, "enroll", A, clip, "--voices", tmp_path / "nowhere")
    assert code == 2 and obj["record"]["error"]["code"] == "input_missing"


def test_relabel_refuses_a_name_with_a_line_break(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a")
    cli_json(capsys, "transcribe", a)
    code, obj, _ = cli_json(capsys, "relabel", a, "--label", "speaker_0", "--name", "Jane\nDoe")
    assert code == 2 and obj["error"]["code"] == "name_invalid"


def test_enroll_result_counts_what_was_added(fx, tmp_path, capsys):
    one, two = audio(tmp_path / "one.m4a"), audio(tmp_path / "two.m4a", seed=2)
    stub_steps.WORK_DIRS.clear()
    code, obj, _ = cli_json(capsys, "enroll", A, one, two)
    assert code == 0, obj
    assert obj["record"]["result"] == {"name": A, "added": 2, "skipped": 0, "seconds": 62.0, "sources": 2}
    work = stub_steps.WORK_DIRS[-1]
    assert work is not None and work.parent.parent == platforms.cache_dir() / "work"      # the run's work folder
    assert not work.exists()
    code, obj, _ = cli_json(capsys, "enroll", A, one)                                     # already there
    assert code == 0 and obj["record"]["result"]["added"] == 0 and "saved" not in obj
    assert obj["nothing_added"] == f"Nothing was added: {A}'s voiceprint already has these recordings."
    assert obj["nothing_added"] == errors.render("already_enrolled", {"name": A})
    code, out, _ = cli(capsys, "enroll", A, one)
    assert out.strip() == obj["nothing_added"] and "Saved." not in out


@pytest.mark.parametrize("before", [False, True], ids=["new-print", "existing-print"])
def test_a_refused_clip_keeps_none_of_the_others(fx, tmp_path, capsys, before):
    p = vdir() / f"{A.lower()}.json"
    if before:
        voiceprint(vdir(), A)
    was = p.read_bytes() if before else None
    good, short = audio(tmp_path / "good.m4a"), audio(tmp_path / "short.m4a", enroll="little:3.0")
    code, obj, _ = cli_json(capsys, "enroll", A, good, short)
    assert code == 4 and obj["record"]["error"]["code"] == "too_little_speech"
    assert obj["record"]["error"]["message"].startswith("short.m4a has 3 s") and "Nothing was saved." in \
        obj["record"]["error"]["message"]
    assert (p.read_bytes() == was) if before else not p.exists()                        # nothing kept, truly


def test_an_engine_failure_is_a_coded_error(fx, tmp_path, capsys):
    voiceprint(vdir(), A)
    was = (vdir() / f"{A.lower()}.json").read_bytes()
    code, obj, err = cli_json(capsys, "enroll", A, audio(tmp_path / "ok.m4a"), audio(tmp_path / "x.m4a", enroll="engine"))
    assert code == 3 and obj["record"]["error"]["code"] == "pack_broken"
    assert obj["record"]["error"]["message"].startswith("A file of the core pack has changed on disk")
    assert (vdir() / f"{A.lower()}.json").read_bytes() == was and "Traceback" not in err
    a = audio(tmp_path / "t.m4a", enroll="engine")
    rid = cli_json(capsys, "transcribe", a)[1]["records"][0]["id"]
    code, obj, _ = cli_json(capsys, "enroll", A, "--from", rid, "--label", "speaker_0")
    assert code == 3 and obj["record"]["error"]["code"] == "pack_broken"


def test_a_print_made_with_another_model_is_refused_first(fx, tmp_path, capsys):
    voiceprint(vdir(), A, model_sha256="e" * 64)
    was = (vdir() / f"{A.lower()}.json").read_bytes()
    code, obj, _ = cli_json(capsys, "enroll", A, audio(tmp_path / "reading.m4a"))
    assert code == 2 and obj["record"]["error"]["code"] == "voice_other_model"
    assert f"`audiointake voices remove '{A}'` deletes it." in obj["record"]["error"]["message"]
    assert (vdir() / f"{A.lower()}.json").read_bytes() == was
    (vdir() / f"{B.lower()}.json").write_text("not json")
    code, obj, _ = cli_json(capsys, "enroll", B, audio(tmp_path / "reading.m4a"))
    assert code == 2 and obj["record"]["error"]["code"] == "voice_other_model"
    for name, body in ((A, None), (B, None), (C, "{}")):     # the remedy the refusal names works on each kind
        if body is not None:
            (vdir() / f"{name.lower()}.json").write_text(body)
        code, obj, _ = cli_json(capsys, "voices", "remove", name)
        assert code == 0 and not (vdir() / f"{name.lower()}.json").exists(), (name, obj)
    code, obj, _ = cli_json(capsys, "voices", "remove", A)
    assert code == 2 and obj["error"]["code"] == "voice_not_found"


def test_a_clip_without_audio_is_unsupported(fx, tmp_path, capsys):
    notes = tmp_path / "notes.pdf"
    notes.write_text("%PDF-1.7 not audio")
    code, obj, _ = cli_json(capsys, "enroll", A, notes)
    assert code == 2 and obj["record"]["error"]["code"] == "unsupported_input"
    assert not (vdir() / f"{A.lower()}.json").exists()


def test_restore_puts_the_print_back(fx):
    voiceprint(vdir(), A)
    p = vdir() / f"{A.lower()}.json"
    was = voices.snapshot(vdir(), A)
    p.write_text("changed")
    with pytest.raises(RuntimeError):
        voices.restore(vdir(), A, was)                             # needs the lock, like every voiceprint change
    with lock.run_lock():
        voices.restore(vdir(), A, was)
        assert p.read_bytes() == was and stat.S_IMODE(p.stat().st_mode) == 0o600
        voices.restore(vdir(), A, None)
        assert not p.exists()
        voices.restore(vdir(), A, None)                            # nothing there: nothing to do
