"""transcribe, apply and relabel on fakes: the steps, every classic sidecar row, warnings, inputs, resume, plan."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from audiointake import fsutil, outputs, pipeline, platforms, record, voices

from fakes import CANARY, audio, cli, cli_json, stub_steps, use_fakes, voiceprint

GAPFILL = "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill"
A, B, C = "Vandrel Oskel", "Nirvo Crondi", "Quibbet Lumvar"


@pytest.fixture
def fx(monkeypatch, tmp_path):
    stub_steps.CALLS.clear()
    return use_fakes(monkeypatch, tmp_path)


def voices_dir() -> Path:
    return platforms.data_dir() / "voices"


def run(capsys, *args):
    code, obj, err = cli_json(capsys, "transcribe", *args)
    return code, obj["records"], err


def codes(rec, key="warnings"):
    return [w["code"] for w in rec[key]]


def diarmeta(folder: Path, stem: str) -> str:
    return (folder / f"{stem}.diarmeta.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the basic run
def test_default_run_writes_md_and_json(fx, tmp_path, capsys):
    a = audio(tmp_path / "Team sync.m4a")
    code, recs, err = run(capsys, a)
    assert code == 0
    rec = recs[0]
    assert rec["status"] == "done" and rec["exit"] == 0 and rec["command"] == "transcribe"
    md, js = tmp_path / "Team sync (transcript).md", tmp_path / "Team sync (transcript).json"
    assert {o["path"] for o in rec["outputs"]} == {str(md), str(js)}
    for o in rec["outputs"]:
        assert fsutil.sha256_file(o["path"]) == o["sha256"]
    text = md.read_text()
    assert text.startswith("# Team sync\n\n") and "· 2 speakers (0 named)" in text and "**Speaker 1** [00:00]" in text
    doc = json.loads(js.read_text())
    assert outputs.seal_ok(js.read_bytes())
    assert doc["record_id"] == rec["id"] and doc["engines"]["asr_tag"] == GAPFILL
    assert doc["outputs"] == [{"kind": "md", "name": md.name, "sha256": fsutil.sha256_file(md)}]
    assert doc["displays"] == {"speaker_0": "Speaker 1", "speaker_1": "Speaker 2"}
    assert doc["rttm"].startswith("SPEAKER Team sync.16k 1 ")
    assert "1 of 1: Team sync.m4a: transcribing" in err and "1 of 1: Team sync.m4a: speakers" in err
    assert stub_steps.CALLS[-1]["template"] == "Team sync" and stub_steps.CALLS[-1]["fmt"] == "all"
    assert stub_steps.CALLS[-1]["duration_s"] == pytest.approx(60.0)
    assert list((platforms.cache_dir() / "work").iterdir()) == []      # the run's work folder is gone
    assert record.speed_median("core") is not None


def test_txt_and_sidecars(fx, tmp_path, capsys):
    a = audio(tmp_path / "Call.m4a")
    code, recs, _ = run(capsys, a, "--format", "txt,sidecars")
    assert code == 0
    names = sorted(Path(o["path"]).name for o in recs[0]["outputs"])
    assert names == ["Call (transcript).txt", "Call.asr.json", "Call.diar.rttm", "Call.diarmeta.txt", "Call.txt"]
    assert (tmp_path / "Call.txt").read_bytes() == (tmp_path / "Call (transcript).txt").read_bytes()
    assert (tmp_path / "Call.txt").read_text().startswith("Speaker 1: ")
    assert diarmeta(tmp_path, "Call") == (f"speakers=2 runs=1 threshold=0.4 engine=fluidaudio-v0.18.0-nemotron3-offline "
                                          f"spatial=no asr={GAPFILL} named=-\n")
    assert json.loads((tmp_path / "Call.asr.json").read_text())["sentences"]
    assert (tmp_path / "Call.diar.rttm").read_text().startswith("SPEAKER Call.16k 1 ")
    kinds = {o["kind"] for o in recs[0]["outputs"]}
    assert kinds == {"txt", "sidecar"}


# --------------------------------------------------------------------------- every classic diarmeta row
ROWS = [
    # id, setup, desc, speakers=, runs=, named=, .diar.rttm?, .speakers.json?, prints?, extra warning
    ("no-fluidaudiocli", {"fluidaudio": False}, {}, "?", "-", "-", False, False, True, "no_speaker_labels"),
    ("diar-wav-failed", {}, {"fail": {"diar_wav": 1}}, "?", "-", "-", False, False, True, "speakers_failed"),
    ("diarize-failed", {}, {"fail": {"diarize": "exit"}}, "?", "-", "-", False, False, True, "speakers_failed"),
    ("diarize-failed-left-a-file", {}, {"fail": {"diarize": "partial"}}, "?", "-", "-", True, False, True,
     "speakers_failed"),
    ("empty-rttm", {}, {"fail": {"diarize": "empty"}}, "0", "1", "-", False, False, True, None),
    ("diarize-ok-no-file", {}, {"fail": {"diarize": "nofile"}}, "0", "1", "-", False, False, True, None),
    ("counting-crashed", {}, {"fail": {"diarize": "malformed"}, "match": {"speaker_0": A}}, "0", "1", A, True, True,
     True, "merge_failed"),
    ("no-voiceprints", {}, {}, "2", "1", "-", True, False, False, None),
    ("speaker-id-failed", {}, {"fail": {"match": "exit"}}, "2", "1", "-", True, False, True, "speakers_failed"),
    ("speaker-id-failed-left-a-file", {}, {"fail": {"match": "partial"}}, "2", "1", "-", True, True, True,
     "speakers_failed"),
    ("nobody-named", {}, {}, "2", "1", "-", True, True, True, None),
    ("named", {}, {"match": {"speaker_0": A, "speaker_1": B}}, "2", "1", f"{B},{A}", True, True, True, None),
    ("names-unreadable", {}, {"fail": {"match": "garbage"}}, "2", "1", "?", True, True, True, None),
    ("names-missing", {}, {"fail": {"match": "nofile"}}, "2", "1", "?", True, False, True, None),
]


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_every_diarmeta_row(monkeypatch, tmp_path, capsys, row):
    rid, setup, desc, spk, runs, named, has_rttm, has_spk, prints, warn = row
    use_fakes(monkeypatch, tmp_path, **setup)
    if prints:
        voiceprint(voices_dir(), A)
        voiceprint(voices_dir(), B)
        voiceprint(voices_dir(), C)
    a = audio(tmp_path / "Row.m4a", **desc)
    code, recs, _ = run(capsys, a, "--format", "sidecars")
    assert code == 0, recs[0]["error"]
    line = diarmeta(tmp_path, "Row")
    assert line == (f"speakers={spk} runs={runs} threshold=0.4 engine=fluidaudio-v0.18.0-nemotron3-offline "
                    f"spatial=no asr={GAPFILL} named={named}\n")
    assert (tmp_path / "Row.diar.rttm").exists() == has_rttm
    assert (tmp_path / "Row.speakers.json").exists() == has_spk
    assert (tmp_path / "Row.txt").exists() and (tmp_path / "Row.asr.json").exists()
    rec = recs[0]
    if warn:
        assert warn in codes(rec)
    else:
        assert not {"speakers_failed", "merge_failed"} & set(codes(rec))
    txt = (tmp_path / "Row.txt").read_text()
    if warn == "merge_failed":
        assert ":" not in txt.split(" ", 1)[0] and txt.count("\n") == 1      # the ensemble's one-line text
    if rid == "named":
        assert txt.startswith(f"{A}: ") or txt.startswith(f"{B}: ")


def test_spatial_follows_any_four_channel_stream(fx, tmp_path, capsys):
    a = audio(tmp_path / "Memo.qta", channels=[2, 4])
    code, recs, _ = run(capsys, a, "--format", "sidecars")
    assert " spatial=yes " in diarmeta(tmp_path, "Memo")
    assert recs[0]["input"]["spatial"] == "yes" and recs[0]["input"]["channels"] == [2, 4]


# --------------------------------------------------------------------------- failures of the first model
@pytest.mark.parametrize("how", ["exit", "partial"])
def test_asr_failed_writes_nothing(fx, tmp_path, capsys, how):
    a = audio(tmp_path / "Bad.m4a", fail={"primary": how})
    code, recs, err = run(capsys, a)
    assert code == 1
    assert recs[0]["error"]["code"] == "asr_failed" and recs[0]["status"] == "failed" and recs[0]["outputs"] == []
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["Bad.m4a"]
    assert recs[0]["log"] in recs[0]["error"]["message"]
    assert "fake parakeet:" in Path(recs[0]["log"]).read_text()               # the child's stderr tail is logged


def test_decode_failed(fx, tmp_path, capsys):
    a = audio(tmp_path / "Bad.m4a", fail={"decode": 1})
    code, recs, _ = run(capsys, a)
    assert code == 1 and recs[0]["error"]["code"] == "decode_failed"


def test_human_output(fx, tmp_path, capsys):
    a = audio(tmp_path / "Team sync.m4a")
    b = audio(tmp_path / "Bad.m4a", fail={"primary": "exit"})
    code, out, err = cli(capsys, "transcribe", a, b)
    assert code == 1
    assert "Team sync.m4a: 0:01:00, 2 speakers (0 named). Files: Team sync (transcript).md" in out
    assert "Bad.m4a: Transcribing Bad.m4a failed, so nothing was written for it." in err


# --------------------------------------------------------------------------- warnings from the ensemble
def test_fill_failed(fx, tmp_path, capsys):
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a", fail={"fill": "exit"}), "--format", "sidecars")
    assert "fill_failed" in codes(recs[0]) and recs[0]["engines"]["asr_tag"] == "parakeet-ultra-mlx"
    assert "asr=parakeet-ultra-mlx named" in diarmeta(tmp_path, "x")


def test_hungarian_vote_failed_and_thai_failed(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "vote_failed", "thai": {"screened": True, "failed": True}})
    code, recs, _ = run(capsys, a, "--format", "sidecars")
    assert {"hungarian_vote_failed", "thai_failed"} <= set(codes(recs[0]))
    assert "asr=parakeet-ultra-mlx+hu-failed+th-failed named" in diarmeta(tmp_path, "x")


def test_detected_hungarian_without_the_pack(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.31})
    code, recs, _ = run(capsys, a)
    rec = recs[0]
    w = next(x for x in rec["warnings"] if x["code"] == "pack_not_installed")
    assert w["params"]["reason"] == "detected" and w["params"]["code"] == "hu" and w["params"]["size"] == 3.1
    assert w["message"].startswith("About 31 % of the words in this recording look Hungarian.")
    assert f"apply {rec['id']} --lang hu" in w["message"]
    offer = next(o for o in rec["offers"] if o["kind"] == "language")
    assert offer["options"] == ["hu"] and offer["apply_flag"] == "--lang"
    assert offer["summary"][0].startswith("Hungarian is read better with the Hungarian pack")


def test_thai_found_without_the_pack(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", ensemble={"thai": {"screened": True, "screen_only": True, "thai_windows": 2}})
    code, recs, _ = run(capsys, a, "--format", "sidecars")
    w = next(x for x in recs[0]["warnings"] if x["code"] == "pack_not_installed")
    assert w["params"]["code"] == "th" and w["message"].startswith("Part of this recording is Thai")
    assert f"asr={GAPFILL} named" in diarmeta(tmp_path, "x")                     # screen-only: no tag change
    assert recs[0]["signals"]["thai_windows"] == 2


def test_detected_pack_without_room(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, free_bytes=1e9)
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4}))
    w = next(x for x in recs[0]["warnings"] if x["code"] == "pack_not_installed")
    assert w["params"]["reason"] == "no_room"
    assert w["message"] == "Hungarian was detected, but its pack needs 3.8 GB and this Mac has 1 GB free, so the core models read it."
    assert not [o for o in recs[0]["offers"] if o["kind"] == "language"]


# --------------------------------------------------------------------------- --lang
def test_lang_without_its_pack_warns_and_never_downloads(fx, tmp_path, capsys, network_attempts):
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a"), "--lang", "hu")
    assert code == 0
    w = next(x for x in recs[0]["warnings"] if x["code"] == "pack_not_installed")
    assert w["params"]["reason"] == "asked" and w["params"]["code"] == "hu"
    assert w["message"].startswith("You asked for Hungarian, but its pack isn't installed")
    assert [o["options"] for o in recs[0]["offers"] if o["kind"] == "language"] == [["hu"]]
    assert ("resolve", {"need": {"hu_vote"}, "voices_dir": voices_dir(), "names": True, "speakers": True}) in fx.calls
    assert not [c for c in fx.calls if c[0] == "install"]
    assert network_attempts == []
    assert stub_steps.CALLS[-1]["force_lang"] == "hu"
    assert recs[0]["options"]["lang"] == ["hu"]


def test_lang_hu_forces_the_vote_with_the_pack(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, packs=("core", "hu"))
    stub_steps.CALLS.clear()
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a"), "--lang", "hu")
    assert recs[0]["engines"]["asr_tag"] == "whisper-large-v3+ultra/v3-vote" and recs[0]["warnings"] == []
    code, recs, _ = run(capsys, audio(tmp_path / "y.m4a"), "--lang", "hu,en")
    assert stub_steps.CALLS[-1]["force_lang"] is None
    code, recs, _ = run(capsys, audio(tmp_path / "z.m4a"), "--lang", "th")
    assert stub_steps.CALLS[-1]["force_lang"] is None


@pytest.mark.parametrize("value", ["xx", "en,zz"])
def test_unknown_language(fx, tmp_path, capsys, value):
    code, obj, _ = cli_json(capsys, "transcribe", audio(tmp_path / "x.m4a"), "--lang", value)
    assert code == 2 and obj["error"]["code"] == "unknown_language" and obj["records"] == []
    assert "It reads bg, cs, da" in obj["error"]["message"]


# --------------------------------------------------------------------------- speakers and names
def test_speaker_count_differs_and_cap(fx, tmp_path, capsys):
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a"), "--speakers", "3")
    w = next(x for x in recs[0]["warnings"] if x["code"] == "speaker_count_differs")
    assert w["params"]["expected"] == 3 and w["params"]["found"] == 2
    assert f"audiointake relabel {recs[0]['id']} --label speaker_K --name NAME" in w["message"]
    code, recs, _ = run(capsys, audio(tmp_path / "y.m4a", speakers=8, sentences=40, turn=[0.5, 1.5]), "--speakers", "8")
    assert codes(recs[0]) == ["speaker_cap"]


def test_no_names_and_no_speakers(fx, tmp_path, capsys):
    voiceprint(voices_dir(), A)
    a = audio(tmp_path / "x.m4a", match={"speaker_0": A})
    code, recs, _ = run(capsys, a, "--no-names", "--format", "md,sidecars")
    assert not (tmp_path / "x.speakers.json").exists() and recs[0]["speakers"][0].get("name") is None
    assert recs[0]["options"]["names"] is False
    b = audio(tmp_path / "y.m4a")
    code, recs, _ = run(capsys, b, "--no-speakers", "--format", "md,sidecars")
    assert recs[0]["warnings"] == [] and recs[0]["speakers"] == []
    assert diarmeta(tmp_path, "y").startswith("speakers=? runs=- ")
    md = (tmp_path / "y (transcript).md").read_text()
    assert "speaker" not in md.splitlines()[2] and "**" not in md
    assert [c for c in fx.calls if c[0] == "resolve"][-1][1]["speakers"] is False


def test_names_from_a_custom_folder(fx, tmp_path, capsys):
    other = tmp_path / "prints"
    voiceprint(other, A)
    a = audio(tmp_path / "x.m4a", match={"speaker_1": A})
    code, recs, _ = run(capsys, a, "--voices", other)
    rec = recs[0]
    assert rec["options"]["voices"] == "custom"
    assert [s["name"] for s in rec["speakers"] if s.get("name")] == [A]
    doc = json.loads((tmp_path / "x (transcript).json").read_text())
    assert doc["speakers"] == {"speaker_1": {"name": A, "score": 0.71, "speech_s": 13.5, "why": "named"}}
    assert doc["displays"]["speaker_1"] == A
    assert "· 2 speakers (1 named)" in (tmp_path / "x (transcript).md").read_text()


def test_voiceprint_folder_notes(fx, tmp_path, capsys):
    voiceprint(voices_dir(), A)
    voiceprint(voices_dir(), B, model_sha256="f" * 64)
    (voices_dir() / "notes.json").write_text("[1, 2]")
    (voices_dir() / ".hidden.json").write_text("{}")
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a"))
    notes = {n["code"]: n for n in recs[0]["notes"]}
    assert notes["voices_model_mismatch"]["params"]["count"] == 1
    assert notes["voices_skipped"]["params"]["count"] == 1
    assert notes["voices_skipped"]["message"] == "1 file(s) in the voiceprint folder aren't voiceprints and were skipped."


def test_a_folder_with_nobody_to_match_skips_speaker_id_quietly(fx, tmp_path, capsys):
    """Only prints of another model, or files that aren't prints: speaker ID has nobody to match, so it doesn't
    run (the notes say why nobody was named); it no longer runs, fails and warns on every recording."""
    voiceprint(voices_dir(), B, model_sha256="f" * 64)
    (voices_dir() / "notes.json").write_text("[1, 2]")
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a"), "--format", "sidecars")
    assert code == 0 and "speakers_failed" not in codes(recs[0])
    assert {"voices_model_mismatch", "voices_skipped"} <= set(codes(recs[0], "notes"))
    assert diarmeta(tmp_path, "x").endswith(" named=-\n") and not (tmp_path / "x.speakers.json").exists()


def test_remember_offer_for_unnamed_labels_with_enough_speech(fx, tmp_path, capsys):
    voiceprint(voices_dir(), A)
    a = audio(tmp_path / "x.m4a", sentences=40, match={"speaker_0": A})
    code, recs, _ = run(capsys, a)
    rec = recs[0]
    offer = next(o for o in rec["offers"] if o["kind"] == "remember")
    assert offer["options"] == ["speaker_1"] and offer["apply_flag"] is None
    assert offer["summary"][0].startswith("speaker_1 has 13.5 s of clear speech.")
    assert f"--from {rec['id']} --label speaker_1" in offer["summary"][0]


# --------------------------------------------------------------------------- speech
def test_no_speech(fx, tmp_path, capsys):
    code, recs, _ = run(capsys, audio(tmp_path / "Quiet.m4a", sentences=0, speakers=0), "--format", "md,json,sidecars")
    rec = recs[0]
    assert code == 0 and codes(rec, "notes")[-1] == "no_speech"
    assert rec["notes"][-1]["message"] == "No speech was found in Quiet.m4a; its transcript is empty."
    assert (tmp_path / "Quiet.txt").read_text() == "\n"
    assert diarmeta(tmp_path, "Quiet").startswith("speakers=0 runs=1 ")


def test_speech_not_transcribed(fx, tmp_path, capsys):
    code, recs, _ = run(capsys, audio(tmp_path / "x.m4a", sentences=2, diar_duration=200))
    w = next(x for x in recs[0]["warnings"] if x["code"] == "speech_not_transcribed")
    assert w["params"]["minutes"] >= 2
    assert " min " in w["message"] and "produced no words" in w["message"]


def test_untranscribed_arithmetic():
    segs = [{"start": 0.0, "dur": 100.0, "spk": "speaker_0"}, {"start": 50.0, "dur": 100.0, "spk": "speaker_1"}]
    gone, total = pipeline.untranscribed(segs, [{"start": 0, "end": 40, "text": "x"}, {"start": 60, "end": 70,
                                                                                         "text": ""}])
    assert (gone, total) == (110.0, 150.0)


def test_clean_seconds_matches_the_engines_cut():
    segs = [{"start": 0.0, "dur": 12.0, "spk": "a"}, {"start": 11.0, "dur": 4.0, "spk": "b"},
            {"start": 20.0, "dur": 1.6, "spk": "a"}]
    # a: 0.25..10.75 once b's overlap is cut: one 10 s piece, and the 0.5 s left over is under 1.5 s, as is
    # 20.25..21.35; b: 12.25..14.75 once a's overlap is cut
    assert pipeline.clean_seconds(segs, "a") == 10.0
    assert pipeline.clean_seconds(segs, "b") == pytest.approx(2.5)


# --------------------------------------------------------------------------- inputs
def test_input_problems(fx, tmp_path, capsys, monkeypatch):
    folder = tmp_path / "a folder.m4a"
    folder.mkdir()
    unreadable = audio(tmp_path / "locked.m4a")
    unreadable.chmod(0)
    notaudio = tmp_path / "words.m4a"
    notaudio.write_text("plain words, no sound")
    cloud = audio(tmp_path / "cloud.m4a")
    real = fsutil.is_dataless
    monkeypatch.setattr(fsutil, "is_dataless", lambda p: Path(p).name == "cloud.m4a" or real(p))
    try:
        code, recs, _ = run(capsys, tmp_path / "missing.m4a", folder, unreadable, notaudio, cloud)
    finally:
        unreadable.chmod(0o644)
    assert code == 2
    assert [r["error"]["code"] for r in recs] == ["input_missing", "input_not_file", "input_not_readable",
                                                  "unsupported_input", "input_not_downloaded"]
    assert all(r["status"] == "usage" and r["exit"] == 2 for r in recs)
    assert recs[1]["error"]["message"] == "a folder.m4a isn't a regular file. Pass the audio files themselves, not a folder."


def test_output_not_writable(fx, tmp_path, capsys):
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o500)
    try:
        code, recs, _ = run(capsys, audio(tmp_path / "x.m4a"), "--outdir", out)
    finally:
        out.chmod(0o700)
    assert code == 2 and recs[0]["error"]["code"] == "output_not_writable"
    code, recs, _ = run(capsys, audio(tmp_path / "y.m4a"), "--outdir", tmp_path / "absent")
    assert recs[0]["error"]["code"] == "output_not_writable"


def test_outdir_paths_from_and_duplicate_stems(fx, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "a").mkdir()
    one = audio(tmp_path / "a" / "x.m4a")
    two = audio(tmp_path / "x.mp3")
    lst = tmp_path / "list.txt"
    lst.write_text(f"{two}\n\n", encoding="utf-8")
    code, recs, _ = run(capsys, one, "--paths-from", lst, "--outdir", out)
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["x.m4a (transcript).json", "x.m4a (transcript).md",
                                                     "x.mp3 (transcript).json", "x.mp3 (transcript).md"]


def test_paths_from_missing(fx, tmp_path, capsys):
    code, obj, _ = cli_json(capsys, "transcribe", "--paths-from", tmp_path / "nope.txt")
    assert code == 2 and obj["error"]["code"] == "input_missing"
    code, obj, _ = cli_json(capsys, "transcribe")
    assert code == 2 and obj["error"]["code"] == "bad_arguments"


def test_exit_priority_across_inputs(fx, tmp_path, capsys):
    ok = audio(tmp_path / "ok.m4a")
    bad = audio(tmp_path / "bad.m4a", fail={"primary": "exit"})
    code, recs, _ = run(capsys, ok, bad, tmp_path / "missing.m4a")
    assert [r["exit"] for r in recs] == [0, 1, 2] and code == 2
    code, recs, _ = run(capsys, bad, audio(tmp_path / "ok2.m4a"))
    assert code == 1


def test_recorded_at_sources(fx, tmp_path, capsys):
    code, recs, _ = run(capsys, audio(tmp_path / "20261116 093000 Standup.m4a"),
                        audio(tmp_path / "meta.m4a", creation_time="2026-03-04T05:06:07"), audio(tmp_path / "plain.m4a"))
    srcs = [r["input"]["recorded_at_source"] for r in recs]
    assert srcs == ["filename", "metadata", "file_date"]
    assert recs[0]["input"]["recorded_at"] == "2026-11-16T09:30:00"
    assert "recorded_at_from_file_date" in codes(recs[2], "notes") and not codes(recs[0], "notes")
    assert (tmp_path / "20261116 093000 Standup (transcript).md").read_text().splitlines()[2].startswith("2026-11-16 · ")


def test_input_changed_during_the_run(fx, tmp_path, capsys, monkeypatch):
    a = audio(tmp_path / "x.m4a")
    stub = stub_steps.namespace()

    def changing(*args, **kw):
        res = stub.transcribe(*args, **kw)
        a.write_text(a.read_text() + " ")
        return res
    monkeypatch.setattr(pipeline, "engine", lambda: SimpleNamespace(**{**vars(stub), "transcribe": changing}))
    code, recs, _ = run(capsys, a)
    assert code == 5 and recs[0]["error"]["code"] == "input_changed" and recs[0]["status"] == "stale"
    assert f"since record {recs[0]['id']} was made" in recs[0]["error"]["message"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["x.m4a"]


# --------------------------------------------------------------------------- existing outputs and resume
def test_output_exists_and_overwrite(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a")
    (tmp_path / "x (transcript).md").write_text("mine")
    code, recs, _ = run(capsys, a)
    assert code == 2 and recs[0]["error"]["code"] == "output_exists"
    assert (tmp_path / "x (transcript).md").read_text() == "mine"
    code, recs, _ = run(capsys, a, "--overwrite")
    assert code == 0 and (tmp_path / "x (transcript).md").read_text().startswith("# x")


def test_sidecars_always_need_overwrite(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a")
    assert run(capsys, a, "--format", "sidecars")[0] == 0
    code, recs, _ = run(capsys, a, "--format", "sidecars")
    assert code == 2 and recs[0]["error"]["code"] == "output_exists"
    assert run(capsys, a, "--format", "sidecars", "--overwrite")[0] == 0


def test_resume_already_done_then_voices_changed(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", match={"speaker_0": A})
    code, recs, _ = run(capsys, a)
    first = recs[0]
    n = len(stub_steps.CALLS)
    code, recs, _ = run(capsys, a)
    assert code == 0 and codes(recs[0], "notes")[-1] == "already_done" and len(stub_steps.CALLS) == n
    assert recs[0]["parent"] == first["id"]
    assert {o["kind"] for o in recs[0]["outputs"]} == {"md", "json"}
    voiceprint(voices_dir(), A)
    code, recs, _ = run(capsys, a)
    assert codes(recs[0], "notes")[-1] == "voices_changed" and len(stub_steps.CALLS) == n
    assert [o["kind"] for o in recs[0]["offers"]] == ["names"]
    assert f"apply {recs[0]['id']} --choice names" in recs[0]["notes"][-1]["message"]
    (tmp_path / "x (transcript).md").write_text("edited")
    code, recs, _ = run(capsys, a)
    assert code == 2 and recs[0]["error"]["code"] == "output_exists"


def test_a_rerun_words_the_remember_offer_with_its_seconds(fx, tmp_path, capsys):
    voiceprint(voices_dir(), A)                       # the speaker worker runs, so labels have their seconds
    a = audio(tmp_path / "x.m4a")
    first = run(capsys, a)[1][0]
    remember = next(o for o in first["offers"] if o["kind"] == "remember")
    code, recs, _ = run(capsys, a)
    assert code == 0 and codes(recs[0], "notes")[-1] == "already_done" and recs[0]["speakers"] == []
    again = next(o for o in recs[0]["offers"] if o["kind"] == "remember")
    assert again["summary"] == [x.replace(first["id"], recs[0]["id"]) for x in remember["summary"]]
    assert " has 0 s " not in " ".join(again["summary"])     # the seconds came along with the offer


def test_other_options_are_not_a_resume(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a")
    run(capsys, a)
    code, recs, _ = run(capsys, a, "--no-names")
    assert code == 2 and recs[0]["error"]["code"] == "output_exists"


def test_plan_writes_only_its_record(fx, tmp_path, capsys):
    run(capsys, audio(tmp_path / "seed.m4a"))
    a = audio(tmp_path / "x.m4a", duration=600.0)
    n = len(stub_steps.CALLS)
    code, recs, _ = run(capsys, a, "--plan", "--lang", "th")
    rec = recs[0]
    assert code == 0 and rec["status"] == "plan" and rec["outputs"] == [] and rec["log"] is None
    assert rec["plan"]["routes"] == ["asr", "diarization"] and rec["plan"]["estimate_s"] > 0
    assert "pack_not_installed" in codes(rec) and len(stub_steps.CALLS) == n
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("x")) == ["x.m4a"]


# --------------------------------------------------------------------------- apply
def test_apply_lang_installs_and_redoes(fx, tmp_path, capsys):
    voiceprint(voices_dir(), A)                       # the speaker worker runs, so labels have their seconds
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4})
    first = run(capsys, a)[1][0]
    code, obj, err = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 0, obj
    rec = obj["records"][0]
    assert rec["command"] == "apply" and rec["parent"] == first["id"]
    assert ("install", {"pack_ids": ["hu"]}) in fx.calls
    assert ("install_options", {"speakers": False, "canary": False}) in fx.calls   # only the pack asked for
    note = next(n for n in rec["notes"] if n["code"] == "pack_fetched")
    assert note["message"] == "Downloaded the hu pack (3.1 GB)."
    assert stub_steps.CALLS[-1]["force_lang"] == "hu"
    assert rec["engines"]["asr_tag"] == "whisper-large-v3+ultra/v3-vote"
    doc = json.loads((tmp_path / "x (transcript).json").read_text())
    assert doc["record_id"] == rec["id"]
    assert {o["path"] for o in rec["outputs"]} == {o["path"] for o in first["outputs"]}
    remember = [o["options"] for o in first["offers"] if o["kind"] == "remember"]
    assert remember and [o["options"] for o in rec["offers"] if o["kind"] == "remember"] == remember


def test_apply_refusals(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4})
    first = run(capsys, a)[1][0]
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "th")
    assert code == 2 and obj["error"]["code"] == "not_an_option" and "hu" in obj["error"]["message"]
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--choice", "speakers")
    assert code == 2 and obj["error"]["code"] == "no_offer"
    assert obj["error"]["message"] == (f"Record {first['id']} has no speakers offer, so there's nothing for "
                                       "`--choice speakers` to act on.")
    (tmp_path / "x (transcript).md").write_text("my edits")
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 5 and obj["records"][0]["error"]["code"] == "output_changed"
    assert (tmp_path / "x (transcript).md").read_text() == "my edits"
    sealed = a.read_bytes()
    a.write_bytes(sealed + b" ")
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 5 and obj["error"]["code"] == "input_changed"
    a.write_bytes(sealed)
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu", "--overwrite")
    assert code == 0
    newest = obj["records"][0]["id"]
    for ref in (str(a), first["id"]):                  # acted on: neither the newest nor the first offers it now
        code, obj, _ = cli_json(capsys, "apply", ref, "--lang", "hu")
        assert code == 2 and obj["error"]["code"] == "no_offer"
        assert obj["error"]["message"].startswith(f"Record {newest} has no language offer")


def test_apply_speakers_adds_labels_without_transcribing_again(monkeypatch, tmp_path, capsys):
    fs = use_fakes(monkeypatch, tmp_path, fluidaudio=False)
    a = audio(tmp_path / "x.m4a")
    first = run(capsys, a)[1][0]
    assert "no_speaker_labels" in codes(first) and [o["kind"] for o in first["offers"]] == ["speakers"]
    assert json.loads((tmp_path / "x (transcript).json").read_text())["rttm"] is None
    fs.fluidaudio = True
    n = len(stub_steps.CALLS)
    code, obj, _ = cli_json(capsys, "apply", tmp_path / "x (transcript).json", "--choice", "speakers")
    assert code == 0, obj
    assert len(stub_steps.CALLS) == n                                     # no second transcription
    doc = json.loads((tmp_path / "x (transcript).json").read_text())
    assert doc["rttm"] and doc["displays"] == {"speaker_0": "Speaker 1", "speaker_1": "Speaker 2"}
    assert "**Speaker 1**" in (tmp_path / "x (transcript).md").read_text()
    assert obj["records"][0]["engines"]["diarizer"] == "fluidaudio-v0.18.0-nemotron3-offline"


def test_apply_names_updates_names_and_keeps_numbers(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", speakers=3, sentences=20, match={"speaker_1": B})
    run(capsys, a)
    before = json.loads((tmp_path / "x (transcript).json").read_text())["displays"]
    voiceprint(voices_dir(), B)
    rec = run(capsys, a)[1][0]
    assert codes(rec, "notes")[-1] == "voices_changed"
    a.write_text(json.dumps({**json.loads(a.read_text())}))              # same bytes: still the sealed input
    n = len(stub_steps.CALLS)
    code, obj, _ = cli_json(capsys, "apply", rec["id"], "--choice", "names")
    assert code == 0, obj
    assert len(stub_steps.CALLS) == n
    after = json.loads((tmp_path / "x (transcript).json").read_text())["displays"]
    assert after["speaker_1"] == B
    for lab, disp in before.items():
        if lab != "speaker_1":
            assert after[lab] == disp                                     # everyone else keeps their number


def test_apply_after_the_record_expired(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4})
    first = run(capsys, a)[1][0]
    record.record_path(first["id"]).unlink()
    code, obj, _ = cli_json(capsys, "apply", tmp_path / "x (transcript).json", "--lang", "hu")
    assert code == 0, obj
    assert obj["records"][0]["parent"] == first["id"]


def test_apply_on_an_edited_transcript(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4})
    first = run(capsys, a)[1][0]
    jp = tmp_path / "x (transcript).json"
    jp.write_text(jp.read_text().replace('"displays": {', '"displays": {"speaker_9": "x", '))
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 5 and obj["error"]["code"] == "transcript_edited"
    assert "audiointake transcribe x.m4a --overwrite" in obj["error"]["message"]


# --------------------------------------------------------------------------- relabel
def test_relabel_keeps_every_other_number(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", speakers=3, sentences=24)
    first = run(capsys, a, "--format", "md,json,txt")[1][0]
    before = json.loads((tmp_path / "x (transcript).json").read_text())["displays"]
    assert sorted(before.values()) == ["Speaker 1", "Speaker 2", "Speaker 3"]
    target = next(lab for lab, d in before.items() if d == "Speaker 1")
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", target, "--name", C)
    assert code == 0, obj
    rec = obj["record"]
    assert rec["command"] == "relabel" and rec["parent"] == first["id"]
    doc = json.loads((tmp_path / "x (transcript).json").read_text())
    assert doc["names_override"] == {target: C} and doc["record_id"] == rec["id"]
    for lab, disp in before.items():
        assert doc["displays"][lab] == (C if lab == target else disp)
    txt = (tmp_path / "x (transcript).txt").read_text()
    assert f"{C}: " in txt and "Speaker 1:" not in txt and "Speaker 2:" in txt and "Speaker 3:" in txt
    remember = [o for o in rec["offers"] if o["kind"] == "remember"]
    # every unnamed or user-named label with 8 s; the newly named one included, and the JSON keeps it
    assert remember and target in remember[0]["options"] and set(remember[0]["options"]) <= set(before)
    assert [o["options"] for o in doc["offers"] if o["kind"] == "remember"] == [remember[0]["options"]]
    second = [lab for lab in before if lab != target][0]
    code, obj, _ = cli_json(capsys, "relabel", str(a), "--label", second, "--name", A)
    doc = json.loads((tmp_path / "x (transcript).json").read_text())
    assert doc["names_override"] == {target: C, second: A}


def test_relabel_refusals(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a")
    first = run(capsys, a)[1][0]
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_5", "--name", A)
    assert code == 2 and obj["error"]["code"] == "unknown_label"
    assert obj["error"]["message"] == f"Record {first['id']} has no speaker speaker_5. Its speakers are speaker_0, speaker_1."
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_0", "--name", ".hidden")
    assert code == 2 and obj["error"]["code"] == "name_invalid"
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_0", "--label", "speaker_1",
                            "--name", A)
    assert code == 2 and obj["error"]["code"] == "bad_arguments"
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_9", "--name", A)
    assert code == 2 and obj["error"]["code"] == "bad_arguments"
    (tmp_path / "x (transcript).md").write_text("mine")
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_0", "--name", A)
    assert code == 5 and obj["error"]["code"] == "output_changed"
    assert (tmp_path / "x (transcript).md").read_text() == "mine"


# --------------------------------------------------------------------------- offers along a transcript's chain
TJ = "x (transcript).json"


def kinds(rec):
    return [o["kind"] for o in rec["offers"]]


def expire_all():
    for p in record.records_dir().glob("r-*.json"):
        p.unlink()


@pytest.mark.parametrize("kept", [True, False], ids=["kept", "expired"])
@pytest.mark.parametrize("form", ["newest-id", "first-id", "audio", "json"])
def test_the_language_offer_survives_relabel_and_notes(fx, tmp_path, capsys, form, kept):
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4})
    first = run(capsys, a)[1][0]
    code, obj, _ = cli_json(capsys, "relabel", str(a), "--label", "speaker_0", "--name", A)
    assert code == 0, obj
    rel = obj["record"]
    assert [o["options"] for o in rel["offers"] if o["kind"] == "language"] == [["hu"]]
    doc = json.loads((tmp_path / TJ).read_text())
    assert [o["options"] for o in doc["offers"] if o["kind"] == "language"] == [["hu"]]    # the JSON keeps it
    notes = tmp_path / "x (notes).md"
    notes.write_text("# x\n\nA meeting.\n")
    code, obj, _ = cli_json(capsys, "notes", str(a), "--into", notes)
    assert code == 0 and "language" in kinds(obj["record"])
    if not kept:
        expire_all()
    ref = {"newest-id": rel["id"], "first-id": first["id"], "audio": str(a), "json": str(tmp_path / TJ)}[form]
    code, obj, _ = cli_json(capsys, "apply", ref, "--lang", "hu")
    if not kept and form.endswith("-id"):
        assert code == 2 and obj["error"]["code"] == "unknown_record"
        return
    assert code == 0, obj
    rec = obj["records"][0]
    assert rec["parent"] == rel["id"] and rec["engines"]["asr_tag"] == "whisper-large-v3+ultra/v3-vote"
    assert "language" not in kinds(rec)                                               # acted on


def test_the_speakers_offer_survives_notes(monkeypatch, tmp_path, capsys):            # no speaker program, then notes
    fs = use_fakes(monkeypatch, tmp_path, fluidaudio=False)
    a = audio(tmp_path / "x.m4a")
    first = run(capsys, a)[1][0]
    assert "speakers" in kinds(first)
    notes = tmp_path / "x (notes).md"
    notes.write_text("# x\n")
    code, obj, _ = cli_json(capsys, "notes", str(a), "--into", notes)
    assert code == 0 and "speakers" in kinds(obj["record"])
    fs.fluidaudio = True
    code, obj, _ = cli_json(capsys, "apply", str(a), "--choice", "speakers")
    assert code == 0, obj
    assert "speakers" not in kinds(obj["records"][0]) and json.loads((tmp_path / TJ).read_text())["rttm"]
    code, obj, _ = cli_json(capsys, "apply", str(a), "--choice", "speakers")           # settled now
    assert code == 2 and obj["error"]["code"] == "no_offer"


def test_the_names_offer_survives_a_relabel_by_the_first_id(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", speakers=3, sentences=20, match={"speaker_1": B})
    first = run(capsys, a)[1][0]
    voiceprint(voices_dir(), B)
    again = run(capsys, a)[1][0]
    assert codes(again, "notes")[-1] == "voices_changed" and again["parent"] == first["id"]
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_2", "--name", C)
    assert code == 0 and "names" in kinds(obj["record"]), obj
    code, obj, _ = cli_json(capsys, "apply", str(a), "--choice", "names")
    assert code == 0, obj
    doc = json.loads((tmp_path / TJ).read_text())
    assert doc["displays"]["speaker_1"] == B and doc["displays"]["speaker_2"] == C       # the user's name stays
    assert "names" not in doc["offers"]


def test_names_applied_then_a_rerun_is_already_done(fx, tmp_path, capsys):            # a names apply settles
    a = audio(tmp_path / "x.m4a", match={"speaker_0": A})
    run(capsys, a)
    voiceprint(voices_dir(), A)
    rec = run(capsys, a)[1][0]
    assert codes(rec, "notes")[-1] == "voices_changed"
    code, obj, _ = cli_json(capsys, "apply", rec["id"], "--choice", "names")
    assert code == 0, obj
    applied = obj["records"][0]
    assert applied["voices_digest"] == voices.digest(voices_dir()) != ""
    doc = json.loads((tmp_path / TJ).read_text())
    assert doc["voices_digest"] == applied["voices_digest"] and doc["displays"]["speaker_0"] == A
    rec = run(capsys, a)[1][0]
    assert codes(rec, "notes")[-1] == "already_done" and "names" not in kinds(rec)


def test_names_are_matched_in_the_folder_the_rerun_named(fx, tmp_path, capsys):
    a = audio(tmp_path / "x.m4a", match={"speaker_0": A})
    run(capsys, a)
    other = tmp_path / "prints"
    voiceprint(other, A)
    rec = run(capsys, a, "--voices", other)[1][0]
    assert codes(rec, "notes")[-1] == "voices_changed"
    code, obj, _ = cli_json(capsys, "apply", str(a), "--choice", "names")
    assert code == 0, obj
    assert [c for c in fx.calls if c[0] == "resolve"][-1][1]["voices_dir"] == other
    assert json.loads((tmp_path / TJ).read_text())["displays"]["speaker_0"] == A


# --------------------------------------------------------------------------- apply --lang and setup's install
HU = {"hu_route": "no_pack", "hu_share": 0.4}


def test_apply_lang_installs_nothing_when_the_pack_is_ready(monkeypatch, tmp_path, capsys):
    fs = use_fakes(monkeypatch, tmp_path)
    first = run(capsys, audio(tmp_path / "x.m4a", ensemble=HU))[1][0]
    fs.packs.add("hu")                                  # installed since, by `setup --pack hu`
    fs._stamp()
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 0, obj
    assert not [c for c in fs.calls if c[0] == "install"] and "pack_fetched" not in codes(obj["records"][0], "notes")


def test_apply_lang_goes_on_when_only_the_speaker_program_failed(monkeypatch, tmp_path, capsys):   # no Command Line Tools
    got = {"exit": 3, "errors": [{"code": "clt_missing", "params": {}}], "warnings": [],
           "packs": [{"id": "core", "installed": True, "verified": True, "fetched_bytes": 0},
                     {"id": "hu", "installed": True, "verified": True, "fetched_bytes": int(3.1e9)}]}
    use_fakes(monkeypatch, tmp_path, fluidaudio=False, install_result=got)
    first = run(capsys, audio(tmp_path / "x.m4a", ensemble=HU))[1][0]
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 0, obj
    rec = obj["records"][0]
    assert "pack_fetched" in codes(rec, "notes") and "no_speaker_labels" in codes(rec) and "speakers" in kinds(rec)
    assert rec["engines"]["asr_tag"] == "whisper-large-v3+ultra/v3-vote"


def test_a_commit_that_fails_half_way_puts_back_what_it_replaced(fx, tmp_path, capsys, monkeypatch):
    """relabel replaces the .md, then the JSON write fails (a full disk): the .md gets its old bytes back, so the
    JSON still lists what is beside it, and the next follow-up works instead of saying output_changed forever."""
    import errno
    first = run(capsys, audio(tmp_path / "x.m4a"))[1][0]
    md, js = tmp_path / "x (transcript).md", tmp_path / TJ
    was_md, was_js, mode = md.read_bytes(), js.read_bytes(), md.stat().st_mode
    real = fsutil.atomic_write

    def full_disk(path, data, **kw):
        if str(path) == str(js):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, data, **kw)
    monkeypatch.setattr(fsutil, "atomic_write", full_disk)
    with pytest.raises(OSError):
        pipeline.commit_outputs([("md", md, b"new md"), ("json", js, b"{}")], overwrite=False,
                                replace={str(md): fsutil.sha256_bytes(was_md), str(js): fsutil.sha256_bytes(was_js)})
    assert md.read_bytes() == was_md and js.read_bytes() == was_js and md.stat().st_mode == mode
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(fsutil.PARTIAL_PREFIX)]
    monkeypatch.setattr(fsutil, "atomic_write", real)
    code, obj, _ = cli_json(capsys, "relabel", first["id"], "--label", "speaker_0", "--name", A)
    assert code == 0, obj


def test_apply_lang_keeps_the_transcripts_other_codes(fx, tmp_path, capsys):
    """Several codes never force a route: `apply --lang hu` on a `--lang en,hu` transcript re-runs with en,hu."""
    first = run(capsys, audio(tmp_path / "x.m4a"), "--lang", "en,hu")[1][0]
    assert [o["options"] for o in first["offers"] if o["kind"] == "language"] == [["hu"]]
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 0, obj
    rec = obj["records"][0]
    assert stub_steps.CALLS[-1]["force_lang"] is None and rec["options"]["lang"] == ["en", "hu"]
    assert rec["options_digest"] == first["options_digest"]
    assert ("resolve", {"need": {"hu_vote"}, "voices_dir": voices_dir(), "names": True, "speakers": True}) in fx.calls
    assert "pack_not_installed" not in codes(rec) and "language" not in kinds(rec)


def test_apply_lang_keeps_another_missing_pack_on_offer(fx, tmp_path, capsys):
    first = run(capsys, audio(tmp_path / "x.m4a"), "--lang", "hu,th")[1][0]
    assert [o["options"] for o in first["offers"] if o["kind"] == "language"] == [["hu", "th"]]
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 0, obj
    rec = obj["records"][0]
    assert stub_steps.CALLS[-1]["force_lang"] is None and rec["options"]["lang"] == ["hu", "th"]
    w = [x["params"]["code"] for x in rec["warnings"] if x["code"] == "pack_not_installed"]
    assert w == ["th"] and [o["options"] for o in rec["offers"] if o["kind"] == "language"] == [["th"]]
    doc = json.loads((tmp_path / TJ).read_text())
    assert [o["options"] for o in doc["offers"] if o["kind"] == "language"] == [["th"]]
    assert pipeline.redo_langs([], "hu") == ("hu",) and pipeline.redo_langs(["hu"], "hu") == ("hu",)


def test_apply_lang_that_cannot_install_leaves_a_record(monkeypatch, tmp_path, capsys):
    got = {"exit": 3, "warnings": [],
           "errors": [{"code": "clt_missing", "params": {}},
                      {"code": "download_failed", "params": {"model": "whisper-large-v3", "host": "huggingface.co"}}],
           "packs": [{"id": "core", "installed": True, "verified": True}, {"id": "hu", "installed": False}]}
    use_fakes(monkeypatch, tmp_path, install_result=got)
    first = run(capsys, audio(tmp_path / "x.m4a", ensemble=HU))[1][0]
    code, obj, _ = cli_json(capsys, "apply", first["id"], "--lang", "hu")
    assert code == 3
    rec = obj["records"][0]
    assert rec["command"] == "apply" and rec["parent"] == first["id"] and rec["status"] == "failed"
    assert rec["error"]["code"] == "download_failed" and "huggingface.co" in rec["error"]["message"]
    assert record.load(rec["id"])["error"]["code"] == "download_failed"
    assert "language" in kinds(rec)                                                    # still open
    assert json.loads((tmp_path / TJ).read_text())["record_id"] == first["id"]          # nothing redone


def test_a_detected_pack_quotes_its_own_size(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, fluidaudio=False)            # the speaker program's source isn't the pack's
    rec = run(capsys, audio(tmp_path / "x.m4a", ensemble=HU))[1][0]
    w = next(w for w in rec["warnings"] if w["code"] == "pack_not_installed")
    assert w["params"]["size"] == pytest.approx(3.1) and "(+3.1 GB)" in w["message"]


# --------------------------------------------------------------------------- a batch keeps going
def test_a_vanishing_input_fails_alone(fx, tmp_path, capsys):
    a = audio(tmp_path / "a.m4a", fail={"vanish": True})
    b = audio(tmp_path / "b.m4a")
    code, recs, _ = run(capsys, a, b)
    assert code == 5
    assert recs[0]["error"]["code"] == "input_changed" and recs[0]["outputs"] == []
    assert recs[1]["status"] == "done" and (tmp_path / "b (transcript).md").exists()
    assert not (tmp_path / "a (transcript).md").exists()


def test_an_unexpected_exception_fails_that_input_only(fx, tmp_path, capsys, monkeypatch):
    real = pipeline._speaker_problems

    def boom(res, rec, o):
        if res.name == "a.m4a":
            raise RuntimeError("words " + CANARY)
        return real(res, rec, o)
    monkeypatch.setattr(pipeline, "_speaker_problems", boom)
    code, recs, err = run(capsys, audio(tmp_path / "a.m4a"), audio(tmp_path / "b.m4a"))
    assert code == 1
    assert recs[0]["error"]["code"] == "internal_error" and recs[0]["status"] == "failed" and recs[0]["outputs"] == []
    assert recs[0]["error"]["message"].endswith(f"The log at {recs[0]['log']} says where; please report it with "
                                                "that log.")
    log = Path(recs[0]["log"]).read_text()
    assert "unexpected RuntimeError at" in log and CANARY not in log and "words" not in log
    assert recs[1]["status"] == "done" and CANARY not in err


def test_the_same_name_in_two_folders_never_shares_an_output(fx, tmp_path, capsys):
    out = tmp_path / "out"
    for d in ("a", "b", "out"):
        (tmp_path / d).mkdir()
    x1, x2 = audio(tmp_path / "a" / "x.m4a", seed=1), audio(tmp_path / "b" / "x.m4a", seed=2)
    code, recs, _ = run(capsys, x1, x2, "--outdir", out, "--overwrite")
    assert code == 2
    assert recs[0]["status"] == "done" and recs[1]["error"]["code"] == "output_exists"
    assert json.loads((out / "x.m4a (transcript).json").read_text())["record_id"] == recs[0]["id"]


def test_a_folder_moved_with_its_transcript(fx, tmp_path, capsys):
    f1, f2 = tmp_path / "f1", tmp_path / "f2"
    f1.mkdir()
    run(capsys, audio(f1 / "x.m4a"))
    f1.rename(f2)
    code, obj, _ = cli_json(capsys, "relabel", f2 / TJ, "--label", "speaker_0", "--name", A)
    assert code == 0, obj
    assert {Path(o["path"]).parent for o in obj["record"]["outputs"]} == {f2}
    assert obj["record"]["input"]["path"] == str(f2 / "x.m4a")
    code, obj, _ = cli_json(capsys, "enroll", B, "--from", f2 / TJ, "--label", "speaker_1")
    assert code == 0, obj                                         # the recording is found beside it


def test_follow_up_messages_never_keep_a_placeholder(monkeypatch, tmp_path, capsys):
    fs = use_fakes(monkeypatch, tmp_path, fluidaudio=False)
    a = audio(tmp_path / "x.m4a", fail={"diarize": "exit"})
    run(capsys, a)
    fs.fluidaudio = True
    code, obj, _ = cli_json(capsys, "apply", str(a), "--choice", "speakers")
    rec = obj["records"][0]
    assert code == 0 and "speakers_failed" in codes(rec)
    msgs = [w["message"] for k in ("warnings", "notes") for w in rec[k]] + [x for o in rec["offers"] for x in o["summary"]]
    assert msgs and not [m for m in msgs if "{" in m]
    assert Path(rec["log"]).is_file() and "fluidaudiocli:" in Path(rec["log"]).read_text()


def test_merge_failed_keeps_the_ensembles_text_bytes(fx, tmp_path, capsys):
    a = audio(tmp_path / "Call.m4a", fail={"diarize": "malformed"}, txt_end="\r\n")
    code, recs, _ = run(capsys, a, "--format", "txt,sidecars")
    assert code == 0 and "merge_failed" in codes(recs[0])
    assert (tmp_path / "Call.txt").read_bytes().endswith(b"\r\n")                    # moved as it is, no newline change
    assert (tmp_path / "Call (transcript).txt").read_bytes() == (tmp_path / "Call.txt").read_bytes()


def test_the_fluidaudio_tag_comes_from_the_runtime(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, fluidaudio=False, fluidaudio_tag="v9.9.9")
    run(capsys, audio(tmp_path / "Call.m4a"), "--format", "sidecars")
    assert "engine=fluidaudio-v9.9.9-nemotron3-offline " in diarmeta(tmp_path, "Call")


def test_an_unknown_length_gets_the_engines_long_cap():
    from audiointake.steps.ensemble import NO_DURATION_TIMEOUT
    assert pipeline._timeout("fluidaudio", None) == pipeline._timeout("ffmpeg", 0.0) == NO_DURATION_TIMEOUT
    assert pipeline._timeout("fluidaudio", 100.0) == 300.0 and pipeline._timeout("ffmpeg", 4000.0) == 1000.0


def test_a_custom_folder_only_an_expired_record_knew_names_nobody(monkeypatch, tmp_path, capsys):
    fs = use_fakes(monkeypatch, tmp_path, fluidaudio=False)
    other = tmp_path / "prints"
    voiceprint(other, A)
    voiceprint(voices_dir(), A)                                      # the default folder must not be used instead
    a = audio(tmp_path / "x.m4a", match={"speaker_0": A})
    run(capsys, a, "--voices", other)
    expire_all()
    fs.fluidaudio = True
    code, obj, _ = cli_json(capsys, "apply", tmp_path / TJ, "--choice", "speakers")
    assert code == 0, obj
    last = [c for c in fs.calls if c[0] == "resolve"][-1][1]
    assert last["voices_dir"] is None and last["names"] is False
    doc = json.loads((tmp_path / TJ).read_text())
    assert A not in doc["displays"].values() and doc["options"]["voices"] == "custom"
