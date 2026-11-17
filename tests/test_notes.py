"""notes --into and notes --check: the transcript is appended word for word, and edits are told apart."""
from __future__ import annotations

import json

import pytest

from audiointake import fsutil, notes, record, render

from fakes import audio, cli, cli_json, use_fakes

HEAD = "# Team sync\n\n## Summary\n\nPlanned the release.\n\n## Action items\n\n- [ ] ship it: Jane Doe, due 2026-11-20\n"


@pytest.fixture
def made(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path)
    a = audio(tmp_path / "Team sync.m4a")
    code, obj, _ = cli_json(capsys, "transcribe", a)
    assert code == 0
    n = tmp_path / "Team sync (notes).md"
    n.write_text(HEAD, encoding="utf-8")
    return a, obj["records"][0], n


def test_into_appends_the_transcript_word_for_word(made, tmp_path, capsys):
    a, rec, n = made
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", n)
    assert code == 0
    doc = json.loads((tmp_path / "Team sync (transcript).json").read_text())
    text = n.read_text(encoding="utf-8")
    assert text == HEAD.rstrip("\n") + "\n\n## Transcript\n\n" + render.dialogue(doc["turns"],
                                                                                 duration_s=doc["input"]["duration_s"])
    child = obj["record"]
    assert child["command"] == "notes" and child["parent"] == rec["id"]
    assert child["outputs"] == [{"kind": "notes", "path": str(n), "sha256": fsutil.sha256_file(n)}]


@pytest.mark.parametrize("ref", ["audio", "json"])
def test_into_takes_a_path_for_ref(made, tmp_path, capsys, ref):
    a, rec, n = made
    r = a if ref == "audio" else tmp_path / "Team sync (transcript).json"
    code, obj, _ = cli_json(capsys, "notes", r, "--into", n)
    assert code == 0 and obj["record"]["parent"] == rec["id"]


def test_into_refuses_a_head_with_a_transcript(made, capsys):
    a, rec, n = made
    n.write_text(HEAD + "\n## Transcript\n\nold\n")
    before = n.read_bytes()
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", n)
    assert code == 2 and obj["error"]["code"] == "head_invalid"
    assert n.read_bytes() == before


def test_into_refuses_an_edited_transcript_json(made, tmp_path, capsys):
    a, rec, n = made
    jp = tmp_path / "Team sync (transcript).json"
    jp.write_text(jp.read_text().replace('"display": "Speaker 1"', '"display": "Someone"', 1))
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", n)
    assert code == 5 and obj["error"]["code"] == "transcript_edited"
    assert n.read_text() == HEAD


def test_into_needs_the_notes_file(made, tmp_path, capsys):
    a, rec, n = made
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", tmp_path / "absent.md")
    assert code == 2 and obj["error"]["code"] == "input_missing"


def test_check_tells_edited_from_untouched(made, capsys):
    a, rec, n = made
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--check", n)
    assert code == 5 and obj["error"]["code"] == "output_changed"          # never written by audiointake
    cli_json(capsys, "notes", rec["id"], "--into", n)
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--check", n)
    assert code == 0 and obj["unchanged"] is True
    code, out, _ = cli(capsys, "notes", str(a), "--check", n)
    assert code == 0 and "is unchanged" in out
    n.write_text(n.read_text() + "\nmy own line\n")
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--check", n)
    assert code == 5 and obj["error"]["code"] == "output_changed"
    assert obj["error"]["message"].startswith("Team sync (notes).md was changed after audiointake wrote it")


def test_notes_after_the_record_expired_and_after_a_relabel(made, tmp_path, capsys):
    a, rec, n = made
    jp = tmp_path / "Team sync (transcript).json"
    record.record_path(rec["id"]).unlink()
    code, obj, _ = cli_json(capsys, "notes", jp, "--into", n)
    assert code == 0
    n.write_text(HEAD)
    code, obj, _ = cli_json(capsys, "relabel", jp, "--label", "speaker_0", "--name", "Jane Doe")
    assert code == 0
    code, obj, _ = cli_json(capsys, "notes", obj["record"]["id"], "--into", n)
    assert code == 0 and "**Jane Doe**" in n.read_text()


def test_input_changed_since_the_transcript(made, capsys):
    a, rec, n = made
    a.write_text(a.read_text() + " ")
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", n)
    assert code == 5 and obj["error"]["code"] == "input_changed"
    a.unlink()
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", n)      # the audio isn't read: carry on
    assert code == 0


def test_compose_and_read_head(tmp_path):
    assert notes.compose("# T\n\n", [{"label": None, "display": "", "start": 0, "end": 1, "text": "Hi."}],
                         duration_s=10) == "# T\n\n## Transcript\n\n[00:00]\nHi.\n"
    p = tmp_path / "n.md"
    p.write_text("# T\n##  Transcript \n")
    with pytest.raises(Exception) as e:
        notes.read_head(p)
    assert e.value.code == "head_invalid"


@pytest.mark.parametrize("which", ["json", "md", "audio", "named-like-a-transcript"])
def test_into_refuses_the_transcripts_own_files(made, tmp_path, capsys, which):
    a, rec, _ = made
    target = {"json": tmp_path / "Team sync (transcript).json", "md": tmp_path / "Team sync (transcript).md",
              "audio": a, "named-like-a-transcript": tmp_path / "Other (transcript).txt"}[which]
    if not target.exists():
        target.write_text(HEAD, encoding="utf-8")
    before = target.read_bytes()
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", target)
    assert code == 2 and obj["error"]["code"] == "bad_arguments"
    assert obj["error"]["message"].startswith(f"--into takes the notes file, not {target.name}")
    assert target.read_bytes() == before
    assert cli_json(capsys, "record", "find", a)[0] == 0                          # the transcript is intact


def test_the_notes_record_carries_the_open_offers(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, fluidaudio=False)
    a = audio(tmp_path / "x.m4a")
    cli_json(capsys, "transcribe", a)
    n = tmp_path / "x (notes).md"
    n.write_text(HEAD, encoding="utf-8")
    code, obj, _ = cli_json(capsys, "notes", a, "--into", n)
    assert code == 0 and [o["kind"] for o in obj["record"]["offers"]] == ["speakers"]
    assert obj["record"]["offers"][0]["summary"][0].startswith("Speaker labels can be added")


def test_check_after_the_notes_record_expired_is_the_safe_answer(made, tmp_path, capsys):
    a, rec, n = made
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--into", n)
    record.record_path(obj["record"]["id"]).unlink()
    code, obj, _ = cli_json(capsys, "notes", a, "--check", n)
    assert code == 5 and obj["error"]["code"] == "output_changed"                 # documented: the skill leaves it
    assert n.read_text(encoding="utf-8").startswith(HEAD.rstrip("\n"))


def test_a_signal_while_the_notes_are_written_waits_for_their_record(made, capsys, monkeypatch):
    import signal
    a, rec, n = made
    real = fsutil.atomic_write
    sent: list[str] = []

    def write(path, data, **kw):
        if not sent and str(path) == str(n):
            sent.append(str(path))
            signal.raise_signal(signal.SIGTERM)                # lands between the notes file and its record
        return real(path, data, **kw)
    monkeypatch.setattr(fsutil, "atomic_write", write)
    code, out, _ = cli(capsys, "notes", rec["id"], "--into", n, "--json")
    assert code == 143 and sent and out == ""
    assert notes.newest_notes(str(n)) == fsutil.sha256_file(n)      # the record came with the file
    monkeypatch.setattr(fsutil, "atomic_write", real)
    code, obj, _ = cli_json(capsys, "notes", rec["id"], "--check", n)
    assert code == 0 and obj["unchanged"] is True

