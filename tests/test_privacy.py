"""No transcript text in a record, a log, stderr or --json; no child's stdout in a log; nobody named who wasn't
heard.

Every text field the fake engines write carries ``CANARY``, and every fake child prints ``STDOUT_MARK`` with
transcript text to stdout. A voiceprint for someone who is never heard (``ABSENT``) is in the folder; the fake
matcher still lists them in its roster, its best guesses and its runners-up, as the real one does.
"""
from __future__ import annotations

import json

import pytest

from audiointake import platforms

from fakes import CANARY, STDOUT_MARK, audio, cli, cli_json, use_fakes, voiceprint

HEARD, ABSENT = "Vandrel Oskel", "Ostrel Absentia"


def everything_kept() -> str:
    """Every record and every run log, as text."""
    parts = []
    for folder in (platforms.data_dir() / "records", platforms.logs_dir()):
        if folder.is_dir():
            parts += [p.read_text(encoding="utf-8", errors="replace") for p in sorted(folder.iterdir())]
    return "\n".join(parts)


@pytest.fixture
def setup_voices(monkeypatch, tmp_path):
    use_fakes(monkeypatch, tmp_path)
    folder = platforms.data_dir() / "voices"
    voiceprint(folder, HEARD)
    voiceprint(folder, ABSENT)
    return tmp_path


def assert_clean(*texts: str, names: bool = True) -> None:
    for t in texts:
        assert CANARY not in t
        assert STDOUT_MARK not in t
        if names:
            assert ABSENT not in t


def test_transcribe_keeps_text_out_of_records_logs_stderr_and_json(setup_voices, capsys):
    tmp = setup_voices
    a = audio(tmp / "Team sync.m4a", sentences=30, match={"speaker_0": HEARD})
    code, obj, err = cli_json(capsys, "transcribe", a)
    assert code == 0
    out = json.dumps(obj, ensure_ascii=False)
    assert HEARD in out                                                    # a heard name may appear
    assert_clean(out, err, everything_kept())
    doc_text = (tmp / "Team sync (transcript).json").read_text(encoding="utf-8")
    assert CANARY in doc_text                                              # the transcript itself holds the text
    assert ABSENT not in doc_text and "enrolled" not in doc_text and "best_guess" not in doc_text
    assert "runner_up" not in doc_text
    assert ABSENT not in (tmp / "Team sync (transcript).md").read_text(encoding="utf-8")
    code, out, err = cli(capsys, "transcribe", a, "--overwrite")
    assert_clean(out, err, everything_kept())


def test_failures_log_stderr_tails_but_never_text(setup_voices, capsys):
    tmp = setup_voices
    for i, fail in enumerate(({"primary": "exit"}, {"fill": "exit"}, {"diarize": "partial"}, {"match": "partial"},
                              {"diarize": "malformed"})):
        a = audio(tmp / f"f{i}.m4a", fail=fail, match={"speaker_0": HEARD})
        code, obj, err = cli_json(capsys, "transcribe", a)
        assert_clean(json.dumps(obj, ensure_ascii=False), err)
    logs = everything_kept()
    assert_clean(logs)
    assert "fake parakeet: the model failed to load" in logs              # stderr tails are kept, stdout never


def test_sidecars_are_the_only_place_the_roster_goes(setup_voices, capsys):
    tmp = setup_voices
    a = audio(tmp / "x.m4a", match={"speaker_0": HEARD})
    code, obj, err = cli_json(capsys, "transcribe", a, "--format", "sidecars")
    assert ABSENT in (tmp / "x.speakers.json").read_text(encoding="utf-8")     # verbatim, for parity only
    assert_clean(json.dumps(obj, ensure_ascii=False), err, everything_kept())


def test_follow_up_commands_keep_text_out(setup_voices, capsys):
    tmp = setup_voices
    a = audio(tmp / "x.m4a", sentences=30, match={"speaker_0": HEARD}, ensemble={"hu_route": "no_pack",
                                                                                "hu_share": 0.4})
    rid = cli_json(capsys, "transcribe", a)[1]["records"][0]["id"]
    notes = tmp / "x (notes).md"
    notes.write_text("# x\n\n## Summary\n\nA meeting.\n")
    results = [cli_json(capsys, "relabel", rid, "--label", "speaker_1", "--name", "Jane Doe"),
               cli_json(capsys, "notes", str(a), "--into", notes),
               cli_json(capsys, "notes", str(a), "--check", notes),
               cli_json(capsys, "record", "show", rid), cli_json(capsys, "record", "find", str(a)),
               cli_json(capsys, "apply", rid, "--lang", "hu"),
               cli_json(capsys, "enroll", "Jane Doe", "--from", str(a), "--label", "speaker_1")]
    for code, obj, err in results:
        assert code == 0, obj
        assert_clean(json.dumps(obj, ensure_ascii=False), err)
    assert_clean(everything_kept())
    code, out, err = cli(capsys, "record", "show", rid)
    assert_clean(out, err)


def test_the_transcript_json_names_no_home_folder(setup_voices, capsys):
    tmp = setup_voices
    a = audio(tmp / "x.m4a")
    cli_json(capsys, "transcribe", a)
    doc = json.loads((tmp / "x (transcript).json").read_text(encoding="utf-8"))
    assert doc["options"]["voices"] == "default" and "voices_dir" not in doc["options"]
    assert str(platforms.data_dir()) not in (tmp / "x (transcript).json").read_text(encoding="utf-8")
    custom = tmp / "client prints"
    custom.mkdir()
    b = audio(tmp / "y.m4a")
    rec = cli_json(capsys, "transcribe", b, "--voices", custom)[1]["records"][0]
    assert rec["options"]["voices"] == "custom" and rec["options"]["voices_dir"] == str(custom)   # record only
    text = (tmp / "y (transcript).json").read_text(encoding="utf-8")
    assert json.loads(text)["options"]["voices"] == "custom" and str(custom) not in text
