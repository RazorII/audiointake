"""Records: their fields, saving, messages, pruning, the run log, speed.json, and REF resolution."""
from __future__ import annotations

import json
import os
import re
import stat
import time

import pytest

from audiointake import errors, lock, outputs, platforms, record
from audiointake.contracts import RECORD_ID_RE
from audiointake.errors import Stale, UsageError, problem

from fakes import CANARY, audio, cli_json, use_fakes

FIELDS = ("record_version", "id", "parent", "tool", "command", "status", "exit", "created", "input", "options",
          "options_digest", "voices_digest", "outputs", "engines", "signals", "speakers", "counts", "timing",
          "warnings", "notes", "offers", "error")


@pytest.fixture
def fx(monkeypatch, tmp_path):
    return use_fakes(monkeypatch, tmp_path)


def test_new_has_every_field_of_version_1(fx):
    rec = record.new("transcribe", parent="r-20260101-00000000")
    for f in FIELDS:
        assert f in rec
    assert re.match(RECORD_ID_RE, rec["id"])
    assert rec["record_version"] == 1 and rec["parent"] == "r-20260101-00000000"


def test_save_needs_the_lock(fx):
    rec = record.finish(record.new("transcribe"), 0)
    with pytest.raises(RuntimeError):
        record.save(rec)


def test_save_is_private_and_renders_core_words(fx):
    rec = record.new("transcribe")
    rec["log"] = "/tmp/x.log"
    record.add(rec, problem("merge_failed"))
    record.add(rec, problem("merge_failed"))                       # kept once
    record.add(rec, problem("speakers"))
    record.add(rec, problem("pack_not_installed", reason="asked", code="hu", language="Hungarian", size=3.1))
    record.add(rec, problem("language", code="hu", language="Hungarian", size=3.1))
    record.add(rec, problem("pack_broken", pack="th", file="config.json", language="Thai"))
    record.finish(rec, 0)
    with lock.run_lock():
        p = record.save(rec)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    saved = json.loads(p.read_text())
    w = {x["code"]: x for x in saved["warnings"]}
    assert set(w) == {"merge_failed", "pack_not_installed", "pack_broken"}
    assert w["merge_failed"]["params"]["log"] == "/tmp/x.log"              # added by the record writer
    assert w["merge_failed"]["message"] == errors.render("merge_failed", {"log": "/tmp/x.log"})
    assert w["pack_not_installed"]["params"]["id"] == rec["id"]
    assert f"apply {rec['id']} --lang hu" in w["pack_not_installed"]["message"]
    assert "Until then, Thai is read with the core models." in w["pack_broken"]["message"]
    offers = {o["kind"]: o for o in saved["offers"]}
    assert offers["language"]["options"] == ["hu"] and offers["language"]["apply_flag"] == "--lang"
    assert offers["language"]["summary"] == [errors.render("language", {"id": rec["id"], "code": "hu",
                                                                        "language": "Hungarian", "size": 3.1})]
    assert offers["speakers"]["apply_flag"] == "--choice speakers"


def test_core_pack_broken_is_an_error(fx):
    rec = record.new("transcribe")
    record.add(rec, problem("pack_broken", pack="core", file="x"))
    assert rec["error"]["code"] == "pack_broken" and rec["warnings"] == []


@pytest.mark.parametrize("ref", ["r-20260101-0000000", "nonsense", "r-20260101-00000000"])
def test_load_unknown(fx, ref):
    with pytest.raises(UsageError) as e:
        record.load(ref)
    assert e.value.code == "unknown_record" and e.value.exit == 2


def test_prune_after_90_days(fx):
    old, new = record.new("transcribe"), record.new("transcribe")
    for r in (old, new):
        record.finish(r, 0)
    with lock.run_lock():
        po, pn = record.save(old), record.save(new)
        logo = record.log_path(old["id"])
        record.RunLog(old["id"])("step")
        t = time.time() - 91 * 86400
        os.utime(po, (t, t))
        os.utime(logo, (t, t))
        assert record.prune() == 2
    assert not po.exists() and not logo.exists() and pn.exists()
    with pytest.raises(RuntimeError):
        record.prune()


def test_run_log_is_private_and_appends(fx):
    log = record.RunLog("r-20260101-00000001")
    log("one")
    log("two")
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.path.parent.stat().st_mode) == 0o700
    lines = log.path.read_text().splitlines()
    assert [ln.strip().split("s ", 1)[1] for ln in lines] == ["one", "two"]
    assert log.path.parent == platforms.logs_dir()


def test_speed_keeps_the_last_ten(fx):
    with lock.run_lock():
        for v in range(1, 13):
            record.speed_add("core", float(v))
    assert record.speed_load()["core"] == [float(v) for v in range(3, 13)]
    assert record.speed_median("core") == 7.5
    assert record.speed_median("hu_vote") is None


# --------------------------------------------------------------------------- REF resolution
def _transcribe(capsys, tmp_path, name="Team sync.m4a", **desc):
    a = audio(tmp_path / name, **desc)
    code, obj, _ = cli_json(capsys, "transcribe", a)
    assert code == 0
    return a, obj["records"][0]


def test_resolve_by_id_audio_and_json(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    jp = tmp_path / "Team sync (transcript).json"
    for ref in (rec["id"], str(a), str(jp)):
        r = record.resolve(ref)
        assert r.id == rec["id"] and r.json_path == jp and r.audio == a
        assert r.doc["record_id"] == rec["id"]
        assert r.named["id"] == rec["id"]


def test_resolve_from_the_json_alone_after_the_record_expired(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    record.record_path(rec["id"]).unlink()
    jp = tmp_path / "Team sync (transcript).json"
    r = record.resolve(str(jp))
    assert r.record is None and r.id == rec["id"] and r.audio == a
    r = record.resolve(str(a))
    assert r.json_path == jp
    with pytest.raises(UsageError) as e:
        record.resolve(rec["id"])
    assert e.value.code == "unknown_record"


def test_resolve_refuses_an_edited_transcript(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    jp = tmp_path / "Team sync (transcript).json"
    jp.write_bytes(jp.read_bytes().replace(b'"names_override": {}', b'"names_override": {"speaker_0": "X"}'))
    with pytest.raises(Stale) as e:
        record.resolve(rec["id"])
    assert e.value.code == "transcript_edited" and e.value.exit == 5


def test_resolve_refuses_a_resealed_transcript_its_record_disowns(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    jp = tmp_path / "Team sync (transcript).json"
    doc = outputs.parse_transcript(jp.read_bytes())
    doc["displays"] = {"speaker_0": "Someone"}
    jp.write_bytes(outputs.transcript_bytes(doc))                  # a valid seal, but not the bytes the run wrote
    with pytest.raises(Stale) as e:
        record.resolve(str(jp))
    assert e.value.code == "transcript_edited"


def test_resolve_without_a_transcript(fx, tmp_path, capsys):
    a = audio(tmp_path / "plain.m4a")
    code, obj, _ = cli_json(capsys, "transcribe", a, "--format", "md")
    rec = obj["records"][0]
    with pytest.raises(UsageError) as e:
        record.resolve(rec["id"], command="relabel")
    assert e.value.code == "no_transcript_json" and "relabel" in e.value.message
    with pytest.raises(UsageError) as e:
        record.resolve(str(tmp_path / "nothing.m4a"))
    assert e.value.code == "unknown_record"


def test_reseal(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    r = record.resolve(rec["id"])
    assert record.reseal(r, required=True) == a
    a.write_text(a.read_text() + " ")
    with pytest.raises(Stale) as e:
        record.reseal(r, required=True)
    assert e.value.code == "input_changed"
    a.unlink()
    assert record.reseal(r, required=False) is None
    with pytest.raises(UsageError) as e:
        record.reseal(r, required=True)
    assert e.value.code == "input_missing"


def test_find_and_show_carry_no_text(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    found = record.find(str(a))
    assert found["id"] == rec["id"] and found["labels"] == ["speaker_0", "speaker_1"]
    assert found["displays"] == {"speaker_0": "Speaker 1", "speaker_1": "Speaker 2"}
    assert CANARY not in json.dumps(found)
    assert record.show(str(a))["id"] == rec["id"]
    assert record.show(rec["id"])["id"] == rec["id"]


def test_record_cli(fx, tmp_path, capsys):
    a, rec = _transcribe(capsys, tmp_path)
    code, obj, _ = cli_json(capsys, "record", "show", rec["id"])
    assert code == 0 and obj["record"]["id"] == rec["id"]
    code, obj, _ = cli_json(capsys, "record", "find", tmp_path / "Team sync (transcript).json")
    assert code == 0 and obj["labels"] == ["speaker_0", "speaker_1"]
    code, obj, _ = cli_json(capsys, "record", "show", "r-20260101-0badc0de")
    assert code == 2 and obj["error"]["code"] == "unknown_record"


def test_newest_first_even_within_one_second(fx):
    made = []
    with lock.run_lock():
        for _ in range(12):
            r = record.finish(record.new("transcribe"), 0)
            record.save(r)
            made.append(r["id"])
    assert [r["id"] for r in record.iter_records()] == made[::-1]


def test_an_audio_ref_finds_its_transcript_past_later_runs(fx, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    a = audio(tmp_path / "a.m4a")
    rid = cli_json(capsys, "transcribe", a, "--outdir", out)[1]["records"][0]["id"]
    assert cli_json(capsys, "transcribe", a, "--plan")[0] == 0                      # a record with no JSON
    assert cli_json(capsys, "transcribe", a, "--outdir", out, "--lang", "de")[0] == 2   # another one
    code, obj, _ = cli_json(capsys, "record", "find", a)
    assert code == 0 and obj["id"] == rid and obj["path"] == str(out / "a (transcript).json")
    code, obj, _ = cli_json(capsys, "relabel", a, "--label", "speaker_0", "--name", "Jane Doe")
    assert code == 0, obj


def test_records_read_rttm_with_the_engines_one_reader():
    line = "SPEAKER Team sync.16k 1 0.320 4.100 <NA> <NA> speaker_0 <NA> <NA>\n"
    assert record.segments(line) == [{"start": 0.32, "dur": 4.1, "spk": "speaker_0"}]
    assert record.segments(line + "SPEAKER cut short\n") == []          # what the engine can't read has no turns
    assert record.labels_of(None) == [] and record.segments("") == []


def test_each_language_option_renders_its_own_size():
    rec = record.new("transcribe")
    for code, size in (("hu", 3.1), ("th", 6.2)):
        record.add(rec, problem("language", code=code, language={"hu": "Hungarian", "th": "Thai"}[code], size=size))
    record.finalize(rec)
    o = rec["offers"][0]
    assert o["options"] == ["hu", "th"] and o["params"]["sizes"] == {"hu": 3.1, "th": 6.2}
    assert "--lang hu` downloads it (+3.1 GB)" in o["summary"][0] and o["summary"][0].startswith("Hungarian")
    assert "--lang th` downloads it (+6.2 GB)" in o["summary"][1] and o["summary"][1].startswith("Thai")
    carried = record.new("notes")                       # a later record of the chain words it the same way
    record.carry_offers(carried, rec["offers"])
    record.finalize(carried)
    assert [x.replace(rec["id"], "ID") for x in o["summary"]] == \
        [x.replace(carried["id"], "ID") for x in carried["offers"][0]["summary"]]


def test_a_carried_remember_offer_keeps_each_labels_seconds():
    rec = record.new("transcribe")
    record.add_offer(rec, "remember", ["speaker_0", "speaker_2"], params={"speech_s": {"speaker_0": 41.5,
                                                                                      "speaker_2": 9.0}})
    later = record.new("transcribe")                     # already done: no speakers list of its own
    record.carry_offers(later, rec["offers"])
    record.finalize(later)
    assert later["offers"][0]["summary"][0].startswith("speaker_0 has 41.5 s of clear speech.")
    assert later["offers"][0]["summary"][1].startswith("speaker_2 has 9 s of clear speech.")

