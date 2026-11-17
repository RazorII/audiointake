"""steps.rttm, the merge's library form and slim's library form, beyond the goldens."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures import build

from audiointake.contracts import Merged, Turn
from audiointake.steps import merge_speakers as M
from audiointake.steps import rttm, slim_asr

FIX = Path(__file__).parent / "fixtures" / "steps"


# --------------------------------------------------------------------------- rttm
def test_text_and_path_readers_agree(tmp_path):
    for n in (1, 2, 3, 4):
        text = (FIX / f"diar-{n}.rttm.txt").read_text(encoding="utf-8")
        p = tmp_path / "x.rttm"
        p.write_text(text, encoding="utf-8")
        a, b = rttm.segments_from_text(text), rttm.segments_from_path(p)
        assert a == b and a and rttm.segments(p) == a


def test_file_ids_with_spaces_and_line_endings():
    line = build.rttm_line("Team sync one.16k", 1.25, 2.5, "speaker_3")
    assert rttm.parse_line(line) == {"start": 1.25, "dur": 2.5, "spk": "speaker_3"}
    crlf = (line + line).replace("\n", "\r\n")
    cr = (line + line).replace("\n", "\r")
    assert rttm.segments_from_text(crlf) == rttm.segments_from_text(cr) == [rttm.parse_line(line)] * 2
    assert rttm.parse_line("") is None and rttm.parse_line("SPKR-INFO x") is None


@pytest.mark.parametrize("bad", ["SPEAKER a 1 0.5 1.0", "SPEAKER a 1 x 1.0 <NA> <NA> speaker_0 <NA> <NA>"])
def test_broken_lines_raise(bad):
    with pytest.raises(ValueError):
        rttm.parse_line(bad)
    with pytest.raises(ValueError):
        rttm.segments_from_text(bad + "\n")


# --------------------------------------------------------------------------- merge()
def _asr(n=1):
    return json.loads((FIX / f"asr-m{n}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("display", ["", None])
def test_an_empty_display_counts_as_no_name(display):
    """As the command line drops a sidecar label without a name: never two labels sharing one "Speaker N"."""
    text = (FIX / "diar-1.rttm.txt").read_text(encoding="utf-8")
    plain = M.merge(_asr(), text, {})
    for label in ("speaker_0", "speaker_1", "speaker_2"):
        m = M.merge(_asr(), text, {label: display})
        assert m.lines == plain.lines and [t.display for t in m.turns] == [t.display for t in plain.turns]


def test_merge_turns_carry_times_and_labels():
    text = (FIX / "diar-1.rttm.txt").read_text(encoding="utf-8")
    m = M.merge(_asr(), text, {"speaker_0": "Ana"})
    assert isinstance(m, Merged) and all(isinstance(t, Turn) for t in m.turns)
    labels = {t.label for t in m.turns}
    assert labels <= {"speaker_0", "speaker_1", "speaker_2", None}
    for t in m.turns:
        assert t.display == ("Ana" if t.label == "speaker_0" else t.display)
        if t.label is not None and t.label != "speaker_0":
            assert t.display.startswith("Speaker ")
    assert m.turns[0].start == _asr()["sentences"][0]["tokens"][0]["start"]
    assert m.turns[-1].end <= _asr()["sentences"][-1]["end"]


def test_a_display_for_every_label_keeps_numbers():
    """relabel passes a display for every label: nobody is renumbered."""
    text = (FIX / "diar-1.rttm.txt").read_text(encoding="utf-8")
    before = M.merge(_asr(), text, None)
    shown = {t.label: t.display for t in before.turns if t.label}
    names = dict(shown, speaker_0="Ana")
    after = M.merge(_asr(), text, names)
    for t in after.turns:
        if t.label and t.label != "speaker_0":
            assert t.display == shown[t.label]


def test_merge_without_diarization_has_sentence_turns():
    a = _asr()
    for r in (None, "", "\n"):
        m = M.merge(a, r, None)
        assert m.lines == [" ".join(s["text"].strip() for s in a["sentences"]).strip()]
        assert [(t.label, t.display, t.start) for t in m.turns] == [(None, "", s["start"]) for s in a["sentences"]]


def test_merge_with_no_sentences_is_not_a_failure():
    with pytest.raises(M.NoSentences):
        M.merge({"text": "", "sentences": []}, None, None)
    assert issubclass(M.NoSentences, ValueError)


def test_count_speakers():
    for n, want in ((1, 3), (2, 2), (3, 4)):
        assert M.count_speakers((FIX / f"diar-{n}.rttm.txt").read_text(encoding="utf-8")) == want
    assert M.count_speakers("") == 0 and M.count_speakers(None) == 0


def test_cli_keeps_the_plain_text_when_the_asr_is_empty(tmp_path, capsys):
    a = tmp_path / "a.json"
    a.write_text('{"text": "", "sentences": []}', encoding="utf-8")
    out = tmp_path / "o.txt"
    assert M.main([str(a), "", str(out)]) == 1                 # nothing to keep: a failure
    out.write_text("plain\n", encoding="utf-8")
    assert M.main([str(a), "", str(out)]) == 0 and out.read_text(encoding="utf-8") == "plain\n"
    assert M.main([str(a)]) == 1 and "usage:" in capsys.readouterr().err


def test_slim_library_and_sentences():
    d = json.loads((FIX / "out-thai-1.json").read_text(encoding="utf-8"))
    s = slim_asr.slim(d)
    assert "\n" not in s and json.loads(s)["sentences"]
    th = [x for x in json.loads(s)["sentences"] if "w" not in x]
    assert th                                                   # the Thai sentences carry no words
    assert slim_asr.slim_sentence({"start": 1.23456, "end": 2, "text": "x"}) == {"start": 1.235, "end": 2.0, "text": "x"}


@pytest.mark.parametrize("mod", ["merge_speakers", "slim_asr"])
def test_modules_run_as_programs(mod, tmp_path):
    r = subprocess.run([sys.executable, "-I", "-B", "-m", f"audiointake.steps.{mod}"], capture_output=True, text=True)
    assert r.returncode == 1 and "usage:" in r.stderr
