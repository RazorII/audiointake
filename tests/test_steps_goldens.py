"""The step modules against their goldens: every output byte for byte as the engine they were ported from wrote it.

The goldens in tests/fixtures/steps/ were made by running the reference implementation on synthetic inputs (made-up words,
placeholder names, tones for voices); manifest.json lists every case with its command line, the files it read and
the output, stdout and stderr it gave. Audio is never stored: the wavs are rebuilt here from their seeds.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from fixtures import build
from fixtures.steps import synth

from audiointake.steps import ensemble, merge_speakers, slim_asr, speaker_id, thai

FIX = Path(__file__).parent / "fixtures" / "steps"
MANIFEST = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
CASES = MANIFEST["cases"]


def _cases(kind):
    return [pytest.param(c, id=f"{kind}-{i}") for i, c in enumerate(x for x in CASES if x["kind"] == kind)]


def _arg(tok, tmp: Path, out: Path, wav: Path | None = None) -> str:
    if tok == "{out}":
        return str(out)
    if tok == "{wav}":
        return str(wav)
    if tok.startswith("{rttm:"):
        src = FIX / tok[6:-1]
        dst = tmp / (src.name[: -len(".rttm.txt")] + ".rttm")
        shutil.copyfile(src, dst)
        return str(dst)
    if tok.startswith("{names:"):
        return str(FIX / tok[7:-1])
    if tok and (FIX / tok).is_file():
        return str(FIX / tok)
    return tok


def test_manifest_has_every_kind():
    kinds = {c["kind"] for c in CASES}
    assert kinds == {"ensemble", "lang", "merge", "slim", "thai_worker", "speaker_id"}
    subs = {c["cli"][0] for c in CASES if c["kind"] == "ensemble"}
    assert subs == {"gapfill", "hu", "thai", "lang"}


@pytest.mark.parametrize("case", _cases("ensemble"))
def test_ensemble_golden(case, tmp_path, capsys):
    out = tmp_path / "out.json"
    wav = None
    if "wav" in case:
        wav = build.write_wav(tmp_path / "e.wav", case["wav"]["seconds"], kind=case["wav"]["kind"])
    args = []
    for tok in case["cli"]:
        if isinstance(tok, dict):                         # a seeded language-share input
            p = tmp_path / "lang.json"
            p.write_text(json.dumps(synth.lang_input(tok["lang_seed"], sorted(ensemble.HU_WORDS)),
                                    ensure_ascii=False), encoding="utf-8")
            args.append(str(p))
        else:
            args.append(_arg(tok, tmp_path, out, wav))
    assert ensemble.main(args) == 0
    got = capsys.readouterr()
    assert got.out == case["stdout"]
    assert got.err == case["stderr"]
    if "expect" in case:
        assert out.read_bytes() == (FIX / case["expect"]).read_bytes()


def test_lang_table_equals_the_original_on_200_seeded_inputs(tmp_path):
    golden = json.loads((FIX / "lang-200.json").read_text(encoding="utf-8"))
    assert golden["seeds"] == 200 and len(golden["results"]) == 200
    fw = sorted(ensemble.HU_WORDS)
    votes = 0
    for seed, (code, share) in enumerate(golden["results"]):
        p = tmp_path / f"l{seed}.json"
        p.write_text(json.dumps(synth.lang_input(seed, fw), ensure_ascii=False), encoding="utf-8")
        got = ensemble.lang(p)
        assert (got[0], repr(got[1])) == (code, share), seed
        votes += code == "hu"
    assert 20 <= votes <= 180          # both outcomes are well represented


@pytest.mark.parametrize("case", _cases("merge"))
def test_merge_golden(case, tmp_path, capsys):
    out = tmp_path / "out.txt"
    out.write_text("previous\n", encoding="utf-8")
    args = [_arg(t, tmp_path, out) for t in case["cli"]]
    rc = merge_speakers.main(args)
    got = capsys.readouterr()
    assert rc == case["rc"]
    assert got.out == case["stdout"]
    if case.get("stderr_prefix"):
        assert got.err.startswith(case["stderr"].split(": Expecting")[0])
    else:
        assert got.err == case["stderr"]
    if "expect" in case:
        assert out.read_bytes() == (FIX / case["expect"]).read_bytes()


@pytest.mark.parametrize("case", [c for c in _cases("merge") if c.values[0].get("expect")
                                  and "{names:speakers-bad.json}" not in c.values[0]["cli"]])
def test_merge_library_gives_the_command_lines_lines(case, tmp_path):
    """merge() returns exactly the lines the command line writes, whatever names, RTTM and rule it is given."""
    c = case
    cli = c["cli"]
    asr = json.loads((FIX / cli[0]).read_text(encoding="utf-8"))
    rttm_text = (FIX / cli[1][6:-1]).read_text(encoding="utf-8") if cli[1].startswith("{rttm:") else None
    names = None
    if "--names" in cli:
        sc = json.loads((FIX / cli[cli.index("--names") + 1][7:-1]).read_text(encoding="utf-8"))
        names = {k: v["name"] for k, v in sc["labels"].items() if v.get("name")}
    m = merge_speakers.merge(asr, rttm_text, names, split="--sentences" not in cli)
    assert m.text.encode("utf-8") == (FIX / c["expect"]).read_bytes()
    assert m.lines == m.text[:-1].split("\n")
    if rttm_text:
        assert len(m.turns) == len(m.lines)
        for line, t in zip(m.lines, m.turns):
            assert line == f"{t.display}: {t.text}"
            assert t.start <= t.end


@pytest.mark.parametrize("case", _cases("slim"))
def test_slim_golden(case, tmp_path, capsys):
    out = tmp_path / "out.asr.json"
    assert slim_asr.main([_arg(t, tmp_path, out) for t in case["cli"]]) == 0
    got = capsys.readouterr()
    assert (got.out, got.err) == (case["stdout"], case["stderr"])
    want = (FIX / case["expect"]).read_bytes()
    assert out.read_bytes() == want
    assert slim_asr.slim(json.loads((FIX / case["cli"][0]).read_text(encoding="utf-8"))).encode("utf-8") == want


def test_thai_worker_pure_parts_golden(tmp_path):
    np = pytest.importorskip("numpy")
    g = json.loads((FIX / "thai-worker.json").read_text(encoding="utf-8"))
    spec, want = g["spec"], g["expect"]
    assert [thai.collapse(t) for t in spec["collapse"]] == want["collapse"]
    rt = [json.loads(json.dumps(thai.turns_from_rttm(FIX / p))) for p in spec["rttm"]]
    assert rt == want["turns_from_rttm"]
    assert [thai.english(u, v, n) for u, v, n in spec["english"]] == want["english"]
    assert [thai.is_thai(*x) for x in spec["is_thai"]] == want["is_thai"]

    class Au:
        wav = "/nonexistent.wav"
    got = []
    for db, rttm_name in spec["turns"]:
        au = Au()
        au.db = np.asarray(db, dtype=np.float64)
        au.thr = float(np.percentile(au.db, 10) + 10)
        got.append(json.loads(json.dumps(thai.turns(au, str(FIX / rttm_name) if rttm_name else None, str(tmp_path)))))
    assert got == want["turns"]
    assert {t[1] for t in got} == {"energy", "nemotron"}


# --------------------------------------------------------------------------- speaker ID
SPK = json.loads((FIX / "speaker-id.json").read_text(encoding="utf-8"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ffmpeg() -> str | None:
    for p in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if p and os.access(p, os.X_OK):
            return p
    return None


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    ex = synth.FakeExtractor()
    monkeypatch.setattr(speaker_id, "_extractor", lambda model: ex)
    monkeypatch.setattr(speaker_id, "_today", lambda: SPK["day"])
    m = tmp_path / "model.onnx"
    m.write_bytes(b"not a model")
    return m


@pytest.mark.parametrize("case", [pytest.param(c, id=c["id"]) for c in SPK["cases"]])
def test_speaker_id_golden(case, tmp_path, fake_model, capsys):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    voices = tmp_path / "voices"
    voices.mkdir()
    before = {}
    for name, model_ok, vecs, sources in case["prints"]:
        p = synth.voiceprint(name, vecs, speaker_id.MODEL_SHA256 if model_ok else "0" * 64, sources=sources,
                             day=SPK["day"])
        text = json.dumps(p, indent=1)
        (voices / f"{name.lower()}.json").write_text(text, encoding="utf-8")
        before[f"{name.lower()}.json"] = text
    work = tmp_path / "work"
    work.mkdir()
    subs = {"{voices}": str(voices), "{work}": str(work)}
    for n, rec in SPK["recordings"].items():
        turns, stretches = synth.voice_layout(rec["seed"], SPK["labels"], rec["seconds"])
        w = work / f"spk-{n}.wav"
        w.write_bytes(synth.tone_track_wav(stretches))
        subs[f"{{wav:{n}}}"] = str(w)
        subs[f"{{rttm:{rec['rttm']}}}"] = str(FIX / rec["rttm"])
    extra = []
    if "clip" in case:
        ff = _ffmpeg()
        if ff is None:
            pytest.skip("ffmpeg isn't installed")
        secs, amp = case["clip"]
        clip = Path(case["args"][2].replace("{work}", str(work)))
        clip.write_bytes(synth.tone_track_wav([(secs, 330.0, amp)]))
        extra = ["--ffmpeg", ff]

    def fill(s):
        for k, v in subs.items():
            s = s.replace(k, v)
        return s
    rc = speaker_id.main([case["kind"], *[fill(a) for a in case["args"]], "--voices", str(voices), "--model",
                          str(fake_model), *extra])
    got = capsys.readouterr()
    assert rc == case["rc"]
    assert got.out == fill(case["stdout"])
    want_err = fill(case["stderr"])
    if "held back" in want_err:
        assert got.err.startswith(want_err.rstrip("\n").rstrip("."))
        assert "If the user confirms it by ear anyway, rerun with --force" in got.err
    else:
        assert got.err == want_err
    files = {}
    for f in sorted(voices.glob("*.json")):
        t = f.read_text(encoding="utf-8")
        if before.get(f.name) != t:
            files[f.name] = _sha(t)
    if "--out" in case["args"]:
        p = Path(fill(case["args"][case["args"].index("--out") + 1]))
        files["sidecar"] = _sha(p.read_text(encoding="utf-8")) if p.exists() else None
    assert files == case["files_sha256"]


def test_speaker_id_pure_parts_golden():
    pytest.importorskip("numpy")
    spec, want = SPK["spec"], SPK["expect"]
    got = [[[repr(a), repr(b)] for a, b in speaker_id.clean_pieces([tuple(s) for s in segs], label)]
           for segs, label in spec["clean_pieces"]]
    assert got == want["clean_pieces"]
    tv, pl = [], []
    for pv in spec["two_voices"]:
        pvs = [(speaker_id._unit(v), s) for v, s in pv]
        inter, minor = speaker_id.two_voices(pvs)
        emb, used = speaker_id.pool(pvs)
        tv.append([repr(float(inter)), repr(float(minor))])
        pl.append([[repr(float(x)) for x in emb], repr(float(used))])
    assert tv == want["two_voices"]
    assert pl == want["pool"]
    sentences, codes = [], []
    for name, vec, pv, prints in spec["problems"]:
        pr = {n: speaker_id._unit(v) for n, v in prints.items()}
        pvs = [(speaker_id._unit(v), s) for v, s in pv] if pv else None
        held = speaker_id.enrollment_problems(name, speaker_id._unit(vec), pvs, prints=pr)
        sentences.append([h.sentence for h in held])
        codes.append([h.problem.code for h in held])
    assert sentences == want["problems"]
    order = ("not_matching_print", "matches_someone_else", "mixed_voices", "two_voices")
    for cs, ss in zip(codes, sentences):
        assert cs == sorted(cs, key=order.index)
        for c, s in zip(cs, ss):
            assert {"not_matching_print": s.startswith("it scores "),
                    "matches_someone_else": s.startswith("audiointake would already name this voice "),
                    "mixed_voices": "don't sound like" in s,
                    "two_voices": s.startswith("it sounds like two voices")}[c]
    assert {c for cs in codes for c in cs} == set(order)          # every refusal is exercised
