"""contracts.py and the package root: the shapes every module codes against, and telemetry off on import."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import audiointake
from audiointake import contracts as c


def test_version_line_is_the_only_version_source():
    init = Path(audiointake.__file__).read_text(encoding="utf-8")
    assert re.search(r'^__version__ = "\d+\.\d+\.\d+"', init, re.M)
    assert audiointake.__version__ == "0.1.0"


def test_import_turns_telemetry_off_first():
    env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY")}
    env["ORT_DISABLE_TELEMETRY"] = "0"
    code = "import os, json, audiointake; print(json.dumps([os.environ.get('ORT_DISABLE_TELEMETRY'), " \
           "os.environ.get('HF_HUB_DISABLE_TELEMETRY')]))"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["1", "1"]


def test_routes_and_patterns():
    assert c.ROUTES == ("hu_vote", "thai_screen", "thai_turns", "es_vote")
    label = re.compile(c.LABEL_RE)
    assert all(label.match(f"speaker_{k}") for k in range(8))
    assert not any(label.match(x) for x in ("speaker_8", "speaker_10", "Speaker_1", "speaker_1 ", "spk_1"))
    rid = re.compile(c.RECORD_ID_RE)
    assert rid.match("r-20261116-1a2b3c4d")
    assert not rid.match("r-2026111-1a2b3c4d") and not rid.match("r-20261116-1A2B3C4D")


def test_dataclass_fields_in_order():
    def names(cls):
        return [f.name for f in dataclasses.fields(cls)]
    assert names(c.Model) == ["key", "tag", "path", "revision"]
    assert names(c.Tools) == ["ffmpeg", "ffprobe", "python", "parakeet_cli", "whisper_cli", "fluidaudiocli",
                              "fluidaudio_tag", "caffeinate"]
    assert names(c.Runtime) == ["tools", "models", "routes", "env", "work_root", "voices_dir"]
    assert names(c.ThaiSummary) == ["screened", "windows", "thai_windows", "turns", "seconds", "screen_only",
                                    "failed"]
    assert names(c.EnsembleResult) == ["tag", "json_path", "txt_path", "hu_share", "hu_detected", "hu_route",
                                       "fill_ok", "thai"]
    assert names(c.Turn) == ["label", "display", "start", "end", "text"]
    assert names(c.Merged) == ["lines", "text", "turns"]
    assert names(c.Problem) == ["code", "params"]


def test_frozen_and_defaults():
    t = c.Tools(ffmpeg=Path("/f"), ffprobe=None, python=Path("/p"), parakeet_cli=Path("/a"), whisper_cli=Path("/w"),
                fluidaudiocli=None, fluidaudio_tag="v0.18.0")
    assert t.caffeinate == Path("/usr/bin/caffeinate")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.ffmpeg = Path("/x")                                   # type: ignore[misc]
    e = c.EnsembleResult("parakeet-ultra-mlx", None, None, 0.0, False, "none", True)
    assert e.thai == c.ThaiSummary(False) and e.thai is not c.EnsembleResult("x", None, None, 0, False, "none",
                                                                             True).thai
    assert c.Problem("fill_failed").params == {}
    assert c.ThaiSummary(True).failed is False


def test_problems_hash_by_value():
    a = c.Problem("speaker_count_differs", {"expected": 4, "found": 3})
    b = c.Problem("speaker_count_differs", {"found": 3, "expected": 4})
    assert a == b and hash(a) == hash(b) and len({a, b, c.Problem("fill_failed")}) == 2


def test_resolve_protocol_signature():
    import inspect
    sig = inspect.signature(c.Resolve.__call__)
    assert [p.name for p in sig.parameters.values()] == ["self", "need", "voices_dir", "names", "speakers"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in list(sig.parameters.values())[1:])


def test_languages():
    assert len(c.CORE_LANGS) == 25 and c.LANGS == c.CORE_LANGS + ("th",)
    assert set(c.LANGUAGE_NAMES) == set(c.LANGS)
    assert c.PACKS == ("core", "hu", "th") and set(c.PACK_OF_LANG.values()) <= set(c.PACKS)
    assert set(c.ROUTE_OF_LANG) == set(c.PACK_OF_LANG) and set(c.ROUTE_OF_LANG.values()) <= set(c.ROUTES)


def test_the_resolve_contract_is_written_down():
    doc = c.__doc__ or ""
    for words in ("``need``", "never downloads anything", "``speakers=False``", "``work_root``",
                  "older than a day", "``errors.with_context``"):
        assert words in doc
