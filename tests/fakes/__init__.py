"""Fakes for the CLI and pipeline tests: stand-in programs, the engine's and setup's functions, made-up audio.

* **Fake audio.** An input "recording" is a small JSON file (``fake_audio``) describing itself: its length,
  channels, how many sentences the fake recogniser hears, how many speakers the fake diarizer finds, who the
  fake speaker matcher names, and which step should fail and how. The fake ffmpeg copies that description into
  every wav it "converts", so each later step reads the same description.
* **Fake programs** (``fake_*.py``) stand in for ``parakeet-mlx``, ``mlx_whisper``, ``fluidaudiocli``, ffmpeg,
  ffprobe and the tool's own Python (the speaker-ID worker). ``install_shims`` writes small executable shims
  for them into a test's folder; each starts this interpreter isolated (``-I``), writes no bytecode, and runs
  the fake. Every text field a fake writes carries ``CANARY``; the fakes also print transcript text with
  ``STDOUT_MARK`` to stdout, which no log may ever contain.
* **Stubs** with the engine's (``stub_steps``) and setup's (``stub_setup``) exact signatures replace
  ``pipeline.engine`` and ``pipeline.setup_api`` (``use_fakes``), so these tests never import the engine's or
  setup's own modules.

Everything is made up: invented words, placeholder names, generated numbers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
CANARY = "qzcanaryqz"                      # in every text field a fake writes
STDOUT_MARK = "qzstdoutmarkqz"             # what fake children print to stdout
WORDS = tuple(w + CANARY for w in ("zorvel", "brimdal", "quessa", "toraph", "velmine", "snarth", "kelvo",
                                   "plimbo", "dravik", "fennow"))
PROGRAMS = {"ffmpeg": "fake_ffmpeg", "ffprobe": "fake_ffprobe", "parakeet-mlx": "fake_parakeet",
            "mlx_whisper": "fake_whisper", "fluidaudiocli": "fake_fluidaudio", "python": "fake_python"}
MODEL_SHA = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"


def audio(path: Path, **desc: Any) -> Path:
    """Write a fake recording: a JSON description (defaults: 60 s mono, 6 sentences, 2 speakers, seed 1)."""
    d = {"fake_audio": 1, "duration": 60.0, "channels": [1], "creation_time": None, "sentences": 6, "seed": 1,
         "speakers": 2, "match": {}, "fail": {}}
    d.update(desc)
    path = Path(path)
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


def read_desc(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return d if isinstance(d, dict) and d.get("fake_audio") == 1 else None


def install_shims(bin_dir: Path) -> Path:
    """Executable shims for every fake program in ``bin_dir``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, module in PROGRAMS.items():
        p = bin_dir / name
        p.write_text(f"#!{sys.executable} -I\n"
                     "import sys\n"
                     "sys.dont_write_bytecode = True\n"
                     f"sys.path.insert(0, {str(TESTS)!r})\n"
                     f"from fakes import {module} as m\n"
                     "sys.exit(m.main(sys.argv[1:]))\n", encoding="utf-8")
        p.chmod(0o755)
    return bin_dir


def voiceprint(folder: Path, name: str, *, model_sha256: str = MODEL_SHA, updated: str = "2026-01-02",
               seconds: float = 30.0, source: str = "sample one.m4a") -> Path:
    """A made-up voiceprint file (``name.lower() + ".json"``) with a short placeholder vector."""
    folder.mkdir(parents=True, exist_ok=True)
    d = {"name": name, "model_sha256": model_sha256,
         "vectors": [{"vec": [0.1, 0.2, 0.3], "source": source, "seconds": seconds, "lang": "EN",
                      "added": updated}], "updated": updated}
    p = folder / f"{name.lower()}.json"
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return p


def use_fakes(monkeypatch: Any, tmp_path: Path, **setup: Any) -> Any:
    """Point audiointake at per-test folders and at the fakes; returns the ``stub_setup.FakeSetup``."""
    from audiointake import cli, pipeline

    from . import stub_setup, stub_steps
    home = tmp_path / "home"
    for sub in ("data", "cache", "home", "hf"):
        (tmp_path / sub).mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AUDIOINTAKE_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AUDIOINTAKE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    fs = stub_setup.FakeSetup(tmp_path / "fake", **setup)
    monkeypatch.setattr(pipeline, "setup_api", fs.namespace)
    monkeypatch.setattr(pipeline, "engine", stub_steps.namespace)
    monkeypatch.setattr(cli, "CAFFEINATE", tmp_path / "no-caffeinate")
    return fs


def cli(capsys: Any, *args: Any) -> tuple[int, str, str]:
    """Run ``audiointake ARGS`` in this process; (exit, stdout, stderr)."""
    from audiointake import cli as cli_mod
    code = cli_mod.main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


def cli_json(capsys: Any, *args: Any) -> tuple[int, dict[str, Any], str]:
    """Run ``audiointake ARGS --json``; (exit, the parsed JSON, stderr). The JSON's exit must equal the process's."""
    code, out, err = cli(capsys, *args, "--json")
    obj = json.loads(out)
    assert obj["exit"] == code, (obj["exit"], code)
    return code, obj, err
