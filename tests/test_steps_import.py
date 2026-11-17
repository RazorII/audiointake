"""Every steps module imports where mlx can't: a stand-in mlx (and mlx_whisper) that raises on import goes first on
sys.path, in a fresh interpreter, and every module still imports and every command line still prints its usage."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
MODULES = ("ensemble", "thai", "speaker_id", "merge_speakers", "slim_asr", "rttm")


@pytest.fixture
def no_mlx(tmp_path):
    for name in ("mlx", "mlx_whisper"):
        d = tmp_path / "stub" / name
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(f"raise ImportError('{name} is stubbed out for this test')\n", encoding="utf-8")
    return tmp_path / "stub"


def _py(stub: Path, code: str) -> subprocess.CompletedProcess:
    pre = f"import sys; sys.path[:0] = [{str(stub)!r}, {str(SRC)!r}]\n"
    return subprocess.run([sys.executable, "-I", "-B", "-c", pre + textwrap.dedent(code)], capture_output=True,
                          text=True)


def test_every_steps_module_imports_without_mlx(no_mlx):
    r = _py(no_mlx, f"""
        import importlib
        try:
            import mlx
        except ImportError:
            pass
        else:
            raise SystemExit("the stub didn't take")
        for m in {MODULES!r}:
            importlib.import_module("audiointake.steps." + m)
        bad = sorted(k for k in sys.modules if k.split(".")[0] in ("mlx", "mlx_whisper", "numpy", "sherpa_onnx"))
        print("ok", bad)
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok []"


# Each command line, run with no arguments, says how it is used and exits non-zero (rttm.py has no command line).
USAGE = {"ensemble": "ensemble transcribe <audio> --output-dir D", "thai": "usage: python -I -B -m audiointake.steps.thai",
         "speaker_id": "speaker_id: usage: speaker_id match|enroll-label|enroll|list|check",
         "merge_speakers": "usage: merge_speakers <asr.json>", "slim_asr": "usage: slim_asr <parakeet.json>"}


@pytest.mark.parametrize("module", sorted(USAGE))
def test_every_command_line_prints_its_usage_without_mlx(no_mlx, module):
    r = _py(no_mlx, f"""
        import runpy
        sys.argv = [{module!r}]
        runpy.run_module("audiointake.steps." + {module!r}, run_name="__main__")
    """)
    assert r.returncode in (1, 2), r.stderr
    assert USAGE[module] in r.stderr


def test_the_thai_worker_says_what_it_needs_without_mlx(no_mlx, tmp_path):
    r = _py(no_mlx, f"""
        from audiointake.steps import thai
        try:
            thai.Audio({str(tmp_path / 'x.wav')!r})
        except ImportError as e:
            print("import error:", e)
    """)
    assert r.returncode == 0, r.stderr
    assert "stubbed out" in r.stdout
