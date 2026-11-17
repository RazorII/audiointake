"""A stopped steps program stops its children. Every child leads its own process group (proc.run), so a SIGTERM to
the program's group doesn't reach them by itself: the Thai worker's diarizer, the module command line's recogniser
and the speaker worker's ffmpeg are each stood in for by a script that writes its pid and sleeps, the program's
group is sent SIGTERM, and the stand-in must be gone, with the program's work folder and temporary files.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from audiointake.steps import ensemble

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"


def _sleeper(path: Path, pidfile: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho $$ > "{pidfile}"\nexec /bin/sleep 47\n', encoding="utf-8")
    path.chmod(0o755)
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_and_check(code: str, pidfile: Path, tmpdir: Path) -> tuple[int, str]:
    """Run ``code`` as a program in its own session, wait for the stand-in's pid, SIGTERM the program's group;
    the program's exit status and stderr. The stand-in must be gone within a few seconds."""
    pre = f"import sys; sys.path[:0] = [{str(SRC)!r}, {str(TESTS)!r}]\n"
    env = dict(os.environ, TMPDIR=str(tmpdir))
    p = subprocess.Popen([sys.executable, "-I", "-B", "-c", pre + textwrap.dedent(code)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    gpid = None
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end and p.poll() is None:
            if pidfile.exists() and pidfile.read_text().strip():
                gpid = int(pidfile.read_text().strip())
                break
            time.sleep(0.05)
        assert gpid is not None, p.communicate(timeout=5)
        assert _alive(gpid)
        os.killpg(p.pid, signal.SIGTERM)
        _, err = p.communicate(timeout=30)
        end = time.monotonic() + 5
        while _alive(gpid) and time.monotonic() < end:
            time.sleep(0.05)
        assert not _alive(gpid), "the stand-in child outlived the program"
        return p.returncode, err
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        if gpid is not None and _alive(gpid):
            os.kill(gpid, signal.SIGKILL)


def test_the_thai_worker_takes_its_diarizer_with_it(tmp_path):
    fa = _sleeper(tmp_path / "bin" / "fluidaudiocli", tmp_path / "pid")
    (tmp_path / "typhoon").mkdir()
    (tmp_path / "typhoon" / "weights.safetensors").write_bytes(b"x")
    (tmp_path / "u.json").write_text('{"text": "", "sentences": []}', encoding="utf-8")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    rc, err = _stop_and_check(f"""
        import numpy as np
        from audiointake.steps import thai

        class Au:                                    # 10 s, voiced throughout: the screen finds Thai
            def __init__(self, wav):
                self.dur, self.db, self.thr = 10.0, np.full(500, -20.0), -30.0
                self.a = np.zeros(160000, dtype=np.float32)

            def clip(self, s, e):
                return self.a[int(s * 16000):int(e * 16000)]

            def voiced(self, s, e):
                return 1.0
        thai.Audio = Au
        thai.LID = lambda m: (lambda au, spans: [{{"th": 0.9, "en": 0.05}} for _ in spans])
        sys.exit(thai.main(["run", {str(tmp_path / 'a.wav')!r}, {str(tmp_path / 'u.json')!r}, "",
                            {str(tmp_path / 'out.json')!r}, "--lid-model", "m", "--fluidaudio", {str(fa)!r},
                            "--nemotron", {str(tmp_path)!r}, "--typhoon", {str(tmp_path / 'typhoon')!r},
                            "--typhoon-tag", "t"]))
    """, tmp_path / "pid", tmpdir)
    assert rc == 128 + signal.SIGTERM, err
    assert list(tmpdir.iterdir()) == []                  # its thai.* folder went too
    assert not (tmp_path / "out.json").exists()


def test_the_module_command_line_takes_its_recogniser_and_work_folder_with_it(tmp_path):
    root = tmp_path / "rig"
    _sleeper(root / "bin" / "parakeet-mlx", tmp_path / "pid")
    audio = tmp_path / "in.m4a"
    audio.write_bytes(b"x")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    rc, err = _stop_and_check(f"""
        import dataclasses, types
        from pathlib import Path
        from fixtures import build
        from audiointake.contracts import Model
        import audiointake.setup
        root = Path({str(root)!r})
        rt = build.fake_runtime(root, ffprobe=False)
        rt = dataclasses.replace(rt, models={{"ultra": Model("ultra", "parakeet-ultra-mlx", root / "m", "0" * 40)}})
        mod = types.ModuleType("audiointake.setup.runtime")
        mod.resolve = lambda **kw: (rt, [])
        sys.modules["audiointake.setup.runtime"] = mod
        audiointake.setup.runtime = mod
        from audiointake.steps import ensemble
        sys.exit(ensemble.main(["transcribe", {str(audio)!r}, "--output-dir", {str(tmp_path / 'dest')!r},
                                "--output-template", "T"]))
    """, tmp_path / "pid", tmpdir)
    assert rc == 128 + signal.SIGTERM, err
    assert not (root / "work").exists()                  # the run's work folder is removed
    assert not (tmp_path / "dest").exists()


def test_the_speaker_worker_takes_its_ffmpeg_with_it(tmp_path):
    ff = _sleeper(tmp_path / "bin" / "ffmpeg", tmp_path / "pid")
    clip = tmp_path / "c1.wav"
    clip.write_bytes(b"x")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    rc, err = _stop_and_check(f"""
        from audiointake.steps import speaker_id as S
        S._extractor = lambda m: object()
        sys.exit(S.main(["enroll", "Brimdal", "en", {str(clip)!r}, "--ffmpeg", {str(ff)!r},
                         "--voices", {str(tmp_path / 'v')!r}, "--model", "m.onnx"]))
    """, tmp_path / "pid", tmpdir)
    assert rc == 128 + signal.SIGTERM, err
    assert list(tmpdir.iterdir()) == []                  # the clip's 16 kHz copy went too
    assert not (tmp_path / "v").exists()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_run_program_turns_a_signal_into_its_exit_status_and_puts_the_handlers_back(sig):
    before = {s: signal.getsignal(s) for s in ensemble.PROGRAM_SIGNALS}
    reached = []

    def body() -> int:
        try:
            os.kill(os.getpid(), sig)
            time.sleep(5)
            return 0
        finally:
            reached.append("finally")
    assert ensemble.run_program(body) == 128 + sig
    assert reached == ["finally"]
    assert {s: signal.getsignal(s) for s in ensemble.PROGRAM_SIGNALS} == before
