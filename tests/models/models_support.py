"""Shared helpers for the real-model tests in tests/models (marker ``models``; scripts/models-local.sh runs them).

Nothing here imports audiointake at module level, and nothing skips at module level: a module-level skip counts as a
skip under ``-m models``, and the models gate fails on any skip. Each test skips itself (per-test ``skipif`` and the
``models_env`` and ``pinned_audio`` fixtures) when it can't run here.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
needs_apple_silicon = pytest.mark.skipif(not APPLE_SILICON, reason="the real models run only on a Mac with Apple silicon")
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
# Every network operation is denied, and a connection to any IP address or a name lookup (the mDNSResponder socket)
# is answered with SIGKILL: an attempt that a library would quietly fall back from (huggingface_hub reads its local
# cache after a connection error) kills the run instead, so the test fails. SIGKILL is not put on a bare network*:
# on macOS that modifier also fires on a hard link (os.link), which filelock makes when huggingface_hub imports it.
DENY_NETWORK = ("(version 1)(allow default)(deny network*)"
                "(deny network-outbound (remote ip) (with send-signal SIGKILL))"
                '(deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder"))'
                " (with send-signal SIGKILL))")
TIMEOUT = 1800

VOTE_TAG = "whisper-large-v3+ultra/v3-vote"
THAI_TAG_SUFFIX = "+typhoon-whisper-large-v3-thai"

# FLEURS clips of tools/fetch_test_audio.py and the ceilings of their corpus error rate (%) through audiointake:
# the rate the reference engine measured on exactly these clips (bench/fleurs_routes.py, 2026-11-17, same code path:
# the whole transcription for de, es and th, the vote route for the --lang hu clips). The ceiling allows for other
# Apple-silicon chips: max(2 points, a quarter of the rate) for de, es and th, rounded up. For hu the ceiling must
# also sit below the core route's rate on the same clips (14.71 %: 30 errors in 204 words), so a run that silently
# fell back to core fails on the number as well as on the route: measured + max(1 point, a tenth), rounded up.
FLEURS_CHECKS = {
    # lang: (ids, extra transcribe flags, measured rate, ceiling, route check)
    "de_de": ("1662 1665 1670 1687 1688".split(), [], 2.63, 4.7, None),           # 4 errors in 152 words
    "es_419": ("1662 1665 1670 1687 1688".split(), [], 0.63, 2.7, None),          # 1 error in 159 words
    "hu_hu": ("1660 1670 1676 1677 1680 1683 1684 1694 1713".split(), ["--lang", "hu"], 12.25, 13.5, "vote"),
    "th_th": ("1663 1672 1673 1675".split(), [], 9.11, 11.4, "thai_turns"),        # 39 in 428 characters
}


def check_route(rec: dict, route: str | None, what: str) -> None:
    """Assert that a record took the route its pack adds (never core with a warning, never a fallback)."""
    if route == "vote":
        assert rec["signals"]["hu_route"] == "vote", f"{what}: the vote route didn't run"
        assert rec["engines"]["asr_tag"] == VOTE_TAG, f"{what}: tag {rec['engines']['asr_tag']}"
    elif route == "thai_turns":
        assert rec["signals"]["thai_screened"] is True, f"{what}: no language screen"
        assert rec["signals"]["thai_turns"] > 0, f"{what}: no turn went through the turn route"
        assert rec["engines"]["asr_tag"].endswith(THAI_TAG_SUFFIX), f"{what}: tag {rec['engines']['asr_tag']}"
    codes = {w["code"] for w in rec.get("warnings", [])}
    fell_back = codes & {"pack_not_installed", "pack_broken", "hungarian_vote_failed", "thai_failed", "fill_failed"}
    assert not fell_back, f"{what}: {sorted(fell_back)}"


def bench_module(name: str):
    """bench/<name>.py, imported by path (bench/ isn't a package)."""
    import importlib.util
    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- per-child peak memory
# macOS keeps, for every process, the lifetime maximum of its physical footprint (the figure Activity Monitor calls
# Memory, Metal/MLX allocations included; resident set size leaves those out). /usr/bin/time -l reports it only for
# its direct child, which here is the orchestrator, not a model child. So a sampler thread walks the CLI's process
# tree every SAMPLE_S while it runs and reads each descendant's ri_lifetime_max_phys_footprint (proc_pid_rusage,
# RUSAGE_INFO_V4): a high-water mark, so a sample at any moment after the peak still sees it. A child that lives less
# than SAMPLE_S can be missed; model children run for seconds.
SAMPLE_S = 0.05
_RUSAGE_INFO_V4 = 4
_LIFETIME_MAX = 2 + 28      # uint64 index in rusage_info_v4: the 16-byte uuid, then ri_lifetime_max_phys_footprint
_CTL_KERN, _KERN_PROCARGS2 = 1, 49


def _libs():
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    return libc, libproc


def _lifetime_max_footprint(libproc, pid: int) -> int | None:
    buf = (ctypes.c_uint64 * 64)()
    if libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
        return None
    return int(buf[_LIFETIME_MAX])


def _children(libproc, pid: int) -> list[int]:
    buf = (ctypes.c_int * 4096)()
    n = libproc.proc_listchildpids(pid, buf, ctypes.sizeof(buf))
    return [buf[i] for i in range(max(n, 0)) if buf[i] > 0]


def _argv(libc, pid: int) -> list[str]:
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0 or not size.value:
        return []
    raw_buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, raw_buf, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        return []
    raw = raw_buf.raw[: size.value]
    argc, rest = int.from_bytes(raw[:4], "little"), raw[4:]
    i = rest.find(b"\0")                                   # the executable path, then NUL padding, then argv
    while 0 <= i < len(rest) and rest[i] == 0:
        i += 1
    return [a.decode("utf-8", "replace") for a in rest[max(i, 0):].split(b"\0")[:argc]]


def child_kind(argv: list[str]) -> str:
    """What a process is, from its argv, without any path or argument: a module run with -m, a script's name, or the
    executable's name (parakeet-mlx, mlx_whisper, audiointake.steps.thai, fluidaudiocli, ffmpeg, ...)."""
    if not argv:
        return "?"
    b0 = os.path.basename(argv[0])
    if b0.lower().startswith("python"):
        if "-m" in argv[:-1]:
            return argv[argv.index("-m") + 1]
        if "-c" in argv:
            return "python -c"
        rest = [a for a in argv[1:] if not a.startswith("-")]
        return os.path.basename(rest[0]) if rest else b0
    return b0


def run_sampled(argv: list[str], **kw) -> tuple[subprocess.CompletedProcess, list[dict]]:
    """subprocess.run with the per-child sampler: (result, [{"kind", "peak_footprint_bytes"}] for every descendant
    process seen, the CLI itself excluded)."""
    libc, libproc = _libs()
    timeout = kw.pop("timeout", TIMEOUT)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
    seen: dict[int, list] = {}
    stop = threading.Event()

    def walk() -> None:
        while not stop.is_set():
            stack = _children(libproc, proc.pid)
            while stack:
                pid = stack.pop()
                fp = _lifetime_max_footprint(libproc, pid)
                if fp is not None:
                    entry = seen.setdefault(pid, [0, None])
                    # read on every sample: a child seen between fork and exec still carries its parent's argv
                    av = _argv(libc, pid)
                    if av:
                        entry[1] = child_kind(av)
                    entry[0] = max(entry[0], fp)
                stack += _children(libproc, pid)
            stop.wait(SAMPLE_S)

    t = threading.Thread(target=walk, daemon=True)
    t.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    finally:
        stop.set()
        t.join()
        if proc.poll() is None:
            proc.kill()
    result = subprocess.CompletedProcess(argv, proc.returncode, out, err)
    return result, [{"kind": k or "?", "peak_footprint_bytes": fp} for fp, k in seen.values()]


def peak_by_kind(children: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in children:
        out[c["kind"]] = max(out.get(c["kind"], 0), c["peak_footprint_bytes"])
    return out


# --------------------------------------------------------------------------- the CLI as a child
def audiointake(env: dict[str, str], *args: str, wrap: list[str] | None = None, cwd: Path | None = None,
                sample: bool = False):
    """Run the installed CLI as a child (the interpreter running the tests), never with a terminal or stdin. With
    sample=True, returns (result, children) from run_sampled."""
    argv = [*(wrap or []), sys.executable, "-m", "audiointake", *args]
    if sample:
        return run_sampled(argv, env=env, stdin=subprocess.DEVNULL, cwd=cwd)
    return subprocess.run(argv, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          timeout=TIMEOUT, cwd=cwd)


def transcribe(env: dict[str, str], audio: Path, outdir: Path, *flags: str, fmt: str = "md,json",
               wrap: list[str] | None = None, sample: bool = False):
    """One `transcribe --json`; returns (result, record), or (result, record, children) with sample=True."""
    outdir.mkdir(parents=True, exist_ok=True)
    got = audiointake(env, "transcribe", str(audio), "--outdir", str(outdir), "--format", fmt, *flags, "--json",
                      wrap=wrap, sample=sample)
    r, children = got if sample else (got, None)
    assert r.returncode == 0, f"transcribe exited {r.returncode}; stderr tail: {r.stderr[-1500:]}"
    data = json.loads(r.stdout)
    assert data["exit"] == 0 and len(data["records"]) == 1, data
    return (r, data["records"][0], children) if sample else (r, data["records"][0])


def enroll_probe(env: dict[str, str], audio: Path, work: Path, wrap: list[str] | None = None) -> Path:
    """A voiceprint folder holding one print, enrolled from the label with the most speech in audio's own transcript
    (so a later transcription with --voices runs the speaker-recognition worker and names someone)."""
    voices = work / "voices"
    voices.mkdir(parents=True, exist_ok=True)
    _, rec = transcribe(env, audio, work / "enroll-from", "--no-names", fmt="json", wrap=wrap)
    label = max(rec["speakers"], key=lambda s: s.get("speech_s") or 0)["label"]
    stem = audio.name[: audio.name.rfind(".")]
    r = audiointake(env, "enroll", "Probe", "--from", str(work / "enroll-from" / f"{stem} (transcript).json"),
                    "--label", label, "--force", "--voices", str(voices), "--json", wrap=wrap)
    assert r.returncode == 0, f"enroll exited {r.returncode}; stderr tail: {r.stderr[-800:]}"
    assert len(list(voices.glob("*.json"))) == 1, "enroll wrote no voiceprint"
    return voices


def transcript_text(outdir: Path, stem: str) -> str:
    """The one-line text the engine writes for a transcript (its sentences joined)."""
    tj = json.loads((outdir / f"{stem} (transcript).json").read_text(encoding="utf-8"))
    return " ".join(s["text"].strip() for s in tj["asr"]["sentences"]).strip()


def cut_wav(src: Path, dest: Path, start: float, end: float, src_start: float) -> Path:
    """Seconds start-end of the meeting from a 16 kHz mono excerpt that begins src_start seconds into it."""
    with wave.open(str(src), "rb") as w:
        w.setpos(int(round((start - src_start) * w.getframerate())))
        frames = w.readframes(int((end - start) * w.getframerate()))
        params = w.getparams()
    with wave.open(str(dest), "wb") as o:
        o.setparams(params)
        o.writeframes(frames)
    return dest


def ffmpeg() -> str:
    import shutil
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def build_fixtures(pinned_audio, out: Path) -> Path:
    """bench/fixtures.py's path fixtures, built from the pinned test audio into out (skips when any clip is absent)."""
    import conftest
    ami = pinned_audio("ami-es2004a-825-915")
    for key in conftest._fetch_tool().AUDIO:
        if key.startswith("fleurs-"):
            pinned_audio(key)
    fx = bench_module("fixtures")
    assert fx.build(out, ami.parents[1] / "fleurs", ami, ffmpeg(), 825.0) == 0
    return out


def fleurs_refs(pinned_audio, lang: str) -> str:
    """Every pinned FLEURS reference of a language, joined (the fixtures are made of those clips)."""
    import conftest
    keys = [k for k in conftest._fetch_tool().AUDIO if k.startswith(f"fleurs-{lang}-") and k.endswith("-ref")]
    return "\n".join(pinned_audio(k).read_text(encoding="utf-8") for k in sorted(keys))


def words_of(text: str, min_len: int = 6) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) >= min_len}


def report(entry: dict) -> None:
    """Append one JSON line to $AUDIOINTAKE_MODELS_REPORT (from the real environment), if scripts/models-local.sh set it."""
    import conftest
    path = conftest.REAL_ENV.get("AUDIOINTAKE_MODELS_REPORT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
