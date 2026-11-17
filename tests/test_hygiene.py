"""Repo hygiene, checked on the tree beside this file. Generic patterns only: names and terms that must never
be published are checked by a scanner outside this repository.

* **Only ``proc.py`` starts processes.** No other module in ``src/`` imports ``subprocess`` (or ``pty``,
  ``_posixsubprocess``, ``multiprocessing``, ``_multiprocessing``, ``concurrent.futures.process``,
  ``asyncio.subprocess``, ``webbrowser``), uses ``os.system``, ``os.popen``, ``os.exec*``, ``os.spawn*``,
  ``os.posix_spawn*``, ``os.fork*``, ``asyncio.create_subprocess_*`` or ``ProcessPoolExecutor``, calls a method
  named like one of those on any object (``loop.subprocess_exec``, a C library's ``system``), star-imports
  ``os`` or ``asyncio``, imports a module or reads an ``os``, ``asyncio`` or ``platform`` attribute by a name that
  isn't written out, or uses a ``platform`` helper outside ``PLATFORM_ALLOWED`` (the others can start
  ``uname``). Read statically, with ``ast``; what that can't see (a name built at run time and handed to ctypes)
  is the reviewer's job.
* **A Python child never writes bytecode.** Every argument list in ``src/`` that starts Python with ``-I`` also
  passes ``-B``: ``-I`` makes Python ignore ``PYTHONDONTWRITEBYTECODE``, so without ``-B`` a run from a source
  folder leaves ``__pycache__`` beside the code.
* **Every module imports anywhere, and touches no network doing it.** Each module in ``src/`` is imported in a
  fresh interpreter where audiointake code may not import MLX, mlx-whisper or parakeet-mlx at module level (they
  need the Mac's GPU, and go inside the function that uses them), nor an extra (``jsonschema``, ``pyarrow``,
  ``pythainlp``, ``pytest``); sockets raise there, and any attempt fails the test. A sibling that another part
  of the build hasn't written yet is allowed while it's a planned module (``PLANNED``) not on disk. Where MLX
  runs on this Mac, ``steps.thai`` and ``setup.convert`` are imported again together with every module their
  functions import, so a typo in a deferred import can't hide.
* **Telemetry is off once ``import audiointake`` has run**: ``ORT_DISABLE_TELEMETRY`` and
  ``HF_HUB_DISABLE_TELEMETRY`` are ``1``.
* **No private paths**: no home folder other than the placeholder ``/Users/example/``, no Linux home, no
  scratch-folder path, in any file that could be committed.
* **No audio, whatever the extension**: no file starts like WAV, an MP4 family file (``ftyp``), Ogg, FLAC, ID3,
  CAF, AIFF, WebM or Matroska (EBML), AMR, WMA (ASF), or an MPEG audio or ADTS frame (MP3 without a tag, raw AAC).
* **No voiceprint, speaker file, transcript or record, whatever the name.** Every committed text file that isn't
  code is read as JSON (a whole document, or one per line, after any byte-order mark): no object may hold both
  ``vectors`` and ``model_sha256`` (a voiceprint) or an ``enrolled`` key (a speaker-ID roster), anywhere; outside
  ``tests/fixtures/`` and ``tests/fakes/`` none may be an audiointake transcript (``"format":
  "audiointake-transcript"``) or a record (``record_version`` with an ``r-`` id); and outside
  ``tests/fixtures/steps/`` none may be a recogniser's output (``sentences`` or ``segments`` of text). A file
  that doesn't parse (a trailing comma, say) is still searched for the voiceprint and roster keys as text.
* **No RTTM line** outside ``tests/fixtures/steps/*.rttm.txt``, where the synthetic goldens live: in any file, no
  line that starts like one (the file id may hold spaces, as recordings' names do), so a block pasted into code
  is caught too; and in any file that isn't code, nowhere, not even inside a JSON string, the diarizer's own
  shape (decimal times and its ``<NA>`` fields). A one-line example inside a string in test code is how a
  parser is tested, so code is held to the first rule only.
* **Nothing in ``src/`` or ``tests/`` is silently ignored by ``.gitignore``** (a fixture folder named after an
  ignored pattern would vanish from every commit); build byproducts such as ``__pycache__`` aside.

Files are the ones git would commit: tracked, or untracked and not ignored (``git ls-files``), or every file
outside ``.git`` when this isn't a git checkout.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "audiointake"

BANNED_MODULES = {"subprocess", "pty", "_posixsubprocess", "multiprocessing", "_multiprocessing",
                  "concurrent.futures.process", "asyncio.subprocess", "webbrowser"}
BANNED_ATTRS = {
    "os": {"system", "popen", "fork", "forkpty", "startfile"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
    "concurrent.futures": {"ProcessPoolExecutor"},
}
BANNED_OS_PREFIXES = ("exec", "spawn", "posix_spawn")
# Method names that start a process whatever object they're called on (an event loop, a ctypes library).
BANNED_ANY_ATTR = {"system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "startfile",
                   "subprocess_exec", "subprocess_shell", "create_subprocess_exec", "create_subprocess_shell"}
NO_STAR = {"os", "asyncio", "subprocess", "pty", "platform", "concurrent.futures"}
NO_DYNAMIC_GETATTR = {"os", "asyncio", "platform", "concurrent.futures"}
PLATFORM_ALLOWED = {"mac_ver", "machine", "system", "release", "python_version", "python_version_tuple",
                    "python_implementation"}           # read os.uname, sys or a plist; never start a process
PROCESS_MODULE = "proc.py"                             # the one module that starts processes

GPU_ONLY = {"mlx", "mlx_whisper", "parakeet_mlx", "mlx_metal"}
OPTIONAL = {"jsonschema", "pyarrow", "pythainlp", "pytest"}
PLANNED = {f"audiointake.{m}" for m in (
    "__main__ contracts errors argv platforms proc fsutil lock "
    "cli pipeline probe outputs render record notes voices doctor about "
    "steps steps.ensemble steps.thai steps.speaker_id steps.merge_speakers steps.slim_asr steps.rttm "
    "setup setup.manifest setup.store setup.download setup.convert setup.fluidaudio setup.swiftpm "
    "setup.smoke setup.runtime setup.ffmpeg").split()}
MLX_CHECKED = ("audiointake.steps.thai", "audiointake.setup.convert")

# Built so that this file doesn't hold the strings it looks for.
HOME_MAC = "/" + "Users/"
HOME_PLACEHOLDER = HOME_MAC + "example/"
HOME_LINUX = "/" + "home/"
SCRATCH_FORMS = ("/." + "scratch/", "~/." + "scratch")
AUDIO_MAGIC = ((0, b"RIFF"), (4, b"ftyp"), (0, b"OggS"), (0, b"fLaC"), (0, b"ID3"), (0, b"caff"), (0, b"FORM"),
               (0, b"\x1a\x45\xdf\xa3"),                                    # EBML: WebM, Matroska
               (0, b"#!AMR"),
               (0, bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")))      # ASF: WMA, WMV
# An RTTM line, at the start of a line: the file id runs up to the channel and the two times, so it may hold spaces.
RTTM_LINE = re.compile(r"^\s*SP" r"EAKER\s+\S.*?\s+\d+\s+-?\d+(?:\.\d+)?\s+\d+(?:\.\d+)?(?:\s|$)")
# The diarizer's own shape anywhere, even inside a string: decimal times and the <NA> fields around the speaker.
RTTM_ANYWHERE = re.compile(r"SP" r"EAKER\s+\S[^\n\"]*?\s+\d+\s+-?\d+\.\d+\s+\d+\.\d+\s+<NA>\s+<NA>\s+\S+\s+<NA>")
RTTM_HOME = ("tests", "fixtures", "steps")
CODE_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini"}
SYNTHETIC_HOMES = (("tests", "fixtures"), ("tests", "fakes"))
IGNORED_BYPRODUCTS = re.compile(r"(?:^|/)(?:__pycache__/|\.pytest_cache/|[^/]*\.egg-info/)|\.pyc$|(?:^|/)\.DS_Store$")


# --------------------------------------------------------------------------- who starts processes

def _source_files(pkg: Path = PKG) -> list[Path]:
    return sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path, src: Path = SRC) -> str:
    parts = list(path.relative_to(src).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _banned_module(name: str) -> bool:
    parts = name.lower().split(".")
    return any(".".join(parts[:i]) in BANNED_MODULES for i in range(1, len(parts) + 1))


def _banned_attr(module: str, attr: str) -> bool:
    if attr in BANNED_ATTRS.get(module, ()):
        return True
    if module == "platform":
        return attr not in PLATFORM_ALLOWED
    return module == "os" and attr.startswith(BANNED_OS_PREFIXES)


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def violations(path: Path, pkg: Path = PKG) -> list[str]:
    """Every process start in one file, as "file:line: what"; none for the package's own proc.py."""
    try:
        if path.resolve() == (pkg / PROCESS_MODULE).resolve():
            return []
    except OSError:
        pass
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.name
    found: list[str] = []
    aliases: dict[str, str] = {}

    def report(node: ast.AST, what: str) -> None:
        found.append(f"{rel}:{getattr(node, 'lineno', 0)}: {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _banned_module(a.name):
                    report(node, f"import {a.name}")
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _banned_module(node.module):
                report(node, f"from {node.module} import ...")
            for a in node.names:
                if a.name == "*" and node.module in NO_STAR:
                    report(node, f"from {node.module} import *")
                if _banned_attr(node.module, a.name) or _banned_module(f"{node.module}.{a.name}"):
                    report(node, f"from {node.module} import {a.name}")
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _dotted(node)
            module = ".".join((aliases[chain[0]].split(".") + chain[1:])[:-1]) if chain and chain[0] in aliases else ""
            if module and _banned_attr(module, node.attr):
                report(node, ".".join(chain))
            elif node.attr in BANNED_ANY_ATTR and not (module == "platform" and node.attr in PLATFORM_ALLOWED):
                report(node, f"a call named {node.attr}")
        elif isinstance(node, ast.Call):
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            args = node.args
            name_arg = args[0] if args else next((k.value for k in node.keywords if k.arg == "name"), None)
            if fname in ("__import__", "import_module") and name_arg is not None:
                if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
                    report(node, f"{fname}() of a name that isn't written out")
                elif _banned_module(name_arg.value):
                    report(node, f"{fname}({name_arg.value!r})")
            if fname == "getattr" and len(args) >= 2:
                chain = _dotted(args[0])
                module = ".".join(aliases[chain[0]].split(".") + chain[1:]) if chain and chain[0] in aliases else ""
                if not (isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)):
                    if module in NO_DYNAMIC_GETATTR:
                        report(node, f"getattr({'.'.join(chain)}, <a name that isn't written out>)")
                elif module and _banned_attr(module, args[1].value):
                    report(node, f"getattr({'.'.join(chain)}, {args[1].value!r})")
    return found


def python_children_without_b(path: Path) -> list[str]:
    """Every argument list (a list or tuple literal) that passes ``-I`` without ``-B``, as "file:line"."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.name
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            flags = {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if "-I" in flags and "-B" not in flags:
                found.append(f"{rel}:{node.lineno}: -I without -B")
    return found


def test_the_package_and_its_process_module_are_there():
    assert (PKG / "__init__.py").is_file() and (PKG / PROCESS_MODULE).is_file()


def test_only_proc_starts_processes():
    found = [v for path in _source_files() for v in violations(path)]
    assert not found, "only src/audiointake/proc.py may start processes:\n" + "\n".join(found)


def test_proc_is_where_subprocess_is():
    src = (PKG / PROCESS_MODULE).read_text(encoding="utf-8")
    assert "import subprocess" in src


PLANTED_BAD = {
    "import subprocess": "import subprocess",
    "import subprocess as sp\nsp.run(['x'])": "import subprocess",
    "from subprocess import run": "from subprocess import",
    "import multiprocessing": "import multiprocessing",
    "from concurrent.futures import ProcessPoolExecutor": "import ProcessPoolExecutor",
    "import concurrent.futures as cf\ncf.ProcessPoolExecutor()": "cf.ProcessPoolExecutor",
    "import os\nos.system('true')": "os.system",
    "import os\nos.popen('true')": "os.popen",
    "import os as o\no.execv('x', [])": "o.execv",
    "from os import posix_spawn": "from os import posix_spawn",
    "import os\nos.spawnv(0, 'x', [])": "os.spawnv",
    "import os\ngetattr(os, 'system')('x')": "getattr(os, 'system')",
    "import asyncio\nasyncio.create_subprocess_exec('x')": "asyncio.create_subprocess_exec",
    "import platform\nplatform.platform()": "platform.platform",
    "from platform import processor": "from platform import processor",
    "import importlib\nimportlib.import_module('subprocess')": "import_module('subprocess')",
    "import pty": "import pty",
    "from asyncio.subprocess import create_subprocess_exec": "from asyncio.subprocess import",
    "from asyncio import subprocess as sp\nsp.create_subprocess_exec('x')": "from asyncio import subprocess",
    "import asyncio\nasyncio.get_event_loop().subprocess_exec(None, 'x')": "a call named subprocess_exec",
    "def f(loop):\n    loop.subprocess_shell(None, 'x')": "a call named subprocess_shell",
    "from os import *\nsystem('x')": "from os import *",
    "import webbrowser\nwebbrowser.open('x')": "import webbrowser",
    "import ctypes\nlibc = ctypes.CDLL(None)\nlibc.system(b'x')": "a call named system",
    "__import__('sub' + 'process')": "__import__() of a name that isn't written out",
    "import importlib\nn = 'x'\nimportlib.import_module(n)": "import_module() of a name that isn't written out",
    "import os\nn = 'sys' + 'tem'\ngetattr(os, n)('x')": "getattr(os, <a name that isn't written out>)",
}
PLANTED_OK = [
    "import platform\nplatform.mac_ver()\nplatform.machine()\nplatform.system()",
    "import os\nos.kill(1, 0)\nos.killpg(1, 0)\nos.getppid()",
    "from concurrent.futures import ThreadPoolExecutor",
    "import threading\nthreading.Thread(target=print)",
    "import os\nflags = os.O_WRONLY | getattr(os, 'O_CLOEXEC', 0)",
    "import importlib\nimportlib.import_module('json')",
    "def f(obj, name):\n    return getattr(obj, name)",
    "import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\nfn = libc.sysctlbyname",
]


@pytest.mark.parametrize("code", list(PLANTED_BAD), ids=range(len(PLANTED_BAD)))
def test_the_static_ban_catches_planted_process_starts(tmp_path: Path, code: str):
    path = tmp_path / "planted.py"
    path.write_text(code + "\n", encoding="utf-8")
    found = violations(path)
    assert found and PLANTED_BAD[code] in "\n".join(found), found


@pytest.mark.parametrize("code", PLANTED_OK, ids=range(len(PLANTED_OK)))
def test_the_static_ban_allows_signals_threads_and_safe_platform_helpers(tmp_path: Path, code: str):
    path = tmp_path / "planted.py"
    path.write_text(code + "\n", encoding="utf-8")
    assert violations(path) == []


def test_a_proc_py_outside_the_package_is_not_exempt(tmp_path: Path):
    path = tmp_path / PROCESS_MODULE
    path.write_text("import subprocess\n", encoding="utf-8")
    assert violations(path) != []


def test_every_python_child_passes_minus_b():
    found = [v for path in _source_files() for v in python_children_without_b(path)]
    assert not found, ("a Python child started with -I ignores PYTHONDONTWRITEBYTECODE; add -B beside it:\n"
                       + "\n".join(found))


def test_the_minus_b_rule(tmp_path: Path):
    path = tmp_path / "m.py"
    path.write_text("a = [py, '-I', '-m', 'x']\nb = (py, '-I', '-B', '-m', 'x')\nc = [py, '-m', 'x']\n"
                    "d = ['-IB']\n", encoding="utf-8")
    assert python_children_without_b(path) == ["m.py:1: -I without -B"]


# --------------------------------------------------------------------------- imports

_SWEEP = r"""
import importlib, importlib.abc, json, os, re, sys
src, names, blocked = sys.argv[1], json.loads(sys.argv[2]), set(json.loads(sys.argv[3]))
sys.path.insert(0, src)
attempts, current = [], [None]
NET = {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
       "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.getnameinfo",
       "urllib.Request", "http.client.connect"}
def hook(event, args):
    if event in NET:
        attempts.append({"module": current[0], "event": event, "args": repr(args)[:160]})
        raise OSError("network blocked while importing")
sys.addaudithook(hook)
def importer():
    f = sys._getframe(2)
    while f is not None:
        name = f.f_globals.get("__name__") or ""
        if not (name.startswith("importlib") or name.startswith("_frozen_importlib")):
            return name
        f = f.f_back
    return ""
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in blocked:
            return None
        who = importer()
        if not (who == "audiointake" or who.startswith("audiointake.")):
            return None
        raise ModuleNotFoundError("No module named " + repr(fullname) + " (blocked: imported at import time by " + who + ")", name=fullname)
sys.meta_path.insert(0, Block())
results = {}
for name in names:
    current[0] = name
    try:
        importlib.import_module(name)
        results[name] = "ok"
    except ModuleNotFoundError as exc:
        results[name] = "missing module: " + str(exc.name) + " | " + str(exc)[:200]
    except ImportError as exc:
        m = re.match(r"cannot import name '([A-Za-z_][A-Za-z0-9_]*)'", str(exc))
        if m and exc.name:
            results[name] = "missing module: " + exc.name + "." + m.group(1) + " | " + str(exc)[:200]
        else:
            results[name] = "error: ImportError: " + str(exc)[:300]
    except BaseException as exc:
        results[name] = "error: " + type(exc).__name__ + ": " + str(exc)[:300]
print("@@SWEEP@@" + json.dumps({"results": results, "attempts": attempts}))
"""


def _sweep(names: list[str], blocked: set[str], src: Path = SRC) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", _SWEEP, str(src), json.dumps(names),
                           json.dumps(sorted(blocked))],
                          capture_output=True, text=True, encoding="utf-8", timeout=600, env=env)
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@SWEEP@@")]
    assert marker, f"the import sweep didn't finish (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
    return json.loads(marker[-1][len("@@SWEEP@@"):])


def _on_disk(name: str, src: Path = SRC) -> bool:
    base = src.joinpath(*name.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def import_problems(results: dict[str, str], src: Path = SRC) -> dict[str, str]:
    """The modules whose import failed for a reason this test doesn't excuse: a missing sibling is excused only
    while it's planned and not on disk yet."""
    bad: dict[str, str] = {}
    for name, result in results.items():
        if result == "ok":
            continue
        if result.startswith("missing module: "):
            missing = result[len("missing module: "):].split(" | ", 1)[0]
            if missing in PLANNED and not _on_disk(missing, src):
                continue
        bad[name] = result
    return bad


def test_every_module_imports_without_the_gpu_and_touches_no_network():
    names = [_module_name(p) for p in _source_files()]
    out = _sweep(names, GPU_ONLY | OPTIONAL)
    assert not out["attempts"], "network at import time:\n" + "\n".join(
        f"{a['module']}: {a['event']} {a['args']}" for a in out["attempts"])
    bad = import_problems(out["results"])
    assert not bad, ("modules that fail to import (MLX or an extra imported at module level, a missing dependency, "
                     "or a sibling that isn't planned):\n" + "\n".join(f"{n}: {r}" for n, r in bad.items()))


def test_the_sweep_blocks_a_planted_module_level_gpu_import(tmp_path: Path):
    pkg = tmp_path / "src" / "audiointake"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "eager.py").write_text("import mlx.core\n", encoding="utf-8")
    (pkg / "lazy.py").write_text("def f():\n    import mlx.core\n", encoding="utf-8")
    (pkg / "sibling.py").write_text("from . import pipeline\n", encoding="utf-8")
    (pkg / "unplanned.py").write_text("from . import nowhere\n", encoding="utf-8")
    (pkg / "network.py").write_text("import socket\ntry:\n    socket.getaddrinfo('example.com', 443)\n"
                                    "except OSError:\n    pass\n", encoding="utf-8")
    names = ["audiointake.eager", "audiointake.lazy", "audiointake.sibling", "audiointake.unplanned",
             "audiointake.network"]
    out = _sweep(names, GPU_ONLY, src=tmp_path / "src")
    bad = import_problems(out["results"], src=tmp_path / "src")
    assert set(bad) == {"audiointake.eager", "audiointake.unplanned"}, out
    assert "blocked: imported at import time by audiointake.eager" in bad["audiointake.eager"]
    assert [a["module"] for a in out["attempts"]] == ["audiointake.network"]


def _mlx_works() -> bool:
    probe = "import mlx.core as mx\nmx.eval(mx.ones(4).sum())\n"
    try:
        r = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def deferred_imports(path: Path) -> set[str]:
    """The absolute module names a file imports inside its functions."""
    names: set[str] = set()
    for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Import):
                    names |= {a.name for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module)
    return names


def test_deferred_imports_are_read_from_function_bodies(tmp_path: Path):
    path = tmp_path / "m.py"
    path.write_text("import json\ndef f():\n    import mlx.core as mx\n    from mlx_whisper import audio\n"
                    "    from . import x\n", encoding="utf-8")
    assert deferred_imports(path) == {"mlx.core", "mlx_whisper"}


@pytest.mark.parametrize("module", MLX_CHECKED)
def test_gpu_modules_import_with_their_deferred_imports_where_mlx_runs(module: str):
    path = SRC.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():
        pytest.skip(f"{module} isn't written yet")
    if not _mlx_works():
        pytest.skip("MLX can't run here (no Metal device), so its deferred imports can't be checked")
    names = [module] + sorted(deferred_imports(path))
    out = _sweep(names, set())
    bad = {n: r for n, r in out["results"].items() if r != "ok"}
    assert not bad and not out["attempts"], (bad, out["attempts"])


def test_import_audiointake_turns_telemetry_off():
    env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY")}
    code = ("import os, sys\nsys.path.insert(0, sys.argv[1])\nimport audiointake\n"
            "print(os.environ.get('ORT_DISABLE_TELEMETRY'), os.environ.get('HF_HUB_DISABLE_TELEMETRY'))\n")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(SRC)], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1", "1"]


# --------------------------------------------------------------------------- the files git would commit

def repo_files(root: Path = ROOT) -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, check=True, timeout=60).stdout
        names = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n]
        return sorted({root / n for n in names if (root / n).is_file() and not (root / n).is_symlink()})
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        skip = {".git", "__pycache__", ".pytest_cache", ".venv", "dist", "build"}
        return sorted(p for p in root.rglob("*") if p.is_file() and not skip & set(p.relative_to(root).parts))


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def path_problems(files: list[Path], root: Path) -> list[str]:
    problems = []
    for p in files:
        text = p.read_bytes().decode("utf-8", "replace")
        for n, line in enumerate(text.splitlines(), 1):
            if line.replace(HOME_PLACEHOLDER, "").count(HOME_MAC):
                problems.append(f"{_rel(p, root)}:{n}: a home folder other than {HOME_PLACEHOLDER}")
            if HOME_LINUX in line:
                problems.append(f"{_rel(p, root)}:{n}: a Linux home folder")
            if any(form in line for form in SCRATCH_FORMS):
                problems.append(f"{_rel(p, root)}:{n}: a scratch-folder path")
    return problems


def _mpeg_frame(head: bytes) -> bool:
    """An MPEG audio frame (MP3 without an ID3 tag) or an ADTS frame (raw AAC): 11 sync bits, a layer that isn't
    the reserved 00, and not UTF-16's byte-order mark (FF FE), which a text file may start with."""
    return (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] != 0xFE
            and (head[1] & 0x06 != 0 or head[1] & 0xF6 == 0xF0))


def audio_problems(files: list[Path], root: Path) -> list[str]:
    problems = []
    for p in files:
        with open(p, "rb") as f:
            head = f.read(16)
        for offset, magic in AUDIO_MAGIC:
            if head[offset:offset + len(magic)] == magic:
                shown = magic.decode("latin-1")
                problems.append(f"{_rel(p, root)}: starts like audio ({shown if shown.isprintable() else magic.hex()})")
                break
        else:
            if _mpeg_frame(head):
                problems.append(f"{_rel(p, root)}: starts like audio (an MPEG or ADTS frame)")
    return problems


def _objects(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _objects(v)
    elif isinstance(x, list):
        for v in x:
            yield from _objects(v)


VOICEPRINT_TEXT = (re.compile(r'"vectors"\s*:'), re.compile(r'"model_sha256"\s*:'))
ROSTER_TEXT = re.compile(r'"enrolled"\s*:')


def _text_of(p: Path) -> str | None:
    """A file's text without a byte-order mark, or None for a binary file."""
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", "replace").lstrip("﻿")


def _documents(text: str) -> list:
    try:
        return [json.loads(text)]
    except ValueError:
        pass
    docs = []
    for line in text.splitlines():
        if line.lstrip().startswith(("{", "[")):
            try:
                docs.append(json.loads(line))
            except ValueError:
                pass
    return docs


def _under(rel: Path, homes: tuple[tuple[str, ...], ...]) -> bool:
    return any(rel.parts[:len(h)] == h for h in homes)


def _shape(obj: dict, rel: Path) -> str | None:
    if "vectors" in obj and "model_sha256" in obj:
        return "a voiceprint (vectors, model_sha256)"
    if "enrolled" in obj:
        return "a speaker-ID file (enrolled)"
    if not _under(rel, SYNTHETIC_HOMES):
        if obj.get("format") == "audiointake-transcript":
            return "an audiointake transcript (format audiointake-transcript)"
        if "record_version" in obj and str(obj.get("id", "")).startswith("r-"):
            return "an audiointake record (record_version)"
    if not _under(rel, (RTTM_HOME,)):
        for key in ("sentences", "segments"):
            items = obj.get(key)
            if isinstance(items, list) and items and all(isinstance(x, dict) and "text" in x for x in items):
                return f"a recogniser's output ({key} of text)"
    return None


def json_problems(files: list[Path], root: Path) -> list[str]:
    """Voiceprints, speaker rosters, transcripts, records and recogniser output, whatever the file is called."""
    problems = []
    for p in files:
        if p.suffix.lower() in CODE_SUFFIXES:
            continue
        text = _text_of(p)
        if text is None:
            continue
        rel = p.relative_to(root)
        found = next((s for doc in _documents(text) for obj in _objects(doc) if (s := _shape(obj, rel))), None)
        if found is None and all(rx.search(text) for rx in VOICEPRINT_TEXT):
            found = "a voiceprint (vectors, model_sha256)"
        if found is None and ROSTER_TEXT.search(text):
            found = "a speaker-ID file (enrolled)"
        if found:
            problems.append(f"{_rel(p, root)}: shaped like {found}")
    return problems


def rttm_problems(files: list[Path], root: Path) -> list[str]:
    problems = []
    for p in files:
        rel = p.relative_to(root)
        if rel.parts[:3] == RTTM_HOME and p.name.endswith(".rttm.txt"):
            continue
        text = p.read_bytes().decode("utf-8", "replace")
        data = p.suffix.lower() not in CODE_SUFFIXES
        for n, line in enumerate(text.splitlines(), 1):
            if RTTM_LINE.match(line) or (data and RTTM_ANYWHERE.search(line.replace("\\n", "\n"))):
                problems.append(f"{_rel(p, root)}:{n}: an RTTM line outside tests/fixtures/steps/*.rttm.txt")
                break
    return problems


@pytest.fixture(scope="module")
def files() -> list[Path]:
    found = repo_files()
    assert any(p.name == "pyproject.toml" for p in found), "the file listing found no pyproject.toml"
    return found


def test_no_private_paths(files: list[Path]):
    assert path_problems(files, ROOT) == []


def test_no_audio_whatever_the_extension(files: list[Path]):
    assert audio_problems(files, ROOT) == []


def test_no_voiceprint_or_speaker_file(files: list[Path]):
    assert json_problems(files, ROOT) == []


def test_no_rttm_line_outside_the_step_goldens(files: list[Path]):
    assert rttm_problems(files, ROOT) == []


def test_nothing_in_src_or_tests_is_silently_ignored():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
                              "--", "src", "tests"], capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"not a git checkout ({exc})")
    ignored = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n and not IGNORED_BYPRODUCTS.search(n)]
    assert ignored == [], ".gitignore hides these from every commit; rename them: " + ", ".join(ignored)


# --- the scanners catch what they're for

def test_the_path_scan(tmp_path: Path):
    good, bad = tmp_path / "good.md", tmp_path / "bad.py"
    good.write_text(f"see {HOME_PLACEHOLDER}Music/x.m4a and ~/.audiointake-dev/venv\n", encoding="utf-8")
    bad.write_text(f"a = '{HOME_MAC}someone/x'\nb = '{HOME_LINUX}runner/x'\nc = '~/.{'scratch'}/run'\n"
                   f"d = '{HOME_PLACEHOLDER}ok {HOME_MAC}someone/y'\n", encoding="utf-8")
    found = path_problems([good, bad], tmp_path)
    assert [f.split(": ", 1)[0] for f in found] == ["bad.py:1", "bad.py:2", "bad.py:3", "bad.py:4"], found


AUDIO_HEADS = {
    "wav": b"RIFF\x24\x00\x00\x00WAVEfmt ", "m4a": b"\x00\x00\x00\x20ftypM4A ", "ogg": b"OggS\x00\x02",
    "flac": b"fLaC\x00\x00\x00\x22", "id3": b"ID3\x04\x00", "caf": b"caff\x00\x01", "aiff": b"FORM\x00\x00\x00\x04AIFF",
    "webm": b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81", "amr": b"#!AMR\n", "wma": bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c"),
    "mp3-no-tag": b"\xff\xfb\x90\x64", "mp3-mpeg2": b"\xff\xf3\x48\xc4", "aac-adts": b"\xff\xf1\x50\x80",
    "aac-adts-mpeg2": b"\xff\xf9\x50\x80",
}


@pytest.mark.parametrize("head", list(AUDIO_HEADS.values()), ids=list(AUDIO_HEADS))
def test_the_audio_scan_reads_the_first_bytes_whatever_the_name(tmp_path: Path, head: bytes):
    p = tmp_path / "notes.dat"
    p.write_bytes(head + b"\x00" * 64)
    assert audio_problems([p], tmp_path) != []


@pytest.mark.parametrize("head", [b'{"riff": "OggS"}', b"\xff\xfeU\x00T\x00F\x00", b"\xef\xbb\xbf# notes\n",
                                  b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff\xe0\x00\x10JFIF"],
                         ids=["json", "utf-16-text", "utf-8-bom", "png", "jpeg"])
def test_the_audio_scan_lets_other_files_through(tmp_path: Path, head: bytes):
    ok = tmp_path / "ok.json"
    ok.write_bytes(head + b"\x00" * 16)
    assert audio_problems([ok], tmp_path) == []


VOICEPRINT = {"name": "Jane Doe", "model_sha256": "0" * 64, "vectors": [{"vec": [0.1, 0.2]}], "updated": "2026-01-02"}


def test_the_json_scan(tmp_path: Path):
    vp, sp, ok = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    vp.write_text(json.dumps(VOICEPRINT), encoding="utf-8")
    sp.write_text(json.dumps({"model": "x", "labels": {}, "enrolled": ["Jane Doe"]}), encoding="utf-8")
    ok.write_text(json.dumps({"labels": {"speaker_0": {"name": None}}, "vectors_note": 1}), encoding="utf-8")
    assert json_problems([vp, sp, ok], tmp_path) == [
        "a.json: shaped like a voiceprint (vectors, model_sha256)", "b.json: shaped like a speaker-ID file (enrolled)"]


def test_the_json_scan_ignores_the_name_and_reads_what_does_not_parse(tmp_path: Path):
    pretty = json.dumps(VOICEPRINT, indent=1)
    plants = {"vp.txt": pretty, "vp_noext": pretty, "bom.json": "﻿" + pretty,
              "comma.json": pretty[:-2] + ",\n}", "roster.notes": '{"enrolled": ["Jane Doe"],}',
              "lines.log": "start\n" + json.dumps(VOICEPRINT) + "\nend\n"}
    for name, text in plants.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    found = json_problems(sorted(tmp_path / n for n in plants), tmp_path)
    assert sorted(f.split(":", 1)[0] for f in found) == sorted(plants), found
    code = tmp_path / "helper.py"
    code.write_text(f"VP = {json.dumps(VOICEPRINT)}\nR = {{\"enrolled\": []}}\n", encoding="utf-8")
    assert json_problems([code], tmp_path) == []                     # code may build a voiceprint to test with


def test_the_json_scan_finds_transcripts_records_and_recogniser_output_outside_the_fixtures(tmp_path: Path):
    docs = {
        "t.dat": {"format": "audiointake-transcript", "version": 1, "turns": []},
        "r.txt": {"record_version": 1, "id": "r-20260101-00000000", "speakers": []},
        "asr.out": {"text": "x", "sentences": [{"text": "x", "start": 0.0, "end": 1.0, "tokens": []}]},
        "w.out": {"text": "x", "segments": [{"text": "x", "start": 0.0, "end": 1.0}]},
    }
    for home in ((), ("tests", "fixtures"), ("tests", "fakes"), RTTM_HOME):
        folder = tmp_path.joinpath(*home)
        folder.mkdir(parents=True, exist_ok=True)
        for name, d in docs.items():
            (folder / name).write_text(json.dumps(d), encoding="utf-8")
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    found = {f.split(": ", 1)[0] for f in json_problems(files, tmp_path)}
    assert found == {"t.dat", "r.txt", "asr.out", "w.out", "tests/fixtures/asr.out", "tests/fixtures/w.out",
                     "tests/fakes/asr.out", "tests/fakes/w.out"}, found
    other = tmp_path / "ok.json"
    other.write_text(json.dumps({"sentences": ["not a recogniser's"], "format": "md", "record_version": 1}), encoding="utf-8")
    assert json_problems([other], tmp_path) == []


def test_the_rttm_scan(tmp_path: Path):
    line = "SP" "EAKER meeting.16k 1 0.320 4.100 <NA> <NA> speaker_0 <NA> <NA>\n"
    spaced = "SP" "EAKER Team sync.16k 1 0.320 4.100 <NA> <NA> speaker_0 <NA> <NA>\n"
    home = tmp_path.joinpath(*RTTM_HOME)
    home.mkdir(parents=True)
    golden, stray, other, named = home / "x.rttm.txt", tmp_path / "x.rttm.txt", home / "x.txt", tmp_path / "n.md"
    for p in (golden, stray, other):
        p.write_text(line, encoding="utf-8")
    named.write_text("notes\n" + spaced, encoding="utf-8")
    inline = tmp_path / "doc.md"
    inline.write_text('{"rttm": "' + spaced.strip() + '\\n' + line.strip() + '"}\n', encoding="utf-8")
    code = tmp_path / "t.py"
    code.write_text(f"X = {spaced.strip()!r}\n", encoding="utf-8")                      # a parser's example: fine
    block = tmp_path / "b.py"
    block.write_text(f'def f():\n    return """\n    {spaced}    {line}    """\n', encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text(f"turns: {spaced}", encoding="utf-8")
    synthetic = tmp_path / "s.py"
    synthetic.write_text('X = "SP' 'EAKER c.16k 1 0 1 <NA> <NA> speaker_0 <NA> <NA>\\nSP' 'EAKER c.16k 1 1 1 <NA> <NA> '
                         'speaker_1 <NA> <NA>\\n"\nY = "SP' 'EAKER <file id> 1 <start> <dur> <NA> <NA> speaker_K"\n',
                         encoding="utf-8")
    found = rttm_problems([golden, stray, other, named, inline, code, block, log, synthetic], tmp_path)
    assert [f.split(": ", 1)[0] for f in found] == ["x.rttm.txt:1", "tests/fixtures/steps/x.txt:1", "n.md:2",
                                                    "doc.md:1", "b.py:3", "run.log:1"], found


def test_the_ignored_filter_keeps_content_and_drops_byproducts():
    keep = ["tests/fakes/voices/a.json", "tests/fixtures/steps/x.rttm", "src/audiointake/build/x.py"]
    drop = ["src/audiointake/__pycache__/x.cpython-314.pyc", "tests/.DS_Store", "src/audiointake.egg-info/PKG-INFO"]
    assert [n for n in keep + drop if not IGNORED_BYPRODUCTS.search(n)] == keep
