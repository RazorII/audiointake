"""Every write into a user's folder goes through one function, called from a few named places.

A static scan of ``src/audiointake`` with ``ast``:

* **``fsutil.atomic_write``** is referred to only inside the owners in ``ATOMIC_OWNERS``: the one writer of a
  transcript's outputs (``pipeline.commit_outputs``, which ``apply`` and ``relabel`` go through too), the notes
  writer, the voiceprint writers and the record writer (the data folder). Any other reference fails: a call, a
  bare reference, an import of the name (aliased or not), the string ``"atomic_write"``, and any ``getattr`` on
  ``fsutil`` whose name isn't a constant.
* **Write and delete primitives** (``open`` for writing, ``write_text``/``write_bytes``/``touch``/``mkdir``/
  ``unlink``/``rename``, ``os.replace``/``rename``/``remove``/``unlink``/``link``/``mkdir``/``makedirs``/``write``,
  ``os.open`` with any flag that can write, ``shutil`` copies, moves and ``rmtree``, ``tempfile``'s files) appear
  only where ``WRITERS`` allows them, each with the reason: inside audiointake's own folders (data, cache, logs,
  the run's work folder) and never in a user's folder.

So a new writer has to be added here on purpose, with its reason. The planted-code tests at the end prove the
scan catches each form.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "audiointake"
ATOMIC = "atomic_write"
ANY = "*"

# module (relative to src/audiointake; a trailing "/" is a package) -> functions that may refer to atomic_write
ATOMIC_OWNERS: dict[str, frozenset[str]] = {
    "pipeline.py": frozenset({"commit_outputs",             # the transcript's outputs (transcribe, apply, relabel)
                              "_put_back"}),                # and putting back a file a failed commit had replaced
    "notes.py": frozenset({"into"}),                        # the notes file Claude wrote the head of
    "voices.py": frozenset({"rename", "restore"}),          # a voiceprint file, renamed or put back after a refusal
    "record.py": frozenset({"save", "speed_add"}),          # <data>/records/<id>.json and <data>/speed.json
    "steps/speaker_id.py": frozenset({ANY}),                # the engine's voiceprint writer (enroll)
    "setup/": frozenset({ANY}),                             # setup's stamps and ledgers in <data>
}
ATOMIC_HOME = "fsutil.py"

# module -> functions that may use write primitives (ANY: the whole module), and where they write
WRITERS: dict[str, tuple[frozenset[str], str]] = {
    "fsutil.py": (frozenset({ANY}), "atomic_write itself, its partial file, private folders, the write probe"),
    "lock.py": (frozenset({ANY}), "<data>/run.lock"),
    "proc.py": (frozenset({ANY}), "a build child's output log in setup's build folder"),
    "record.py": (frozenset({"save", "prune", "__call__", "remove", "speed_add"}),
                  "records and speed.json in <data>, run logs in ~/Library/Logs/audiointake"),
    "pipeline.py": (frozenset({"abort", "cleanup", "_work", "_write_work", "_remove_work", "_unlink_work",
                               "commit_outputs", "_put_back"}),
                    "the run's work folder; commit_outputs and _put_back: undoing a commit that failed half-way "
                    "(the files it created go, the files it replaced get their old bytes back)"),
    "probe.py": (frozenset({"_with_ffmpeg"}), "ffmpeg's stderr file in the run's work folder"),
    "voices.py": (frozenset({"remove", "rename", "restore"}), "the voiceprint folder"),
    "steps/": (frozenset({ANY}), "the engine: its outdir (always in the run's work folder) and voiceprints"),
    "setup/": (frozenset({ANY}), "setup: <data>, <cache>, the Hugging Face cache, build folders"),
    "doctor.py": (frozenset({ANY}), "doctor's self-test, through setup's own folders"),
}

WRITE_ATTRS = frozenset({"write_bytes", "write_text", "touch", "unlink", "rmdir", "rename", "replace", "mkdir",
                         "symlink_to", "hardlink_to", "chmod"})
MODULE_CALLS = {
    "os": frozenset({"replace", "rename", "renames", "remove", "unlink", "rmdir", "removedirs", "link", "symlink",
                     "truncate", "ftruncate", "mkdir", "makedirs", "write", "pwrite", "writev", "mkfifo", "chmod"}),
    "shutil": frozenset({"move", "copy", "copy2", "copyfile", "copytree", "rmtree", "copyfileobj"}),
    "tempfile": frozenset({"mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory", "SpooledTemporaryFile"}),
}
READ_ONLY_FLAGS = frozenset({"O_RDONLY", "O_NONBLOCK", "O_NOCTTY", "O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY"})


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    owner: str
    what: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} in {self.owner or '<module>'}: {self.what}"


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(parts)]


def _mode_writes(call: ast.Call, pos: int) -> bool:
    mode = call.args[pos] if len(call.args) > pos else None
    for kw in call.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(c in mode.value for c in "wax+")
    return True


def _flags_read_only(node: ast.AST, os_names: set[str]) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flags_read_only(node.left, os_names) and _flags_read_only(node.right, os_names)
    d = _dotted(node)
    if d and len(d) == 2 and d[0] in os_names:
        return d[1] in READ_ONLY_FLAGS
    if (isinstance(node, ast.Call) and _dotted(node.func) == ["getattr"] and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant) and node.args[1].value in READ_ONLY_FLAGS):
        return True
    return False


class _Scan(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.owner: list[str] = []
        self.atomic: list[Finding] = []
        self.writes: list[Finding] = []
        self.modules: dict[str, str] = {"os": "os", "shutil": "shutil", "tempfile": "tempfile"}
        self.fsutil_names: set[str] = {"fsutil"}
        self.direct: dict[str, tuple[str, str]] = {}           # name -> (module, function) from "from X import f"
        self.pkgmods: set[str] = set()                          # audiointake's own modules ("from . import voices")

    def _here(self) -> str:
        return self.owner[-1] if self.owner else ""

    def _def(self, node: ast.AST, name: str) -> None:
        self.owner.append(name)
        self.generic_visit(node)
        self.owner.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._def(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _atomic(self, node: ast.AST, what: str) -> None:
        self.atomic.append(Finding(self.rel, getattr(node, "lineno", 0), self._here(), what))

    def _write(self, node: ast.AST, what: str) -> None:
        self.writes.append(Finding(self.rel, getattr(node, "lineno", 0), self._here(), what))

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            local = a.asname or a.name.split(".")[0]
            if a.name in self.modules.values() or a.name in ("os", "shutil", "tempfile"):
                self.modules[local] = a.name
            if a.name.endswith("fsutil"):
                self.fsutil_names.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            local = a.asname or a.name
            if a.name == ATOMIC:
                self._atomic(node, f"imports {ATOMIC} as {local}")
            if a.name == "fsutil":
                self.fsutil_names.add(local)
            if node.module in ("os", "shutil", "tempfile"):
                self.direct[local] = (node.module, a.name)
            if node.level and not node.module:
                self.pkgmods.add(local)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == ATOMIC:
            self._atomic(node, f"refers to {ATOMIC}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == ATOMIC:
            self._atomic(node, f"refers to .{ATOMIC}")
        if node.attr == "__dict__" and isinstance(node.value, ast.Name) and node.value.id in self.fsutil_names:
            self._atomic(node, "reads fsutil.__dict__")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == ATOMIC:
            self._atomic(node, f"the string {ATOMIC!r}")

    @staticmethod
    def _path_shaped(node: ast.Call) -> bool:
        """``Path.replace`` and ``Path.rename`` take exactly one target; ``str.replace(a, b)`` and
        ``datetime.replace(tzinfo=...)`` don't look like that."""
        if node.func.attr in ("replace", "rename"):                    # type: ignore[union-attr]
            return len(node.args) == 1 and not node.keywords
        return True

    def visit_Call(self, node: ast.Call) -> None:
        d = _dotted(node.func) or []
        if d == ["getattr"] and node.args and isinstance(node.args[0], ast.Name) \
                and node.args[0].id in self.fsutil_names and not (len(node.args) > 1 and isinstance(node.args[1], ast.Constant)):
            self._atomic(node, "getattr on fsutil with a computed name")
        if d in (["open"], ["io", "open"], ["codecs", "open"]) and _mode_writes(node, 1):
            self._write(node, "open() for writing")
        if d[:1] and d[0] in self.modules and len(d) == 2:
            mod = self.modules[d[0]]
            if mod == "os" and d[1] == "open":
                flags = node.args[1] if len(node.args) > 1 else None
                if flags is None or not _flags_read_only(flags, {k for k, v in self.modules.items() if v == "os"}):
                    self._write(node, "os.open with flags that can write")
            elif mod == "os" and d[1] == "fdopen" and _mode_writes(node, 1):
                self._write(node, "os.fdopen for writing")
            elif d[1] in MODULE_CALLS.get(mod, ()):
                self._write(node, f"{mod}.{d[1]}()")
        elif len(d) == 1 and d[0] in self.direct and self.direct[d[0]][1] in MODULE_CALLS.get(self.direct[d[0]][0], ()):
            self._write(node, f"{'.'.join(self.direct[d[0]])}()")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in WRITE_ATTRS \
                and not (d[:1] and (d[0] in self.modules or d[0] in self.pkgmods)) and self._path_shaped(node):
            self._write(node, f".{node.func.attr}()")
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open" and _mode_writes(node, 0):
            self._write(node, ".open() for writing")
        self.generic_visit(node)


def _key(rel: str, table: dict) -> str | None:
    if rel in table:
        return rel
    for k in table:
        if k.endswith("/") and rel.startswith(k):
            return k
    return None


def scan_source(rel: str, source: str) -> tuple[list[Finding], list[Finding]]:
    """(atomic_write references outside their owners, write primitives outside their writers) for one module."""
    s = _Scan(rel)
    s.visit(ast.parse(source))
    bad_atomic = []
    if rel != ATOMIC_HOME:
        k = _key(rel, ATOMIC_OWNERS)
        owners = ATOMIC_OWNERS.get(k, frozenset()) if k else frozenset()
        bad_atomic = [f for f in s.atomic if ANY not in owners and f.owner not in owners]
    k = _key(rel, WRITERS)
    allowed = WRITERS[k][0] if k else frozenset()
    bad_writes = [f for f in s.writes if ANY not in allowed and f.owner not in allowed]
    return bad_atomic, bad_writes


def _modules() -> list[tuple[str, Path]]:
    return sorted((p.relative_to(PKG).as_posix(), p) for p in PKG.rglob("*.py"))


def test_only_the_named_writers_use_atomic_write():
    found = []
    for rel, p in _modules():
        found += scan_source(rel, p.read_text(encoding="utf-8"))[0]
    assert [str(f) for f in found] == []


def test_no_module_writes_outside_its_own_folders():
    found = []
    for rel, p in _modules():
        found += scan_source(rel, p.read_text(encoding="utf-8"))[1]
    assert [str(f) for f in found] == []


def test_the_owners_really_write():
    """The table isn't stale: each U3 owner named above does refer to atomic_write."""
    for rel in ("pipeline.py", "notes.py", "voices.py", "record.py"):
        s = _Scan(rel)
        s.visit(ast.parse((PKG / rel).read_text(encoding="utf-8")))
        assert {f.owner for f in s.atomic} == set(ATOMIC_OWNERS[rel]), rel


def test_the_cli_writes_nothing_itself():
    for rel in ("cli.py", "outputs.py", "render.py"):
        s = _Scan(rel)
        s.visit(ast.parse((PKG / rel).read_text(encoding="utf-8")))
        assert s.atomic == [] and s.writes == [], rel


# --------------------------------------------------------------------------- the scan catches each form
PLANTED_ATOMIC = [
    "from . import fsutil\ndef f(p):\n    fsutil.atomic_write(p, b'', overwrite=True)\n",
    "from .fsutil import atomic_write\n",
    "from .fsutil import atomic_write as aw\n",
    "from . import fsutil\nw = fsutil.atomic_write\n",
    "from . import fsutil\ndef f(n):\n    return getattr(fsutil, n)\n",
    "from . import fsutil\nname = 'atomic_write'\n",
    "from . import fsutil as fu\ndef f():\n    return fu.__dict__\n",
]
PLANTED_WRITES = [
    "def f(p):\n    open(p, 'w').write('x')\n",
    "def f(p, m):\n    open(p, m)\n",
    "def f(p):\n    p.write_text('x')\n",
    "def f(p):\n    p.write_bytes(b'x')\n",
    "def f(p):\n    p.unlink()\n",
    "def f(p):\n    p.mkdir()\n",
    "import os\ndef f(a, b):\n    os.replace(a, b)\n",
    "import os as o\ndef f(a):\n    o.remove(a)\n",
    "import shutil\ndef f(a):\n    shutil.rmtree(a)\n",
    "from shutil import copyfile\ndef f(a, b):\n    copyfile(a, b)\n",
    "import os\ndef f(p):\n    os.open(p, os.O_WRONLY | os.O_CREAT)\n",
    "import os\ndef f(p, flags):\n    os.open(p, flags)\n",
    "import tempfile\ndef f():\n    tempfile.mkstemp()\n",
    "def f(p):\n    p.open('a')\n",
    "def f(p, q):\n    p.replace(q)\n",
    "def f(p, q):\n    p.rename(q)\n",
]


@pytest.mark.parametrize("src", PLANTED_ATOMIC)
def test_planted_atomic_references_are_caught(src):
    assert scan_source("cli.py", src)[0], src


@pytest.mark.parametrize("src", PLANTED_WRITES)
def test_planted_writes_are_caught(src):
    assert scan_source("cli.py", src)[1], src
    assert scan_source("steps/ensemble.py", src)[1] == []         # allowed there, with its reason


def test_reads_are_not_writes():
    src = ("import os\nfrom . import voices\ndef f(p, d):\n    open(p).read()\n    open(p, 'rb')\n    p.read_text()\n"
           "    os.close(os.open(p, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)))\n    'a'.replace('a', 'b')\n"
           "    d.replace(tzinfo=None)\n    voices.rename(p, 'a', 'b')\n")
    assert scan_source("cli.py", src) == ([], [])


def test_an_owner_outside_its_function_is_caught():
    src = "from . import fsutil\ndef helper(p):\n    fsutil.atomic_write(p, b'', overwrite=False)\n"
    assert scan_source("pipeline.py", src)[0]
    assert scan_source("pipeline.py", src.replace("helper", "commit_outputs"))[0] == []
