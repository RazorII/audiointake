"""The pack manifests (``setup/packs/<id>.json``, schema 1): a strict, hand-written loader.

A pack is data only: which models it needs, pinned to a 40-hex revision (Hugging Face) or a release tag
(GitHub), each file with its size, SHA-256 and the name the Hugging Face cache stores it under (``blob``: the LFS
object id, which is the SHA-256, for large files; the git blob id for small ones). Nothing in a manifest is ever
imported or executed, and no manifest text reaches the user except ids, tags and the licence and credit lines
``about`` shows.

The loader refuses, each with the JSON path of the problem: anything that isn't UTF-8 without a BOM, a file over
64 KiB, duplicate keys, NaN or infinities, unknown keys, a revision or git id that isn't 40-hex, a SHA-256 that
isn't 64-hex, a repository that isn't ``owner/name``, a file path that is absolute, has ``..`` or a backslash, or
ends in a code suffix (``.py``, ``.so``, ``.dylib``, ``.sh`` and the like), and a canary URL off its one host.
Models live on ``huggingface.co`` (``kind: hf``) or ``github.com`` release assets (``kind: github_release``); a
``converted`` model is made at setup from another model of the same pack (its ``convert.from``), which is then
only a source: never kept by a run, and downloaded into a temporary folder when the Hugging Face cache doesn't
already hold it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from ..contracts import PACKS, ROUTES

SCHEMA = 1
MAX_BYTES = 64 * 1024
KINDS = ("hf", "github_release", "converted")
SMOKE_KINDS = ("parakeet", "whisper", "whisper_lid", "typhoon", "nemotron", "campplus")
CONVERT_TOOLS = ("whisper_hf2mlx",)
CANARY_HOSTS = ("groups.inf.ed.ac.uk",)
HF_HOST = "huggingface.co"
GITHUB_HOST = "github.com"
# Where a download may be redirected to, by the host it starts on (".hf.co": any host under hf.co).
REDIRECT_HOSTS: dict[str, tuple[str, ...]] = {
    HF_HOST: (HF_HOST, ".hf.co"),
    GITHUB_HOST: (GITHUB_HOST, "objects.githubusercontent.com", "release-assets.githubusercontent.com"),
    "groups.inf.ed.ac.uk": ("groups.inf.ed.ac.uk",),
}
CODE_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".pyw", ".pyz", ".pth", ".so", ".dylib", ".dll", ".exe", ".com",
                 ".bat", ".cmd", ".ps1", ".sh", ".zsh", ".command", ".app", ".jar", ".msi", ".pkg")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_RELEASE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
_WORD = re.compile(r"^[^\s]{1,40}$")


class ManifestError(ValueError):
    """A manifest that the loader refuses; the message starts with the JSON path."""


@dataclass(frozen=True)
class FileSpec:
    path: str           # relative to the repository (hf) or the release (github_release)
    size: int
    sha256: str
    blob: str           # the Hugging Face cache's blob name; for a release asset, its SHA-256

    @property
    def lfs(self) -> bool:
        """Whether the file is an LFS file (its blob name is its SHA-256), so its git id isn't the content's."""
        return self.blob == self.sha256


@dataclass(frozen=True)
class OutputSpec:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Convert:
    from_key: str
    tool: str
    output: tuple[OutputSpec, ...]


@dataclass(frozen=True)
class Link:
    source: str         # a repository path (a file or a folder) in the model's snapshot
    name: str           # the link's name in the assembled folder


@dataclass(frozen=True)
class Layout:
    dir_name: str
    links: tuple[Link, ...]


@dataclass(frozen=True)
class Smoke:
    kind: str
    seconds: float


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str
    tag: str
    licence: str
    credit: str
    repo: str | None = None
    revision: str | None = None
    release_tag: str | None = None
    files: tuple[FileSpec, ...] = ()
    convert: Convert | None = None
    layout: Layout | None = None
    smoke: Smoke | None = None

    @property
    def host(self) -> str | None:
        return HF_HOST if self.kind == "hf" else GITHUB_HOST if self.kind == "github_release" else None

    def url(self, f: FileSpec) -> str:
        """The download URL of one of this model's files (pinned revision or release tag; never a branch)."""
        if self.kind == "hf":
            return f"https://{HF_HOST}/{self.repo}/resolve/{self.revision}/{f.path}"
        if self.kind == "github_release":
            return f"https://{GITHUB_HOST}/{self.repo}/releases/download/{self.release_tag}/{f.path}"
        raise ValueError(f"a {self.kind} model has no download URL")

    @property
    def fetch_bytes(self) -> int:
        return sum(f.size for f in self.files)


@dataclass(frozen=True)
class Canary:
    url: str
    start: int
    end: int            # exclusive
    sha256: str         # of the fetched bytes [start, end)
    rate: int
    channels: int
    bits: int
    file: str           # the kept clip's name under <data>/canary/
    must_find: tuple[str, ...]
    min_speakers: int

    @property
    def host(self) -> str:
        return self.url.split("/", 3)[2]


@dataclass(frozen=True)
class Pack:
    schema: int
    id: str
    version: int
    title: str
    language: str | None
    requires: tuple[str, ...]
    routes: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    canary: Canary | None

    def model(self, key: str) -> ModelSpec:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(key)

    @property
    def sources(self) -> frozenset[str]:
        """Keys of models that only feed a conversion (never kept for runs)."""
        return frozenset(m.convert.from_key for m in self.models if m.convert is not None)


# --------------------------------------------------------------------------- parsing
def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ManifestError(f"$: duplicate key {k!r}")
        out[k] = v
    return out


def _no_constant(name: str) -> Any:
    raise ManifestError(f"$: {name} isn't allowed")


def _obj(v: Any, path: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ManifestError(f"{path}: must be an object")
    unknown = set(v) - set(required) - set(optional)
    if unknown:
        raise ManifestError(f"{path}: unknown key(s) {sorted(unknown)}")
    missing = [k for k in required if k not in v]
    if missing:
        raise ManifestError(f"{path}: missing key(s) {missing}")
    return v


def _str(v: Any, path: str, pattern: re.Pattern[str] | None = None, *, max_len: int = 400) -> str:
    if not isinstance(v, str) or not v or len(v) > max_len:
        raise ManifestError(f"{path}: must be a non-empty string of at most {max_len} characters")
    if pattern is not None and not pattern.fullmatch(v):
        raise ManifestError(f"{path}: {v!r} isn't a valid value here")
    return v


def _int(v: Any, path: str, lo: int = 0, hi: int = 1 << 40) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ManifestError(f"{path}: must be an integer in [{lo}, {hi}]")
    return v


def _num(v: Any, path: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
        raise ManifestError(f"{path}: must be a number in [{lo}, {hi}]")
    return float(v)


def _list(v: Any, path: str, *, min_len: int = 0, max_len: int = 64) -> list[Any]:
    if not isinstance(v, list) or not min_len <= len(v) <= max_len:
        raise ManifestError(f"{path}: must be a list of {min_len} to {max_len} items")
    return v


def _relpath(v: Any, path: str) -> str:
    s = _str(v, path, max_len=300)
    parts = s.split("/")
    if s.startswith("/") or "\\" in s or "\x00" in s or any(p in ("", ".", "..") for p in parts):
        raise ManifestError(f"{path}: {s!r} must be a relative path without '.', '..', '\\\\' or empty parts")
    low = s.casefold()
    if low.endswith(CODE_SUFFIXES) or re.search(r"\.so(\.[0-9]+)+$", low):
        raise ManifestError(f"{path}: {s!r} has a code suffix; a pack holds data only")
    return s


def _file(v: Any, path: str) -> FileSpec:
    d = _obj(v, path, ("path", "size", "sha256", "blob"))
    sha = _str(d["sha256"], f"{path}.sha256", _HEX64)
    blob = _str(d["blob"], f"{path}.blob")
    if not (_HEX64.fullmatch(blob) or _HEX40.fullmatch(blob)):
        raise ManifestError(f"{path}.blob: must be a 64-hex LFS id or a 40-hex git id")
    if _HEX64.fullmatch(blob) and blob != sha:
        raise ManifestError(f"{path}.blob: a 64-hex blob is the file's SHA-256 and must equal it")
    return FileSpec(_relpath(d["path"], f"{path}.path"), _int(d["size"], f"{path}.size", 1), sha, blob)


def _model(v: Any, path: str) -> ModelSpec:
    d = _obj(v, path, ("key", "kind", "tag", "licence", "credit"),
             ("repo", "revision", "release_tag", "files", "convert", "layout", "smoke"))
    key = _str(d["key"], f"{path}.key", _KEY)
    kind = _str(d["kind"], f"{path}.kind")
    if kind not in KINDS:
        raise ManifestError(f"{path}.kind: must be one of {KINDS}")
    tag = _str(d["tag"], f"{path}.tag", _TAG)
    licence = _str(d["licence"], f"{path}.licence", max_len=60)
    credit = _str(d["credit"], f"{path}.credit", max_len=300)
    repo = revision = release_tag = None
    files: tuple[FileSpec, ...] = ()
    convert = layout = smoke = None
    if kind in ("hf", "github_release"):
        repo = _str(d.get("repo"), f"{path}.repo", _REPO)
        if kind == "hf":
            revision = _str(d.get("revision"), f"{path}.revision", _HEX40)
            if "release_tag" in d:
                raise ManifestError(f"{path}.release_tag: only for github_release")
        else:
            release_tag = _str(d.get("release_tag"), f"{path}.release_tag", _RELEASE_TAG)
            if "revision" in d:
                raise ManifestError(f"{path}.revision: only for hf")
        files = tuple(_file(f, f"{path}.files[{i}]")
                      for i, f in enumerate(_list(d.get("files"), f"{path}.files", min_len=1)))
        if kind == "github_release" and any(not f.lfs for f in files):
            raise ManifestError(f"{path}.files: a release asset's blob is its SHA-256")
        seen = set()
        for i, f in enumerate(files):
            if f.path.casefold() in seen:
                raise ManifestError(f"{path}.files[{i}].path: listed twice")
            seen.add(f.path.casefold())
        if "convert" in d:
            raise ManifestError(f"{path}.convert: only for converted models")
    else:
        for k in ("repo", "revision", "release_tag", "files"):
            if k in d:
                raise ManifestError(f"{path}.{k}: a converted model takes its files from convert.from")
        c = _obj(d.get("convert"), f"{path}.convert", ("from", "tool", "output"))
        tool = _str(c["tool"], f"{path}.convert.tool")
        if tool not in CONVERT_TOOLS:
            raise ManifestError(f"{path}.convert.tool: must be one of {CONVERT_TOOLS}")
        outs = []
        for i, o in enumerate(_list(c["output"], f"{path}.convert.output", min_len=1, max_len=8)):
            od = _obj(o, f"{path}.convert.output[{i}]", ("path", "size", "sha256"))
            outs.append(OutputSpec(_relpath(od["path"], f"{path}.convert.output[{i}].path"),
                                   _int(od["size"], f"{path}.convert.output[{i}].size", 1),
                                   _str(od["sha256"], f"{path}.convert.output[{i}].sha256", _HEX64)))
        convert = Convert(_str(c["from"], f"{path}.convert.from", _KEY), tool, tuple(outs))
    if "layout" in d:
        ld = _obj(d["layout"], f"{path}.layout", ("dir_name",), ("links",))
        links = tuple(Link(_relpath(_obj(x, f"{path}.layout.links[{i}]", ("from", "to"))["from"],
                                    f"{path}.layout.links[{i}].from"),
                           _str(x["to"], f"{path}.layout.links[{i}].to", _NAME))
                      for i, x in enumerate(_list(ld.get("links", []), f"{path}.layout.links", max_len=16)))
        if any("/" in ln.name for ln in links):
            raise ManifestError(f"{path}.layout.links: a link's name is one path component")
        layout = Layout(_str(ld["dir_name"], f"{path}.layout.dir_name", _NAME), links)
        if kind == "hf":
            known = [f.path for f in files]
            for i, ln in enumerate(links):
                if not any(p == ln.source or p.startswith(ln.source + "/") for p in known):
                    raise ManifestError(f"{path}.layout.links[{i}].from: names no file of this model")
    if "smoke" in d:
        sd = _obj(d["smoke"], f"{path}.smoke", ("kind", "seconds"))
        sk = _str(sd["kind"], f"{path}.smoke.kind")
        if sk not in SMOKE_KINDS:
            raise ManifestError(f"{path}.smoke.kind: must be one of {SMOKE_KINDS}")
        smoke = Smoke(sk, _num(sd["seconds"], f"{path}.smoke.seconds", 1, 30))
    return ModelSpec(key, kind, tag, licence, credit, repo, revision, release_tag, files, convert, layout, smoke)


def _canary(v: Any, path: str) -> Canary:
    d = _obj(v, path, ("url", "range", "sha256", "wav", "file", "must_find", "min_speakers"))
    url = _str(d["url"], f"{path}.url", max_len=300)
    m = re.fullmatch(r"https://([a-z0-9.-]+)/[A-Za-z0-9_./-]+", url)
    if not m or m.group(1) not in CANARY_HOSTS or ".." in url:
        raise ManifestError(f"{path}.url: must be an https URL on {CANARY_HOSTS}")
    rng = _list(d["range"], f"{path}.range", min_len=2, max_len=2)
    start, end = _int(rng[0], f"{path}.range[0]"), _int(rng[1], f"{path}.range[1]")
    if end <= start or end - start > 8 * 1024 * 1024:
        raise ManifestError(f"{path}.range: must be [start, end) with end > start and at most 8 MiB")
    w = _obj(d["wav"], f"{path}.wav", ("rate", "channels", "bits"))
    if (w["rate"], w["channels"], w["bits"]) != (16000, 1, 16):
        raise ManifestError(f"{path}.wav: must be 16000 Hz, 1 channel, 16 bits")
    words = tuple(_str(x, f"{path}.must_find[{i}]", _WORD)
                  for i, x in enumerate(_list(d["must_find"], f"{path}.must_find", min_len=1, max_len=8)))
    name = _str(d["file"], f"{path}.file", _NAME)
    if not name.endswith(".wav"):
        raise ManifestError(f"{path}.file: must end in .wav")
    return Canary(url, start, end, _str(d["sha256"], f"{path}.sha256", _HEX64), 16000, 1, 16, name, words,
                  _int(d["min_speakers"], f"{path}.min_speakers", 1, 8))


def parse(data: bytes) -> Pack:
    """A Pack from a manifest's bytes, or ManifestError naming the first problem."""
    if not isinstance(data, (bytes, bytearray)):
        raise ManifestError("$: a manifest is bytes")
    if len(data) > MAX_BYTES:
        raise ManifestError(f"$: larger than {MAX_BYTES} bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ManifestError("$: starts with a byte-order mark")
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"$: not UTF-8 ({exc.reason})") from None
    try:
        raw = json.loads(text, object_pairs_hook=_no_dupes, parse_constant=_no_constant)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"$: not JSON ({exc.msg} at line {exc.lineno})") from None
    d = _obj(raw, "$", ("schema", "id", "version", "title", "requires", "routes", "models"), ("language", "canary"))
    if d["schema"] != SCHEMA or isinstance(d["schema"], bool):
        raise ManifestError(f"$.schema: this core reads schema {SCHEMA}")
    pid = _str(d["id"], "$.id")
    if pid not in PACKS:
        raise ManifestError(f"$.id: must be one of {PACKS}")
    requires = tuple(_str(x, f"$.requires[{i}]") for i, x in enumerate(_list(d["requires"], "$.requires", max_len=4)))
    if any(r not in PACKS or r == pid for r in requires):
        raise ManifestError("$.requires: names an unknown pack or the pack itself")
    routes = tuple(_str(x, f"$.routes[{i}]") for i, x in enumerate(_list(d["routes"], "$.routes", max_len=4)))
    if any(r not in ROUTES for r in routes):
        raise ManifestError(f"$.routes: every route must be one of {ROUTES}")
    models = tuple(_model(m, f"$.models[{i}]") for i, m in enumerate(_list(d["models"], "$.models", min_len=1)))
    keys = [m.key for m in models]
    if len(set(keys)) != len(keys):
        raise ManifestError("$.models: a key is used twice")
    for i, m in enumerate(models):
        if m.convert is not None:
            src = next((x for x in models if x.key == m.convert.from_key), None)
            if src is None or src.kind != "hf" or src.convert is not None:
                raise ManifestError(f"$.models[{i}].convert.from: must name an hf model of this pack")
    canary = _canary(d["canary"], "$.canary") if "canary" in d else None
    language = _str(d["language"], "$.language", re.compile(r"^[a-z]{2}$")) if "language" in d else None
    return Pack(SCHEMA, pid, _int(d["version"], "$.version", 1, 10_000), _str(d["title"], "$.title", max_len=120),
                language, requires, routes, models, canary)


_CACHE: dict[str, Pack] = {}


def load(pack_id: str) -> Pack:
    """The shipped manifest of a pack (``setup/packs/<id>.json``), parsed once per process."""
    if pack_id not in PACKS:
        raise KeyError(pack_id)
    if pack_id not in _CACHE:
        data = resources.files("audiointake.setup").joinpath("packs", f"{pack_id}.json").read_bytes()
        p = parse(data)
        if p.id != pack_id:
            raise ManifestError(f"$.id: packs/{pack_id}.json says it is {p.id!r}")
        _CACHE[pack_id] = p
    return _CACHE[pack_id]


def all_packs() -> list[Pack]:
    return [load(i) for i in PACKS]


def expand(pack_ids: list[str] | tuple[str, ...]) -> list[str]:
    """The packs to install for a request, with what they require, in ``PACKS`` order; core always comes first."""
    want = {"core"}
    todo = list(pack_ids)
    while todo:
        p = todo.pop()
        if p not in PACKS:
            raise KeyError(p)
        if p not in want:
            want.add(p)
        todo.extend(r for r in load(p).requires if r not in want)
    return [p for p in PACKS if p in want]


def models_of(pack_ids: list[str] | tuple[str, ...]) -> list[ModelSpec]:
    """Every model the packs name, each once (a model shared by two packs has the same key and pins)."""
    seen: dict[str, ModelSpec] = {}
    for pid in pack_ids:
        for m in load(pid).models:
            if m.key in seen and seen[m.key] != m:
                raise ManifestError(f"model {m.key!r} is pinned differently in two packs")
            seen.setdefault(m.key, m)
    return list(seen.values())


def host_allowed(host: str, allow: tuple[str, ...] | list[str]) -> bool:
    """The one host rule: ``host`` is one of ``allow``, or under an entry that starts with a dot (".hf.co")."""
    h = host.lower().rstrip(".")
    return any(h == a or (a.startswith(".") and h.endswith(a) and len(h) > len(a)) for a in allow)


def redirect_allowed(start_host: str, host: str) -> bool:
    """Whether a download that started on ``start_host`` may follow a redirect to ``host``."""
    return host_allowed(host, REDIRECT_HOSTS.get(start_host, ()))
