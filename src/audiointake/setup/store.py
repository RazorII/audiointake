"""Install, verify and remove packs: the models setup downloads, converts and assembles, and the verified ledger.

Where things go:

* Hugging Face models live in the standard Hugging Face cache at their pinned revisions, in huggingface_hub's own
  layout (``models--<org>--<name>/blobs/<blob>`` plus the relative ``snapshots/<revision>/<path>`` link), so a Mac
  that already has them downloads nothing and other tools share them.
* What can't live there lives in ``<data>`` (``platforms.data_dir()``): ``models/campplus/`` (the CAM++ release
  asset), ``models/<dir_name>/`` folders a manifest's ``layout`` assembles (symlinks into the cache in the layout
  FluidAudio's ``--models`` wants), and converted models (``models/typhoon-whisper-large-v3-<rev7>/``). A
  conversion's source is read from the Hugging Face cache when it holds it, else downloaded into
  ``<cache>/build/<key>/`` and deleted once the conversion has verified.
* ``<data>/packs/<id>.json`` is a pack's stamp: written only once every file of the pack has been hashed in full
  and every model loaded on generated audio. It records each file's path, size, inode, modification time, blob
  name and hash: the verified ledger a run checks cheaply. ``<data>/downloads.json`` lists what audiointake
  itself downloaded or made, which is all ``packs remove``, ``uninstall`` and the pruning after a pin bump ever
  delete; a model that was in the cache before audiointake stays.
* A stamp also pins what it was made from: the pack's version and, per model, its revision and its files'
  hashes. A stamp that no longer matches the shipped manifest (a release that changed a pin) is *outdated*: core
  outdated stops a run (exit 3) and ``doctor`` says so the same way; an outdated hu or th pack is switched off
  with a warning, as a broken one is (``pack_check``, the one check ``resolve``, ``verify`` and ``status`` share).
* ``uninstall`` deletes only the names audiointake creates (``OWNED_DATA`` in ``<data>``, ``OWNED_CACHE`` in
  ``<cache>``), so pointing ``AUDIOINTAKE_HOME`` or ``AUDIOINTAKE_CACHE`` at a shared folder loses nothing else.

Every function that writes assumes the run lock is held (``lock.require_held``); the CLI takes it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .. import __version__, argv, errors, lock, platforms, proc
from ..contracts import PACK_TERMS, PACKS, Problem
from ..errors import SetupError, problem
from . import manifest
from .manifest import ModelSpec, Pack

STAMP_SCHEMA = 1
DOWNLOADS_SCHEMA = 1
GB = 1_000_000_000
MB = 1_000_000
WATCH_EVERY_S = 10.0           # the watchdog samples the partial file this often
STALL_S = 120.0                # and kills a child whose partial hasn't grown for this long
ATTEMPTS = 3
SLOW_BPS = 256 * 1024          # an attempt's cap: 600 s plus the file's size at this rate
FAST_BPS, SLOWEST_BPS = 20 * MB, 2 * MB          # setup --plan's time range
STALE_PARTIAL_S = 24 * 3600
PACK_NAMES = {"core": "core", "hu": "Hungarian", "th": "Thai"}
# PACK_TERMS (contracts): terms a pack's models come with beyond their licence, which setup --plan states before
# anything is downloaded (THIRD-PARTY.md has them in full).
Progress = Callable[[str], None]
# What audiointake creates directly in <data> and <cache>: all uninstall ever deletes there (run.lock stays).
OWNED_DATA = ("packs", "models", "tools", "canary", "records", "voices", "downloads.json", "speed.json")
OWNED_CACHE = ("work", "build")
CAFFEINATE = Path("/usr/bin/caffeinate")


def code_or(preferred: str, fallback: str) -> str:
    """``preferred`` once errors.CODES has it, else ``fallback`` (for codes proposed but not yet registered)."""
    return preferred if preferred in errors.CODES else fallback


# --------------------------------------------------------------------------- paths
def data() -> Path:
    return platforms.data_dir()


def stamp_path(pack_id: str) -> Path:
    return data() / "packs" / f"{pack_id}.json"


def downloads_path() -> Path:
    return data() / "downloads.json"


def models_dir() -> Path:
    return data() / "models"


def canary_path(canary: manifest.Canary) -> Path:
    return data() / "canary" / canary.file


def repo_dir(repo: str) -> Path:
    return platforms.hf_cache() / ("models--" + repo.replace("/", "--"))


def blob_path(repo: str, blob: str) -> Path:
    return repo_dir(repo) / "blobs" / blob


def snapshot_dir(repo: str, revision: str) -> Path:
    return repo_dir(repo) / "snapshots" / revision


def hf_lock_path(repo: str, blob: str) -> Path:
    return platforms.hf_cache() / ".locks" / ("models--" + repo.replace("/", "--")) / f"{blob}.lock"


def source_dir(m: ModelSpec) -> Path:
    """Where a conversion source is downloaded when the Hugging Face cache doesn't hold it."""
    return platforms.cache_dir() / "build" / f"{m.key}-{(m.revision or '')[:7]}"


def file_dest(m: ModelSpec, f: manifest.FileSpec, *, transient: bool = False) -> tuple[Path, Path | None]:
    """(where the bytes go, the link that names them or None) for one file of a model."""
    if m.kind == "hf" and not transient:
        return blob_path(m.repo, f.blob), snapshot_dir(m.repo, m.revision) / f.path
    if m.kind == "hf":
        return source_dir(m) / f.path, None
    return models_dir() / m.key / f.path, None


def model_path(m: ModelSpec) -> Path:
    """The path a run hands the model's program: a snapshot folder, an assembled or converted folder, a file."""
    if m.layout is not None:
        return models_dir() / m.layout.dir_name
    if m.kind == "hf":
        return snapshot_dir(m.repo, m.revision)
    if m.kind == "github_release":
        return models_dir() / m.key / m.files[0].path
    raise ValueError(f"{m.key}: a converted model needs a layout dir_name")


# --------------------------------------------------------------------------- small json files
def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "rb") as fh:
            d = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _write_json(path: Path, obj: dict) -> None:
    """Replace ``path`` atomically (temp file in the same folder, fsync, rename); folders 0700, file 0600."""
    from .. import fsutil

    fsutil.private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def read_stamp(pack_id: str) -> dict | None:
    s = _read_json(stamp_path(pack_id))
    if not s or s.get("schema") != STAMP_SCHEMA or s.get("id") != pack_id:
        return None
    return s


def read_downloads() -> dict:
    d = _read_json(downloads_path())
    if not d or d.get("schema") != DOWNLOADS_SCHEMA or not isinstance(d.get("items"), list):
        return {"schema": DOWNLOADS_SCHEMA, "items": []}
    return d


def _record_downloads(items: list[dict]) -> None:
    if not items:
        return
    d = read_downloads()
    have = {json.dumps(x, sort_keys=True) for x in d["items"]}
    for it in items:
        k = json.dumps(it, sort_keys=True)
        if k not in have:
            d["items"].append(it)
            have.add(k)
    _write_json(downloads_path(), d)


# --------------------------------------------------------------------------- the verified ledger
def _hash(path: Path, git: bool) -> str:
    from .download import _hash as h

    return h(path, git)


def ledger_entry(path: Path, digest: str, *, git: bool, blob: str | None) -> dict:
    st = os.stat(path)
    return {"path": str(path), "digest": digest, "git": git, "blob": blob, "size": st.st_size, "ino": st.st_ino,
            "mtime_ns": st.st_mtime_ns}


def check_entry(e: dict, *, full: bool = False) -> str:
    """"ok", "refreshed" (the file's stat changed and a re-hash matched: ``e`` is updated in place) or "broken"
    (missing, another blob, another size, or another hash)."""
    p = Path(e["path"])
    try:
        st = os.stat(p)
    except OSError:
        return "broken"
    if e.get("blob") and p.is_symlink() and Path(os.readlink(p)).name != e["blob"]:
        return "broken"
    if st.st_size != e["size"]:
        return "broken"
    same = (st.st_ino, st.st_mtime_ns) == (e["ino"], e["mtime_ns"])
    if same and not full:
        return "ok"
    if _hash(p, bool(e.get("git"))) != e["digest"]:
        return "broken"
    if same:
        return "ok"
    e.update(ino=st.st_ino, mtime_ns=st.st_mtime_ns)
    return "refreshed"


def check_links(model: dict) -> bool:
    for ln in model.get("links", []):
        p = Path(ln["path"])
        if not p.is_symlink() or os.readlink(p) != ln["target"] or not p.exists():
            return False
    return True


def verify_stamp(stamp: dict, *, full: bool = False) -> tuple[list[str], bool]:
    """(names of broken files, whether any entry was refreshed). Entries are updated in place on refresh."""
    broken: list[str] = []
    refreshed = False
    for key, m in sorted(stamp.get("models", {}).items()):
        for e in m.get("files", []):
            r = check_entry(e, full=full)
            if r == "broken":
                broken.append(Path(e["path"]).name)
            refreshed = refreshed or r == "refreshed"
        if not check_links(m):
            broken.append(key)
    return broken, refreshed


def save_refreshed(pack_id: str, stamp: dict) -> bool:
    """Write a refreshed ledger, only while the run lock is held (else the next locked command does it)."""
    if not lock.held():
        return False
    _write_json(stamp_path(pack_id), stamp)
    return True


def pin_of(m: ModelSpec, pk: Pack) -> tuple[str | None, tuple[str, ...]]:
    """What a model is pinned to in the shipped manifest, as a stamp records it: (revision, sorted file digests).
    A converted model: its source's revision and its outputs' hashes; a release asset: its SHA-256 twice."""
    if m.convert is not None:
        return pk.model(m.convert.from_key).revision, tuple(sorted(o.sha256 for o in m.convert.output))
    if m.kind == "hf":
        return m.revision, tuple(sorted(f.sha256 if f.lfs else f.blob for f in m.files))
    return m.files[0].sha256, tuple(sorted(f.sha256 for f in m.files))


def outdated(stamp: dict, pk: Pack) -> str | None:
    """Why a stamp no longer matches the shipped manifest (None when it does): another pack version, a model
    missing, or a model whose revision or file hashes changed (a pin bump, with or without a version bump)."""
    if stamp.get("version") != pk.version:
        return f"pack version {stamp.get('version')}, now {pk.version}"
    models = stamp.get("models", {})
    for m in pk.models:
        if m.key in pk.sources:
            continue
        sm = models.get(m.key)
        if not isinstance(sm, dict):
            return f"{m.tag} missing"
        rev, digests = pin_of(m, pk)
        if sm.get("revision") != rev or tuple(sorted(e.get("digest", "") for e in sm.get("files", []))) != digests:
            return f"{m.tag} pinned to another version"
    return None


def pack_check(pack_id: str, *, full: bool = False) -> tuple[dict | None, list[str], str | None]:
    """(stamp, broken file names, why it is outdated) for one pack: the one check ``runtime.resolve``,
    ``verify`` (so ``doctor``) and ``status`` share. An outdated stamp isn't hashed; a refreshed ledger is saved
    while the run lock is held."""
    st = read_stamp(pack_id)
    if st is None:
        return None, [], None
    why = outdated(st, manifest.load(pack_id))
    if why is not None:
        return st, [], why
    broken, refreshed = verify_stamp(st, full=full)
    if refreshed and not broken:
        save_refreshed(pack_id, st)
    return st, broken, None


def pack_problem(pack_id: str, broken: list[str], why: str | None) -> Problem | None:
    """The problem for a pack that failed ``pack_check``: ``pack_outdated`` (or, until that code is registered,
    ``pack_broken`` naming why) or ``pack_broken`` naming the first broken file; None for a good pack."""
    from ..contracts import LANGUAGE_NAMES

    if why is None and not broken:
        return None
    lang = manifest.load(pack_id).language or pack_id
    language = LANGUAGE_NAMES.get(lang, PACK_NAMES.get(pack_id, pack_id))
    if why is not None:
        return problem(code_or("pack_outdated", "pack_broken"), pack=pack_id, file=why, language=language)
    return problem("pack_broken", pack=pack_id, file=broken[0], language=language)


# --------------------------------------------------------------------------- cheap presence (for plans)
def _file_present(m: ModelSpec, f: manifest.FileSpec, *, transient: bool = False) -> bool:
    dest, link = file_dest(m, f, transient=transient)
    try:
        if link is not None:
            if not link.is_symlink() or Path(os.readlink(link)).name != f.blob:
                return False
            return os.stat(link).st_size == f.size
        return os.stat(dest).st_size == f.size
    except OSError:
        return False


def _converted_present(m: ModelSpec) -> bool:
    d = model_path(m)
    return all((d / o.path).is_file() and (d / o.path).stat().st_size == o.size for o in m.convert.output)


def _source_cached(src: ModelSpec) -> bool:
    return all(_file_present(src, f) for f in src.files)


# --------------------------------------------------------------------------- status and plan
def installed() -> list[str]:
    return [p for p in PACKS if read_stamp(p) is not None]


def status() -> list[dict]:
    """One row per pack: id, title, installed (a stamp exists), verified (its cheap check passes), version,
    routes, disk_bytes (the sizes its stamp records)."""
    rows = []
    for pid in PACKS:
        pk = manifest.load(pid)
        st, broken, why = pack_check(pid)
        rows.append({"id": pid, "title": pk.title, "installed": st is not None,
                     "verified": st is not None and not broken and why is None, "outdated": why is not None,
                     "version": st.get("version") if st else None, "routes": list(pk.routes),
                     "disk_bytes": _stamp_bytes(st)})
    return rows


def plan(pack_ids: Iterable[str]) -> dict:
    """What installing these packs (plus what they require) would fetch and need, net of what is already here:
    ``{fetch_bytes, disk_needed_bytes, disk_free_bytes, minutes_low, minutes_high, hosts, packs: {id: bytes},
    terms}``, where ``terms`` lists the terms these packs' models come with beyond their licence (``PACK_TERMS``).
    Writes nothing and opens no connection."""
    from . import fluidaudio, swiftpm

    ids = manifest.expand(list(pack_ids))
    fetch = keep = peak = 0
    hosts: set[str] = set()
    per_pack: dict[str, int] = {}
    counted: set[str] = set()
    convert_s = (0.0, 0.0)
    for pid in ids:
        pk = manifest.load(pid)
        n = 0
        for m in pk.models:
            if m.key in counted or m.key in pk.sources:
                continue
            counted.add(m.key)
            if m.convert is not None:
                if _converted_present(m):
                    continue
                out = sum(o.size for o in m.convert.output)
                keep += out
                src = pk.model(m.convert.from_key)
                if not _source_cached(src):
                    s = sum(f.size for f in src.files if not _file_present(src, f, transient=True))
                    fetch, n, peak = fetch + s, n + s, max(peak, s)      # the output is already in keep
                    hosts.update(_hosts_for(src))
                convert_s = (convert_s[0] + 20, convert_s[1] + 120)
                continue
            for f in m.files:
                if not _file_present(m, f):
                    fetch, n, keep = fetch + f.size, n + f.size, keep + f.size
                    hosts.update(_hosts_for(m))
        if pk.canary is not None and not canary_path(pk.canary).is_file():
            clip = pk.canary.end - pk.canary.start
            fetch, n, keep = fetch + clip, n + clip, keep + clip + 44
            hosts.add(pk.canary.host)
        per_pack[pid] = n
    build_s = (0.0, 0.0)
    if fluidaudio.check() is None:
        fa = fluidaudio.SOURCE_FETCH_BYTES + sum(a["size"] for a in swiftpm.pinned_artifacts()
                                                  if not swiftpm.cached(a["url"], a["checksum"]))
        fetch, keep, peak = fetch + fa, keep + fluidaudio.KEPT_BYTES, max(peak, fluidaudio.BUILD_PEAK_BYTES)
        per_pack["core"] = per_pack.get("core", 0) + fa
        hosts.update(manifest.REDIRECT_HOSTS[manifest.GITHUB_HOST])
        build_s = fluidaudio.BUILD_SECONDS
    need = keep + peak
    free = min(platforms.free_bytes(data()), platforms.free_bytes(platforms.hf_cache()))
    lo = fetch / FAST_BPS + build_s[0] + convert_s[0]
    hi = fetch / SLOWEST_BPS + build_s[1] + convert_s[1]
    return {"fetch_bytes": fetch, "disk_needed_bytes": need, "disk_free_bytes": free,
            "minutes_low": max(1, round(lo / 60)) if lo else 0, "minutes_high": max(1, round(hi / 60)) if hi else 0,
            "hosts": sorted(("hosts under " + h[1:]) if h.startswith(".") else h for h in hosts),
            "packs": per_pack, "terms": [PACK_TERMS[p] for p in ids if p in PACK_TERMS]}


def _hosts_for(m: ModelSpec) -> tuple[str, ...]:
    return manifest.REDIRECT_HOSTS.get(m.host or "", ())


# --------------------------------------------------------------------------- environments for setup's children
def setup_env(tmp: Path) -> dict[str, str]:
    """The setup children's environment: nothing else from the user's environment, no token of any kind. git
    reads neither the system's nor the user's configuration (``~/.gitconfig`` and the XDG one), so a URL rewrite
    to SSH, an extra header or a credential helper there never reaches setup's fetch from GitHub."""
    from . import runtime

    env = {"PATH": f"{runtime.scripts_dir()}:/usr/bin:/bin", "HOME": str(Path.home()), "TMPDIR": str(tmp),
           "USER": os.environ.get("USER", ""), "LANG": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "ORT_DISABLE_TELEMETRY": "1",
           "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    for k in ("https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    if os.environ.get("DEVELOPER_DIR"):
        env["DEVELOPER_DIR"] = os.environ["DEVELOPER_DIR"]
    return env


def python() -> Path:
    return Path(sys.executable)


def programs(*extra: Path | None) -> frozenset[Path]:
    """Setup's process allowlist (DESIGN 7.2): this Python, the toolchain's git, swift and xcode-select,
    caffeinate, the tool environment's parakeet-mlx and mlx_whisper, and what the caller adds (the ffmpeg found,
    the speaker program just built). Every process setup, doctor and about start is checked against it."""
    from . import fluidaudio, runtime

    s = runtime.scripts_dir()
    base = (python(), fluidaudio.GIT, fluidaudio.SWIFT, fluidaudio.XCODE_SELECT, CAFFEINATE, s / "parakeet-mlx",
            s / "mlx_whisper")
    return frozenset(Path(p) for p in (*base, *extra) if p is not None)


def _under(p: Path, root: Path) -> bool:
    try:
        Path(p).relative_to(root)
        return True
    except ValueError:
        return False


def private_parent(p: Path) -> None:
    """Create ``p``'s folder: 0700 (and any missing parent) when it is under ``<data>``, else as umask has it."""
    from .. import fsutil

    if _under(p.parent, data()):
        fsutil.private_dir(p.parent)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- one download, watched
class DownloadFailed(Exception):
    def __init__(self, code: str, params: dict) -> None:
        super().__init__(code)
        self.code, self.params = code, params


def _watchdog(partial: Path, size: int, label: str, progress: Progress | None,
              alive: Path | None = None) -> Callable[[], bool]:
    """A stall predicate: progress is the partial file growing or the child's heartbeat file being touched (it
    touches it while waiting for the lock or hashing, never while a network read hangs)."""
    state: dict[str, Any] = {"t": time.monotonic(), "len": -1, "grew": time.monotonic(), "beat": None}

    def stalled() -> bool:
        now = time.monotonic()
        if now - state["t"] < WATCH_EVERY_S:
            return False
        state["t"] = now
        try:
            n = partial.stat().st_size
        except OSError:
            n = 0
        beat = None
        if alive is not None:
            try:
                beat = alive.stat().st_mtime_ns
            except OSError:
                beat = None
        if n > state["len"]:
            state["len"], state["grew"] = n, now
            if progress is not None and size >= 50 * MB:
                progress(f"{label}: {_human(n)} of {_human(size)}")
        if beat is not None and beat != state["beat"]:
            state["beat"], state["grew"] = beat, now
        return now - state["grew"] >= STALL_S

    return stalled


def _human(n: int) -> str:
    return f"{n / GB:.2f} GB" if n >= GB else f"{n // MB} MB"


def fetch_file(url: str, dest: Path, size: int, digest: str, *, git: bool = False, lock_file: Path | None = None,
               link: Path | None = None, byte_range: tuple[int, int] | None = None, label: str = "",
               progress: Progress | None = None, allow: tuple[str, ...] | None = None,
               extra_args: tuple[str, ...] = ()) -> dict:
    """Download one file through the child downloader, watched, up to ``ATTEMPTS`` times, each resuming from the
    partial file. Returns the child's status dict; raises DownloadFailed(code, params)."""
    host = (url.split("/", 3)[2] if "://" in url else "").split(":")[0]
    partial = dest.with_name(dest.name + ".audiointake-partial")
    cmd: list[str] = [str(python()), "-I", "-B", "-m", "audiointake.setup.download", "--url", url, "--dest", str(dest),
                      "--size", str(size), "--git-blob" if git else "--sha256", digest, *extra_args]
    if lock_file is not None:
        cmd += ["--lock", str(lock_file)]
    if link is not None:
        cmd += ["--link", str(link)]
    if byte_range is not None:
        cmd += ["--range", str(byte_range[0]), str(byte_range[1])]
    for a in allow or ():
        cmd += ["--allow-host", a]
    from .download import alive_path

    tmp = dest.parent
    private_parent(dest)
    env = setup_env(tmp)
    last: proc.Completed | None = None
    for attempt in range(1, ATTEMPTS + 1):
        last = proc.run(cmd, env=env, timeout=600 + size / SLOW_BPS, allowed=programs(),
                        stall=_watchdog(partial, size, label, progress, alive_path(dest)))
        if last.ok:
            try:
                return json.loads(last.stdout.decode("utf-8").strip().splitlines()[-1])
            except (ValueError, IndexError):
                # the file is in place and checked, but who put it there is unknown: never claim it as ours
                return {"status": "present", "fetched": 0, "hosts": [], "headers": []}
        if last.returncode == 10:                  # a whole file (or the server's own size) disagreed with the pin
            raise DownloadFailed("model_changed", {"file": Path(dest).name if not link else link.name, "host": host,
                                                   "version": __version__})
        if last.returncode in (2, 12):
            break
        # anything else is the network's: a stall, a dropped connection, or a server that answered a resume with
        # the wrong byte range (14: the child dropped the partial), so the next attempt resumes or starts again
    raise DownloadFailed("download_failed", {"model": label or Path(dest).name, "host": host,
                                              "detail": " | ".join((last.stderr_tail if last else [])[-1:])})


# --------------------------------------------------------------------------- install
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SWEPT_FILES = ("*.audiointake-partial", "*.audiointake-alive", "*.audiointake-bad")


def sweep_partials() -> int:
    """Delete what an interrupted setup left, once it is older than a day: download partials, heartbeat files and
    (from earlier versions) mismatched downloads in the folders setup writes, and ``.partial-*`` / ``.old-*``
    folders beside the models and the speaker program. Returns how many."""
    from . import fluidaudio

    roots = [platforms.hf_cache(), models_dir(), platforms.cache_dir() / "build", data() / "canary"]
    n = 0
    cutoff = time.time() - STALE_PARTIAL_S
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in SWEPT_FILES:
            for p in root.rglob(pattern):
                try:
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()
                        n += 1
                except OSError:
                    pass
    for root in (models_dir(), fluidaudio.root()):
        if not root.is_dir():
            continue
        for p in root.iterdir():
            try:
                if p.name.startswith((".partial-", ".old-")) and p.is_dir() and not p.is_symlink() \
                        and p.stat().st_mtime < cutoff:
                    shutil.rmtree(p, ignore_errors=True)
                    n += 1
            except OSError:
                pass
    return n


def _install_files(m: ModelSpec, *, transient: bool, progress: Progress, report: dict) -> list[dict]:
    """Fetch (or find) and check every file of a model; ledger entries for them. Raises DownloadFailed."""
    entries = []
    for f in m.files:
        dest, link = file_dest(m, f, transient=transient)
        lock_file = hf_lock_path(m.repo, f.blob) if (m.kind == "hf" and not transient) else None
        if lock_file is None:
            lock_file = dest.with_name(f".{dest.name}.lock")
        out = fetch_file(m.url(f), dest, f.size, f.sha256 if f.lfs else f.blob, git=not f.lfs, lock_file=lock_file,
                         link=link, label=m.tag, progress=progress)
        report["fetched_bytes"] += int(out.get("fetched", 0))
        report["hosts"].update(out.get("hosts", []))
        report["headers"].update(out.get("headers", []))
        if out.get("status") == "downloaded" and not transient:
            report["downloads"].append({"kind": "hf", "repo": m.repo, "revision": m.revision, "path": f.path,
                                        "blob": f.blob, "blob_path": str(dest), "link": str(link)}
                                       if m.kind == "hf" else {"kind": "data", "path": str(dest)})
        p = link if link is not None else dest
        entries.append(ledger_entry(p, f.sha256 if f.lfs else f.blob, git=not f.lfs,
                                    blob=f.blob if link is not None else None))
    return entries


def _assemble(m: ModelSpec) -> list[dict]:
    """The layout folder of symlinks into the model's snapshot (absolute targets), built aside and swapped in."""
    final = model_path(m)
    want = [(ln.name, str(snapshot_dir(m.repo, m.revision) / ln.source)) for ln in m.layout.links]
    if final.is_dir() and all((final / n).is_symlink() and os.readlink(final / n) == t for n, t in want):
        return [{"path": str(final / n), "target": t} for n, t in want]
    from .. import fsutil

    part = final.with_name(f".partial-{final.name}-{os.getpid()}")
    shutil.rmtree(part, ignore_errors=True)
    fsutil.private_dir(part)
    for n, t in want:
        os.symlink(t, part / n)
    _swap_in(part, final)
    return [{"path": str(final / n), "target": t} for n, t in want]


def _swap_in(part: Path, final: Path) -> None:
    old = None
    if final.exists() or final.is_symlink():
        old = final.with_name(f".old-{final.name}-{os.getpid()}")
        os.replace(final, old)
    os.replace(part, final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def _convert(m: ModelSpec, pk: Pack, *, progress: Progress, report: dict) -> list[dict]:
    """Make a converted model (in a child) from its source, check every output against its pin, swap it in."""
    final = model_path(m)
    outs = m.convert.output
    if final.is_dir() and all(Path(final / o.path).is_file() and (final / o.path).stat().st_size == o.size
                              and _hash(final / o.path, False) == o.sha256 for o in outs):
        return [ledger_entry(final / o.path, o.sha256, git=False, blob=None) for o in outs]
    src = pk.model(m.convert.from_key)
    transient = not _source_cached(src)
    _install_files(src, transient=transient, progress=progress, report=report)
    source = source_dir(src) if transient else snapshot_dir(src.repo, src.revision)
    from .. import fsutil

    part = final.with_name(f".partial-{final.name}-{os.getpid()}")
    shutil.rmtree(part, ignore_errors=True)
    fsutil.private_dir(part)                                # the converter writes into it; swapped in whole
    progress(f"{m.tag}: converting")
    weights = [x for f in src.files if (x := f.path).endswith(".safetensors")]
    cmd = [str(python()), "-I", "-B", "-m", "audiointake.setup.convert", "--source", str(source), "--out", str(part)]
    for w in weights:
        cmd += ["--weights", w]
    t0 = time.monotonic()
    r = proc.run(cmd, env=setup_env(final.parent), timeout=3600, allowed=programs(), stdout="capture")
    report["convert_seconds"] = round(time.monotonic() - t0, 1)
    if not r.ok:
        shutil.rmtree(part, ignore_errors=True)
        raise DownloadFailed(code_or("model_load_failed", "pack_broken"),
                             {"pack": pk.id, "model": m.tag, "file": m.tag, "language": pk.title,
                              "exit": r.returncode, "timed_out": int(r.timed_out)})
    for o in outs:
        p = part / o.path
        if not p.is_file() or p.stat().st_size != o.size or _hash(p, False) != o.sha256:
            shutil.rmtree(part, ignore_errors=True)
            raise DownloadFailed("model_changed", {"file": o.path, "host": "this Mac", "version": __version__})
    _swap_in(part, final)
    if transient:
        shutil.rmtree(source_dir(src), ignore_errors=True)
    report["downloads"].append({"kind": "data", "path": str(final)})
    return [ledger_entry(final / o.path, o.sha256, git=False, blob=None) for o in outs]


def _stamp_model(m: ModelSpec, files: list[dict], links: list[dict] | None = None, *,
                 revision: str | None = None) -> dict:
    rev = revision or (m.revision if m.kind == "hf" else m.files[0].sha256 if m.kind == "github_release" else None)
    d = {"kind": m.kind, "tag": m.tag, "revision": rev, "path": str(model_path(m)), "files": files}
    if links:
        d["links"] = links
    return d


def _canary_command(clip: Path, outdir: Path) -> list[str]:
    """The canary child: ``audiointake transcribe`` on the kept clip (so setup never imports the pipeline), with
    ``--no-names`` so the test recording is never matched against the user's voiceprints."""
    return [str(python()), "-I", "-B", "-m", "audiointake", "transcribe", str(clip), "--outdir", str(outdir),
            "--format", "json", "--no-names", "--json"]


def _canary_env(work: Path, ffmpeg: Path | None = None) -> dict[str, str]:
    """A run child's environment (``runtime.run_env``: the tool's scripts, ffmpeg's folder, then the system's
    two; nothing else of the user's, no developer folder), plus the folders this setup uses, so the child
    finds the same packs and the same ffmpeg."""
    from . import runtime

    env = runtime.run_env(work, ffmpeg if ffmpeg is not None else Path("/usr/bin/ffmpeg"))
    env.update(AUDIOINTAKE_HOME=str(data()), AUDIOINTAKE_CACHE=str(platforms.cache_dir()),
               HF_HUB_CACHE=str(platforms.hf_cache()))
    return env


def fetch_canary(c: manifest.Canary, *, progress: Progress) -> tuple[Path, dict | None]:
    """The canary clip, fetched once by byte range and kept (so later self-tests are offline), and the
    downloader's report when it was fetched now. DownloadFailed."""
    dest = canary_path(c)
    if dest.is_file():
        return dest, None
    raw = dest.with_suffix(".pcm")
    out = fetch_file(c.url, raw, c.end - c.start, c.sha256, byte_range=(c.start, c.end), label="test recording",
                     progress=progress, lock_file=raw.with_name(f".{raw.name}.lock"))
    from . import smoke

    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(smoke.wav_header(c.end - c.start, rate=c.rate, channels=c.channels, bits=c.bits)
                    + raw.read_bytes())
    os.replace(tmp, dest)
    raw.unlink()
    raw.with_name(f".{raw.name}.lock").unlink(missing_ok=True)
    return dest, out


def run_canary(c: manifest.Canary, *, speakers: bool, progress: Progress, fetch: bool = True,
               ffmpeg: Path | None = None) -> dict:
    """Run the canary through ``audiointake transcribe`` as a child; ``{ok, rtf, code?, ...}``. With ``fetch`` the
    clip is fetched once if it isn't kept yet; without it (``doctor --selftest``) nothing goes online. The child
    writes one record and one speed figure, as any finished transcription does (the canary's speed seeds
    speed.json)."""
    from . import ffmpeg as ffm, smoke

    if ffmpeg is None:
        ffmpeg = ffm.find()[0]

    if not fetch and not canary_path(c).is_file():
        return {"ok": False, "rtf": None, "code": "canary_unavailable", "host": c.host}
    try:
        clip, fetched = fetch_canary(c, progress=progress)
    except DownloadFailed as exc:
        return {"ok": False, "rtf": None, "code": "canary_unavailable", "host": exc.params.get("host", c.host)}
    work = platforms.cache_dir() / "work" / f"canary-{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    from .. import fsutil

    fsutil.private_dir(work)
    progress("test recording: transcribing")
    t0 = time.monotonic()
    try:
        env = _canary_env(work, ffmpeg)
        r = proc.run(_canary_command(clip, work), env=lock.child_env(env) if lock.held() else env, timeout=900,
                     allowed=programs(ffmpeg), stdout="capture")
        seconds = time.monotonic() - t0
        result = smoke.canary_check(work, clip, c, speakers=speakers)
        result.update(rtf=round(seconds / ((c.end - c.start) / (2 * c.rate)), 3), exit=r.returncode)
        if fetched is not None:
            result["fetch"] = {"bytes": fetched.get("fetched", 0), "hosts": fetched.get("hosts", []),
                               "headers": fetched.get("headers", [])}
        if r.returncode != 0:
            result["ok"] = False
            result.setdefault("code", "transcribe_failed")
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def canary_problem(c: dict | None) -> Problem | None:
    """The warning a canary result gives: ``canary_unavailable`` when the clip couldn't be had, ``canary_failed``
    (once that code is registered) when it ran and didn't pass; None when it passed or was skipped."""
    if not c or c.get("ok") is not False:
        return None
    if c.get("code") == "canary_unavailable":
        return problem("canary_unavailable", host=c.get("host", ""))
    if "canary_failed" in errors.CODES:
        found_speakers = c.get("speakers")
        return problem("canary_failed", found=c.get("found", 0), of=c.get("of", 0),
                       speakers=found_speakers if found_speakers is not None else "?")
    return None


def install(pack_ids: list[str], *, progress: Callable[[str], None], speakers: bool = True,
            canary: bool = True) -> dict:
    """Install packs (and what they require); returns setup's report (DESIGN 6.5's shape, plus ``hosts_seen``,
    ``headers_sent``, ``fetched_bytes`` and ``convert_seconds``). Holds caffeinate for its length; needs the lock.

    ``exit`` is 3 when anything failed, as ``audiointake setup`` reports it, including a speaker program that
    couldn't be built (``clt_missing`` and the like), which leaves every pack installed. A caller that installs
    one pack for its own sake reads that pack's own verdict: its ``packs[]`` row (``verified``) and the errors
    whose ``params.pack`` names it. ``speakers=False`` leaves the speaker program alone and ``canary=False`` skips
    the test recording; ``apply --lang`` passes both, so a pack it installs is never failed by a missing
    Command Line Tools or a test recording."""
    lock.require_held()
    from . import ffmpeg as ffm, fluidaudio, smoke

    res: dict[str, Any] = {"exit": 0, "platform": {}, "prerequisites": [], "plan": {}, "packs": [],
                           "speakers": {}, "canary": {}, "warnings": [], "errors": [], "hosts_seen": [],
                           "headers_sent": []}
    errs: list[Problem] = []
    warns: list[Problem] = []

    def done() -> dict:
        res["errors"] = [{"code": p.code, "params": dict(p.params)} for p in errs]
        res["warnings"] = [{"code": p.code, "params": dict(p.params)} for p in warns]
        res["exit"] = errors.EXIT_SETUP if errs else errors.EXIT_OK
        return res

    try:
        res["platform"] = platforms.require_supported()
    except SetupError as exc:
        res["platform"] = platforms.info()
        errs.append(exc.problem())
        return done()
    warns += platforms.warnings()
    ids = manifest.expand(list(pack_ids))
    ff, fp = ffm.find(ensure_fallback=True)
    res["prerequisites"] = ffm.prerequisites(ff, fp)
    if ff is None:
        errs.append(problem("ffmpeg_missing"))
    swept = sweep_partials()
    pl = plan(ids)
    res["plan"] = pl
    if pl["disk_needed_bytes"] > pl["disk_free_bytes"]:
        errs.append(problem("no_room", what="Setting up audiointake", need=pl["disk_needed_bytes"] / GB,
                            free=pl["disk_free_bytes"] / GB))
        return done()
    report = {"fetched_bytes": 0, "hosts": set(), "headers": set(), "downloads": [], "swept": swept}
    stamps: dict[str, dict] = {}
    failed: set[str] = set()
    smokes: dict[str, dict] = {}
    fetched_by: dict[str, int] = {}
    with proc.keep_awake():
        for pid in ids:
            pk = manifest.load(pid)
            if any(r in failed for r in pk.requires):
                failed.add(pid)
                continue
            before = report["fetched_bytes"]
            st = {"schema": STAMP_SCHEMA, "id": pid, "version": pk.version, "core_version": __version__,
                  "macos": res["platform"].get("macos", ""), "created": _now(), "models": {}, "smoke": {}}
            try:
                for m in pk.models:
                    if m.key in pk.sources:
                        continue
                    progress(f"{m.tag}: checking")
                    if m.convert is not None:
                        st["models"][m.key] = _stamp_model(m, _convert(m, pk, progress=progress, report=report),
                                                           revision=pk.model(m.convert.from_key).revision)
                    else:
                        files = _install_files(m, transient=False, progress=progress, report=report)
                        links = _assemble(m) if m.layout is not None else None
                        st["models"][m.key] = _stamp_model(m, files, links)
            except DownloadFailed as exc:
                errs.append(problem(exc.code, **{**exc.params, "pack": pid}))
                failed.add(pid)
                continue
            stamps[pid] = st
            fetched_by[pid] = report["fetched_bytes"] - before
        _record_downloads(report["downloads"])
        if ff is not None:
            from . import runtime

            for pid, st in stamps.items():
                pk = manifest.load(pid)
                for m in pk.models:
                    if m.key in pk.sources or m.smoke is None or m.smoke.kind == "nemotron":
                        continue
                    if m.key not in smokes:
                        progress(f"{m.tag}: test run")
                        smokes[m.key] = smoke.model_smoke(m.smoke, model_path(m), ff, runtime.scripts_dir())
                    st["smoke"][m.key] = smokes[m.key]
                    if not smokes[m.key]["ok"]:
                        errs.append(problem(code_or("model_load_failed", "pack_broken"), pack=pid, model=m.tag,
                                            file=m.tag, language=PACK_NAMES.get(pid, pid)))
                        failed.add(pid)
        # the speaker program (core): built from its pinned commit, smoke-tested on our Nemotron folder
        spk: dict[str, Any] = {"tag": fluidaudio.TAG, "commit": fluidaudio.COMMIT,
                               "recipe": fluidaudio.RECIPE, "built": False, "reused": False, "seconds": None}
        if not speakers:
            spk["skipped"] = True
        elif "core" in stamps:
            nemo = manifest.load("core").model("nemotron")
            fluidaudio.LAST_HOSTS.clear()
            fluidaudio.LAST_DOWNLOADS.clear()
            try:
                t0 = time.monotonic()
                before = fluidaudio.check()
                fluidaudio.LAST_SMOKE.clear()
                binary = fluidaudio.ensure(build=True, progress=progress, warnings=warns,
                                           smoke_models=model_path(nemo) if ff is not None else None, ffmpeg=ff)
                after = fluidaudio.check()
                rebuilt = after is not None and (before is None or before.get("built_at") != after.get("built_at"))
                spk.update(built=rebuilt, reused=after is not None and not rebuilt, path=str(binary),
                           seconds=round(time.monotonic() - t0, 1),
                           build_seconds=(after or {}).get("build_seconds") if rebuilt else None)
                if not rebuilt and after is not None and after.get("tag_moved"):
                    warns.append(problem("fluidaudio_tag_moved", tag=fluidaudio.TAG, commit=fluidaudio.COMMIT[:12]))
                if fluidaudio.LAST_SMOKE:
                    stamps["core"]["smoke"]["nemotron"] = dict(fluidaudio.LAST_SMOKE)
            except SetupError as exc:
                spk["code"] = exc.code
                errs.append(exc.problem())
            report["hosts"].update(fluidaudio.LAST_HOSTS)
            report["downloads"].extend({"kind": "data", "path": d} for d in fluidaudio.LAST_DOWNLOADS)
        res["speakers"] = spk
        _record_downloads(report["downloads"])
        for pid in ids:                      # a pack whose required pack failed fails with it
            if any(r in failed for r in manifest.load(pid).requires):
                failed.add(pid)
        verified_keys: set[str] = set()
        for pid in ids:
            if pid in stamps and pid not in failed and ff is not None:
                _write_json(stamp_path(pid), stamps[pid])
                verified_keys.update(stamps[pid]["models"])
            elif pid in failed and stamp_path(pid).exists():
                stamp_path(pid).unlink()      # no stale stamp: the next setup redoes this pack
        core = manifest.load("core")
        if not canary:
            res["canary"] = {"ok": None, "rtf": None, "skipped": True}
        elif "core" in stamps and "core" not in failed and ff is not None and core.canary is not None:
            c = run_canary(core.canary, speakers=fluidaudio.check() is not None, progress=progress, ffmpeg=ff)
            res["canary"] = c
            report["hosts"].update((c.get("fetch") or {}).get("hosts", []))
            cp = canary_problem(c)
            if cp is not None:
                warns.append(cp)
        res["hosts_seen"] = sorted(report["hosts"])
        res["headers_sent"] = sorted(report["headers"])
        res["pruned"] = prune(verified_keys)
    res["packs"] = [{"id": pid, "installed": read_stamp(pid) is not None, "verified": pid in stamps
                     and pid not in failed and ff is not None, "fetched_bytes": fetched_by.get(pid, 0),
                     "disk_bytes": _stamp_bytes(read_stamp(pid))} for pid in ids]
    res["fetched_bytes"] = report["fetched_bytes"]
    res["convert_seconds"] = report.get("convert_seconds")
    return done()


def _stamp_bytes(st: dict | None) -> int:
    if not st:
        return 0
    return sum(e["size"] for m in st.get("models", {}).values() for e in m.get("files", []))


# --------------------------------------------------------------------------- verify, remove, prune, uninstall
def verify(full: bool) -> list[Problem]:
    """Problems with the installed packs, from the same ``pack_check`` a run uses: ``pack_outdated`` (or, until
    that code is registered, ``pack_broken``) for a stamp the shipped manifest no longer matches, ``pack_broken``
    for a changed file (cheap check, or a full re-hash with ``full``). A refreshed ledger is saved when the lock
    is held."""
    out: list[Problem] = []
    for pid in PACKS:
        st, broken, why = pack_check(pid, full=full)
        if st is None:
            continue
        p = pack_problem(pid, broken, why)
        if p is not None:
            out.append(p)
    return out


def _needed_keys(except_pack: str | None = None) -> set[str]:
    keys: set[str] = set()
    for pid in installed():
        if pid != except_pack:
            keys.update(m.key for m in manifest.load(pid).models)
    return keys


def _delete_item(it: dict) -> int:
    """Delete one downloads.json item; returns the bytes freed."""
    freed = 0
    if it.get("kind") == "hf":
        link, blob = Path(it["link"]), Path(it["blob_path"])
        if link.is_symlink():
            link.unlink()
        repo_snaps = blob.parent.parent / "snapshots"
        still = repo_snaps.is_dir() and any(p.is_symlink() and Path(os.readlink(p)).name == blob.name
                                            for p in repo_snaps.rglob("*"))
        if blob.is_file() and not still:
            freed += blob.stat().st_size
            blob.unlink()
    elif it.get("kind") == "data":
        p = Path(it["path"])
        if p.is_dir() and not p.is_symlink():
            freed += sum(x.stat().st_size for x in p.rglob("*") if x.is_file() and not x.is_symlink())
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file() or p.is_symlink():
            freed += p.stat().st_size if p.is_file() else 0
            p.unlink()
    return freed


def _item_key(it: dict) -> str | None:
    """The model key a downloads.json item belongs to."""
    for pk in manifest.all_packs():
        for m in pk.models:
            if it.get("kind") == "hf" and m.kind == "hf" and it.get("repo") == m.repo and it.get("revision") == m.revision:
                return m.key
            if it.get("kind") == "data" and Path(it["path"]) in (model_path(m), *(
                    (models_dir() / m.key / f.path) for f in m.files if m.kind == "github_release")):
                return m.key
    return None


def remove(pack_id: str) -> dict:
    """Remove a pack: its stamp, and only the files downloads.json lists that no other installed pack needs."""
    if pack_id == "core":
        raise errors.UsageError("core_required")
    if pack_id not in PACKS:
        raise errors.UsageError("bad_arguments", detail=f"there's no pack {pack_id!r}", command="packs remove")
    lock.require_held()
    pk = manifest.load(pack_id)
    keep = _needed_keys(except_pack=pack_id)
    mine = {m.key for m in pk.models} - keep
    d = read_downloads()
    left, freed, removed = [], 0, 0
    for it in d["items"]:
        if _item_key(it) in mine:
            freed += _delete_item(it)
            removed += 1
        else:
            left.append(it)
    if stamp_path(pack_id).exists():
        stamp_path(pack_id).unlink()
    d["items"] = left
    _write_json(downloads_path(), d)
    return {"id": pack_id, "removed": removed, "freed_bytes": freed, "kept_shared": sorted({m.key for m in pk.models} & keep)}


def _snapshots_in_use() -> set[tuple[str, str]]:
    """(repo folder, revision) of every snapshot a pack stamp on this Mac still points at."""
    used: set[tuple[str, str]] = set()
    for pid in PACKS:
        st = read_stamp(pid)
        for m in (st or {}).get("models", {}).values():
            for e in m.get("files", []):
                parts = Path(e.get("path", "")).parts
                if "snapshots" in parts:
                    i = parts.index("snapshots")
                    if 0 < i < len(parts) - 1:
                        used.add((parts[i - 1], parts[i + 1]))
    return used


def prune(verified_keys: Iterable[str]) -> dict:
    """After a pin bump: delete what downloads.json lists for an older revision of a model whose current pin was
    just installed and verified (``verified_keys``), unless a pack stamp on this Mac still points at it. Nothing
    of a model this setup didn't verify is touched, so a core-only setup never prunes the hu or th models."""
    keys = set(verified_keys)
    current = {m.repo: m.revision for pk in manifest.all_packs() for m in pk.models
               if m.kind == "hf" and m.key in keys}
    in_use = _snapshots_in_use()
    d = read_downloads()
    left, freed, removed = [], 0, 0
    for it in d["items"]:
        repo, rev = it.get("repo"), it.get("revision")
        if it.get("kind") == "hf" and repo in current and rev != current[repo] \
                and ("models--" + str(repo).replace("/", "--"), rev) not in in_use:
            freed += _delete_item(it)
            removed += 1
        else:
            left.append(it)
    if removed:
        d["items"] = left
        _write_json(downloads_path(), d)
    return {"removed": removed, "freed_bytes": freed}


def _size(p: Path) -> int:
    if p.is_symlink():
        return 0
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(x.stat().st_size for x in p.rglob("*") if x.is_file() and not x.is_symlink())
    return 0


UNINSTALL_COMMAND = "uv tool uninstall audiointake"


def _ours(name: str, owned: tuple[str, ...]) -> bool:
    """Whether an entry of ``<data>`` or ``<cache>`` is one audiointake creates: an owned name, a ``.partial-``
    or ``.old-`` leftover, or one of setup's own temporary files (``.<owned name>.<pid>.tmp``)."""
    if name in owned or name.startswith((".partial-", ".old-")):
        return True
    return name.startswith(".") and name.endswith(".tmp") and any(name[1:].startswith(o + ".") for o in owned)


def uninstall(*, keep_voices: bool, plan: bool) -> dict:
    """Remove what audiointake made: what downloads.json lists (models it downloaded, never one that was in the
    Hugging Face cache before it; the build component it put in SwiftPM's cache), the names it creates in
    ``<data>`` (``OWNED_DATA``) and ``<cache>`` (``OWNED_CACHE``), and the logs. Anything else in those folders
    (``AUDIOINTAKE_HOME`` or ``AUDIOINTAKE_CACHE`` may point at a shared folder) is listed under ``left`` and kept.
    ``plan`` only lists. ``run.lock`` stays (deleting a lock file races with its next taker)."""
    if not plan:
        lock.require_held()
    items = read_downloads()["items"]
    hf = [it for it in items if it.get("kind") == "hf"]
    outside = [Path(it["path"]) for it in items if it.get("kind") == "data" and it.get("path")
               and not _under(Path(it["path"]), data()) and not _under(Path(it["path"]), platforms.cache_dir())]
    paths: list[Path] = []
    left: list[Path] = []
    owned_data = tuple(n for n in OWNED_DATA if not (keep_voices and n == "voices"))
    for root, owned in ((data(), owned_data), (platforms.cache_dir(), OWNED_CACHE)):
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.name == lock.LOCK_NAME or (keep_voices and p.name == "voices"):
                continue
            (paths if _ours(p.name, owned) else left).append(p)
    if platforms.logs_dir().exists():
        paths.append(platforms.logs_dir())
    for p in outside:                         # with the lock file the downloader left beside it
        paths += [x for x in (p, p.with_name(f".{p.name}.lock")) if x.exists() or x.is_symlink()]
    out = {"plan": plan, "keep_voices": keep_voices,
           "remove": [{"path": str(p), "bytes": _size(p)} for p in paths],
           "models": [{"repo": it["repo"], "path": it["path"], "bytes": _size(Path(it["blob_path"]))} for it in hf],
           "left": [{"path": str(p), "bytes": _size(p)} for p in left],
           "then": UNINSTALL_COMMAND}
    out["bytes"] = sum(x["bytes"] for x in out["remove"]) + sum(x["bytes"] for x in out["models"])
    if plan:
        return out
    freed = sum(_delete_item(it) for it in hf)
    for p in paths:
        freed += _size(p)
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink()
    cache = platforms.cache_dir()
    try:
        if cache.is_dir() and not any(cache.iterdir()):
            cache.rmdir()                     # empty now: it held only what audiointake made
    except OSError:
        pass
    out["freed_bytes"] = freed
    return out
