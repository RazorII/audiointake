"""The speaker program: FluidAudio's ``fluidaudiocli``, built on this Mac from a pinned commit and keyed to it.

The build lives in ``<data>/tools/fluidaudio/<tag>-<commit[:12]>-r<recipe>/`` with ``bin/fluidaudiocli`` and
``BUILD.json`` (schema, tag, commit, recipe, the binary's SHA-256, when and how long it was built, the Swift
toolchain and macOS it was built with). A package update with the same pin and recipe reuses it; a pin or recipe
bump makes a new key and a new build; a macOS or Xcode update doesn't rebuild, a failed
smoke does. ``check()`` is the cheap test every run makes: BUILD.json names this pin, the binary is executable and
its SHA-256 matches (about 20 ms for its 24 MB).

The build (setup only, under the run lock, in ``<cache>/build/fluidaudio-<key>/`` so a rerun after a cap or a
crash continues where SwiftPM left off):

1. The source is fetched by its commit (``git fetch --depth 1 URL <commit>``), never by its tag, so an upstream
   retag can't strand setup; ``git ls-remote`` then checks the tag, and a tag that now names another commit is the
   warning ``fluidaudio_tag_moved`` (the pinned commit is built anyway). If GitHub ever refuses a fetch by commit,
   the fallback is a shallow clone of the tag that must check out to the pinned commit.
2. SwiftPM's binary artifacts are prefetched (``swiftpm.prefetch``), ``swift package resolve`` runs (15 min cap)
   and the prefetch runs again.
3. ``swift build -c release --product fluidaudiocli`` (45 min cap, default traits). A failure while the toolchain
   still holds stale ``*.private.swiftinterface`` files is retried once with ``SWIFTPM_CUSTOM_LIBS_DIR`` pointing
   at a pruned copy of the manifest libraries.
4. The binary (and only the resource bundles ``BUNDLES`` names: ``nemotron3-diarize`` needs none) is copied aside
   with BUILD.json, smoke-tested, and swapped into its key; the build folder is deleted; older keys but the
   previous one are pruned, so rolling back is instant. LuxTTS's pronunciation data (GPL and CC-BY-SA derived) is
   never copied.

Before any of it, ``xcode-select -p`` must name an existing folder holding a ``swift`` (``platforms.swift_binary``):
only then is ``/usr/bin/swift`` run, so a Mac without the Command Line Tools never sees Apple's install window
mid-run. Every cap kills the whole process group. git runs with neither the system's nor the user's configuration,
https only and no SSH, so nothing in ``~/.gitconfig`` (a URL rewrite, a header, a credential helper) reaches the
fetch. Whether the tag had moved is kept in BUILD.json (``tag_moved``), so a reused build and ``doctor``, which
stays offline, still report it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .. import argv, fsutil, lock, platforms, proc
from ..contracts import Problem
from ..errors import SetupError, problem

TAG = "v0.18.0"
COMMIT = "e15287985625dbf544a343df679436789ffb03fb"
FLUIDAUDIO_PIN = (TAG, COMMIT)
RECIPE = 1                       # bumped whenever build flags, traits or the copied files change
FLUIDAUDIO_RECIPE = RECIPE
REPO_URL = "https://github.com/FluidInference/FluidAudio.git"
PRODUCT = "fluidaudiocli"
BUNDLES: tuple[str, ...] = ()    # resource bundles nemotron3-diarize needs next to the binary (measured: none)
NEVER_COPY = ("FluidAudio_LuxTtsG2pResources.bundle",)
SCHEMA = 1
GIT_PROTOCOLS = "https"          # git may fetch over these only (a test swaps in "file" for its local repository)
GIT = Path("/usr/bin/git")
SWIFT = Path("/usr/bin/swift")
XCODE_SELECT = Path("/usr/bin/xcode-select")
GIT_CAP_S, RESOLVE_CAP_S, BUILD_CAP_S, SMOKE_CAP_S = 300.0, 900.0, 2700.0, 900.0
# Planning figures (setup --plan): the source fetch, what is kept, the build folder at its largest, the time.
SOURCE_FETCH_BYTES = 24_000_000
KEPT_BYTES = 25_000_000
BUILD_PEAK_BYTES = 640_000_000
BUILD_SECONDS = (100.0, 600.0)
Progress = Callable[[str], None]
LAST_SMOKE: dict = {}            # the most recent smoke's result, for setup's report
LAST_HOSTS: set[str] = set()     # hosts the most recent build contacted (git and the prefetch), for setup's report
LAST_DOWNLOADS: list[str] = []   # files the most recent build downloaded outside <data> (SwiftPM's cache)


def key(tag: str | None = None, commit: str | None = None, recipe: int | None = None) -> str:
    tag, commit, recipe = tag or TAG, commit or COMMIT, RECIPE if recipe is None else recipe
    return f"{tag}-{commit[:12]}-r{recipe}"


def root() -> Path:
    return platforms.data_dir() / "tools" / "fluidaudio"


def key_dir() -> Path:
    return root() / key()


def build_dir() -> Path:
    return platforms.cache_dir() / "build" / f"fluidaudio-{key()}"


def build_log() -> Path:
    return build_dir() / "build.log"


def _sha(p: Path) -> str:
    from .download import _hash

    return _hash(p, False)


def check() -> dict | None:
    """The reusable build for the current pin and recipe (``BUILD.json`` plus ``path``), or None."""
    d = key_dir()
    exe = d / "bin" / PRODUCT
    try:
        meta = json.loads((d / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA or meta.get("tag") != TAG \
            or meta.get("commit") != COMMIT or meta.get("recipe") != RECIPE:
        return None
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        return None
    try:
        if _sha(exe) != meta.get("binary_sha256"):
            return None
    except OSError:
        return None
    return {**meta, "path": str(exe)}


# --------------------------------------------------------------------------- the toolchain
def _env(tmp: Path, toolchain_bin: Path | None = None) -> dict[str, str]:
    from . import store

    env = store.setup_env(tmp)
    env["GIT_ALLOW_PROTOCOL"] = GIT_PROTOCOLS
    env["GIT_SSH_COMMAND"] = "/usr/bin/false"
    # git runs in the build folder and never looks above it for a repository, so the configuration of a repository
    # the user happens to run setup from (or a home folder kept in git) doesn't apply either
    env["GIT_CEILING_DIRECTORIES"] = str(Path(tmp).parent)
    if toolchain_bin is not None:
        head, _, rest = env["PATH"].partition(":")
        env["PATH"] = f"{head}:{toolchain_bin}:{rest}"
    return env


def _programs(*extra: Path | None) -> frozenset[Path]:
    from . import store

    return store.programs(*extra)


def _host(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0] if "://" in url and not url.startswith("file:") else ""


def toolchain() -> dict:
    """``{developer_dir, swift, code}``: code None, ``clt_missing``, ``xcode_license`` or ``swift_too_old``.

    ``/usr/bin/swift`` is run only after ``xcode-select -p`` named a folder that holds a swift."""
    import tempfile

    env = _env(Path(tempfile.gettempdir()))
    try:
        r = proc.run([XCODE_SELECT, "-p"], env=env, timeout=30, allowed=_programs())
    except OSError:
        return {"developer_dir": None, "swift": "", "code": "clt_missing"}
    dev = platforms.developer_dir(r.stdout.decode("utf-8", "replace"), r.returncode)
    if dev is None or platforms.swift_binary(dev) is None:
        return {"developer_dir": str(dev) if dev else None, "swift": "", "code": "clt_missing"}
    r2 = proc.run([SWIFT, "--version"], env=env, timeout=60, allowed=_programs())
    out = r2.stdout.decode("utf-8", "replace") + "\n" + "\n".join(r2.stderr_tail)
    code, version = platforms.swift_problem(r2.returncode, out)
    return {"developer_dir": str(dev), "swift": version, "code": code,
            "toolchain_bin": str(platforms.swift_binary(dev).parent)}


# --------------------------------------------------------------------------- building
def _git(args: list[str], cwd: Path | None, env: dict, log: Path, cap: float = GIT_CAP_S) -> proc.Completed:
    return proc.run([GIT, *args], env=env, timeout=cap, allowed=_programs(), cwd=cwd or Path(env["TMPDIR"]),
                    stdout="discard", output_log=log)


def _head(src: Path, env: dict, log: Path) -> str:
    if not (src / ".git").exists():
        return ""
    r = proc.run([GIT, "-C", str(src), "rev-parse", "HEAD"], env=env, timeout=60, allowed=_programs())
    return r.stdout.decode("ascii", "replace").strip() if r.ok else ""


def fetch_source(src: Path, env: dict, log: Path) -> None:
    """The pinned commit checked out (detached) in ``src``; SetupError speakers_build_failed when it can't be."""
    if _head(src, env, log) == COMMIT:
        return
    if src.exists() and not (src / ".git").exists():
        shutil.rmtree(src, ignore_errors=True)
    if _host(REPO_URL):
        LAST_HOSTS.add(_host(REPO_URL))
    ok = (src / ".git").exists() or _git(["init", "-q", str(src)], None, env, log).ok
    ok = ok and _git(["-C", str(src), "fetch", "-q", "--depth", "1", REPO_URL, COMMIT], None, env, log).ok
    ok = ok and _git(["-C", str(src), "checkout", "-q", "--detach", "FETCH_HEAD"], None, env, log).ok
    if not ok or _head(src, env, log) != COMMIT:
        # fallback: a shallow clone of the tag, accepted only if it is the pinned commit
        alt = src.with_name(src.name + "-tag")
        shutil.rmtree(alt, ignore_errors=True)
        cloned = _git(["clone", "-q", "--depth", "1", "--branch", TAG, REPO_URL, str(alt)], None, env, log).ok
        if not cloned or _head(alt, env, log) != COMMIT:
            shutil.rmtree(alt, ignore_errors=True)
            raise SetupError("speakers_build_failed", tag=TAG, path=str(log))
        shutil.rmtree(src, ignore_errors=True)
        os.replace(alt, src)


def tag_moved(env: dict, log: Path) -> bool | None:
    """Whether the tag now names another commit upstream (None when that can't be checked)."""
    if _host(REPO_URL):
        LAST_HOSTS.add(_host(REPO_URL))
    r = proc.run([GIT, "ls-remote", REPO_URL, f"refs/tags/{TAG}", f"refs/tags/{TAG}^{{}}"], env=env,
                 timeout=GIT_CAP_S, allowed=_programs(), cwd=Path(env["TMPDIR"]))
    if not r.ok:
        return None
    refs = dict(reversed(line.split("\t", 1)) for line in r.stdout.decode("ascii", "replace").splitlines()
                if "\t" in line)
    peeled = refs.get(f"refs/tags/{TAG}^{{}}") or refs.get(f"refs/tags/{TAG}")
    return None if not peeled else peeled != COMMIT


def _stale_interfaces(developer_dir: str) -> Path | None:
    pm = Path(developer_dir) / "usr" / "lib" / "swift" / "pm"
    mod = pm / "ManifestAPI" / "PackageDescription.swiftmodule"
    return pm if mod.is_dir() and any(mod.glob("*.private.swiftinterface")) else None


def _pruned_libs(pm: Path) -> Path:
    fixed = platforms.cache_dir() / "build" / "swiftpm-libs-fixed"
    shutil.rmtree(fixed, ignore_errors=True)
    fixed.mkdir(parents=True)
    for part in ("ManifestAPI", "PluginAPI"):
        if (pm / part).is_dir():
            shutil.copytree(pm / part, fixed / part, symlinks=True)
    for p in (fixed / "ManifestAPI" / "PackageDescription.swiftmodule").glob("*.private.swiftinterface"):
        p.unlink()
    return fixed


def _build(tc: dict, progress: Progress, warnings: list[Problem] | None) -> Path:
    bd = fsutil.private_dir(build_dir())
    log = build_log()
    src = bd / "src"
    env = _env(bd, Path(tc["toolchain_bin"]))
    t0 = time.monotonic()
    progress(f"speaker program (FluidAudio {TAG}): fetching its source")
    fetch_source(src, env, log)
    moved = tag_moved(env, log)
    if moved and warnings is not None:
        warnings.append(problem("fluidaudio_tag_moved", tag=TAG, commit=COMMIT[:12]))
    _prefetch(src, progress)
    progress(f"speaker program (FluidAudio {TAG}): resolving")
    proc.run([SWIFT, "package", "resolve"], env=env, timeout=RESOLVE_CAP_S, allowed=_programs(), cwd=src,
             stdout="discard", output_log=log)
    _prefetch(src, progress)
    progress(f"speaker program (FluidAudio {TAG}): building (a few minutes)")
    cmd = [SWIFT, "build", "-c", "release", "--product", PRODUCT]
    r = proc.run(cmd, env=env, timeout=BUILD_CAP_S, allowed=_programs(), cwd=src, stdout="discard", output_log=log)
    if r.timed_out:
        raise SetupError("build_timeout", minutes=int(BUILD_CAP_S // 60))
    if not r.ok and tc.get("developer_dir") and (pm := _stale_interfaces(tc["developer_dir"])) is not None:
        env2 = {**env, "SWIFTPM_CUSTOM_LIBS_DIR": str(_pruned_libs(pm))}
        r = proc.run(cmd, env=env2, timeout=BUILD_CAP_S, allowed=_programs(), cwd=src, stdout="discard",
                     output_log=log)
        if r.timed_out:
            raise SetupError("build_timeout", minutes=int(BUILD_CAP_S // 60))
    products = src / ".build" / "release"
    exe = products / PRODUCT
    if not r.ok or not exe.is_file():
        raise SetupError("speakers_build_failed", tag=TAG, path=str(log))
    return _stage(products, tc, round(time.monotonic() - t0, 1), moved)


def _prefetch(src: Path, progress: Progress) -> None:
    """``swiftpm.prefetch``, noting the hosts it contacted and the files it downloaded for setup's report."""
    from . import swiftpm

    for a in swiftpm.prefetch(src, progress=progress):
        LAST_HOSTS.update(a.get("hosts") or [])
        if a.get("downloaded") and a.get("path") and a["path"] not in LAST_DOWNLOADS:
            LAST_DOWNLOADS.append(a["path"])


def _stage(products: Path, tc: dict, seconds: float, moved: bool | None = None) -> Path:
    """Copy the binary (and BUNDLES) into ``.partial-<key>-<pid>`` with BUILD.json; returns that folder (0700)."""
    r = fsutil.private_dir(root())
    part = r / f".partial-{key()}-{os.getpid()}"
    shutil.rmtree(part, ignore_errors=True)
    fsutil.private_dir(part / "bin")
    shutil.copy2(products / PRODUCT, part / "bin" / PRODUCT)
    for b in BUNDLES:
        if b in NEVER_COPY:
            continue
        if (products / b).is_dir():
            shutil.copytree(products / b, part / "bin" / b, symlinks=True)
    meta = {"schema": SCHEMA, "tag": TAG, "commit": COMMIT, "recipe": RECIPE,
            "binary_sha256": _sha(part / "bin" / PRODUCT),
            "built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "build_seconds": seconds, "swift": tc.get("swift", ""), "developer_dir": tc.get("developer_dir"),
            "macos": platforms.info().get("macos", ""), "arch": platforms.machine(),
            "tag_moved": moved, "tag_checked_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (part / "BUILD.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return part


def _swap(part: Path) -> Path:
    final = key_dir()
    if final.exists():
        old = final.with_name(f".old-{final.name}-{os.getpid()}")
        os.replace(final, old)
        os.replace(part, final)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(part, final)
    return final / "bin" / PRODUCT


def prune_keys() -> list[str]:
    """Delete key folders other than the current one and the newest previous one; returns their names."""
    r = root()
    if not r.is_dir():
        return []
    others = []
    for d in r.iterdir():
        if d.is_dir() and not d.name.startswith(".") and d.name != key():
            try:
                built = json.loads((d / "BUILD.json").read_text(encoding="utf-8")).get("built_at", "")
            except (OSError, ValueError):
                built = ""
            others.append((built, d))
    others.sort(reverse=True)
    gone = []
    for _, d in others[1:]:
        shutil.rmtree(d, ignore_errors=True)
        gone.append(d.name)
    for d in r.glob(".partial-*"):
        if d.is_dir() and not d.name.endswith(f"-{os.getpid()}"):
            shutil.rmtree(d, ignore_errors=True)
    return gone


def smoke(binary: Path, models: Path, ffmpeg: Path) -> dict:
    """``nemotron3-diarize`` on a generated 3 s sine with ``--models`` our folder (pre-warms CoreML too)."""
    import tempfile

    from . import runtime

    with tempfile.TemporaryDirectory(prefix="audiointake-fa-") as tmp:
        t = Path(tmp)
        wav, rttm = t / "tone.16k.wav", t / "tone.rttm"
        env = runtime.run_env(t, ffmpeg)
        g = proc.run(argv.tone_wav(ffmpeg, wav, seconds=3, kind="sine"), env=env, timeout=120,
                     allowed=_programs(ffmpeg), stdout="discard")
        if not g.ok:
            return {"ok": False, "seconds": round(g.seconds, 2), "code": "ffmpeg_failed"}
        t0 = time.monotonic()
        r = proc.run(argv.diarize(binary, wav, rttm, models=models), env=env, timeout=SMOKE_CAP_S,
                     allowed=_programs(ffmpeg, binary), stdout="discard")
        sec = round(time.monotonic() - t0, 2)
        if not r.ok:
            out = {"ok": False, "seconds": sec, "code": "timeout" if r.timed_out else f"exit_{r.returncode}"}
        else:
            out = {"ok": rttm.exists(), "seconds": sec}
        LAST_SMOKE.clear()
        LAST_SMOKE.update(out)
        return out


def ensure(*, build: bool, progress: Progress | None = None, warnings: list[Problem] | None = None,
           smoke_models: Path | None = None, ffmpeg: Path | None = None) -> Path | None:
    """The keyed binary: reused when ``check()`` passes; else, with ``build``, built (lock held), smoke-tested on
    ``smoke_models`` when given, and swapped in. None without ``build`` when there's nothing to reuse. Raises
    SetupError (``clt_missing``, ``xcode_license``, ``swift_too_old``, ``speakers_build_failed``,
    ``build_timeout``).

    A build whose smoke fails is rebuilt beside it and replaced only once the new one passes its own smoke; if
    the rebuild can't be made (no network, no Command Line Tools, a failed build), the old build stays in place
    and the error is raised, so a passing moment of CoreML trouble never leaves the Mac without speaker labels."""
    say = progress or (lambda _m: None)
    info = check()
    if info is not None:
        exe = Path(info["path"])
        if smoke_models is None or ffmpeg is None or smoke(exe, smoke_models, ffmpeg)["ok"]:
            return exe
        if not build:
            return None
    if not build:
        return None
    lock.require_held()
    tc = toolchain()
    if tc["code"] is not None:
        raise SetupError(tc["code"], swift=tc.get("swift", ""))
    part = _build(tc, say, warnings)
    if smoke_models is not None and ffmpeg is not None:
        s = smoke(part / "bin" / PRODUCT, smoke_models, ffmpeg)
        if not s["ok"]:
            shutil.rmtree(part, ignore_errors=True)
            raise SetupError("speakers_build_failed", tag=TAG, path=str(build_log()))
    exe = _swap(part)
    shutil.rmtree(build_dir(), ignore_errors=True)
    prune_keys()
    return exe
