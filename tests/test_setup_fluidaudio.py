"""The speaker program's keyed build: reuse, pin and recipe bumps, tampering, a moved tag, caps, resume, the
toolchain checks. A local git repository stands in for FluidAudio and a fake ``swift`` for the compiler."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from audiointake import lock, platforms
from audiointake.errors import SetupError
from audiointake.setup import fluidaudio as fa

GIT = "/usr/bin/git"


def _git(*args, cwd=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run([GIT, *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\nimport os, sys, time\n{body}\n")
    path.chmod(0o755)
    return path


FAKE_SWIFT = r'''
import json
here = os.path.dirname(os.path.abspath(__file__))
try:
    ctl = json.load(open(os.path.join(here, "control.json")))     # the setup environment is scrubbed: a file steers us
except OSError:
    ctl = {}
args = sys.argv[1:]
open(os.path.join(here, "swift.log"), "a").write(" ".join(args) + "\n")
if args == ["--version"]:
    print(ctl.get("version", "swift-driver version: 1.0 Apple Swift version 6.4.2 (x)"))
    sys.exit(int(ctl.get("version_exit", 0)))
if args[:2] == ["package", "resolve"]:
    sys.exit(0)
if args[:1] == ["build"]:
    if ctl.get("sleep"):
        time.sleep(float(ctl["sleep"]))
    if ctl.get("needs_libs") and not os.environ.get("SWIFTPM_CUSTOM_LIBS_DIR"):
        print("Invalid manifest", file=sys.stderr)
        sys.exit(1)
    out = os.path.join(".build", "release")
    os.makedirs(out, exist_ok=True)
    version = open("VERSION").read().strip()
    p = os.path.join(out, "fluidaudiocli")
    open(p, "w").write("#!/bin/sh\n# built from " + version + "\n" + 'while [ $# -gt 0 ]; do if [ "$1" = --output ]; then shift; echo "SPEAKER x 1 0 1 <NA> <NA> speaker_0 <NA> <NA>" > "$1"; fi; shift; done\n')
    os.chmod(p, 0o755)
    os.makedirs(os.path.join(out, "FluidAudio_LuxTtsG2pResources.bundle"), exist_ok=True)
    sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    for name, sub in (("AUDIOINTAKE_HOME", "data"), ("AUDIOINTAKE_CACHE", "cache")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(name, str(tmp_path / sub))
    up = tmp_path / "up"
    up.mkdir()
    _git("init", "-q", str(up))
    (up / "VERSION").write_text("pinned\n")
    (up / "Package.swift").write_text("// no binary targets\n")
    _git("add", ".", cwd=up)
    _git("commit", "-qm", "pinned", cwd=up)
    pinned = _git("rev-parse", "HEAD", cwd=up)
    _git("tag", "-a", "v0.18.0", "-m", "t", cwd=up)
    dev = tmp_path / "dev"
    swift = _script(dev / "usr" / "bin" / "swift", FAKE_SWIFT)
    xs = _script(tmp_path / "bin" / "xcode-select", f"print({str(dev)!r})")
    monkeypatch.setattr(fa, "REPO_URL", f"file://{up}")
    monkeypatch.setattr(fa, "GIT_PROTOCOLS", "file")            # the stand-in upstream is a local repository
    monkeypatch.setattr(fa, "COMMIT", pinned)
    monkeypatch.setattr(fa, "SWIFT", swift)
    monkeypatch.setattr(fa, "XCODE_SELECT", xs)
    ffmpeg = _script(tmp_path / "bin" / "ffmpeg", "open(sys.argv[-1], 'wb').write(b'RIFF')")
    w = type("W", (), {})()
    w.up, w.pinned, w.dev, w.swift, w.ffmpeg, w.tmp = up, pinned, dev, swift, ffmpeg, tmp_path
    w.log = swift.parent / "swift.log"
    w.ctl = lambda **kw: (swift.parent / "control.json").write_text(json.dumps(kw))
    return w


def _ensure(**kw):
    with lock.run_lock():
        return fa.ensure(build=True, **kw)


def test_nothing_built_means_check_is_none(world):
    assert fa.check() is None and fa.ensure(build=False) is None


def test_build_then_reuse(world):
    warns = []
    exe = _ensure(warnings=warns)
    assert exe == fa.key_dir() / "bin" / "fluidaudiocli" and "built from pinned" in exe.read_text()
    meta = json.loads((fa.key_dir() / "BUILD.json").read_text())
    assert (meta["tag"], meta["commit"], meta["recipe"]) == (fa.TAG, world.pinned, fa.RECIPE)
    assert meta["binary_sha256"] == hashlib.sha256(exe.read_bytes()).hexdigest()
    assert meta["swift"] == "6.4" and meta["developer_dir"] == str(world.dev)
    assert not (exe.parent / "FluidAudio_LuxTtsG2pResources.bundle").exists()        # never copied
    assert not fa.build_dir().exists() and warns == []                                # the build folder went
    builds = world.log.read_text().count("build -c release")
    assert _ensure() == exe and fa.check()["path"] == str(exe)
    assert world.log.read_text().count("build -c release") == builds  # reused, not rebuilt


def test_a_pin_bump_reports_no_build_and_builds_a_new_key(world, monkeypatch):
    _ensure()
    old = fa.key_dir()
    monkeypatch.setattr(fa, "TAG", "v0.18.3")
    assert fa.check() is None and fa.key_dir() != old
    assert fa.key_dir().name.startswith("v0.18.3-")
    builds = world.log.read_text().count("build -c release")
    exe = _ensure()
    assert world.log.read_text().count("build -c release") == builds + 1        # built for the new key
    assert exe.parent.parent == fa.key_dir() and fa.check()["tag"] == "v0.18.3"
    assert old.exists()                                                           # the previous key stays


def test_a_recipe_bump_makes_a_new_key_and_keeps_the_previous_one(world, monkeypatch):
    first = _ensure()
    monkeypatch.setattr(fa, "RECIPE", fa.RECIPE + 1)
    assert fa.check() is None
    second = _ensure()
    assert second != first and second.parent.parent.name.endswith(f"-r{fa.RECIPE}")
    assert first.exists()                                  # the previous key stays for an instant rollback
    monkeypatch.setattr(fa, "RECIPE", fa.RECIPE + 1)
    _ensure()
    assert not first.exists()                              # only the newest previous one is kept


def test_a_tampered_binary_is_caught(world):
    exe = _ensure()
    exe.write_text(exe.read_text() + "# changed\n")
    assert fa.check() is None


def test_a_moved_tag_warns_and_the_pinned_commit_is_built(world):
    (world.up / "VERSION").write_text("moved\n")
    _git("commit", "-qam", "moved", cwd=world.up)
    _git("tag", "-f", "-a", "v0.18.0", "-m", "moved", cwd=world.up)
    warns = []
    exe = _ensure(warnings=warns)
    assert [w.code for w in warns] == ["fluidaudio_tag_moved"]
    assert warns[0].params["commit"] == world.pinned[:12]
    assert "built from pinned" in exe.read_text()
    meta = json.loads((fa.key_dir() / "BUILD.json").read_text())
    assert meta["commit"] == world.pinned and meta["tag_moved"] is True       # kept, so doctor can say so offline
    assert fa.check()["tag_moved"] is True


def test_a_build_cap_is_build_timeout_and_the_next_run_resumes(world, monkeypatch):
    monkeypatch.setattr(fa, "BUILD_CAP_S", 1.0)
    world.ctl(sleep=30)
    with pytest.raises(SetupError) as e:
        _ensure()
    assert e.value.code == "build_timeout" and e.value.exit == 3
    src = fa.build_dir() / "src"
    assert (src / ".git").exists()                          # the build folder stays for the next run
    world.ctl()
    monkeypatch.setattr(fa, "BUILD_CAP_S", 60.0)
    monkeypatch.setattr(fa, "REPO_URL", "file:///nonexistent/repo")   # a refetch would fail: the source is reused
    assert _ensure().exists()


def test_a_failed_build_is_speakers_build_failed_with_its_log(world, monkeypatch):
    world.ctl(needs_libs=True)
    with pytest.raises(SetupError) as e:
        _ensure()
    assert e.value.code == "speakers_build_failed" and Path(e.value.detail["path"]).is_file()
    assert "Invalid manifest" in Path(e.value.detail["path"]).read_text()


def test_stale_private_interfaces_retry_with_pruned_manifest_libs(world, monkeypatch):
    mod = world.dev / "usr" / "lib" / "swift" / "pm" / "ManifestAPI" / "PackageDescription.swiftmodule"
    mod.mkdir(parents=True)
    (mod / "arm64.private.swiftinterface").write_text("stale")
    (mod / "arm64.swiftinterface").write_text("ok")
    world.ctl(needs_libs=True)
    assert _ensure().exists()
    fixed = platforms.cache_dir() / "build" / "swiftpm-libs-fixed" / "ManifestAPI" / "PackageDescription.swiftmodule"
    assert (fixed / "arm64.swiftinterface").exists() and not (fixed / "arm64.private.swiftinterface").exists()


def test_building_needs_the_lock(world):
    with pytest.raises(RuntimeError, match="run_lock"):
        fa.ensure(build=True)


def test_no_command_line_tools_never_runs_swift(world, monkeypatch, tmp_path):
    xs = _script(tmp_path / "bin2" / "xcode-select", "print('xcode-select: error', file=sys.stderr); sys.exit(2)")
    monkeypatch.setattr(fa, "XCODE_SELECT", xs)
    with pytest.raises(SetupError) as e:
        _ensure()
    assert e.value.code == "clt_missing"
    assert not world.log.exists()
    empty = tmp_path / "empty-dev"
    empty.mkdir()
    monkeypatch.setattr(fa, "XCODE_SELECT", _script(tmp_path / "bin3" / "xcode-select", f"print({str(empty)!r})"))
    assert fa.toolchain()["code"] == "clt_missing" and not world.log.exists()


def test_xcode_licence_and_old_swift_are_detected(world, monkeypatch):
    world.ctl(version_exit=69, version="You have not agreed to the Xcode license agreements.")
    with pytest.raises(SetupError) as e:
        _ensure()
    assert e.value.code == "xcode_license"
    world.ctl(version="Apple Swift version 5.10 (x)")
    assert fa.toolchain()["code"] == "swift_too_old"


def test_a_failed_smoke_rebuilds_and_a_passing_one_is_recorded(world):
    exe = _ensure(smoke_models=world.tmp, ffmpeg=world.ffmpeg)
    assert fa.LAST_SMOKE["ok"] is True
    exe.write_text("#!/bin/sh\nexit 1\n")
    meta = json.loads((fa.key_dir() / "BUILD.json").read_text())
    meta["binary_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()      # passes check(), fails its smoke
    (fa.key_dir() / "BUILD.json").write_text(json.dumps(meta))
    again = _ensure(smoke_models=world.tmp, ffmpeg=world.ffmpeg)
    assert "built from pinned" in again.read_text() and fa.LAST_SMOKE["ok"] is True


def test_the_shipped_pin():
    assert fa.FLUIDAUDIO_PIN == ("v0.18.0", "e15287985625dbf544a343df679436789ffb03fb")
    assert fa.key() == "v0.18.0-e15287985625-r1" and fa.BUNDLES == ()
    assert fa.GIT == Path("/usr/bin/git") and fa.SWIFT == Path("/usr/bin/swift")


def test_a_failed_smoke_keeps_the_old_build_when_the_rebuild_cant_be_made(world, monkeypatch, tmp_path):
    exe = _ensure(smoke_models=world.tmp, ffmpeg=world.ffmpeg)
    exe.write_text("#!/bin/sh\nexit 1\n")
    meta = json.loads((fa.key_dir() / "BUILD.json").read_text())
    meta["binary_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    (fa.key_dir() / "BUILD.json").write_text(json.dumps(meta))
    monkeypatch.setattr(fa, "XCODE_SELECT", _script(tmp_path / "noclt" / "xcode-select", "sys.exit(2)"))
    with pytest.raises(SetupError) as e:
        _ensure(smoke_models=world.tmp, ffmpeg=world.ffmpeg)
    assert e.value.code == "clt_missing"
    assert fa.check() is not None and exe.exists()           # still there: runs keep their speaker labels


def test_a_build_reports_the_hosts_it_contacted(world, monkeypatch):
    fa.LAST_HOSTS.clear()
    monkeypatch.setattr(fa, "REPO_URL", "https://example.invalid/FluidInference/FluidAudio.git")
    assert fa._host(fa.REPO_URL) == "example.invalid"
    monkeypatch.setattr(fa, "REPO_URL", f"file://{world.up}")
    assert fa._host(fa.REPO_URL) == ""
    _ensure()
    assert fa.LAST_HOSTS == set()                             # a local repository: no host


def test_git_ignores_the_users_git_configuration(world, tmp_path, monkeypatch):
    """A global insteadOf to SSH (or any header or helper) in ~/.gitconfig or the XDG file never applies."""
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    rules = '[url "git@github.com:"]\n\tinsteadOf = https://github.com/\n[http]\n\textraHeader = X-Test: 1\n'
    (home / ".gitconfig").write_text(rules)
    (home / ".config" / "git" / "config").write_text(rules)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    env = fa._env(tmp_path)
    url = "https://github.com/FluidInference/FluidAudio.git"
    r = subprocess.run([GIT, "ls-remote", "--get-url", url], env=env, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == url
    shown = subprocess.run([GIT, "config", "--list", "--show-origin"], env=env, capture_output=True, text=True)
    assert "insteadof" not in shown.stdout.lower() and "extraheader" not in shown.stdout.lower()
    assert env["GIT_ALLOW_PROTOCOL"] == "file" and env["GIT_SSH_COMMAND"] == "/usr/bin/false"
    assert env["GIT_CEILING_DIRECTORIES"] == str(tmp_path.parent)
    hostile = {**{k: v for k, v in env.items() if not k.startswith("GIT_CONFIG")}}
    r2 = subprocess.run([GIT, "ls-remote", "--get-url", url], env=hostile, capture_output=True, text=True)
    assert r2.stdout.strip() != url                           # the check would catch a regression


def test_the_shipped_git_protocols_are_https_only():
    assert 'GIT_PROTOCOLS = "https"' in Path(fa.__file__).read_text(encoding="utf-8")
