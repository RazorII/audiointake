"""platforms.py: the platform checks and their codes, machine facts without processes, the Swift helpers, folders."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from audiointake import __version__, platforms
from audiointake.errors import EXIT_SETUP, SetupError

ON_APPLE_SILICON = sys.platform == "darwin" and os.uname().machine == "arm64"


# --------------------------------------------------------------------------- the checks
@pytest.mark.host_memory
@pytest.mark.skipif(not ON_APPLE_SILICON, reason="needs a Mac with Apple silicon")
def test_this_mac_is_supported():
    info = platforms.require_supported()
    assert set(info) == {"os", "macos", "chip", "memory_gb", "translated"}
    assert info["os"] == "macOS" and info["translated"] is False
    assert isinstance(info["memory_gb"], int) and info["memory_gb"] >= 8
    assert info["chip"].startswith("Apple")
    assert platforms.macos_version() >= platforms.MIN_MACOS


def _raises(code, monkeypatch, **patches):
    with monkeypatch.context() as m:
        for name, value in patches.items():
            if name == "sys_platform":
                m.setattr(sys, "platform", value)
            else:
                m.setattr(platforms, name, value)
        with pytest.raises(SetupError) as ei:
            platforms.require_supported()
    assert ei.value.code == code and ei.value.exit == EXIT_SETUP
    return ei.value


def test_not_macos(monkeypatch):
    e = _raises("platform_not_macos", monkeypatch, sys_platform="linux")
    assert e.detail["os"] == "Linux" and "Linux" in e.message


def test_intel_mac(monkeypatch):
    _raises("platform_not_apple_silicon", monkeypatch, sys_platform="darwin", machine=lambda: "x86_64",
            translated=lambda: False)


def test_rosetta(monkeypatch):
    e = _raises("platform_rosetta", monkeypatch, sys_platform="darwin", machine=lambda: "x86_64",
                translated=lambda: True)
    assert f"v{__version__}.tar.gz" in e.message


def test_macos_too_old(monkeypatch):
    class FakePlatform:
        @staticmethod
        def mac_ver():
            return ("13.6.1", ("", "", ""), "arm64")
    e = _raises("macos_too_old", monkeypatch, sys_platform="darwin", machine=lambda: "arm64",
                translated=lambda: False, _platform=FakePlatform)
    assert e.detail["macos"] == "13.6.1" and "macOS 13.6.1" in e.message


def test_unreadable_macos_version_doesnt_refuse(monkeypatch):
    class FakePlatform:
        @staticmethod
        def mac_ver():
            return ("", ("", "", ""), "")
    with monkeypatch.context() as m:
        m.setattr(sys, "platform", "darwin")
        m.setattr(platforms, "machine", lambda: "arm64")
        m.setattr(platforms, "translated", lambda: False)
        m.setattr(platforms, "_platform", FakePlatform)
        assert platforms.require_supported()["macos"] == ""


@pytest.mark.parametrize("gb,warn", [(8, True), (16, True), (18, False), (24, False), (64, False), (None, False)])
def test_memory_warning(monkeypatch, gb, warn):
    monkeypatch.setattr(platforms, "memory_gb", lambda: gb)
    got = platforms.warnings()
    if warn:
        assert len(got) == 1 and got[0].code == "memory_unmeasured" and got[0].params == {"gb": gb}
    else:
        assert got == []


# --------------------------------------------------------------------------- machine facts
@pytest.mark.skipif(sys.platform != "darwin", reason="sysctl is macOS's")
def test_sysctl_reads_without_a_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(subprocess, "Popen", boom)
    assert platforms.memory_bytes() > 2**30
    assert platforms.translated() in (True, False)
    assert isinstance(platforms.chip(), str)
    assert platforms._sysctl("no.such.sysctl.name", "int") is None
    assert platforms._sysctl("no.such.sysctl.name", "str") is None


@pytest.mark.skipif(not ON_APPLE_SILICON, reason="needs a native Apple-silicon Python")
def test_native_facts():
    assert platforms.machine() == "arm64" and platforms.translated() is False


def test_off_macos_facts_are_empty(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert platforms._sysctl("hw.memsize", "int64") is None
    assert platforms.macos_version() is None
    assert platforms.os_name() == "Linux"


@pytest.mark.parametrize("text,want", [("27.2.1", (27, 2)), ("14", (14, 0)), ("10.16", (10, 16)), ("", None),
                                       ("x", None)])
def test_parse_version(text, want):
    assert platforms.parse_version(text) == want


def test_free_bytes(tmp_path):
    assert platforms.free_bytes(tmp_path) > 0
    assert platforms.free_bytes(tmp_path / "not" / "yet") == platforms.free_bytes(tmp_path)


# --------------------------------------------------------------------------- the Swift toolchain helpers
def _exe(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


def test_swift_binary_command_line_tools_layout(tmp_path):
    swift = _exe(tmp_path / "CommandLineTools" / "usr" / "bin" / "swift")
    assert platforms.swift_binary(tmp_path / "CommandLineTools") == swift


def test_swift_binary_xcode_layout(tmp_path):
    dev = tmp_path / "Xcode.app" / "Contents" / "Developer"
    swift = _exe(dev / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin" / "swift")
    (dev / "usr" / "bin").mkdir(parents=True)          # Xcode's own usr/bin has no swift
    assert platforms.swift_binary(dev) == swift


def test_swift_binary_missing(tmp_path):
    assert platforms.swift_binary(tmp_path) is None
    (tmp_path / "usr" / "bin").mkdir(parents=True)
    (tmp_path / "usr" / "bin" / "swift").write_text("not executable")
    assert platforms.swift_binary(tmp_path) is None


def test_developer_dir(tmp_path):
    assert platforms.developer_dir(f"{tmp_path}\n", 0) == tmp_path
    assert platforms.developer_dir(f"{tmp_path}\n", 2) is None
    assert platforms.developer_dir("", 0) is None
    assert platforms.developer_dir(f"{tmp_path / 'gone'}\n", 0) is None
    assert platforms.developer_dir("relative/dir\n", 0) is None


SWIFT_642 = ("swift-driver version: 1.168.6 Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)\n"
             "Target: arm64-apple-macosx27.2.0\n")
LICENCE = ("You have not agreed to the Xcode license agreements. Please run 'sudo xcodebuild -license' from within "
           "a Terminal window to review and agree to the Xcode and Apple SDKs license.\n")


@pytest.mark.parametrize("rc,out,want", [
    (0, SWIFT_642, (None, "6.4")),
    (0, "Apple Swift version 6.0 (swiftlang-6.0.0.9.10 clang-1600.0.26.2)\n", (None, "6.0")),
    (0, "Apple Swift version 5.10 (swiftlang-5.10.0.13 clang-1500.3.9.4)\n", ("swift_too_old", "5.10")),
    (69, LICENCE, ("xcode_license", "")),
    (1, LICENCE, ("xcode_license", "")),
    (69, "", ("xcode_license", "")),
    (1, "xcrun: error: invalid active developer path\n", ("clt_missing", "")),
    (0, "something else\n", ("clt_missing", "")),
])
def test_swift_problem(rc, out, want):
    assert platforms.swift_problem(rc, out) == want


@pytest.mark.skipif(not Path("/Library/Developer/CommandLineTools/usr/bin/swift").exists()
                    and not Path("/Applications/Xcode.app").exists(), reason="no Swift toolchain here")
def test_swift_binary_on_this_mac():
    for d in (Path("/Library/Developer/CommandLineTools"), Path("/Applications/Xcode.app/Contents/Developer")):
        if d.is_dir():
            assert platforms.swift_binary(d) is not None


# --------------------------------------------------------------------------- folders
def test_folders_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.delenv("AUDIOINTAKE_HOME", raising=False)
    monkeypatch.delenv("AUDIOINTAKE_CACHE", raising=False)
    assert platforms.data_dir() == tmp_path / "h" / "Library" / "Application Support" / "audiointake"
    assert platforms.cache_dir() == tmp_path / "h" / "Library" / "Caches" / "audiointake"
    assert platforms.logs_dir() == tmp_path / "h" / "Library" / "Logs" / "audiointake"
    monkeypatch.setenv("AUDIOINTAKE_HOME", "~/elsewhere")
    monkeypatch.setenv("AUDIOINTAKE_CACHE", str(tmp_path / "c"))
    assert platforms.data_dir() == tmp_path / "h" / "elsewhere"
    assert platforms.cache_dir() == tmp_path / "c"
    assert not (tmp_path / "h").exists()                 # nothing is created


def test_the_test_session_uses_scratch_folders():
    real = Path(os.path.expanduser("~"))
    for f in (platforms.data_dir, platforms.cache_dir, platforms.logs_dir, platforms.hf_cache):
        assert "audiointake-test-" in str(f()), f.__name__
    assert real == Path(os.environ["HOME"])


HF_CASES = [
    {},
    {"HF_HOME": "{t}/hfhome"},
    {"XDG_CACHE_HOME": "{t}/xdg"},
    {"XDG_CACHE_HOME": "{t}/xdg", "HF_HOME": "{t}/hfhome"},
    {"HUGGINGFACE_HUB_CACHE": "{t}/legacy"},
    {"HUGGINGFACE_HUB_CACHE": "{t}/legacy", "HF_HUB_CACHE": "{t}/hub"},
    {"HF_HUB_CACHE": "{t}/hub", "HF_HOME": "{t}/hfhome"},
    {"HF_HUB_CACHE": "~/tilde"},
]


def _hf_env(case: dict[str, str], tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")}
    env["HOME"] = str(tmp_path / "home")
    env.update({k: v.format(t=tmp_path) for k, v in case.items()})
    return env


@pytest.mark.parametrize("case", HF_CASES)
def test_hf_cache_rules(monkeypatch, tmp_path, case):
    for k, v in _hf_env(case, tmp_path).items():
        monkeypatch.setenv(k, v)
    for k in ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        if k not in case:
            monkeypatch.delenv(k, raising=False)
    got = platforms.hf_cache()
    home = tmp_path / "home"
    want = {
        0: home / ".cache" / "huggingface" / "hub", 1: tmp_path / "hfhome" / "hub",
        2: tmp_path / "xdg" / "huggingface" / "hub", 3: tmp_path / "hfhome" / "hub", 4: tmp_path / "legacy",
        5: tmp_path / "hub", 6: tmp_path / "hub", 7: home / "tilde",
    }[HF_CASES.index(case)]
    assert got == want


@pytest.mark.skipif(importlib.util.find_spec("huggingface_hub") is None, reason="huggingface_hub isn't installed")
@pytest.mark.parametrize("case", HF_CASES)
def test_hf_cache_agrees_with_huggingface_hub(tmp_path, case):
    env = _hf_env(case, tmp_path)
    code = ("import json\nfrom huggingface_hub import constants\n"
            "from audiointake import platforms\n"
            "print(json.dumps([constants.HF_HUB_CACHE, str(platforms.hf_cache())]))")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    theirs, ours = json.loads(r.stdout)
    assert os.path.realpath(os.path.expanduser(theirs)) == os.path.realpath(ours)
