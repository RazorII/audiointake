"""fsutil.py: the atomic no-clobber writer and its no-hard-links fallback, hashes, private folders, stems."""
from __future__ import annotations

import errno
import hashlib
import io
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from audiointake import fsutil
from audiointake.errors import EXIT_USAGE, UsageError


def partials(folder: Path) -> list[str]:
    return [p.name for p in folder.iterdir() if p.name.startswith(fsutil.PARTIAL_PREFIX)]


# --------------------------------------------------------------------------- atomic_write
def test_writes_a_new_file(tmp_path):
    out = tmp_path / "Team sync (transcript).md"
    fsutil.atomic_write(out, "héllo\n".encode(), overwrite=False)
    assert out.read_bytes() == "héllo\n".encode()
    assert partials(tmp_path) == []
    assert fsutil._LIVE_PARTIALS == set()


def test_never_replaces_without_overwrite(tmp_path):
    out = tmp_path / "a.txt"
    out.write_bytes(b"keep me")
    with pytest.raises(UsageError) as ei:
        fsutil.atomic_write(out, b"new", overwrite=False)
    assert ei.value.code == "output_exists" and ei.value.exit == EXIT_USAGE
    assert ei.value.detail["out"] == "a.txt"
    assert out.read_bytes() == b"keep me"
    assert partials(tmp_path) == []


def test_a_dangling_link_counts_as_existing(tmp_path):
    out = tmp_path / "a.txt"
    out.symlink_to(tmp_path / "nowhere")
    with pytest.raises(UsageError):
        fsutil.atomic_write(out, b"new", overwrite=False)
    assert out.is_symlink()


def test_overwrite_replaces(tmp_path):
    out = tmp_path / "a.txt"
    out.write_bytes(b"old")
    ino = out.stat().st_ino
    fsutil.atomic_write(out, b"new", overwrite=True)
    assert out.read_bytes() == b"new"
    assert out.stat().st_ino != ino                  # a new file swapped in, never the old one rewritten
    assert partials(tmp_path) == []


def test_mode_is_exact_whatever_the_umask(tmp_path):
    old = os.umask(0o077)
    try:
        fsutil.atomic_write(tmp_path / "v.json", b"{}", overwrite=False, mode=0o600)
        fsutil.atomic_write(tmp_path / "w.json", b"{}", overwrite=False, mode=0o644)
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "v.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "w.json").stat().st_mode) == 0o644


def test_default_mode_follows_the_umask(tmp_path):
    old = os.umask(0o022)
    try:
        fsutil.atomic_write(tmp_path / "a.md", b"x", overwrite=False)
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "a.md").stat().st_mode) == 0o644


def test_takes_bytes_only(tmp_path):
    with pytest.raises(TypeError):
        fsutil.atomic_write(tmp_path / "a", "text", overwrite=False)     # type: ignore[arg-type]


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fsutil.atomic_write(tmp_path / "no" / "a.txt", b"x", overwrite=False)


def test_a_failed_write_leaves_nothing(tmp_path, monkeypatch):
    def broken(fd, data):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(fsutil, "_write_all", broken)
    with pytest.raises(OSError):
        fsutil.atomic_write(tmp_path / "a.txt", b"x" * 10, overwrite=False)
    assert list(tmp_path.iterdir()) == []
    assert fsutil._LIVE_PARTIALS == set()


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP, errno.EXDEV])
def test_no_hard_links_fallback_creates_exclusively(tmp_path, monkeypatch, code):
    def no_link(src, dst):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(fsutil.os, "link", no_link)
    out = tmp_path / "a.txt"
    fsutil.atomic_write(out, b"fresh", overwrite=False, mode=0o600)
    assert out.read_bytes() == b"fresh" and stat.S_IMODE(out.stat().st_mode) == 0o600
    assert partials(tmp_path) == []
    with pytest.raises(UsageError):                  # and still never clobbers
        fsutil.atomic_write(out, b"again", overwrite=False)
    assert out.read_bytes() == b"fresh"


def test_fallback_loses_a_race_without_clobbering(tmp_path, monkeypatch):
    out = tmp_path / "a.txt"

    def no_link(src, dst):
        Path(dst).write_bytes(b"theirs")             # another writer wins between the check and the link
        raise OSError(errno.EPERM, "Operation not permitted")
    monkeypatch.setattr(fsutil.os, "link", no_link)
    with pytest.raises(UsageError):
        fsutil.atomic_write(out, b"mine", overwrite=False)
    assert out.read_bytes() == b"theirs"
    assert partials(tmp_path) == []


def test_other_link_errors_propagate(tmp_path, monkeypatch):
    def bad(src, dst):
        raise OSError(errno.EIO, "I/O error")
    monkeypatch.setattr(fsutil.os, "link", bad)
    with pytest.raises(OSError) as ei:
        fsutil.atomic_write(tmp_path / "a.txt", b"x", overwrite=False)
    assert ei.value.errno == errno.EIO
    assert list(tmp_path.iterdir()) == []


def test_remove_live_partials(tmp_path):
    p = tmp_path / (fsutil.PARTIAL_PREFIX + "abc")
    p.write_bytes(b"half")
    fsutil._LIVE_PARTIALS.add(p)
    try:
        assert fsutil.remove_live_partials() == 1
        assert not p.exists()
        assert fsutil.remove_live_partials() == 0
    finally:
        fsutil._LIVE_PARTIALS.discard(p)


def test_partial_is_removed_when_the_signal_path_runs_mid_write(tmp_path, monkeypatch):
    seen = {}
    real_fsync = fsutil._fsync_fd

    def interrupt(fd):
        real_fsync(fd)
        seen["removed"] = fsutil.remove_live_partials()
        raise KeyboardInterrupt
    monkeypatch.setattr(fsutil, "_fsync_fd", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fsutil.atomic_write(tmp_path / "a.txt", b"x", overwrite=False)
    assert seen["removed"] == 1
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- probes and hashes
def test_probe_writable(tmp_path):
    assert fsutil.probe_writable(tmp_path) is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
def test_probe_writable_read_only(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        assert fsutil.probe_writable(ro) is False
    finally:
        ro.chmod(0o700)


def test_probe_writable_missing_folder(tmp_path):
    assert fsutil.probe_writable(tmp_path / "missing") is False


def test_sha256(tmp_path):
    data = os.urandom(3 * (1 << 20) + 17)
    f = tmp_path / "blob"
    f.write_bytes(data)
    assert fsutil.sha256_file(f) == hashlib.sha256(data).hexdigest() == fsutil.sha256_bytes(data)


# --------------------------------------------------------------------------- private folders
def test_private_dir_creates_0700_with_parents(tmp_path):
    d = fsutil.private_dir(tmp_path / "a" / "b" / "c")
    for p in (tmp_path / "a", tmp_path / "a" / "b", d):
        assert stat.S_IMODE(p.stat().st_mode) == 0o700


def test_private_dir_fixes_an_existing_folder_but_not_its_parents(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    d = parent / "data"
    d.mkdir()
    d.chmod(0o777)
    assert fsutil.private_dir(d) == d
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_private_dir_refuses_a_file(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        fsutil.private_dir(f)


# --------------------------------------------------------------------------- placeholders, stdio, stems
def test_is_dataless(tmp_path, monkeypatch):
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x")
    assert fsutil.is_dataless(f) is False
    real = os.stat
    fake = types.SimpleNamespace(st_flags=fsutil.SF_DATALESS | 0x20)
    monkeypatch.setattr(fsutil.os, "stat", lambda p, *a, **k: fake if Path(p) == f else real(p, *a, **k))
    assert fsutil.is_dataless(f) is True


def test_is_dataless_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fsutil.is_dataless(tmp_path / "nope")


def test_utf8_stdio(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    err = io.TextIOWrapper(io.BytesIO(), encoding="latin-1")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    fsutil.utf8_stdio()
    assert out.encoding.lower().replace("-", "") == "utf8" and err.encoding.lower().replace("-", "") == "utf8"
    print("Ünïcødé ✓ 会议")                              # would raise under ascii


@pytest.mark.parametrize("name,want", [
    ("Team sync.m4a", "Team sync"), ("a.b.c", "a.b"), ("noext", "noext"), (".hidden", ""), ("file.", "file"),
    ("2026-11-16 09.30 call.qta", "2026-11-16 09.30 call"), ("über.Opus", "über"), ("", ""),
])
def test_stem_is_the_shell_rule(name, want):
    assert fsutil.stem(name) == want
