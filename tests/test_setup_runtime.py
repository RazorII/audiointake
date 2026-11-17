"""``runtime.resolve()``: the verified ledger turned into a Runtime, offline, with the contract's meaning of need,
speakers, names and work_root."""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from audiointake import lock, platforms
from audiointake.contracts import ROUTES, Runtime
from audiointake.errors import SetupError
from audiointake.setup import fluidaudio, runtime, store

from test_setup_store import _children_offline, install, installed  # noqa: F401  (fixtures)


def _entry(pid: str, key: str, suffix: str) -> dict:
    return next(e for e in store.read_stamp(pid)["models"][key]["files"] if e["path"].endswith(suffix))


def test_not_set_up_is_exit_3_with_the_download_size(installed, network_attempts):
    with pytest.raises(SetupError) as e:
        runtime.resolve()
    assert e.value.code == "not_set_up" and e.value.exit == 3 and "size" in e.value.detail
    assert not network_attempts


def test_a_set_up_mac_resolves_offline(installed, network_attempts):
    install()
    rt, probs = runtime.resolve()
    assert isinstance(rt, Runtime) and probs == []
    assert rt.routes == frozenset({"hu_vote", "thai_screen", "thai_turns"})
    assert set(rt.models) == {"ultra", "v3", "nemotron", "campplus", "whisper", "typhoon"}
    assert rt.models["ultra"].path == store.snapshot_dir("test/ultra", "1" * 40)
    assert rt.models["ultra"].revision == "1" * 40 and rt.models["ultra"].tag == "ultra-tag"
    assert rt.models["typhoon"].path == store.models_dir() / "ty-5555555"
    assert rt.tools.fluidaudiocli == installed.fa and rt.tools.fluidaudio_tag == fluidaudio.TAG
    assert rt.tools.ffmpeg == installed.ffmpeg and rt.tools.ffprobe is None
    assert rt.tools.parakeet_cli.name == "parakeet-mlx" and rt.tools.whisper_cli.name == "mlx_whisper"
    env = rt.env
    assert env["HF_HUB_OFFLINE"] == "1" and env["TMPDIR"] == str(rt.work_root)
    assert env["PATH"].split(":")[1] == str(installed.ffmpeg.parent)
    assert not any(k.startswith(("PARAKEET_", "THAI", "WHISPER_", "SPEAKER_ID_")) or "TOKEN" in k for k in env)
    assert stat.S_IMODE(rt.work_root.stat().st_mode) == 0o700
    assert rt.work_root.parent == platforms.cache_dir() / "work"
    assert not network_attempts


def test_a_broken_core_file_is_pack_broken(installed):
    install()
    e = _entry("core", "v3", "model.safetensors")
    blob = Path(e["path"]).resolve()
    blob.write_bytes(b"Q" * blob.stat().st_size)
    with pytest.raises(SetupError) as err:
        runtime.resolve()
    assert err.value.code == "pack_broken" and err.value.detail["pack"] == "core"


def test_a_broken_th_file_turns_the_th_routes_off_with_a_warning(installed):
    install()
    out = store.models_dir() / "ty-5555555" / "weights.safetensors"
    out.write_bytes(b"#" * out.stat().st_size)
    rt, probs = runtime.resolve()
    assert "thai_turns" not in rt.routes and "typhoon" not in rt.models
    assert {"hu_vote", "thai_screen"} <= rt.routes                          # the hu pack still gives these
    [p] = [p for p in probs if p.code == "pack_broken"]
    assert p.params["pack"] == "th" and p.params["language"] == "Thai"


def test_a_moved_inode_with_the_same_content_rehashes_once_then_never(installed, monkeypatch):
    install()
    e = _entry("core", "ultra", "model.safetensors")
    blob = Path(e["path"]).resolve()
    data = blob.read_bytes()
    tmp = blob.with_name("moved")
    tmp.write_bytes(data)
    os.replace(tmp, blob)                                # what a restore or a migration does
    calls = []
    real = store._hash
    monkeypatch.setattr(store, "_hash", lambda p, g: calls.append(Path(p).name) or real(p, g))
    with lock.run_lock():
        rt, probs = runtime.resolve()
    assert probs == [] and len(calls) == 1
    assert _entry("core", "ultra", "model.safetensors")["ino"] == blob.stat().st_ino    # the ledger was refreshed
    with lock.run_lock():
        runtime.resolve()
    assert len(calls) == 1


def test_without_the_lock_a_refresh_isnt_saved(installed, monkeypatch):
    install()
    e = _entry("core", "ultra", "model.safetensors")
    blob = Path(e["path"]).resolve()
    t = time.time() - 50
    os.utime(blob, (t, t))
    runtime.resolve()
    assert _entry("core", "ultra", "model.safetensors")["mtime_ns"] == e["mtime_ns"]


def test_a_needed_route_that_isnt_installed_is_a_warning_and_an_offer_never_a_download(installed, network_attempts,
                                                                                     monkeypatch):
    install(("hu",))
    called = []
    monkeypatch.setattr(store, "fetch_file", lambda *a, **k: called.append(a))
    rt, probs = runtime.resolve(need=frozenset({"thai_turns", "hu_vote"}))
    assert "thai_turns" not in rt.routes and "hu_vote" in rt.routes
    codes = [(p.code, p.params.get("reason"), p.params.get("code")) for p in probs]
    assert codes == [("pack_not_installed", "asked", "th"), ("language", None, "th")]
    assert probs[0].params["language"] == "Thai" and isinstance(probs[0].params["size"], float)
    assert not called and not network_attempts


def test_need_names_routes_only(installed):
    install()
    with pytest.raises(ValueError):
        runtime.resolve(need=frozenset({"thai"}))
    assert set(ROUTES) >= {"hu_vote"}


def test_speakers_off_offers_nothing_and_keeps_the_diarizer_for_thai(installed, monkeypatch, tmp_path):
    install()
    monkeypatch.setattr(fluidaudio, "check", lambda: None)
    rt, probs = runtime.resolve(speakers=False)
    assert rt.tools.fluidaudiocli is None and rt.voices_dir is None and probs == []
    built = tmp_path / "fluidaudiocli"
    monkeypatch.setattr(fluidaudio, "check", lambda: {"path": str(built)})
    rt, probs = runtime.resolve(speakers=False)
    assert rt.tools.fluidaudiocli == built and rt.voices_dir is None and probs == []


def test_no_speaker_program_is_a_warning_and_a_speakers_offer(installed, monkeypatch):
    install()
    monkeypatch.setattr(fluidaudio, "check", lambda: None)
    rt, probs = runtime.resolve()
    assert rt.tools.fluidaudiocli is None
    assert [p.code for p in probs] == ["no_speaker_labels", "speakers"] and probs[0].params["reason"]


def test_voices_folder_rules(installed, tmp_path):
    install()
    default = platforms.data_dir() / "voices"
    rt, _ = runtime.resolve()
    assert rt.voices_dir is None                           # no folder
    default.mkdir()
    (default / ".hidden.json").write_text("{}")
    (default / "notes.txt").write_text("x")
    assert runtime.resolve()[0].voices_dir is None         # no voiceprint (dotfiles don't count)
    (default / "stray.json").write_text("{}")
    (default / "old.json").write_text(_print("Old Print", model="e" * 64))
    assert runtime.resolve()[0].voices_dir is None         # nobody to match: not a print, another model's print
    (default / "jane doe.json").write_text(_print("Jane Doe"))
    assert runtime.resolve()[0].voices_dir == default
    assert runtime.resolve(names=False)[0].voices_dir is None
    other = tmp_path / "prints"
    other.mkdir()
    (other / "a.json").write_text(_print("A Person"))
    assert runtime.resolve(voices_dir=other)[0].voices_dir == other


def _print(name: str, *, model: str | None = None) -> str:
    """A made-up voiceprint file (placeholder vectors, never a real voice)."""
    from audiointake.steps.speaker_id import MODEL_SHA256
    return json.dumps({"name": name, "model_sha256": model or MODEL_SHA256, "updated": "2026-01-01",
                       "vectors": [{"vec": [0.6, 0.8], "source": "made up", "added": "2026-01-01", "seconds": 9.0}]})


def test_old_work_folders_are_pruned_and_each_run_gets_its_own(installed):
    install()
    root = platforms.cache_dir() / "work"
    old = root / "w-old"
    old.mkdir(parents=True)
    t = time.time() - 2 * 86400
    os.utime(old, (t, t))
    fresh = root / "w-fresh"
    fresh.mkdir()
    a = runtime.resolve()[0].work_root
    b = runtime.resolve()[0].work_root
    assert a != b and not old.exists() and fresh.exists()


def test_ffmpeg_missing_is_exit_3(installed, monkeypatch):
    install()
    from audiointake.setup import ffmpeg as ffm

    monkeypatch.setattr(ffm, "find", lambda **kw: (None, None))
    with pytest.raises(SetupError) as e:
        runtime.resolve()
    assert e.value.code == "ffmpeg_missing"


def _bump(monkeypatch, pid: str, **changes) -> None:
    """Ship a changed manifest for ``pid`` (what a release that changes a pack does), files on disk untouched."""
    import dataclasses
    from audiointake.setup import manifest

    pk = manifest.load(pid)
    if "version" in changes:
        pk = dataclasses.replace(pk, version=changes["version"])
    if "revision" in changes:
        key, rev = changes["revision"]
        pk = dataclasses.replace(pk, models=tuple(dataclasses.replace(m, revision=rev) if m.key == key else m
                                                  for m in pk.models))
    monkeypatch.setitem(manifest._CACHE, pid, pk)


def test_core_set_up_by_an_older_release_is_exit_3_not_a_silent_old_model(installed, monkeypatch):
    install()
    _bump(monkeypatch, "core", version=2)
    with pytest.raises(SetupError) as e:
        runtime.resolve()
    assert e.value.exit == 3 and e.value.code == store.code_or("pack_outdated", "pack_broken")
    assert e.value.detail["pack"] == "core" and "version" in e.value.detail["file"]


def test_a_pin_bump_without_a_version_bump_is_caught_too(installed, monkeypatch):
    from audiointake.setup import manifest

    install()
    shipped = manifest.load("core")
    _bump(monkeypatch, "core", revision=("ultra", "a" * 40))
    with pytest.raises(SetupError) as e:
        runtime.resolve()
    assert e.value.detail["file"] == "ultra-tag pinned to another version"
    monkeypatch.setitem(manifest._CACHE, "core", shipped)
    rt, probs = runtime.resolve()
    assert probs == [] and rt.models["ultra"].revision == "1" * 40


def test_an_outdated_th_pack_switches_its_routes_off_with_a_warning(installed, monkeypatch):
    install()
    _bump(monkeypatch, "th", version=7)
    rt, probs = runtime.resolve()
    assert "thai_turns" not in rt.routes and "typhoon" not in rt.models and "hu_vote" in rt.routes
    [p] = [p for p in probs if p.params.get("pack") == "th"]
    assert p.code == store.code_or("pack_outdated", "pack_broken") and p.params["language"] == "Thai"


def test_pack_outdated_is_used_once_registered(installed, monkeypatch):
    from audiointake import errors

    install()
    monkeypatch.setitem(errors.CODES, "pack_outdated", ("error", 3))
    monkeypatch.setitem(errors.MESSAGES, "pack_outdated", "The {pack} pack was set up by an earlier audiointake.")
    _bump(monkeypatch, "core", version=2)
    with pytest.raises(SetupError) as e:
        runtime.resolve()
    assert e.value.code == "pack_outdated"
