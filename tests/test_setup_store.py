"""Installing, verifying and removing packs, on made-up manifests whose tiny files sit in a scratch Hugging Face
cache (so the downloader only ever finds them present: no network). Model runs, the speaker program and the
canary are faked; what is tested is the store: plans, ledgers, stamps, downloads.json, remove, prune, uninstall.

The ``installed`` fixture is shared with test_setup_runtime.py and test_doctor.py, and so is
``_children_offline``: the conftest guard covers this process only, so every child setup starts here gets a dead
loopback proxy for anything that isn't loopback, and an attempt to reach the internet fails fast and visibly.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from audiointake import lock, platforms
from audiointake.errors import SetupError, UsageError
from audiointake.setup import fluidaudio, manifest, smoke, store
from audiointake.setup import ffmpeg as ffm

from test_download import server  # noqa: F401  (a fixture: the loopback fake server)


@pytest.fixture(autouse=True)
def every_setup_message_is_filled(monkeypatch):
    """Every report any test here gets from ``store.install``: no error or warning keeps an unfilled ``{field}``
    when the CLI renders it (setup has no record, so nothing fills ``{id}`` or ``{log}`` later)."""
    import re
    from audiointake import errors
    real = store.install

    def checked(*a, **kw):
        res = real(*a, **kw)
        for key in ("errors", "warnings"):
            for p in res.get(key) or []:
                text = errors.render(p["code"], p.get("params") or {})
                assert not re.search(r"\{[a-z_]+\}", text), (key, p)
        return res
    monkeypatch.setattr(store, "install", checked)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def _f(path: str, data: bytes, lfs: bool = True) -> tuple[dict, bytes]:
    return {"path": path, "size": len(data), "sha256": _sha(data), "blob": _sha(data) if lfs else _git(data)}, data


class World:
    """Made-up packs, their files on disk, and where everything lives for one test."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, list[tuple[dict, bytes]]] = {}
        self.specs: dict[str, dict] = {}

    def hf(self, key: str, repo: str, rev: str, files: list[tuple[dict, bytes]], **extra) -> dict:
        self.files[key] = files
        d = {"key": key, "kind": "hf", "tag": f"{key}-tag", "licence": "MIT", "credit": f"{key} credit",
             "repo": repo, "revision": rev, "files": [f for f, _ in files], **extra}
        self.specs[key] = d
        return d

    def place_hf(self, key: str) -> None:
        d = self.specs[key]
        for f, data in self.files[key]:
            blob = store.blob_path(d["repo"], f["blob"])
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
            link = store.snapshot_dir(d["repo"], d["revision"]) / f["path"]
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                link.unlink()
            os.symlink(os.path.relpath(blob, link.parent), link)


@pytest.fixture(autouse=True)
def _children_offline(_block_network, monkeypatch):
    """Children inherit the proxy variables (setup passes them through): point them at a closed loopback port."""
    for k in ("https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")
    for k in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(k, "127.0.0.1,localhost")


def _fake_ffmpeg(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "ffmpeg"
    p.write_text(f"#!{sys.executable}\nprint('ffmpeg version 9.9-test Copyright (c) the testers')\n")
    p.chmod(0o755)
    return p


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A world with core, hu and th described and their files already on disk (nothing installed yet)."""
    for name, sub in (("AUDIOINTAKE_HOME", "data"), ("AUDIOINTAKE_CACHE", "cache"), ("HF_HUB_CACHE", "hf/hub")):
        (tmp_path / sub).mkdir(parents=True)
        monkeypatch.setenv(name, str(tmp_path / sub))
    w = World(tmp_path)
    core = {"schema": 1, "id": "core", "version": 1, "title": "Core", "requires": [], "routes": [], "models": [
        w.hf("ultra", "test/ultra", "1" * 40, [_f("config.json", b'{"a": 1}', lfs=False),
                                              _f("model.safetensors", b"ultra-weights" * 50)],
             smoke={"kind": "parakeet", "seconds": 3}),
        w.hf("v3", "test/v3", "2" * 40, [_f("config.json", b'{"a": 1}', lfs=False),
                                        _f("model.safetensors", b"v3-weights" * 70)],
             smoke={"kind": "parakeet", "seconds": 3}),
        w.hf("nemotron", "test/nemo", "3" * 40, [_f("monolithic/v2/N.mlmodelc/weight.bin", b"nemo" * 99),
                                                _f("monolithic/v2/N.mlmodelc/model.mil", b"mil", lfs=False),
                                                _f("learnable_sil_emb.bin", b"sil" * 9)],
             layout={"dir_name": "nemo-3333333", "links": [
                 {"from": "monolithic/v2/N.mlmodelc", "to": "N.mlmodelc"},
                 {"from": "learnable_sil_emb.bin", "to": "learnable_sil_emb.bin"}]},
             smoke={"kind": "nemotron", "seconds": 3}),
    ]}
    onnx = b"onnx" * 40
    core["models"].append({"key": "campplus", "kind": "github_release", "tag": "campplus-tag", "licence": "Apache-2.0",
                           "credit": "c", "repo": "test/sherpa", "release_tag": "r1",
                           "files": [{"path": "model.onnx", "size": len(onnx), "sha256": _sha(onnx),
                                      "blob": _sha(onnx)}], "smoke": {"kind": "campplus", "seconds": 3}})
    core["canary"] = {"url": "https://groups.inf.ed.ac.uk/x/a.wav", "range": [44, 44 + 3200], "sha256": "c" * 64,
                      "wav": {"rate": 16000, "channels": 1, "bits": 16}, "file": "c.wav", "must_find": ["alpha"],
                      "min_speakers": 2}
    whisper = w.hf("whisper", "test/whisper", "4" * 40, [_f("config.json", b"{}", lfs=False),
                                                       _f("weights.npz", b"w" * 333)],
                   smoke={"kind": "whisper", "seconds": 3})
    hu = {"schema": 1, "id": "hu", "version": 1, "title": "Hungarian", "language": "hu", "requires": ["core"],
          "routes": ["hu_vote", "thai_screen"], "models": [whisper]}
    out_w, out_c = b"typhoon-mlx" * 30, b'{"x": 1}'
    th = {"schema": 1, "id": "th", "version": 1, "title": "Thai", "language": "th", "requires": ["core"],
          "routes": ["thai_screen", "thai_turns"], "models": [
              whisper,
              w.hf("typhoon_source", "test/ty", "5" * 40, [_f("config.json", b"{}", lfs=False),
                                                          _f("model.safetensors", b"src" * 20)]),
              {"key": "typhoon", "kind": "converted", "tag": "typhoon-tag", "licence": "MIT", "credit": "t",
               "convert": {"from": "typhoon_source", "tool": "whisper_hf2mlx", "output": [
                   {"path": "weights.safetensors", "size": len(out_w), "sha256": _sha(out_w)},
                   {"path": "config.json", "size": len(out_c), "sha256": _sha(out_c)}]},
               "layout": {"dir_name": "ty-5555555"}, "smoke": {"kind": "typhoon", "seconds": 3}}]}
    packs = {p["id"]: manifest.parse(json.dumps(p).encode()) for p in (core, hu, th)}
    monkeypatch.setattr(manifest, "_CACHE", dict(packs))
    for key in ("ultra", "v3", "nemotron", "whisper", "typhoon_source"):
        w.place_hf(key)
    cp = store.models_dir() / "campplus" / "model.onnx"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(onnx)
    ty = store.models_dir() / "ty-5555555"
    ty.mkdir(parents=True)
    (ty / "weights.safetensors").write_bytes(out_w)
    (ty / "config.json").write_bytes(out_c)
    w.ffmpeg = _fake_ffmpeg(tmp_path / "bin")
    w.fa = tmp_path / "fa" / "fluidaudiocli"
    w.fa.parent.mkdir()
    w.fa.write_text("#!/bin/sh\nexit 0\n")
    w.fa.chmod(0o755)
    w.calls = {"smoke": 0, "ensure": 0, "canary": 0}

    def fake_smoke(spec, model, ffmpeg, scripts):
        w.calls["smoke"] += 1
        return {"ok": True, "seconds": 0.01}

    def fake_ensure(*, build, progress=None, warnings=None, smoke_models=None, ffmpeg=None):
        w.calls["ensure"] += 1
        return w.fa

    def fake_canary(c, *, speakers, progress, fetch=True, ffmpeg=None):
        w.calls["canary"] += 1
        return dict(w.canary_result)

    w.canary_result = {"ok": True, "rtf": 0.05}

    monkeypatch.setattr(ffm, "find", lambda **kw: (w.ffmpeg, None))
    monkeypatch.setattr(smoke, "model_smoke", fake_smoke)
    monkeypatch.setattr(fluidaudio, "ensure", fake_ensure)
    monkeypatch.setattr(fluidaudio, "check", lambda: {"path": str(w.fa), "built_at": "t", "build_seconds": 1.0})
    monkeypatch.setattr(store, "run_canary", fake_canary)
    return w


def install(packs=("hu", "th"), **kw):
    with lock.run_lock():
        return store.install(list(packs), progress=lambda m: None, **kw)


# --------------------------------------------------------------------------- install
def test_install_needs_the_run_lock(installed):
    with pytest.raises(RuntimeError, match="run_lock"):
        store.install(["hu"], progress=lambda m: None)


def test_install_stamps_every_pack_and_fetches_nothing_already_here(installed, network_attempts):
    res = install()
    assert res["exit"] == 0, res["errors"]
    assert [p["id"] for p in res["packs"]] == ["core", "hu", "th"]
    assert all(p["installed"] and p["verified"] for p in res["packs"])
    assert res["fetched_bytes"] == 0 and store.read_downloads()["items"] == []
    for pid in ("core", "hu", "th"):
        st = store.read_stamp(pid)
        assert st["version"] == 1 and st["core_version"]
        mode = stat.S_IMODE(store.stamp_path(pid).stat().st_mode)
        assert mode == 0o600
    core = store.read_stamp("core")
    assert set(core["models"]) == {"ultra", "v3", "nemotron", "campplus"}
    assert core["smoke"]["ultra"]["ok"]
    nemo = store.model_path(manifest.load("core").model("nemotron"))
    assert (nemo / "N.mlmodelc").is_symlink() and (nemo / "N.mlmodelc" / "weight.bin").read_bytes() == b"nemo" * 99
    assert "typhoon" in store.read_stamp("th")["models"] and "typhoon_source" not in store.read_stamp("th")["models"]
    assert installed.calls["smoke"] == 5                 # ultra, v3, campplus, whisper (once for hu and th), typhoon
    assert installed.calls["canary"] == 1
    assert not network_attempts


def test_a_second_install_does_nothing_new(installed):
    install()
    first = {p: store.read_stamp(p)["models"] for p in ("core", "hu", "th")}
    res = install()
    assert res["exit"] == 0 and res["fetched_bytes"] == 0
    assert {p: store.read_stamp(p)["models"] for p in ("core", "hu", "th")} == first


def test_a_full_disk_stops_before_any_download(installed, monkeypatch):
    w = installed
    store.blob_path("test/whisper", w.specs["whisper"]["files"][1]["blob"]).unlink()      # 333 bytes to fetch
    monkeypatch.setattr(platforms, "free_bytes", lambda p: 10)
    called = []
    monkeypatch.setattr(store, "fetch_file", lambda *a, **k: called.append(a))
    res = install()
    assert res["exit"] == 3 and res["errors"][0]["code"] == "no_room" and not called
    assert store.read_stamp("core") is None


def _via_fake_server(monkeypatch, fake) -> list[str]:
    """Send every download the store starts to the loopback fake server; returns the URLs it was asked for."""
    real = store.fetch_file
    asked: list[str] = []

    def fetch(url, dest, size, digest, **kw):
        asked.append(url)
        kw.pop("allow", None)
        kw.pop("extra_args", None)
        return real(fake.url + "/file", dest, size, digest, allow=("127.0.0.1",),
                    extra_args=("--insecure-loopback",), **kw)

    monkeypatch.setattr(store, "fetch_file", fetch)
    return asked


def test_a_mismatched_file_is_model_changed_and_that_pack_alone_stops(installed, server, monkeypatch,
                                                                      network_attempts):
    w = installed
    blob = store.blob_path("test/whisper", w.specs["whisper"]["files"][1]["blob"])
    blob.write_bytes(b"x" * blob.stat().st_size)          # same size, other bytes: fetched again
    server.body = b"y" * blob.stat().st_size              # and the server's copy doesn't match the pin either
    asked = _via_fake_server(monkeypatch, server)
    res = install(("hu",))
    assert [(e["code"], e["params"]["pack"]) for e in res["errors"]] == [("model_changed", "hu")]
    assert "https://huggingface.co/test/whisper/resolve/" + "4" * 40 + "/weights.npz" in asked
    assert [r["path"] for r in server.requests] == ["/file"]   # one download, the mismatched one; nothing else
    assert store.read_stamp("core") is not None and store.read_stamp("hu") is None
    assert [p for p in res["packs"] if p["id"] == "hu"][0]["verified"] is False
    assert not list(blob.parent.glob("*.audiointake-*"))      # the mismatched download wasn't kept
    assert not network_attempts


def test_without_the_speaker_program_core_still_installs_and_setup_says_why(installed, monkeypatch):
    def no_clt(**kw):
        raise SetupError("clt_missing")

    monkeypatch.setattr(fluidaudio, "ensure", no_clt)
    res = install(("hu",))
    assert res["exit"] == 3 and res["speakers"]["code"] == "clt_missing"
    assert [e["code"] for e in res["errors"]] == ["clt_missing"]
    assert store.read_stamp("core") is not None


def test_a_failed_smoke_leaves_its_pack_unstamped(installed, monkeypatch):
    monkeypatch.setattr(smoke, "model_smoke",
                        lambda spec, *a: {"ok": spec.kind != "typhoon", "seconds": 0.0, "code": "exit_1"})
    res = install()
    assert res["exit"] == 3 and store.read_stamp("th") is None and store.read_stamp("hu") is not None
    [e] = res["errors"]
    assert e["params"]["pack"] == "th" and e["params"]["model"] == "typhoon-tag"
    assert e["code"] == store.code_or("model_load_failed", "pack_broken")


def test_a_failed_smoke_is_model_load_failed_once_that_code_is_registered(installed, monkeypatch):
    from audiointake import errors

    monkeypatch.setitem(errors.CODES, "model_load_failed", ("error", 3))
    monkeypatch.setattr(smoke, "model_smoke",
                        lambda spec, *a: {"ok": spec.kind != "typhoon", "seconds": 0.0, "code": "exit_1"})
    assert [e["code"] for e in install()["errors"]] == ["model_load_failed"]


def test_apply_lang_installs_a_pack_without_the_speaker_program_or_the_canary(installed, monkeypatch):
    """What apply --lang passes: on a Mac without the Command Line Tools the pack it asked for installs, exit 0."""
    def no_clt(**kw):
        raise SetupError("clt_missing")

    monkeypatch.setattr(fluidaudio, "ensure", no_clt)
    res = install(("hu",), speakers=False, canary=False)
    assert res["exit"] == 0 and res["errors"] == []
    assert res["speakers"]["skipped"] is True and res["canary"]["skipped"] is True
    assert installed.calls["canary"] == 0
    assert {p["id"]: p["verified"] for p in res["packs"]} == {"core": True, "hu": True}
    res = install(("hu",))                                   # setup itself still says what's missing
    assert res["exit"] == 3 and [e["code"] for e in res["errors"]] == ["clt_missing"]
    assert {p["id"]: p["verified"] for p in res["packs"]} == {"core": True, "hu": True}


def test_a_failed_canary_is_a_warning(installed, monkeypatch):
    from audiointake import errors

    installed.canary_result = {"ok": False, "rtf": 0.1, "found": 1, "of": 3, "speakers": 1}
    res = install(("hu",))
    assert res["exit"] == 0 and res["canary"]["ok"] is False
    monkeypatch.setitem(errors.CODES, "canary_failed", ("warning", None))
    res = install(("hu",))
    assert [(w["code"], w["params"]["found"], w["params"]["of"]) for w in res["warnings"]
            if w["code"] == "canary_failed"] == [("canary_failed", 1, 3)]
    installed.canary_result = {"ok": False, "rtf": None, "code": "canary_unavailable", "host": "h.example"}
    res = install(("hu",))
    assert [w["code"] for w in res["warnings"] if w["code"].startswith("canary")] == ["canary_unavailable"]


# --------------------------------------------------------------------------- the ledger
def test_a_stat_change_rehashes_once_and_refreshes_the_ledger(installed, monkeypatch):
    install()
    st = store.read_stamp("core")
    e = next(x for x in st["models"]["ultra"]["files"] if x["path"].endswith("model.safetensors"))
    blob = Path(e["path"]).resolve()
    data = blob.read_bytes()
    blob.unlink()
    blob.write_bytes(data)                                 # same bytes, new inode
    hashes = []
    real = store._hash
    monkeypatch.setattr(store, "_hash", lambda p, g: hashes.append(p) or real(p, g))
    with lock.run_lock():
        assert store.verify(full=False) == []
    assert len(hashes) == 1
    with lock.run_lock():
        assert store.verify(full=False) == []
    assert len(hashes) == 1                                # refreshed: not hashed again


def test_a_changed_file_is_pack_broken(installed):
    install()
    st = store.read_stamp("core")
    e = next(x for x in st["models"]["v3"]["files"] if x["path"].endswith("model.safetensors"))
    blob = Path(e["path"]).resolve()
    blob.write_bytes(b"Z" * blob.stat().st_size)
    probs = store.verify(full=False)
    assert [p.code for p in probs] == ["pack_broken"] and probs[0].params["pack"] == "core"


def test_a_link_to_another_blob_is_pack_broken_without_hashing(installed, monkeypatch):
    install()
    st = store.read_stamp("core")
    e = next(x for x in st["models"]["ultra"]["files"] if x["path"].endswith("config.json"))
    link = Path(e["path"])
    other = link.parent / os.readlink(link.parent / "model.safetensors")
    link.unlink()
    os.symlink(other, link)
    monkeypatch.setattr(store, "_hash", lambda p, g: pytest.fail("hashed a file whose blob name is wrong"))
    assert [p.code for p in store.verify(full=False)] == ["pack_broken"]


def test_full_verify_rehashes_everything(installed, monkeypatch):
    install()
    n = []
    real = store._hash
    monkeypatch.setattr(store, "_hash", lambda p, g: n.append(p) or real(p, g))
    assert store.verify(full=True) == []
    files = sum(len(m["files"]) for p in ("core", "hu", "th") for m in store.read_stamp(p)["models"].values())
    assert len(n) == files


def test_non_lfs_files_are_checked_by_their_git_blob_id(installed):
    install()
    st = store.read_stamp("core")
    e = next(x for x in st["models"]["ultra"]["files"] if x["path"].endswith("config.json"))
    assert e["git"] is True and len(e["digest"]) == 40 and e["digest"] == e["blob"]


# --------------------------------------------------------------------------- status, plan
def test_status_and_plan(installed):
    rows = {r["id"]: r for r in store.status()}
    assert not rows["core"]["installed"]
    install()
    rows = {r["id"]: r for r in store.status()}
    assert all(r["installed"] and r["verified"] for r in rows.values())
    p = store.plan(["hu", "th"])
    assert p["fetch_bytes"] >= 0 and p["packs"]["hu"] == 0 and p["packs"]["th"] == 0
    assert set(p) >= {"fetch_bytes", "disk_needed_bytes", "disk_free_bytes", "minutes_low", "minutes_high", "hosts"}


def test_plan_counts_a_conversion_once(installed):
    """Converting a source that isn't cached: the output is kept and the source is the peak; never both twice."""
    import shutil

    w = installed
    shutil.rmtree(store.models_dir() / "ty-5555555")
    shutil.rmtree(store.repo_dir("test/ty"))
    out = sum(x.size for x in manifest.load("th").model("typhoon").convert.output)
    src = sum(f["size"] for f, _ in w.files["typhoon_source"])
    clip = 3200 + 44
    p = store.plan(["th"])
    assert p["packs"]["th"] == src
    assert p["disk_needed_bytes"] == out + clip + src           # keep (the output, the canary) + the source's peak


def test_plan_counts_what_is_missing(installed):
    w = installed
    blob = store.blob_path("test/whisper", w.specs["whisper"]["files"][1]["blob"])
    blob.unlink()
    (store.models_dir() / "ty-5555555" / "weights.safetensors").unlink()
    p = store.plan(["th"])
    assert p["packs"]["th"] == 333                          # the whisper weights; the typhoon source is cached
    assert "huggingface.co" in p["hosts"] and "hosts under hf.co" in p["hosts"]


def test_the_plan_states_the_thai_models_terms(installed, capsys):
    """The terms the Thai model comes with beyond its licence are in the plan before anything is downloaded,
    whether or not the pack is installed; packs without such terms list none."""
    from fakes import cli, cli_json

    terms = "The Thai model comes with OpenTyphoon's terms (opentyphoon.ai/tac)."
    assert store.PACK_TERMS == {"th": terms}
    assert store.plan(["th"])["terms"] == [terms] == store.plan(["hu", "th"])["terms"]
    assert store.plan(["core"])["terms"] == [] == store.plan(["hu"])["terms"]
    code, out, _ = cli(capsys, "setup", "--plan", "--pack", "th")
    assert code == 0 and terms in out
    code, obj, _ = cli_json(capsys, "setup", "--plan", "--pack", "th")
    assert code == 0 and obj["plan"]["terms"] == [terms]
    install()
    assert store.plan(["th"])["terms"] == [terms]


# --------------------------------------------------------------------------- remove, prune, uninstall
def _downloaded(w: World, key: str) -> None:
    """Record a model's files in downloads.json as if audiointake had downloaded them."""
    d = w.specs[key]
    items = [{"kind": "hf", "repo": d["repo"], "revision": d["revision"], "path": f["path"], "blob": f["blob"],
              "blob_path": str(store.blob_path(d["repo"], f["blob"])),
              "link": str(store.snapshot_dir(d["repo"], d["revision"]) / f["path"])} for f in d["files"]]
    store._record_downloads(items)


def test_core_cant_be_removed(installed):
    install()
    with lock.run_lock(), pytest.raises(UsageError) as e:
        store.remove("core")
    assert e.value.code == "core_required"


def test_remove_keeps_shared_and_foreign_models(installed):
    w = installed
    install()
    _downloaded(w, "whisper")
    store._record_downloads([{"kind": "data", "path": str(store.models_dir() / "ty-5555555")}])
    with lock.run_lock():
        out = store.remove("th")
    assert store.read_stamp("th") is None
    assert not (store.models_dir() / "ty-5555555").exists()           # made by audiointake for th only
    assert "whisper" in out["kept_shared"]                              # hu still needs it
    assert store.blob_path("test/whisper", w.specs["whisper"]["files"][1]["blob"]).exists()
    with lock.run_lock():
        store.remove("hu")
    assert not store.blob_path("test/whisper", w.specs["whisper"]["files"][1]["blob"]).exists()
    assert store.blob_path("test/ultra", w.specs["ultra"]["files"][1]["blob"]).exists()   # never downloaded by us


def _old_revision(repo: str, rev: str = "9" * 40) -> dict:
    old = {"kind": "hf", "repo": repo, "revision": rev, "path": "model.safetensors", "blob": f"old-{rev[:4]}",
           "blob_path": str(store.blob_path(repo, f"old-{rev[:4]}")),
           "link": str(store.snapshot_dir(repo, rev) / "model.safetensors")}
    Path(old["blob_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(old["blob_path"]).write_bytes(b"old")
    Path(old["link"]).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(old["blob_path"], Path(old["link"]).parent), old["link"])
    store._record_downloads([old])
    return old


def test_prune_removes_only_revisions_no_pin_names(installed):
    w = installed
    install()
    old = _old_revision("test/ultra")
    _downloaded(w, "v3")
    with lock.run_lock():
        out = store.prune({"ultra", "v3"})
    assert out["removed"] == 1 and not Path(old["blob_path"]).exists() and not Path(old["link"]).is_symlink()
    assert store.blob_path("test/v3", w.specs["v3"]["files"][1]["blob"]).exists()


def test_prune_touches_only_models_this_setup_verified_and_none_a_stamp_still_uses(installed):
    install()
    old_whisper = _old_revision("test/whisper")
    res = install(("core",))                                   # core only: the whisper model wasn't verified now
    assert res["pruned"]["removed"] == 0 and Path(old_whisper["blob_path"]).exists()
    st = store.read_stamp("hu")                                # a stamp that still points at the old revision
    e = st["models"]["whisper"]["files"][0]
    e["path"] = str(store.snapshot_dir("test/whisper", "9" * 40) / "config.json")
    with lock.run_lock():
        store._write_json(store.stamp_path("hu"), st)
        assert store.prune({"whisper"})["removed"] == 0 and Path(old_whisper["blob_path"]).exists()
    res = install(("hu",))                                     # hu re-verified at the current pin: now it goes
    assert res["pruned"]["removed"] == 1 and not Path(old_whisper["blob_path"]).exists()


def test_uninstall_plan_changes_nothing_and_uninstall_keeps_the_lock_and_foreign_models(installed):
    w = installed
    install()
    _downloaded(w, "v3")
    voices = platforms.data_dir() / "voices"
    voices.mkdir()
    (voices / "jane doe.json").write_text("{}")
    before = sorted(p.name for p in platforms.data_dir().iterdir())
    plan = store.uninstall(keep_voices=True, plan=True)
    assert plan["plan"] and plan["then"] == "uv tool uninstall audiointake"
    assert sorted(p.name for p in platforms.data_dir().iterdir()) == before
    assert any(m["repo"] == "test/v3" for m in plan["models"])
    with lock.run_lock():
        out = store.uninstall(keep_voices=True, plan=False)
    left = sorted(p.name for p in platforms.data_dir().iterdir())
    assert left == ["run.lock", "voices"] and out["freed_bytes"] > 0
    assert not store.blob_path("test/v3", w.specs["v3"]["files"][1]["blob"]).exists()
    assert store.blob_path("test/ultra", w.specs["ultra"]["files"][1]["blob"]).exists()


def test_uninstall_deletes_only_what_audiointake_made(installed, monkeypatch, tmp_path):
    """AUDIOINTAKE_HOME and AUDIOINTAKE_CACHE may be shared folders: nothing else in them is touched."""
    install()
    home, cache = platforms.data_dir(), platforms.cache_dir()
    (home / "unrelated-notes.txt").write_text("mine")
    (home / "photos").mkdir()
    (home / "photos" / "a.jpg").write_bytes(b"jpg")
    (cache / "other-app").mkdir()
    (cache / "other-app" / "x").write_text("theirs")
    (cache / "work" / "w-1").mkdir(parents=True)
    (home / "records").mkdir()
    (home / "records" / "r-20261117-00000000.json").write_text("{}")
    (home / "speed.json").write_text("{}")
    swift_art = tmp_path / "swiftpm-artifacts" / "thing.zip"
    swift_art.parent.mkdir()
    swift_art.write_bytes(b"zip")
    (swift_art.parent / ".thing.zip.lock").write_text("")
    store._record_downloads([{"kind": "data", "path": str(swift_art)}])
    plan = store.uninstall(keep_voices=False, plan=True)
    assert sorted(Path(x["path"]).name for x in plan["left"]) == ["other-app", "photos", "unrelated-notes.txt"]
    assert str(swift_art) in {x["path"] for x in plan["remove"]}
    with lock.run_lock():
        store.uninstall(keep_voices=False, plan=False)
    assert sorted(p.name for p in home.iterdir()) == ["photos", "run.lock", "unrelated-notes.txt"]
    assert (home / "photos" / "a.jpg").read_bytes() == b"jpg"
    assert sorted(p.name for p in cache.iterdir()) == ["other-app"]
    assert not swift_art.exists() and not (swift_art.parent / ".thing.zip.lock").exists()


def test_uninstall_removes_the_cache_folder_when_it_held_only_ours(installed):
    install()
    (platforms.cache_dir() / "work").mkdir(exist_ok=True)
    with lock.run_lock():
        store.uninstall(keep_voices=False, plan=False)
    assert not platforms.cache_dir().exists()


def test_uninstall_needs_the_lock_unless_planning(installed):
    store.uninstall(keep_voices=False, plan=True)
    with pytest.raises(RuntimeError):
        store.uninstall(keep_voices=False, plan=False)


# --------------------------------------------------------------------------- the canary (made-up words only)
def test_canary_check_counts_words_and_speakers(tmp_path):
    c = manifest.load("core").canary
    clip = tmp_path / c.file
    doc = {"turns": [{"text": "x " + " ".join(c.must_find)}],
           "rttm": "SPEAKER c.16k 1 0 1 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER c.16k 1 1 1 <NA> <NA> speaker_1 <NA> <NA>\n"}
    (tmp_path / f"{clip.stem} (transcript).json").write_text(json.dumps(doc))
    r = smoke.canary_check(tmp_path, clip, c, speakers=True)
    assert r == {"ok": True, "found": len(c.must_find), "of": len(c.must_find), "speakers": 2,
                 "min_speakers": c.min_speakers}
    doc["turns"] = [{"text": "nothing here"}]
    (tmp_path / f"{clip.stem} (transcript).json").write_text(json.dumps(doc))
    assert smoke.canary_check(tmp_path, clip, c, speakers=True)["ok"] is False
    (tmp_path / f"{clip.stem} (transcript).json").unlink()
    assert smoke.canary_check(tmp_path, clip, c, speakers=True)["code"] == "no_transcript"


def test_wav_header_is_canonical():
    h = smoke.wav_header(960000)
    assert len(h) == 44 and h[:4] == b"RIFF" and h[8:16] == b"WAVEfmt " and h[36:40] == b"data"
    assert int.from_bytes(h[40:44], "little") == 960000 and int.from_bytes(h[24:28], "little") == 16000


def test_a_failed_core_smoke_fails_every_pack_that_requires_core(installed, monkeypatch):
    install()
    monkeypatch.setattr(smoke, "model_smoke",
                        lambda spec, *a: {"ok": spec.kind != "parakeet", "seconds": 0.0, "code": "exit_1"})
    res = install()
    assert res["exit"] == 3
    assert [store.read_stamp(p) for p in ("core", "hu", "th")] == [None, None, None]      # no stale stamps left


def test_without_ffmpeg_nothing_is_stamped(installed, monkeypatch):
    monkeypatch.setattr(ffm, "find", lambda **kw: (None, None))
    res = install()
    assert res["exit"] == 3 and "ffmpeg_missing" in [e["code"] for e in res["errors"]]
    assert store.read_stamp("core") is None


def test_folders_setup_makes_under_data_are_private(installed, server, monkeypatch):
    """campplus fetched for real (from the loopback server), Nemotron assembled, Typhoon converted (faked child):
    every folder setup makes under <data> is 0700."""
    import shutil
    from audiointake import proc

    shutil.rmtree(store.models_dir())
    server.body = b"onnx" * 40
    _via_fake_server(monkeypatch, server)
    real_run = proc.run
    out_w, out_c = b"typhoon-mlx" * 30, b'{"x": 1}'

    def run(argv, **kw):
        if "audiointake.setup.convert" in argv:
            out = Path(argv[argv.index("--out") + 1])
            (out / "weights.safetensors").write_bytes(out_w)
            (out / "config.json").write_bytes(out_c)
            return proc.Completed(0, b"{}")
        return real_run(argv, **kw)

    monkeypatch.setattr(store.proc, "run", run)
    res = install()
    assert res["exit"] == 0, res["errors"]
    assert [r["path"] for r in server.requests] == ["/file"]   # campplus only
    for d in (store.models_dir(), store.models_dir() / "campplus", store.models_dir() / "nemo-3333333",
              store.models_dir() / "ty-5555555"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d


def test_an_unreadable_child_report_is_never_claimed_as_a_download(installed, monkeypatch, tmp_path):
    from audiointake import proc

    dest = tmp_path / "blob"
    dest.write_bytes(b"here")
    monkeypatch.setattr(store.proc, "run", lambda *a, **k: proc.Completed(0, b""))
    out = store.fetch_file("https://huggingface.co/x/y/resolve/" + "1" * 40 + "/f", dest, 4, _sha(b"here"))
    assert out["status"] == "present" and out["fetched"] == 0


def test_setup_programs_are_one_allowlist(installed):
    from audiointake import proc

    progs = store.programs(Path("/opt/x/ffmpeg"))
    assert {store.python(), fluidaudio.GIT, fluidaudio.SWIFT, fluidaudio.XCODE_SELECT, Path("/usr/bin/caffeinate"),
            Path("/opt/x/ffmpeg")} <= progs
    with pytest.raises(proc.NotAllowed):
        proc.run(["/bin/echo", "x"], env={}, timeout=5, allowed=store.programs())


def test_stale_leftovers_are_swept_and_fresh_ones_kept(installed):
    import time as _t

    old = _t.time() - 2 * 86400
    hub = Path(os.environ["HF_HUB_CACHE"]) / "models--x--y" / "blobs"
    hub.mkdir(parents=True, exist_ok=True)
    files = [hub / "a.audiointake-bad", hub / "b.audiointake-alive", hub / "c.audiointake-partial"]
    for f in files:
        f.write_bytes(b"1")
        os.utime(f, (old, old))
    fresh = hub / "d.audiointake-partial"
    fresh.write_bytes(b"2")
    part = store.models_dir() / ".partial-ty-5555555-99999"
    part.mkdir(parents=True)
    (part / "weights.safetensors").write_bytes(b"x")
    os.utime(part, (old, old))
    fa_old = fluidaudio.root() / ".old-v0-x-r1-1"
    fa_old.mkdir(parents=True)
    os.utime(fa_old, (old, old))
    assert store.sweep_partials() == 5
    assert not any(f.exists() for f in files) and fresh.exists() and not part.exists() and not fa_old.exists()


def test_a_source_not_in_the_cache_is_fetched_to_a_temp_folder_converted_and_deleted(installed, monkeypatch):
    from audiointake import proc

    w = installed
    install(("hu",))
    ty = store.models_dir() / "ty-5555555"
    want = {p.name: p.read_bytes() for p in ty.iterdir()}
    import shutil

    shutil.rmtree(ty)
    shutil.rmtree(store.repo_dir("test/ty"))                       # neither the output nor the source is here
    src_files = {f["path"]: data for f, data in w.files["typhoon_source"]}
    fetched = []
    real_fetch = store.fetch_file
    temp = store.source_dir(manifest.load("th").model("typhoon_source"))

    def fake_fetch(url, dest, size, digest, **kw):
        if Path(dest).parent != temp:
            return real_fetch(url, dest, size, digest, **kw)          # the files already here: checked for real
        fetched.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src_files[dest.name])
        return {"status": "downloaded", "fetched": size, "hosts": ["huggingface.co"], "headers": []}

    real_run = proc.run

    def fake_run(argv, **kw):
        if "audiointake.setup.convert" in argv:
            out = Path(argv[argv.index("--out") + 1])
            source = Path(argv[argv.index("--source") + 1])
            assert source == store.source_dir(manifest.load("th").model("typhoon_source"))
            assert (source / "model.safetensors").exists()
            out.mkdir(parents=True, exist_ok=True)
            for name, data in want.items():
                (out / name).write_bytes(data)
            return proc.Completed(0, b'{"weights": 1, "extra": 0}')
        return real_run(argv, **kw)

    monkeypatch.setattr(store, "fetch_file", fake_fetch)
    monkeypatch.setattr(store.proc, "run", fake_run)
    res = install(("th",))
    assert res["exit"] == 0, res["errors"]
    assert {Path(d).parent for _, d in fetched} == {store.source_dir(manifest.load("th").model("typhoon_source"))}
    assert not store.source_dir(manifest.load("th").model("typhoon_source")).exists()      # deleted after
    assert {p.name: p.read_bytes() for p in ty.iterdir()} == want
    assert {"kind": "data", "path": str(ty)} in store.read_downloads()["items"]
    assert not store.repo_dir("test/ty").exists()                  # the source never went into the cache


def test_a_conversion_that_doesnt_match_its_pin_is_model_changed(installed, monkeypatch):
    from audiointake import proc

    install(("hu",))
    import shutil

    shutil.rmtree(store.models_dir() / "ty-5555555")

    real_run = proc.run

    def fake_run(argv, **kw):
        if "audiointake.setup.convert" not in argv:
            return real_run(argv, **kw)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "weights.safetensors").write_bytes(b"other")
        (out / "config.json").write_bytes(b"{}")
        return proc.Completed(0, b"{}")

    monkeypatch.setattr(store.proc, "run", fake_run)
    res = install(("th",))
    assert res["exit"] == 3 and res["errors"][0]["code"] == "model_changed"
    assert store.read_stamp("th") is None and not (store.models_dir() / "ty-5555555").exists()
    assert not list(store.models_dir().glob(".partial-*"))


def test_a_failed_conversion_names_its_model_and_pack(installed, monkeypatch):
    from audiointake import proc
    import shutil

    install(("hu",))
    shutil.rmtree(store.models_dir() / "ty-5555555")
    real_run = proc.run
    monkeypatch.setattr(store.proc, "run", lambda argv, **kw: proc.Completed(137, b"") if
                        "audiointake.setup.convert" in argv else real_run(argv, **kw))
    res = install(("th",))
    [e] = res["errors"]
    assert e["code"] == store.code_or("model_load_failed", "pack_broken")
    assert (e["params"]["pack"], e["params"]["model"], e["params"]["exit"]) == ("th", "typhoon-tag", 137)
    assert store.read_stamp("th") is None and not list(store.models_dir().glob(".partial-*"))
