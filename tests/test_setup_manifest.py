"""The pack manifests: the shipped ones load and agree with each other; the loader refuses everything else."""
from __future__ import annotations

import json
import re

import pytest

from audiointake.contracts import PACKS, ROUTES
from audiointake.setup import manifest
from audiointake.setup.manifest import ManifestError, parse


def _core() -> dict:
    from importlib import resources

    return json.loads(resources.files("audiointake.setup").joinpath("packs", "core.json").read_bytes())


def _bytes(d: dict) -> bytes:
    return json.dumps(d).encode("utf-8")


def test_the_shipped_packs_load():
    packs = manifest.all_packs()
    assert [p.id for p in packs] == list(PACKS)
    core = manifest.load("core")
    assert [m.key for m in core.models] == ["ultra", "v3", "nemotron", "campplus"]
    assert core.canary is not None and core.canary.end - core.canary.start == 30 * 32000
    assert manifest.load("hu").routes == ("hu_vote", "thai_screen")
    assert manifest.load("th").routes == ("thai_screen", "thai_turns")
    assert manifest.load("th").sources == frozenset({"typhoon_source"})
    for p in packs:
        assert set(p.routes) <= set(ROUTES)


def test_pins_are_exact_and_shared_models_agree():
    core, hu, th = (manifest.load(p) for p in PACKS)
    assert hu.model("whisper") == th.model("whisper")
    for p in (core, hu, th):
        for m in p.models:
            if m.kind == "hf":
                assert re.fullmatch(r"[0-9a-f]{40}", m.revision)
            for f in m.files:
                assert re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0
                assert f.blob == f.sha256 or re.fullmatch(r"[0-9a-f]{40}", f.blob)
    nemo = core.model("nemotron")
    assert nemo.revision == "25a90f97f254428d4b30374b76af9c74fdee8327"
    assert nemo.layout.dir_name == "nemotron3-offline-25a90f9"
    assert {ln.name for ln in nemo.layout.links} == {"Nemotron3Diarizer_offline.mlmodelc", "learnable_sil_emb.bin"}
    ty = th.model("typhoon")
    assert ty.convert.from_key == "typhoon_source" and {o.path for o in ty.convert.output} == {
        "weights.safetensors", "config.json"}
    assert core.model("campplus").files[0].sha256.startswith("aa3cfc16")


def test_the_thai_models_credit_names_opentyphoons_terms():
    """Both Typhoon entries (the download and the converted model ``about`` shows) say what it comes with."""
    th = manifest.load("th")
    for key in ("typhoon_source", "typhoon"):
        assert th.model(key).credit.endswith("; the Thai model comes with OpenTyphoon's terms (opentyphoon.ai/tac)")
    assert not [m.key for p in manifest.all_packs() for m in p.models
                if "OpenTyphoon" in m.credit and not m.key.startswith("typhoon")]


def test_non_lfs_files_carry_git_blob_ids():
    ultra = manifest.load("core").model("ultra")
    cfg = next(f for f in ultra.files if f.path == "config.json")
    assert len(cfg.blob) == 40 and not cfg.lfs
    weights = next(f for f in ultra.files if f.path == "model.safetensors")
    assert weights.lfs


def test_urls_are_pinned_never_a_branch():
    core = manifest.load("core")
    u = core.model("ultra")
    assert u.url(u.files[0]) == f"https://huggingface.co/{u.repo}/resolve/{u.revision}/{u.files[0].path}"
    c = core.model("campplus")
    assert c.url(c.files[0]).startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                                        "speaker-recongition-models/")
    for p in manifest.all_packs():
        for m in p.models:
            for f in m.files:
                assert "/main/" not in m.url(f)


def test_expand_and_models_of():
    assert manifest.expand([]) == ["core"]
    assert manifest.expand(["th"]) == ["core", "th"]
    assert manifest.expand(["th", "hu"]) == ["core", "hu", "th"]
    keys = [m.key for m in manifest.models_of(manifest.expand(["hu", "th"]))]
    assert keys.count("whisper") == 1
    with pytest.raises(KeyError):
        manifest.expand(["xx"])


def test_redirect_hosts():
    assert manifest.redirect_allowed("huggingface.co", "us.aws.cdn.hf.co")
    assert manifest.redirect_allowed("huggingface.co", "cas-bridge.xethub.hf.co")
    assert not manifest.redirect_allowed("huggingface.co", "hf.co.evil.example")
    assert not manifest.redirect_allowed("huggingface.co", "evilhf.co")
    assert manifest.redirect_allowed("github.com", "release-assets.githubusercontent.com")
    assert not manifest.redirect_allowed("github.com", "huggingface.co")


@pytest.mark.parametrize("mutate,where", [
    (lambda d: d.update(extra=1), "unknown key"),
    (lambda d: d.update(schema=2), "schema"),
    (lambda d: d.update(id="xx"), "$.id"),
    (lambda d: d["models"][0].update(revision="main"), "revision"),
    (lambda d: d["models"][0].update(revision="B554592C50B2A48471ADD2DAA3D46FA9F00FEF5E"), "revision"),
    (lambda d: d["models"][0]["files"][0].update(sha256="xyz"), "sha256"),
    (lambda d: d["models"][0]["files"][1].update(blob="0" * 64), "blob"),
    (lambda d: d["models"][0]["files"][0].update(path="../escape.json"), "path"),
    (lambda d: d["models"][0]["files"][0].update(path="/abs.json"), "path"),
    (lambda d: d["models"][0]["files"][0].update(path="loader.py"), "code suffix"),
    (lambda d: d["models"][0]["files"][0].update(path="libx.so.1"), "code suffix"),
    (lambda d: d["models"][0]["files"][0].update(path="a\\b.json"), "path"),
    (lambda d: d["models"][0].update(repo="not a repo"), "repo"),
    (lambda d: d["models"][0].update(kind="url"), "kind"),
    (lambda d: d["models"][0]["files"][0].update(size=0), "size"),
    (lambda d: d["models"][0]["files"][0].update(size=True), "size"),
    (lambda d: d["models"][0].update(release_tag="x"), "release_tag"),
    (lambda d: d["models"][1].update(key="ultra"), "used twice"),
    (lambda d: d["canary"].update(url="https://example.com/x.wav"), "canary.url"),
    (lambda d: d["canary"].update(range=[10, 5]), "range"),
    (lambda d: d["canary"]["wav"].update(rate=44100), "wav"),
    (lambda d: d["models"][2]["layout"]["links"][0].update({"from": "nowhere/x"}), "names no file"),
    (lambda d: d.update(routes=["fly"]), "routes"),
])
def test_refusals(mutate, where):
    d = _core()
    mutate(d)
    with pytest.raises(ManifestError) as e:
        parse(_bytes(d))
    assert where.lower() in str(e.value).lower()


def test_byte_level_refusals():
    good = _bytes(_core())
    parse(good)
    with pytest.raises(ManifestError, match="byte-order mark"):
        parse(b"\xef\xbb\xbf" + good)
    with pytest.raises(ManifestError, match="larger than"):
        parse(b" " * (manifest.MAX_BYTES + 1))
    with pytest.raises(ManifestError, match="not UTF-8"):
        parse(b"\xff\xfe{}")
    with pytest.raises(ManifestError, match="duplicate key"):
        parse(b'{"schema": 1, "schema": 1}')
    with pytest.raises(ManifestError, match="NaN"):
        parse(b'{"schema": NaN}')
    with pytest.raises(ManifestError, match="not JSON"):
        parse(b"{")


def test_converted_models_name_a_source_of_the_same_pack():
    from importlib import resources

    th = json.loads(resources.files("audiointake.setup").joinpath("packs", "th.json").read_bytes())
    th["models"][2]["convert"]["from"] = "whisperx"
    with pytest.raises(ManifestError, match="convert.from"):
        parse(_bytes(th))
    th["models"][2]["convert"]["from"] = "typhoon_source"
    th["models"][2]["repo"] = "a/b"
    with pytest.raises(ManifestError, match="converted model"):
        parse(_bytes(th))
