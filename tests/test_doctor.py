"""``doctor`` and ``about``: what they report, that they write nothing and open no connection."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from audiointake import about, doctor, errors, platforms
from audiointake.errors import SetupError
from audiointake.setup import fluidaudio, store

from test_setup_store import _children_offline, install, installed  # noqa: F401  (fixtures)
from test_setup_runtime import _bump


UNFILLED = re.compile(r"\{[a-z_]+\}")


@pytest.fixture(autouse=True)
def every_message_is_filled(monkeypatch):
    """Every report any test here makes: no message keeps an unfilled ``{field}`` (doctor has no record, so a
    wording that names ``{id}`` would print it as written)."""
    real = doctor.report

    def checked(**kw):
        rep = real(**kw)
        for key in ("problems", "warnings"):
            for p in rep.get(key) or []:
                for text in (p["message"], errors.render(p["code"], p["params"])):
                    assert not UNFILLED.search(text), (key, p)
        return rep
    monkeypatch.setattr(doctor, "report", checked)


@pytest.fixture
def toolchain_ok(monkeypatch):
    monkeypatch.setattr(fluidaudio, "toolchain",
                        lambda: {"developer_dir": "/dev/x", "swift": "6.4", "code": None, "toolchain_bin": "/x"})


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def test_an_unset_mac_is_not_set_up_exit_3(installed, toolchain_ok, network_attempts):
    rep = doctor.report()
    assert rep["exit"] == 3 and [p["code"] for p in rep["problems"]] == ["not_set_up"]
    assert rep["problems"][0]["message"].startswith("audiointake isn't set up yet")
    assert not network_attempts


def test_a_set_up_mac_is_ready(installed, toolchain_ok, network_attempts):
    install()
    before = _files(platforms.data_dir())
    rep = doctor.report(verify=True, selftest=True)
    assert rep["exit"] == 0 and rep["problems"] == [] and rep["warnings"] == []
    assert rep["routes"] == ["hu_vote", "thai_screen", "thai_turns"]
    assert all(p["verified"] for p in rep["packs"])
    assert rep["speakers"]["built"] and rep["speakers"]["tag"] == fluidaudio.TAG
    assert rep["selftest"]["ok"] is True and installed.calls["canary"] == 2       # setup's, then the self-test's
    assert all(not p["outdated"] for p in rep["packs"])
    assert {p["name"] for p in rep["prerequisites"]} == {"ffmpeg", "ffprobe", "swift"}
    assert _files(platforms.data_dir()) == before                                  # doctor writes nothing
    assert not network_attempts
    json.dumps(rep)


def test_a_set_up_mac_with_unmeasured_memory_is_ready_with_a_warning(installed, toolchain_ok, monkeypatch):
    install()
    monkeypatch.setattr(platforms, "memory_gb", lambda: 14)                       # GitHub's hosted Macs
    rep = doctor.report(verify=True)
    assert rep["exit"] == 0 and rep["problems"] == []
    assert [(w["code"], w["params"]) for w in rep["warnings"]] == [("memory_unmeasured", {"gb": 14})]


def test_a_broken_hu_pack_is_a_warning_and_a_broken_core_blocks(installed, toolchain_ok):
    install()
    w = installed
    blob = store.blob_path("test/whisper", w.specs["whisper"]["files"][1]["blob"])
    blob.write_bytes(b"!" * blob.stat().st_size)
    rep = doctor.report(verify=True)
    assert rep["exit"] == 0 and {p["params"]["pack"] for p in rep["warnings"]} == {"hu", "th"}
    blob = store.blob_path("test/v3", w.specs["v3"]["files"][1]["blob"])
    blob.write_bytes(b"!" * blob.stat().st_size)
    rep = doctor.report(verify=True)
    assert rep["exit"] == 3 and rep["problems"][0]["code"] == "pack_broken"


def test_no_speaker_program_says_why(installed, monkeypatch):
    install()
    monkeypatch.setattr(fluidaudio, "check", lambda: None)
    monkeypatch.setattr(fluidaudio, "toolchain", lambda: {"developer_dir": None, "swift": "", "code": "clt_missing"})
    rep = doctor.report()
    assert rep["exit"] == 0
    [w] = rep["warnings"]
    assert w["code"] == "no_speaker_labels" and "Command Line Tools" in w["params"]["reason"]
    assert w["message"].startswith("Transcripts made on this Mac have no speaker labels yet")
    assert [p for p in rep["prerequisites"] if p["name"] == "swift"][0]["code"] == "clt_missing"


def test_an_unsupported_platform_stops_there(installed, monkeypatch):
    def nope():
        raise SetupError("platform_not_macos", os="Linux")

    monkeypatch.setattr(platforms, "require_supported", nope)
    rep = doctor.report()
    assert rep["exit"] == 3 and rep["supported"] is False and rep["problems"][0]["code"] == "platform_not_macos"


def test_doctor_codes_are_closed():
    for c in doctor.BLOCKING:
        assert errors.CODES[c] == ("error", 3)


def test_about_lists_models_licences_hosts_and_folders(installed, network_attempts):
    install()
    before = _files(platforms.data_dir())
    a = about.about()
    assert a["version"] and a["network"] == about.PRIVACY_LINE
    keys = {m["key"] for m in a["models"]}
    assert keys == {"ultra", "v3", "nemotron", "campplus", "whisper", "typhoon"}
    whisper = next(m for m in a["models"] if m["key"] == "whisper")
    assert whisper["packs"] == ["hu", "th"] and whisper["licence"] == "MIT"
    typhoon = next(m for m in a["models"] if m["key"] == "typhoon")
    assert typhoon["converted"] and typhoon["revision"] == "5" * 40
    assert a["fluidaudio"]["licence"]["spdx"] == "Apache-2.0" and a["fluidaudio"]["commit"] == fluidaudio.COMMIT
    hosts = {h["host"] for h in a["hosts"]}
    assert {"huggingface.co", "github.com", "groups.inf.ed.ac.uk", "pypi.org"} <= hosts
    assert {f["path"] for f in a["folders"]} >= {str(platforms.data_dir()), str(platforms.cache_dir())}
    assert next(f for f in a["folders"] if f["path"] == str(platforms.hf_cache()))["bytes"] > 0
    assert set(a["packages"]) == {"mlx", "mlx-metal"}
    assert _files(platforms.data_dir()) == before and not network_attempts
    json.dumps(a)


def test_wheel_tags_read_the_installed_wheel():
    tags = about.wheel_tags("pytest")
    assert tags and all("-" in t for t in tags)
    assert about.wheel_tags("no-such-distribution-here") == []


def test_the_shipped_about_has_real_licences():
    a_models = {m["key"]: m for m in about.about()["models"]}
    assert a_models["ultra"]["licence"] == "CC-BY-4.0" and a_models["nemotron"]["licence"] == "OpenMDW-1.1"
    assert a_models["campplus"]["revision"] == "speaker-recongition-models"
    assert a_models["typhoon"]["bytes"] == 3083280590 + 269


def test_doctor_and_a_run_agree_on_an_outdated_pack(installed, toolchain_ok, monkeypatch):
    """A release that bumps core: transcribe would exit 3, so doctor must too, with the same code."""
    from audiointake.setup import runtime

    install()
    _bump(monkeypatch, "core", version=2)
    rep = doctor.report()
    with pytest.raises(SetupError) as e:
        runtime.resolve()
    assert rep["exit"] == 3 == e.value.exit
    assert [p["code"] for p in rep["problems"]] == [e.value.code]
    core = next(p for p in rep["packs"] if p["id"] == "core")
    assert core["outdated"] and not core["verified"]


def test_doctor_and_a_run_agree_on_an_outdated_th_pack(installed, toolchain_ok, monkeypatch):
    from audiointake.setup import runtime

    install()
    _bump(monkeypatch, "th", version=2)
    rep = doctor.report()
    rt, probs = runtime.resolve()
    assert rep["exit"] == 0 and [(p["code"], p["params"]["pack"]) for p in rep["warnings"]] == \
        [(p.code, p.params["pack"]) for p in probs]
    assert rep["routes"] == sorted(rt.routes) == ["hu_vote", "thai_screen"]


def test_a_self_test_without_its_clip_says_so(installed, toolchain_ok, monkeypatch):
    install()
    monkeypatch.setattr(store, "run_canary", lambda c, **kw: {"ok": False, "rtf": None, "code": "canary_unavailable",
                                                             "host": c.host})
    rep = doctor.report(selftest=True)
    assert rep["exit"] == 0
    assert [(w["code"], w["params"]["host"]) for w in rep["warnings"]] == [("canary_unavailable",
                                                                           "groups.inf.ed.ac.uk")]


def test_a_tag_that_had_moved_when_the_build_was_made_is_reported_offline(installed, toolchain_ok, monkeypatch,
                                                                         network_attempts):
    install()
    info = {"path": str(installed.fa), "built_at": "t", "tag_moved": True}
    monkeypatch.setattr(fluidaudio, "check", lambda: info)
    rep = doctor.report()
    assert [w["code"] for w in rep["warnings"]] == ["fluidaudio_tag_moved"] and not network_attempts
    info["tag_moved"] = False
    assert doctor.report()["warnings"] == []


def test_about_credits_the_thai_model_with_its_terms():
    """``about`` credits each model from the shipped pack manifests, and the Thai model's credit says what it
    comes with beyond its licence, once."""
    rows = [m for m in about.about()["models"] if "OpenTyphoon" in m["credit"]]
    assert [(m["key"], m["licence"], m["packs"]) for m in rows] == [("typhoon", "MIT", ["th"])]
    assert rows[0]["credit"].endswith("; the Thai model comes with OpenTyphoon's terms (opentyphoon.ai/tac)")
    assert rows[0]["credit"].count("OpenTyphoon") == 1


def test_about_names_both_data_sets_and_the_swiftpm_component(installed, monkeypatch, tmp_path):
    from audiointake.setup import swiftpm

    credits = " ".join(about.about()["data_credits"])
    assert "AMI" in credits and "FLEURS" in credits and credits.count("CC BY 4.0") == 2
    # who made them and where they are, as THIRD-PARTY.md's credit line says
    for part in ("collected by the AMI project consortium", "https://groups.inf.ed.ac.uk/ami/corpus/",
                 "FLEURS, by Google", "https://huggingface.co/datasets/google/fleurs"):
        assert part in credits, part
    monkeypatch.setattr(swiftpm, "artifacts_dir", lambda: tmp_path)
    a = swiftpm.pinned_artifacts()[0]
    (tmp_path / swiftpm.cache_name(a["url"])).write_bytes(b"z" * 10)
    row = [f for f in about.about()["folders"] if "SwiftPM" in f["what"]]
    assert len(row) == 1 and row[0]["bytes"] == 10
