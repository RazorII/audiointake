"""The command line: parsing, every command's wiring and output, setup's codes, signals, the lock, and a check
that every code of the registry and every option of every command has a test."""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from audiointake import cli as cli_mod
from audiointake import errors, lock, platforms
from audiointake.contracts import PACK_TERMS
from audiointake.errors import SetupError, problem

from fakes import HERE, audio, cli, cli_json, use_fakes, voiceprint

TESTS = Path(__file__).resolve().parent
A = "Vandrel Oskel"


@pytest.fixture
def fx(monkeypatch, tmp_path):
    return use_fakes(monkeypatch, tmp_path)


# --------------------------------------------------------------------------- parsing
def test_version_and_help(capsys):
    code, out, _ = cli(capsys, "--version")
    assert code == 0 and out.strip() == "audiointake 0.1.0"
    code, out, _ = cli(capsys, "--help")
    assert code == 0 and "transcribe" in out and "Nothing is ever asked" in out
    code, out, _ = cli(capsys, "transcribe", "--help")
    assert code == 0 and "--paths-from" in out


@pytest.mark.parametrize("args", [["transcribe", "x.m4a", "--overwr"], ["frobnicate"], [],
                                  ["transcribe", "x.m4a", "--speakers", "0"], ["apply", "r-20260101-00000000"],
                                  ["apply", "r-20260101-00000000", "--choice", "both"],
                                  ["notes", "r-20260101-00000000", "--into", "a", "--check", "b"],
                                  ["transcribe", "x.m4a", "--format", "pdf"], ["setup", "--pack", "xx"]])
def test_bad_arguments(fx, capsys, args):
    code, obj, _ = cli_json(capsys, *args)
    assert code == 2 and obj["error"]["code"] == "bad_arguments"
    assert "--help` for the options." in obj["error"]["message"]
    if args[:1] in (["transcribe"], ["apply"]):
        assert obj["records"] == []
    code, out, err = cli(capsys, *args)
    assert code == 2 and out == "" and "for the options." in err


def test_no_command_reads_stdin(fx, tmp_path, capsys, monkeypatch):
    class Boom:
        def read(self, *a):
            raise AssertionError("stdin was read")
        readline = readlines = read
    monkeypatch.setattr(sys, "stdin", Boom())
    a = audio(tmp_path / "x.m4a")
    assert cli(capsys, "transcribe", a)[0] == 0
    assert cli(capsys, "transcribe", "-")[0] == 2               # "-" is a missing file, never standard input
    code, obj, _ = cli_json(capsys, "transcribe", "--paths-from", "/dev/stdin")
    assert code == 2 and obj["error"]["code"] == "bad_arguments" and "isn't a regular file" in obj["error"]["message"]


# --------------------------------------------------------------------------- setup, packs, doctor, about, uninstall
def test_setup(fx, capsys):
    code, obj, _ = cli_json(capsys, "setup", "--plan", "--pack", "th")
    assert code == 0 and obj["plan"]["fetch_bytes"] == 9.3e9 and ("plan", {"pack_ids": ["core", "th"]}) in fx.calls
    assert not [c for c in fx.calls if c[0] == "install"]
    code, obj, _ = cli_json(capsys, "setup", "--pack", "hu", "--pack", "th", "--pack", "hu")
    assert code == 0 and ("install", {"pack_ids": ["core", "hu", "th"]}) in fx.calls
    assert not lock.held()


def test_setup_failure_passes_its_exit_through(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, install_result={"exit": 3, "errors": [{"code": "clt_missing", "params": {}}],
                                                     "warnings": [{"code": "memory_unmeasured", "params": {"gb": 8}}]})
    code, out, _ = cli(capsys, "setup")
    assert code == 3
    assert "Setup didn't finish." in out and errors.render("clt_missing") in out
    assert "This Mac has 8 GB of memory." in out


def test_packs(fx, capsys):
    code, obj, _ = cli_json(capsys, "packs", "list")
    assert code == 0 and [p["id"] for p in obj["packs"]] == ["core", "hu", "th"]
    code, obj, _ = cli_json(capsys, "packs", "remove", "core")
    assert code == 2 and obj["error"]["code"] == "core_required"
    code, obj, _ = cli_json(capsys, "packs", "remove", "hu")
    assert code == 0 and ("remove", {"pack_id": "hu"}) in fx.calls


def test_doctor_and_about(monkeypatch, tmp_path, capsys):
    fs = use_fakes(monkeypatch, tmp_path, doctor_result={"exit": 3, "errors": [{"code": "not_set_up",
                                                                                 "params": {"size": 4.3}}],
                                                         "warnings": [
        {"code": "memory_unmeasured", "params": {"gb": 16}},
        {"code": "canary_unavailable", "params": {"host": "example.org"}},
        {"code": "fluidaudio_tag_moved", "params": {"tag": "v0.18.0", "commit": "e15287985625"}}]})
    code, obj, _ = cli_json(capsys, "doctor", "--verify", "--selftest")
    assert code == 3 and ("doctor", {"verify": True, "selftest": True}) in fs.calls
    code, out, _ = cli(capsys, "doctor")
    assert code == 3
    for c, p in (("memory_unmeasured", {"gb": 16}), ("canary_unavailable", {"host": "example.org"}),
                 ("fluidaudio_tag_moved", {"tag": "v0.18.0", "commit": "e15287985625"}), ("not_set_up", {"size": 4.3})):
        assert errors.render(c, p) in out
    code, obj, _ = cli_json(capsys, "about")
    assert code == 0 and obj["version"] == "0.1.0"


def test_uninstall(fx, capsys):
    code, obj, _ = cli_json(capsys, "uninstall", "--plan", "--keep-voices")
    assert code == 0 and ("uninstall", {"keep_voices": True, "plan": True}) in fx.calls
    code, out, _ = cli(capsys, "uninstall")
    assert code == 0 and "To remove the tool itself: uv tool uninstall audiointake" in out


def test_a_thai_language_offer_prints_the_models_terms_under_its_line_only(fx, tmp_path, capsys):
    code, out, _ = cli(capsys, "transcribe", audio(tmp_path / "x.m4a"), "--lang", "hu,th")
    lines = out.splitlines()
    th = next(i for i, line in enumerate(lines) if line.startswith("  - Thai is read better with the Thai pack"))
    assert code == 0 and lines[th + 1] == f"    {PACK_TERMS['th']}"
    assert [line for line in lines if "opentyphoon" in line.lower()] == [lines[th + 1]]    # not under Hungarian
    code, obj, _ = cli_json(capsys, "transcribe", audio(tmp_path / "y.m4a"), "--lang", "hu,th")
    assert [o["options"] for o in obj["records"][0]["offers"] if o["kind"] == "language"] == [["hu", "th"]]
    assert "opentyphoon" not in json.dumps(obj).lower()                                   # the record keeps none


# --------------------------------------------------------------------------- setup's and the platform's codes
SETUP_CODES = sorted(c for c, (kind, ex) in errors.CODES.items() if kind == "error" and ex == 3
                     and not c.startswith("platform_") and c != "macos_too_old")


@pytest.mark.parametrize("code", SETUP_CODES)
def test_setup_errors_stop_transcribe_before_any_audio(monkeypatch, tmp_path, capsys, code):
    use_fakes(monkeypatch, tmp_path, setup_error=code)
    a = audio(tmp_path / "x.m4a")
    rc, obj, _ = cli_json(capsys, "transcribe", a)
    assert rc == 3 and obj["error"]["code"] == code and obj["records"] == []
    assert obj["error"]["message"] == errors.render(code, {"size": 4.3, "pack": "core", "file": "config.json"})
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["x.m4a"]


PLATFORM = [("platform_not_macos", {"os": "Linux"}), ("platform_not_apple_silicon", {}),
            ("platform_rosetta", {"version": "0.1.0"}), ("macos_too_old", {"macos": "13.6"})]


@pytest.mark.parametrize("code,params", PLATFORM)
def test_platform_codes(fx, tmp_path, capsys, monkeypatch, code, params):
    def refuse():
        raise SetupError(code, **params)
    monkeypatch.setattr(platforms, "require_supported", refuse)
    for args in (["transcribe", audio(tmp_path / "x.m4a")], ["setup"], ["enroll", A, "x.m4a"],
                 ["apply", "r-20260101-00000000", "--choice", "names"]):
        rc, obj, _ = cli_json(capsys, *args)
        assert rc == 3 and obj["error"]["code"] == code
        assert obj["error"]["message"] == errors.render(code, params)


def test_resolve_warnings_reach_the_record(monkeypatch, tmp_path, capsys):
    use_fakes(monkeypatch, tmp_path, problems=[problem("pack_broken", pack="th", file="weights.safetensors",
                                                       language="Thai")])
    rc, obj, _ = cli_json(capsys, "transcribe", audio(tmp_path / "x.m4a"))
    w = obj["records"][0]["warnings"]
    assert rc == 0 and w[0]["code"] == "pack_broken" and "Until then, Thai is read with the core models." in w[0]["message"]


def test_relabel_needs_the_transcript_json(fx, tmp_path, capsys):
    rec = cli_json(capsys, "transcribe", audio(tmp_path / "x.m4a"), "--format", "md")[1]["records"][0]
    rc, obj, _ = cli_json(capsys, "relabel", rec["id"], "--label", "speaker_0", "--name", A)
    assert rc == 2 and obj["error"]["code"] == "no_transcript_json"


def test_relabel_overwrite_replaces_an_edited_output(fx, tmp_path, capsys):
    rec = cli_json(capsys, "transcribe", audio(tmp_path / "x.m4a"))[1]["records"][0]
    (tmp_path / "x (transcript).md").write_text("mine")
    rc, obj, _ = cli_json(capsys, "relabel", rec["id"], "--label", "speaker_0", "--name", A, "--overwrite")
    assert rc == 0 and f"**{A}**" in (tmp_path / "x (transcript).md").read_text()


# --------------------------------------------------------------------------- every code and every option has a test
COVERED = {
    "bad_arguments": "test_cli::test_bad_arguments", "input_missing": "test_pipeline::test_input_problems",
    "input_not_file": "test_pipeline::test_input_problems", "input_not_downloaded": "test_pipeline::test_input_problems",
    "input_not_readable": "test_pipeline::test_input_problems", "unsupported_input": "test_pipeline::test_input_problems",
    "output_not_writable": "test_pipeline::test_output_not_writable",
    "output_exists": "test_pipeline::test_output_exists_and_overwrite",
    "unknown_language": "test_pipeline::test_unknown_language", "unknown_record": "test_record::test_load_unknown",
    "no_offer": "test_pipeline::test_apply_refusals", "not_an_option": "test_pipeline::test_apply_refusals",
    "no_transcript_json": "test_cli::test_relabel_needs_the_transcript_json",
    "unknown_label": "test_pipeline::test_relabel_refusals", "name_invalid": "test_voices::test_bad_names",
    "voice_not_found": "test_voices::test_list_remove_rename", "core_required": "test_cli::test_packs",
    "voice_exists": "test_voices::test_list_remove_rename",
    "voice_other_model": "test_voices::test_a_print_made_with_another_model_is_refused_first",
    "internal_error": "test_pipeline::test_an_unexpected_exception_fails_that_input_only",
    "already_enrolled": "test_voices::test_enroll_result_counts_what_was_added",
    "pack_outdated": "test_setup_runtime::test_pack_outdated_is_used_once_registered",
    "model_load_failed": "test_setup_store::test_a_failed_smoke_is_model_load_failed_once_that_code_is_registered",
    "canary_failed": "test_setup_store::test_a_failed_canary_is_a_warning",
    "head_invalid": "test_notes::test_into_refuses_a_head_with_a_transcript",
    # setup's and the platform's codes: the tests where the real producer (runtime.resolve, store, fluidaudio,
    # download, platforms, doctor) raises them, not the CLI tests that only pass a fake's error through
    "platform_not_macos": "test_platforms::test_not_macos", "platform_not_apple_silicon": "test_platforms::test_intel_mac",
    "platform_rosetta": "test_platforms::test_rosetta", "macos_too_old": "test_platforms::test_macos_too_old",
    "not_set_up": "test_setup_runtime::test_not_set_up_is_exit_3_with_the_download_size",
    "ffmpeg_missing": "test_setup_runtime::test_ffmpeg_missing_is_exit_3",
    "clt_missing": "test_setup_fluidaudio::test_no_command_line_tools_never_runs_swift",
    "xcode_license": "test_setup_fluidaudio::test_xcode_licence_and_old_swift_are_detected",
    "swift_too_old": "test_setup_fluidaudio::test_xcode_licence_and_old_swift_are_detected",
    "speakers_build_failed": "test_setup_fluidaudio::test_a_failed_build_is_speakers_build_failed_with_its_log",
    "build_timeout": "test_setup_fluidaudio::test_a_build_cap_is_build_timeout_and_the_next_run_resumes",
    "no_room": "test_setup_store::test_a_full_disk_stops_before_any_download",
    "download_failed": "test_download::test_a_redirect_off_the_allowed_hosts_is_refused",
    "model_changed": "test_setup_store::test_a_mismatched_file_is_model_changed_and_that_pack_alone_stops",
    "pack_broken": "test_setup_runtime::test_a_broken_core_file_is_pack_broken",
    "asr_failed": "test_pipeline::test_asr_failed_writes_nothing", "decode_failed": "test_pipeline::test_decode_failed",
    "too_little_speech": "test_voices::test_enroll_too_little_speech",
    **{c: "test_voices::test_enroll_refusals" for c in ("two_voices", "mixed_voices", "matches_someone_else",
                                                         "not_matching_print")},
    "input_changed": "test_pipeline::test_input_changed_during_the_run",
    "output_changed": "test_pipeline::test_apply_refusals",
    "transcript_edited": "test_pipeline::test_apply_on_an_edited_transcript",
    "pack_not_installed": "test_pipeline::test_lang_without_its_pack_warns_and_never_downloads",
    "speech_not_transcribed": "test_pipeline::test_speech_not_transcribed",
    "no_speaker_labels": "test_pipeline::test_apply_speakers_adds_labels_without_transcribing_again",
    "speakers_failed": "test_pipeline::test_every_diarmeta_row", "merge_failed": "test_pipeline::test_every_diarmeta_row",
    "speaker_cap": "test_pipeline::test_speaker_count_differs_and_cap",
    "speaker_count_differs": "test_pipeline::test_speaker_count_differs_and_cap",
    "fill_failed": "test_pipeline::test_fill_failed",
    "hungarian_vote_failed": "test_pipeline::test_hungarian_vote_failed_and_thai_failed",
    "thai_failed": "test_pipeline::test_hungarian_vote_failed_and_thai_failed",
    "memory_unmeasured": "test_platforms::test_memory_warning",
    "canary_unavailable": "test_doctor::test_a_self_test_without_its_clip_says_so",
    "fluidaudio_tag_moved": "test_setup_fluidaudio::test_a_moved_tag_warns_and_the_pinned_commit_is_built",
    "already_done": "test_pipeline::test_resume_already_done_then_voices_changed",
    "voices_changed": "test_pipeline::test_resume_already_done_then_voices_changed",
    "no_speech": "test_pipeline::test_no_speech", "recorded_at_from_file_date": "test_pipeline::test_recorded_at_sources",
    "voices_model_mismatch": "test_pipeline::test_voiceprint_folder_notes",
    "voices_skipped": "test_pipeline::test_voiceprint_folder_notes",
    "pack_fetched": "test_pipeline::test_apply_lang_installs_and_redoes",
    "language": "test_pipeline::test_detected_hungarian_without_the_pack",
    "speakers": "test_pipeline::test_apply_speakers_adds_labels_without_transcribing_again",
    "names": "test_pipeline::test_resume_already_done_then_voices_changed",
    "remember": "test_pipeline::test_remember_offer_for_unnamed_labels_with_enough_speech",
}


def _test_names(module: str) -> set[str]:
    tree = ast.parse((TESTS / f"{module}.py").read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}


def test_every_code_has_a_test():
    assert set(COVERED) == set(errors.CODES), sorted(set(errors.CODES) ^ set(COVERED))
    for code, where in COVERED.items():
        module, name = where.split("::")
        assert name in _test_names(module), (code, where)
        assert f'"{code}"' in (TESTS / f"{module}.py").read_text(encoding="utf-8"), code


def _options(parser) -> set[str]:
    import argparse
    out: set[str] = set()
    for action in parser._actions:
        out.update(s for s in action.option_strings if s.startswith("--") and s not in ("--help",))
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                out |= _options(sub)
    return out


def test_every_option_has_a_test():
    text = "".join((TESTS / f"test_{m}.py").read_text(encoding="utf-8") for m in
                   ("cli", "pipeline", "record", "outputs", "voices", "notes", "privacy"))
    missing = sorted(o for o in _options(cli_mod.build_parser()) if f'"{o}"' not in text)
    assert missing == []


# --------------------------------------------------------------------------- signals and the lock
def _env(tmp_path: Path) -> dict[str, str]:
    for sub in ("home", "data", "cache", "hf"):
        (tmp_path / sub).mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AUDIOINTAKE_", "HF_"))}
    env.update(HOME=str(tmp_path / "home"), AUDIOINTAKE_HOME=str(tmp_path / "data"),
               AUDIOINTAKE_CACHE=str(tmp_path / "cache"), HF_HUB_CACHE=str(tmp_path / "hf"),
               PYTHONDONTWRITEBYTECODE="1")
    return env


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.cli_subprocess
@pytest.mark.parametrize("sig,want", [(signal.SIGTERM, 143), (signal.SIGHUP, 129), (signal.SIGINT, 130)])
def test_a_signal_mid_run_leaves_nothing(tmp_path, sig, want):
    env = _env(tmp_path)
    marker = tmp_path / "child.pid"
    src = tmp_path / "in"
    src.mkdir()
    a = audio(src / "x.m4a", fail={"primary": "hang"}, marker=str(marker))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"root": str(tmp_path / "fake"), "setup": {}}))
    p = subprocess.Popen([sys.executable, "-I", "-B", str(HERE / "run_cli.py"), str(cfg), "transcribe", str(a), "--json"],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while not marker.exists() and time.monotonic() < deadline and p.poll() is None:
        time.sleep(0.05)
    assert marker.exists(), p.communicate(timeout=10)
    child = int(marker.read_text())
    p.send_signal(sig)
    out, err = p.communicate(timeout=30)
    assert p.returncode == want, err.decode()
    assert out == b""                                                  # no records printed
    assert sorted(x.name for x in src.iterdir()) == ["x.m4a"]           # no output, no partial
    assert not (tmp_path / "data" / "records").exists() or not list((tmp_path / "data" / "records").iterdir())
    work = tmp_path / "cache" / "work"
    assert not work.exists() or list(work.iterdir()) == []
    logs = tmp_path / "home" / "Library" / "Logs" / "audiointake"
    assert not logs.exists() or list(logs.iterdir()) == []
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.1)
    assert not _alive(child)                                            # the model child was stopped too


def test_a_signal_during_the_commit_waits_for_it(fx, tmp_path, capsys, monkeypatch):
    from audiointake import fsutil
    from audiointake import record as record_mod
    a = audio(tmp_path / "x.m4a")
    real = fsutil.atomic_write
    sent: list[str] = []

    def write(path, data, **kw):
        if not sent and str(path).endswith("(transcript).md"):
            sent.append(str(path))
            signal.raise_signal(signal.SIGTERM)                 # arrives between the commit's first and second file
        return real(path, data, **kw)
    monkeypatch.setattr(fsutil, "atomic_write", write)
    code, out, _ = cli(capsys, "transcribe", a, "--json")
    assert code == 143 and sent and out == ""
    recs = list(record_mod.iter_records())
    assert len(recs) == 1 and recs[0]["status"] == "done"          # the commit and its record finished together
    assert sorted(o["kind"] for o in recs[0]["outputs"]) == ["json", "md"]
    for o in recs[0]["outputs"]:
        assert fsutil.sha256_file(o["path"]) == o["sha256"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["x (transcript).json", "x (transcript).md",
                                                                          "x.m4a"]


def test_a_signal_while_a_child_starts_stops_that_child(fx, tmp_path, capsys, monkeypatch):
    """A stop landing between a model child's fork and its registration used to leave that child running (the
    stop couldn't see it); the CLI now holds the signal until the child is registered, then stops it."""
    from audiointake import proc
    real = proc.subprocess.Popen
    started: dict[str, int] = {}

    class SignalOnStart(real):
        def __init__(self, args, **kw):
            super().__init__(args, **kw)
            if not started and Path(str(args[0])).name == "parakeet-mlx":
                started["pid"] = self.pid
                os.kill(os.getpid(), signal.SIGTERM)          # handled before run() registers the child
                time.sleep(0.2)
    monkeypatch.setattr(proc.subprocess, "Popen", SignalOnStart)
    a = audio(tmp_path / "x.m4a", fail={"primary": "hang"})
    code, out, _ = cli(capsys, "transcribe", a, "--json")
    assert code == 143 and out == "" and started
    for _ in range(50):
        if not _alive(started["pid"]):
            break
        time.sleep(0.1)
    assert not _alive(started["pid"])
    assert proc._STARTING == {"depth": 0, "pending": None}
    assert sorted(x.name for x in tmp_path.iterdir() if x.is_file()) == ["x.m4a"]


def test_lang_usage_errors_name_the_command_they_came_from(fx, tmp_path, capsys):
    for args, command in ((["apply", "r-20261117-00000000", "--lang", "hu,th"], "apply"),
                          (["apply", "r-20261117-00000000", "--lang", ""], "apply"),
                          (["enroll", A, str(tmp_path / "c.m4a"), "--lang", ""], "enroll"),
                          (["transcribe", str(tmp_path / "c.m4a"), "--lang", " , "], "transcribe")):
        code, obj, _ = cli_json(capsys, *args)
        assert code == 2 and obj["error"]["code"] == "bad_arguments", (args, obj)
        assert f"Run `audiointake {command} --help`" in obj["error"]["message"], (args, obj)


@pytest.mark.cli_subprocess
def test_a_second_run_waits_for_the_lock(tmp_path):
    env = _env(tmp_path)
    voiceprint(tmp_path / "data" / "voices", A)
    with lock.run_lock(tmp_path / "data"):
        p = subprocess.Popen([sys.executable, "-I", "-B", "-m", "audiointake", "voices", "remove", A, "--json"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        time.sleep(2.0)
        assert p.poll() is None                                          # waiting on the run, not finished
        assert (tmp_path / "data" / "voices" / f"{A.lower()}.json").exists()
    out, err = p.communicate(timeout=30)
    assert p.returncode == 0, err.decode()
    assert err.decode().count(lock.WAIT_MESSAGE) == 1
    assert json.loads(out)["removed"] == A
    assert not (tmp_path / "data" / "voices" / f"{A.lower()}.json").exists()


# --------------------------------------------------------------------------- REF: an id, the recording, the JSON
REF_COMMANDS = {
    "apply": lambda ref, tmp: ["apply", ref, "--lang", "hu"],
    "relabel": lambda ref, tmp: ["relabel", ref, "--label", "speaker_0", "--name", A],
    "notes": lambda ref, tmp: ["notes", ref, "--into", tmp / "x (notes).md"],
    "enroll": lambda ref, tmp: ["enroll", A, "--from", ref, "--label", "speaker_1"],
    "record find": lambda ref, tmp: ["record", "find", ref],
}


@pytest.mark.parametrize("kept", [True, False], ids=["record-kept", "record-expired"])
@pytest.mark.parametrize("form", ["id", "audio", "json"])
@pytest.mark.parametrize("command", sorted(REF_COMMANDS))
def test_every_command_takes_any_ref(fx, tmp_path, capsys, command, form, kept):
    from audiointake import record as record_mod
    a = audio(tmp_path / "x.m4a", ensemble={"hu_route": "no_pack", "hu_share": 0.4})
    rid = cli_json(capsys, "transcribe", a)[1]["records"][0]["id"]
    (tmp_path / "x (notes).md").write_text("# x\n\n## Summary\n\nA meeting.\n")
    if not kept:
        record_mod.record_path(rid).unlink()
    ref = {"id": rid, "audio": str(a), "json": str(tmp_path / "x (transcript).json")}[form]
    code, obj, _ = cli_json(capsys, *REF_COMMANDS[command](ref, tmp_path))
    if form == "id" and not kept:
        assert code == 2 and obj["error"]["code"] == "unknown_record"
        return
    assert code == 0, obj
    if command == "record find":
        assert obj["id"] == rid and obj["labels"] == ["speaker_0", "speaker_1"]
    else:
        child = obj["records"][0] if "records" in obj else obj["record"]
        assert child["parent"] == rid


@pytest.mark.parametrize("kept", [True, False], ids=["record-kept", "record-expired"])
@pytest.mark.parametrize("form", ["id", "audio", "json"])
def test_record_show_and_notes_check_take_any_ref(fx, tmp_path, capsys, form, kept):
    from audiointake import record as record_mod
    a = audio(tmp_path / "x.m4a")
    rid = cli_json(capsys, "transcribe", a)[1]["records"][0]["id"]
    n = tmp_path / "x (notes).md"
    n.write_text("# x\n\n## Summary\n\nA meeting.\n")
    assert cli_json(capsys, "notes", rid, "--into", n)[0] == 0
    if not kept:
        record_mod.record_path(rid).unlink()             # the transcript's record; the notes record is still kept
    ref = {"id": rid, "audio": str(a), "json": str(tmp_path / "x (transcript).json")}[form]
    code, obj, _ = cli_json(capsys, "record", "show", ref)
    if kept:
        assert code == 0 and obj["record"]["id"] == rid
    elif form == "id":
        assert code == 2 and obj["error"]["code"] == "unknown_record"
    else:                                                # the newest record left of that transcript: its notes
        assert code == 0 and obj["record"]["command"] == "notes" and obj["record"]["parent"] == rid
    code, obj, _ = cli_json(capsys, "notes", ref, "--check", n)
    if not kept and form == "id":
        assert code == 2 and obj["error"]["code"] == "unknown_record"
    else:
        assert code == 0 and obj["unchanged"] is True
