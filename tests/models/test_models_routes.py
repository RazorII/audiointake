"""Real models, end to end: the canary clip, speaker counts, and each route a path fixture exercises.

Run by scripts/models-local.sh on a Mac with Apple silicon, against a data folder where `audiointake setup` has
installed every pack (AUDIOINTAKE_TEST_HOME), with the pinned public audio fetched (tools/fetch_test_audio.py). Metrics
and route signals only, never bytes: byte-for-byte comparisons with the reference implementation aren't part of this repository.
"""
from __future__ import annotations

import json

import pytest

from models_support import (build_fixtures, check_route, cut_wav, needs_apple_silicon, report, transcribe,
                            transcript_text)

pytestmark = pytest.mark.models


@needs_apple_silicon
def test_canary_clip_end_to_end(models_env, pinned_audio, tmp_path):
    from importlib import resources
    core = json.loads(resources.files("audiointake.setup").joinpath("packs/core.json").read_text(encoding="utf-8"))
    canary = core["canary"]
    ami = pinned_audio("ami-es2004a-825-915")
    clip = cut_wav(ami, tmp_path / "canary.wav", 825.0, 855.0, 825.0)
    _, rec = transcribe(models_env, clip, tmp_path / "out", fmt="json")
    text = transcript_text(tmp_path / "out", "canary").lower()
    missing = [w for w in canary["must_find"] if w.lower() not in text]
    assert not missing, f"{len(missing)} of the canary's words weren't found"
    assert rec["signals"]["speaker_count"] >= canary.get("min_speakers", 2)


@needs_apple_silicon
def test_ami_excerpt_has_four_speakers(models_env, pinned_audio, tmp_path):
    ami = pinned_audio("ami-es2004a-825-915")
    _, rec = transcribe(models_env, ami, tmp_path / "out", "--no-names", fmt="json")
    assert rec["signals"]["speaker_count"] == 4
    report({"test": "ami_excerpt_speakers", "speaker_count": rec["signals"]["speaker_count"],
            "seconds": rec["timing"]["seconds"], "rtf": rec["timing"].get("rtf")})


@pytest.fixture
def path_fixtures(pinned_audio, tmp_path):
    """Built from the pinned clips (a second or two; skips, naming the fetch command, when any clip is missing)."""
    return build_fixtures(pinned_audio, tmp_path / "fixtures")


@needs_apple_silicon
def test_vote_fixture_takes_the_vote(models_env, path_fixtures, tmp_path):
    _, rec = transcribe(models_env, path_fixtures / "vote.wav", tmp_path / "out", "--no-names", fmt="json")
    check_route(rec, "vote", "vote.wav")


@needs_apple_silicon
def test_turns_fixture_yields_thai_turns(models_env, path_fixtures, tmp_path):
    _, rec = transcribe(models_env, path_fixtures / "turns.wav", tmp_path / "out", "--no-names", fmt="json")
    check_route(rec, "thai_turns", "turns.wav")


@needs_apple_silicon
def test_core_route_and_formats(models_env, path_fixtures, tmp_path):
    for name in ("gapfill-de.wav", "gapfill-es.wav", "aac-m4a.m4a"):
        _, rec = transcribe(models_env, path_fixtures / name, tmp_path / name, "--no-names", fmt="json")
        assert rec["engines"]["asr_tag"] == "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill", name
        check_route(rec, None, name)


@needs_apple_silicon
def test_spatial_and_silence(models_env, path_fixtures, tmp_path):
    _, rec = transcribe(models_env, path_fixtures / "spatial-qta.qta", tmp_path / "spatial", "--no-names", fmt="json")
    assert rec["input"]["spatial"] in (True, "yes")
    # The streams hold different speech: 60 s in the first, 30 s in the 4-channel one. ffmpeg's own choice (no -map,
    # as the reference engine) is the first, so speech is transcribed past the second stream's end.
    tj = json.loads((tmp_path / "spatial" / "spatial-qta (transcript).json").read_text(encoding="utf-8"))
    assert max(s["end"] for s in tj["asr"]["sentences"]) > 35.0
    _, rec = transcribe(models_env, path_fixtures / "silence.wav", tmp_path / "silence", "--no-names", fmt="json")
    assert rec["signals"]["speaker_count"] == 0
    assert "no_speech" in {n["code"] for n in rec["notes"]}
