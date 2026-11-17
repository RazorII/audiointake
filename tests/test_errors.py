"""errors.py: the closed registry, every message and offer template, exit priority, the exception classes."""
from __future__ import annotations

import pickle
import re
import string

import pytest

from audiointake import errors
from audiointake.contracts import Problem

ERRORS = {
    2: ["bad_arguments", "input_missing", "input_not_file", "input_not_downloaded", "input_not_readable",
        "unsupported_input", "output_not_writable", "output_exists", "unknown_language", "unknown_record", "no_offer",
        "not_an_option", "no_transcript_json", "unknown_label", "name_invalid", "voice_not_found", "core_required",
        "head_invalid", "voice_exists", "voice_other_model"],
    3: ["platform_not_macos", "platform_not_apple_silicon", "platform_rosetta", "macos_too_old", "not_set_up",
        "ffmpeg_missing", "clt_missing", "xcode_license", "swift_too_old", "speakers_build_failed", "build_timeout",
        "no_room", "download_failed", "model_changed", "pack_broken", "pack_outdated", "model_load_failed"],
    1: ["asr_failed", "decode_failed", "internal_error"],
    4: ["too_little_speech", "two_voices", "mixed_voices", "matches_someone_else", "not_matching_print"],
    5: ["input_changed", "output_changed", "transcript_edited"],
}
WARNINGS = ["pack_not_installed", "speech_not_transcribed", "no_speaker_labels", "speakers_failed", "merge_failed",
            "speaker_cap", "speaker_count_differs", "fill_failed", "hungarian_vote_failed", "thai_failed",
            "memory_unmeasured", "canary_unavailable", "canary_failed", "fluidaudio_tag_moved"]
NOTES = ["already_done", "voices_changed", "no_speech", "recorded_at_from_file_date", "voices_model_mismatch",
         "voices_skipped", "pack_fetched", "already_enrolled"]
OFFERS = ["language", "speakers", "names", "remember"]

# The fields each template fills (every variant of a code together).
FIELDS = {
    "bad_arguments": {"detail", "command"}, "input_missing": {"file"}, "input_not_file": {"file"},
    "input_not_downloaded": {"file"}, "input_not_readable": {"file"}, "unsupported_input": {"file"},
    "output_not_writable": {"folder"}, "output_exists": {"out"}, "unknown_language": {"code", "codes"},
    "unknown_record": {"id", "path"}, "no_offer": {"id", "kind", "flag"}, "not_an_option": {"option", "id", "options"},
    "no_transcript_json": {"command", "path"}, "unknown_label": {"id", "label", "labels"}, "name_invalid": set(),
    "voice_not_found": {"name"}, "voice_exists": {"name"}, "voice_other_model": {"file", "name"},
    "core_required": set(), "head_invalid": {"path"}, "platform_not_macos": {"os"},
    "platform_not_apple_silicon": set(), "platform_rosetta": {"version"}, "macos_too_old": {"macos"},
    "not_set_up": {"size"}, "ffmpeg_missing": set(), "clt_missing": set(), "xcode_license": set(),
    "swift_too_old": {"swift"}, "speakers_build_failed": {"tag", "path"}, "build_timeout": {"minutes"},
    "no_room": {"what", "need", "free"}, "download_failed": {"model", "host"},
    "model_changed": {"file", "host", "version"}, "pack_broken": {"pack", "file", "language"},
    "pack_outdated": {"pack", "file", "language"}, "model_load_failed": {"model", "pack"},
    "asr_failed": {"file", "log"}, "decode_failed": {"file"}, "internal_error": {"file", "log"},
    "too_little_speech": {"file", "seconds", "name", "label"}, "two_voices": {"file", "name"},
    "mixed_voices": {"percent", "name"}, "matches_someone_else": {"other", "name"},
    "not_matching_print": {"name", "score"}, "input_changed": {"file", "id"}, "output_changed": {"out"},
    "transcript_edited": {"path", "file"},
    "pack_not_installed": {"share", "core_wer", "pack_wer", "id", "size", "language", "code", "need", "free"},
    "speech_not_transcribed": {"minutes", "seconds"}, "no_speaker_labels": {"reason", "id"},
    "speakers_failed": {"step", "fallback", "log"}, "merge_failed": {"log"}, "speaker_cap": set(),
    "speaker_count_differs": {"expected", "found", "id"}, "fill_failed": set(), "hungarian_vote_failed": set(),
    "thai_failed": set(), "memory_unmeasured": {"gb"}, "canary_unavailable": {"host"},
    "canary_failed": {"found", "of", "speakers"},
    "fluidaudio_tag_moved": {"tag", "commit"}, "already_done": {"file"}, "voices_changed": {"file", "id"},
    "no_speech": {"file"}, "recorded_at_from_file_date": set(), "voices_model_mismatch": {"count"},
    "voices_skipped": {"count"}, "pack_fetched": {"pack", "size"}, "already_enrolled": {"name"},
    "language": {"language", "id", "code", "size"}, "speakers": {"id"}, "names": {"id"},
    "remember": {"label", "seconds", "id"},
}


def fields_of(t: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(t) if f}


def all_templates(code: str) -> list[str]:
    if code in errors.OFFERS:
        return [errors.OFFERS[code]]
    if code in errors.VARIANTS:
        return list(errors.VARIANTS[code].values())
    return [errors.MESSAGES[code]]


def test_registry_is_exactly_the_documented_codes():
    want = {c: ("error", ex) for ex, cs in ERRORS.items() for c in cs}
    want |= {c: ("warning", None) for c in WARNINGS}
    want |= {c: ("note", None) for c in NOTES}
    want |= {c: ("offer", None) for c in OFFERS}
    assert errors.CODES == want
    assert errors.ALSO_WARNING == {"pack_broken", "pack_outdated"}
    assert set(errors.OFFER_FLAGS) == set(OFFERS)


def test_every_code_has_words_and_nothing_else_does():
    worded = set(errors.MESSAGES) | set(errors.VARIANTS) | set(errors.OFFERS)
    assert worded == set(errors.CODES)
    assert not (set(errors.MESSAGES) & set(errors.VARIANTS))
    assert set(errors.OFFERS) == set(OFFERS)


@pytest.mark.parametrize("code", sorted(FIELDS))
def test_template_fields(code):
    got = set().union(*(fields_of(t) for t in all_templates(code)))
    assert got == FIELDS[code]


@pytest.mark.parametrize("code", sorted(FIELDS))
def test_renders_fully_with_its_params(code):
    params = {f: f"<{f}>" for f in FIELDS[code]}
    for t in all_templates(code):
        out = t.format_map(params)
        assert "{" not in out and "}" not in out
        assert out.strip() == out and out.endswith((".", ")", "`.", ".)"))


def test_messages_are_plain_and_never_ask():
    for code in errors.CODES:
        for t in all_templates(code):
            assert not re.search(r"\b(would you|do you want|shall I|please confirm)\b", t, re.I), code
            assert "\n" not in t


def test_missing_params_stay_visible():
    assert errors.render("not_set_up") == ("audiointake isn't set up yet. Run `audiointake setup` once: it checks "
                                           "this Mac, downloads the core models (about {size} GB) and builds the "
                                           "speaker program.")


@pytest.mark.parametrize("params,variant", [
    ({"reason": "detected", "code": "hu"}, "detected_hu"), ({"reason": "detected", "code": "th"}, "detected_th"),
    ({"reason": "asked", "code": "hu"}, "asked"), ({"reason": "no_room"}, "no_room"),
    ({"reason": "detected", "code": "zz"}, "asked"), ({}, "asked"),
    ({"reason": "detected", "pack": "hu"}, "asked"),           # the language is `code`, never `pack`
])
def test_pack_not_installed_variants(params, variant):
    assert errors.variant_of("pack_not_installed", params) == variant


def test_pack_not_installed_detected_fills_both_figures():
    params = {"reason": "detected", "code": "hu", "share": 41.6, "core_wer": 20.14, "pack_wer": 9.85, "size": 3.08,
              "id": "r-20261116-1a2b3c4d"}
    out = errors.render("pack_not_installed", params)
    assert "About 42 % of the words" in out
    assert "20.1 % with core, 9.9 % with the pack" in out
    assert "(+3.1 GB)" in out and "{" not in out


def test_pack_wer_is_measured_and_the_pack_is_better():
    core, pack = errors.PACK_WER["hu"]
    assert 0 < pack < core < 100
    out = errors.render("pack_not_installed", {"reason": "detected", "code": "hu", **errors.pack_wer_params("hu")})
    assert "(word errors on public test speech: 14.8 % with core, 12.8 % with the pack)" in out


def test_pack_wer_params():
    assert set(errors.PACK_WER) == {"hu"}
    saved = dict(errors.PACK_WER)
    try:
        errors.PACK_WER["hu"] = None
        assert errors.pack_wer_params("hu") == {} and errors.pack_wer_params("xx") == {}
        errors.PACK_WER["hu"] = (20.1, 9.9)
        assert errors.pack_wer_params("hu") == {"core_wer": 20.1, "pack_wer": 9.9}
    finally:
        errors.PACK_WER.clear()
        errors.PACK_WER.update(saved)


# --------------------------------------------------------------------------- a `code` param (a language code)
@pytest.mark.parametrize("code,params", [
    ("unknown_language", {"code": "hu", "codes": ["de", "fr"]}),
    ("pack_not_installed", {"reason": "asked", "code": "hu", "language": "Hungarian", "size": 3.08,
                            "id": "r-20261116-1a2b3c4d"}),
    ("language", {"code": "hu", "language": "Hungarian", "size": 3.08, "id": "r-20261116-1a2b3c4d"}),
])
def test_a_code_param_never_collides_with_the_problem_code(code, params):
    p = errors.problem(code, **params)
    assert p.code == code and p.params["code"] == "hu"
    words = errors.render(code, p.params)
    assert "{" not in words and "hu" in words
    if errors.CODES[code][0] == "error":
        e = errors.error_for(code, **params)
        assert e.code == code and e.detail["code"] == "hu" and str(e) == words
        e2 = errors.UsageError(code, **params)
        assert e2.code == code and e2.problem() == p
        back = pickle.loads(pickle.dumps(e2))
        assert back.code == code and back.detail == e2.detail and back.message == e2.message
        assert errors.UsageError(code, message="own words", **params).message == "own words"


# --------------------------------------------------------------------------- numbers, lists, context
@pytest.mark.parametrize("field,value,want", [
    ("size", 4.25, "4.3"), ("size", 4, "4"), ("size", 0.38, "0.4"), ("need", 6.01, "6.1"), ("free", 6.09, "6"),
    ("score", 0.4399, "0.43"), ("score", 0.6129, "0.61"), ("seconds", 7.96, "7.9"), ("seconds", 41.0, "41"),
    ("seconds", 12, "12"), ("percent", 19.5, "20"), ("share", 8.4, "8"), ("minutes", 3.9, "3"), ("gb", 16, "16"),
    ("core_wer", 12.65, "12.7"), ("expected", 4, "4"), ("count", 2.0, "2"), ("seconds", -0.04, "0"),
])
def test_numbers_print_one_way(field, value, want):
    assert errors.format_number(field, value) == want


def test_a_value_under_a_threshold_never_prints_as_the_threshold():
    s = errors.render("too_little_speech", {"source": "label", "label": "speaker_1", "seconds": 7.99})
    assert s.startswith("speaker_1 has 7.9 s of speech")
    s = errors.render("not_matching_print", {"name": "Ana", "score": 0.4399})
    assert "(score 0.43;" in s


def test_lists_become_comma_separated_words():
    assert errors.render("unknown_language", {"code": "xx", "codes": ["bg", "cs", "da"]}).endswith(
        "It reads bg, cs, da.")
    p = errors.problem("not_an_option", option="xx", id="r-20261116-1a2b3c4d", options=["de", "fr"])
    assert p.params["options"] == "de, fr"
    assert errors.render("unknown_label", {"id": "r", "label": "speaker_9", "labels": ("speaker_0", "speaker_1")}
                         ).endswith("Its speakers are speaker_0, speaker_1.")


def test_with_context_adds_only_the_fields_a_template_names():
    rid = "r-20261116-1a2b3c4d"
    p = errors.problem("no_speaker_labels", reason="no build")
    q = errors.with_context(p, id=rid, log="/x/r.log")
    assert q.params == {"reason": "no build", "id": rid}
    assert errors.with_context(errors.problem("asr_failed", file="a.m4a"), id=rid, log="/x/r.log").params == {
        "file": "a.m4a", "log": "/x/r.log"}
    same = errors.problem("fill_failed")
    assert errors.with_context(same, id=rid, log="/x") is same
    kept = errors.problem("speakers", id="r-20261116-00000000")
    assert errors.with_context(kept, id=rid).params["id"] == "r-20261116-00000000"
    with pytest.raises(ValueError):
        errors.with_context(same, path="/x")


def test_other_variants():
    assert errors.variant_of("too_little_speech", {"source": "label"}) == "label"
    assert errors.variant_of("too_little_speech", {"label": "speaker_1", "seconds": 3}) == "label"
    assert errors.variant_of("too_little_speech", {"file": "a.m4a", "seconds": 3}) == "clip"
    assert errors.variant_of("pack_broken", {"pack": "core"}) == "core"
    assert errors.variant_of("pack_broken", {"pack": "th"}) == "pack"
    assert errors.variant_of("asr_failed", {}) is None
    w = errors.render("pack_broken", {"pack": "th", "file": "x", "language": "Thai"})
    assert w.endswith("Until then, Thai is read with the core models.")
    e = errors.render("pack_broken", {"pack": "core", "file": "x"})
    assert e.endswith("to repair it.")
    assert errors.variant_of("pack_outdated", {"pack": "core"}) == "core"
    assert errors.variant_of("pack_outdated", {"pack": "hu"}) == "pack"
    assert errors.variant_of("no_speaker_labels", {"reason": "x"}) == "record"
    assert errors.variant_of("no_speaker_labels", {"reason": "x", "scope": "mac"}) == "mac"
    assert "{id}" not in errors.render("no_speaker_labels", {"reason": "x", "scope": "mac"})
    o = errors.render("pack_outdated", {"pack": "hu", "file": "pack version 1, now 2", "language": "Hungarian"})
    assert "(pack version 1, now 2)" in o and o.endswith("Until then, Hungarian is read with the core models.")


def test_offer_render():
    s = errors.render("remember", {"label": "speaker_1", "seconds": 41, "id": "r-20261116-1a2b3c4d"})
    assert s.startswith("speaker_1 has 41 s of clear speech.")
    assert "enroll NAME --from r-20261116-1a2b3c4d --label speaker_1" in s


def test_unknown_code():
    with pytest.raises(KeyError):
        errors.render("nope")
    with pytest.raises(KeyError):
        errors.problem("nope")
    with pytest.raises(ValueError):
        errors.AudioIntakeError("nope")


# --------------------------------------------------------------------------- exits
def test_exit_constants():
    assert (errors.EXIT_OK, errors.EXIT_FAILED, errors.EXIT_USAGE, errors.EXIT_SETUP, errors.EXIT_REFUSED,
            errors.EXIT_STALE) == (0, 1, 2, 3, 4, 5)
    assert errors.EXIT_PRIORITY == (2, 3, 1, 4, 5, 0)


@pytest.mark.parametrize("exits,want", [([], 0), ([0, 0], 0), ([0, 5, 4], 4), ([1, 3], 3), ([5, 1, 2], 2),
                                        ([4, 5], 4), ([0, 1], 1)])
def test_process_exit(exits, want):
    assert errors.process_exit(exits) == want


def test_process_exit_refuses_a_non_record_exit():
    with pytest.raises(ValueError):
        errors.process_exit([0, 143])


# --------------------------------------------------------------------------- exceptions
@pytest.mark.parametrize("cls,ex", [(errors.AudioIntakeError, 1), (errors.UsageError, 2), (errors.SetupError, 3),
                                    (errors.Refused, 4), (errors.Stale, 5), (errors.AsrFailed, 1)])
def test_class_exits(cls, ex):
    assert cls.exit == ex
    assert issubclass(cls, errors.AudioIntakeError)


@pytest.mark.parametrize("code", sorted(c for c, (k, _) in errors.CODES.items() if k == "error"))
def test_error_for_every_error_code(code):
    e = errors.error_for(code, file="a.m4a")
    assert e.exit == errors.CODES[code][1] and e.code == code
    assert e.message == errors.render(code, {"file": "a.m4a"})
    assert isinstance(e, errors.AsrFailed) == (code == "asr_failed")


def test_error_for_refuses_non_errors():
    with pytest.raises(ValueError):
        errors.error_for("fill_failed")


def test_exception_carries_code_detail_and_problem(tmp_path):
    e = errors.UsageError("input_missing", file=tmp_path / "x.m4a", n=3, ok=True)
    assert str(e) == f"{tmp_path / 'x.m4a'} doesn't exist."
    assert e.params is e.detail
    p = e.problem()
    assert p == Problem("input_missing", {"file": str(tmp_path / "x.m4a"), "n": 3, "ok": 1})
    assert errors.UsageError("input_missing", "custom words").message == "custom words"


def test_exceptions_pickle():
    e = errors.Stale("output_changed", out="a (transcript).md")
    back = pickle.loads(pickle.dumps(e))
    assert type(back) is errors.Stale and back.code == e.code and back.message == e.message
    assert back.detail == e.detail
