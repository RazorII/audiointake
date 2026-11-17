"""Exit codes, the exception classes, and the closed registry of codes with every message audiointake prints.

Every error, warning, note and offer audiointake reports is a code in ``CODES``; its words are core's, from the
templates below, filled only from record params (numbers, codes, pack and label ids, file names, names the user
gave) and from ``setup --plan`` (sizes), never from model output. The CLI, the documentation tests and the
Claude skill all read this one table.

A run never asks: there is no "decision needed" exit. What a person could do next travels as an offer on the
record. With several records the process exits with the first of 2, 3, 1, 4, 5, 0 present among them.

Params, the same for every producer:

* ``code`` in params is always a language code (``unknown_language``, ``pack_not_installed``, the ``language``
  offer), never the problem's own code; ``problem()``, the exceptions and ``error_for()`` take the problem's code
  positionally only, so ``code=`` is free for it.
* ``pack_not_installed`` picks its wording from ``reason`` (``detected`` | ``asked`` | ``no_room``) and, for
  ``detected``, from ``code``, the language detected. A detected wording that quotes word-error figures takes them
  as ``core_wer`` and ``pack_wer``, from ``PACK_WER`` (``pack_wer_params``), measured on FLEURS through the
  package and filled in once.
* Producers pass numbers as numbers; ``render`` formats each field one way for everyone (``NUMBER_FORMATS``:
  gigabytes to one decimal, scores and seconds cut, never rounded up, so a value under a threshold never prints
  as the threshold). A list or tuple becomes ``"a, b"`` in params and words.
* ``id`` (the record id) and ``log`` (the run log's path) are never set by the step that finds a problem: the
  record writer adds them to every problem whose template names them (``with_context``) before it saves or
  renders the record.
"""
from __future__ import annotations

import decimal
import math
import string
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import Problem

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_SETUP, EXIT_REFUSED, EXIT_STALE = 0, 1, 2, 3, 4, 5
EXIT_PRIORITY = (2, 3, 1, 4, 5, 0)       # several records: the process exits with the first present
EXIT_SIGHUP, EXIT_SIGINT, EXIT_SIGTERM = 129, 130, 143   # stopped: children killed, work folder and temporary
                                                        # outputs removed, no records (SIGINT is Ctrl-C)

KINDS = ("error", "warning", "note", "offer")

# The closed registry: code -> (kind, exit or None). kind: error | warning | note | offer.
# ``pack_broken`` and ``pack_outdated`` are exit-3 errors for the core pack and warnings for any other pack
# (ALSO_WARNING).
CODES: dict[str, tuple[str, int | None]] = {
    # errors, exit 2: usage, per input where it applies
    "bad_arguments": ("error", EXIT_USAGE),
    "input_missing": ("error", EXIT_USAGE),
    "input_not_file": ("error", EXIT_USAGE),
    "input_not_downloaded": ("error", EXIT_USAGE),
    "input_not_readable": ("error", EXIT_USAGE),
    "unsupported_input": ("error", EXIT_USAGE),
    "output_not_writable": ("error", EXIT_USAGE),
    "output_exists": ("error", EXIT_USAGE),
    "unknown_language": ("error", EXIT_USAGE),
    "unknown_record": ("error", EXIT_USAGE),
    "no_offer": ("error", EXIT_USAGE),
    "not_an_option": ("error", EXIT_USAGE),
    "no_transcript_json": ("error", EXIT_USAGE),
    "unknown_label": ("error", EXIT_USAGE),
    "name_invalid": ("error", EXIT_USAGE),
    "voice_not_found": ("error", EXIT_USAGE),
    "core_required": ("error", EXIT_USAGE),
    "head_invalid": ("error", EXIT_USAGE),
    "voice_exists": ("error", EXIT_USAGE),
    "voice_other_model": ("error", EXIT_USAGE),
    # errors, exit 3: setup
    "platform_not_macos": ("error", EXIT_SETUP),
    "platform_not_apple_silicon": ("error", EXIT_SETUP),
    "platform_rosetta": ("error", EXIT_SETUP),
    "macos_too_old": ("error", EXIT_SETUP),
    "not_set_up": ("error", EXIT_SETUP),
    "ffmpeg_missing": ("error", EXIT_SETUP),
    "clt_missing": ("error", EXIT_SETUP),
    "xcode_license": ("error", EXIT_SETUP),
    "swift_too_old": ("error", EXIT_SETUP),
    "speakers_build_failed": ("error", EXIT_SETUP),
    "build_timeout": ("error", EXIT_SETUP),
    "no_room": ("error", EXIT_SETUP),
    "download_failed": ("error", EXIT_SETUP),
    "model_changed": ("error", EXIT_SETUP),
    "pack_broken": ("error", EXIT_SETUP),
    "pack_outdated": ("error", EXIT_SETUP),
    "model_load_failed": ("error", EXIT_SETUP),
    # errors, exit 1: a file failed and nothing was written for it
    "asr_failed": ("error", EXIT_FAILED),
    "decode_failed": ("error", EXIT_FAILED),
    "internal_error": ("error", EXIT_FAILED),
    # errors, exit 4: refused, nothing written
    "too_little_speech": ("error", EXIT_REFUSED),
    "two_voices": ("error", EXIT_REFUSED),
    "mixed_voices": ("error", EXIT_REFUSED),
    "matches_someone_else": ("error", EXIT_REFUSED),
    "not_matching_print": ("error", EXIT_REFUSED),
    # errors, exit 5: stale
    "input_changed": ("error", EXIT_STALE),
    "output_changed": ("error", EXIT_STALE),
    "transcript_edited": ("error", EXIT_STALE),
    # warnings
    "pack_not_installed": ("warning", None),
    "speech_not_transcribed": ("warning", None),
    "no_speaker_labels": ("warning", None),
    "speakers_failed": ("warning", None),
    "merge_failed": ("warning", None),
    "speaker_cap": ("warning", None),
    "speaker_count_differs": ("warning", None),
    "fill_failed": ("warning", None),
    "hungarian_vote_failed": ("warning", None),
    "thai_failed": ("warning", None),
    "memory_unmeasured": ("warning", None),
    "canary_unavailable": ("warning", None),
    "canary_failed": ("warning", None),
    "fluidaudio_tag_moved": ("warning", None),
    # notes
    "already_done": ("note", None),
    "voices_changed": ("note", None),
    "no_speech": ("note", None),
    "recorded_at_from_file_date": ("note", None),
    "voices_model_mismatch": ("note", None),
    "voices_skipped": ("note", None),
    "pack_fetched": ("note", None),
    "already_enrolled": ("note", None),
    # offers (the kind of an offer on a record; acted on with apply or enroll)
    "language": ("offer", None),
    "speakers": ("offer", None),
    "names": ("offer", None),
    "remember": ("offer", None),
}

ALSO_WARNING = frozenset({"pack_broken", "pack_outdated"})   # an error for core, a warning (route off) for others

# The flag an offer is acted on with (``apply_flag`` on a record's offer); ``remember`` goes through
# ``enroll NAME --from REF --label LABEL`` only, after the user names the person and has their agreement.
OFFER_FLAGS: dict[str, str | None] = {
    "language": "--lang",
    "speakers": "--choice speakers",
    "names": "--choice names",
    "remember": None,
}

# --------------------------------------------------------------------------- messages
# {field} is filled from params; a field missing from params stays as written, so a message never fails to render.
MESSAGES: dict[str, str] = {
    # errors
    "bad_arguments": "{detail}. Run `audiointake {command} --help` for the options.",
    "input_missing": "{file} doesn't exist.",
    "input_not_file": "{file} isn't a regular file. Pass the audio files themselves, not a folder.",
    "input_not_downloaded": "{file} is in iCloud Drive but not downloaded to this Mac yet. In Finder, right-click it "
                            "and choose Download Now, then run this again.",
    "input_not_readable": "macOS didn't let audiointake read {file}. In System Settings → Privacy & Security → Files "
                          "and Folders, allow the app you run it from (Terminal or Claude) to use that folder, then "
                          "run this again.",
    "unsupported_input": "{file} has no audio audiointake can find. It reads any recording ffmpeg can decode: m4a, "
                         "mp3, wav, flac, ogg, opus, webm, mp4, mov, Voice Memos' qta and more.",
    "output_not_writable": "audiointake can't write into {folder}. Allow the app you run it from in System Settings "
                           "→ Privacy & Security → Files and Folders, or use `--outdir` to write somewhere else.",
    "output_exists": "{out} already exists and wasn't made from this recording with these settings, so it was left "
                     "alone. Use `--overwrite` to replace it, or `--outdir` to write elsewhere.",
    "unknown_language": "audiointake doesn't know the language code {code}. It reads {codes}.",
    "unknown_record": "There's no record {id} on this Mac, and no transcript file beside {path} to rebuild it from.",
    "no_offer": "Record {id} has no {kind} offer, so there's nothing for `{flag}` to act on.",
    "not_an_option": "{option} isn't one of the options record {id} offers: {options}.",
    "no_transcript_json": "{command} needs the transcript's JSON file, which this run didn't write or which has "
                          "moved: {path}.",
    "unknown_label": "Record {id} has no speaker {label}. Its speakers are {labels}.",
    "name_invalid": "A name needs 1 to 64 characters, without '/' or line breaks, and not starting with '.'.",
    "voice_not_found": "There's no voiceprint called {name}.",
    "voice_exists": "There's already a voiceprint called {name}.",
    "voice_other_model": "{file} in the voiceprint folder isn't a voiceprint made with this speaker model, so {name} "
                         "can't be added to it. `audiointake voices remove '{name}'` deletes it.",
    "core_required": "The core pack can't be removed: audiointake needs it for every transcript. "
                     "`audiointake uninstall` removes everything.",
    "head_invalid": "{path} should hold only the notes that go above the transcript, and it already has a "
                    "`## Transcript` section. Nothing was changed.",
    "platform_not_macos": "audiointake runs only on a Mac with Apple silicon (M1 or later). This computer runs {os}, "
                          "so nothing was installed.",
    "platform_not_apple_silicon": "audiointake needs a Mac with Apple silicon (M1 or later). This Mac has an Intel "
                                  "processor, and the speech models audiointake uses run only on Apple silicon. "
                                  "Nothing was installed.",
    "platform_rosetta": "This copy of audiointake runs under Rosetta, as an Intel program, so its speech models "
                        "can't use the Mac's GPU. Reinstall it with an Apple-silicon Python: `uv tool install "
                        "--force --python 3.14 https://github.com/RazorII/audiointake/archive/refs/tags/"
                        "v{version}.tar.gz`.",
    "macos_too_old": "audiointake needs macOS 14 Sonoma or later, and this Mac runs macOS {macos}. Update it in "
                     "System Settings → General → Software Update, then run `audiointake setup` again.",
    "not_set_up": "audiointake isn't set up yet. Run `audiointake setup` once: it checks this Mac, downloads the "
                  "core models (about {size} GB) and builds the speaker program.",
    "ffmpeg_missing": "audiointake reads audio with ffmpeg, which isn't installed. Install it with Homebrew: "
                      "`brew install ffmpeg` (no Homebrew yet? https://brew.sh). Then run this again.",
    "clt_missing": "Speaker labels need a small program built on this Mac with Apple's Command Line Tools, which "
                   "aren't installed. Run `xcode-select --install`, click Install in the window that opens, and run "
                   "`audiointake setup` again when it's done. Until then, recordings are transcribed without speaker "
                   "labels.",
    "xcode_license": "The speaker program is built with Xcode, and Xcode's licence hasn't been accepted on this Mac. "
                     "Run `sudo xcodebuild -license accept` in Terminal (it asks for your password), then "
                     "`audiointake setup` again. Until then, recordings are transcribed without speaker labels.",
    "swift_too_old": "The speaker program needs Swift 6.0 or later, and this Mac's Command Line Tools have Swift "
                     "{swift}. Update them in System Settings → General → Software Update, then run `audiointake "
                     "setup` again.",
    "speakers_build_failed": "Building the speaker program (FluidAudio {tag}) failed, so recordings will be "
                             "transcribed without speaker labels. The build log is at {path}. Run `audiointake "
                             "setup` again to retry.",
    "build_timeout": "Building the speaker program stopped after {minutes} minutes without finishing. Run "
                     "`audiointake setup` again; it continues from where it stopped.",
    "no_room": "{what} needs {need} GB of free space, and this Mac has {free} GB. Free some space, then run "
               "`audiointake setup` again.",
    "download_failed": "Couldn't download {model} from {host}. A network filter or company firewall may block that "
                       "host; PRIVACY.md lists every host setup uses. Run `audiointake setup` again to resume.",
    "model_changed": "{file} from {host} doesn't match the fingerprint pinned in audiointake {version}, so it wasn't "
                     "kept. Nothing else changed.",
    "model_load_failed": "The {model} model was downloaded and checked, but it didn't load or convert on this Mac, "
                         "so the {pack} pack wasn't set up. Run `audiointake setup` again.",
    "asr_failed": "Transcribing {file} failed, so nothing was written for it. The log is at {log}.",
    "decode_failed": "ffmpeg couldn't read the audio in {file}. It may be damaged or copy-protected.",
    "internal_error": "Something went wrong inside audiointake while working on {file}, so nothing was written for "
                      "it. The log at {log} says where; please report it with that log.",
    "two_voices": "{file} sounds like two different voices, so it wasn't used for {name}'s voiceprint. Use a "
                  "recording where only {name} talks.",
    "mixed_voices": "{percent} % of this recording doesn't sound like {name}, so it wasn't added.",
    "matches_someone_else": "This voice already matches {other}'s voiceprint, so it wasn't saved as {name}. If it "
                            "really is {name}, it can be added anyway with `--force`.",
    "not_matching_print": "This recording doesn't sound enough like {name}'s existing voiceprint (score {score}; "
                          "naming needs 0.44 and a 0.08 lead), so it wasn't added. If it really is {name}, add it "
                          "with `--force`.",
    "input_changed": "{file} changed while it was being worked on, or since record {id} was made, so nothing was "
                     "written. Run this again.",
    "output_changed": "{out} was changed after audiointake wrote it, so it was left alone. Move it aside to keep "
                      "your version, or pass `--overwrite` to replace it.",
    "transcript_edited": "{path} was changed after audiointake wrote it, so it can't be trusted to rebuild from, and "
                         "nothing was changed. `audiointake transcribe {file} --overwrite` makes a fresh one.",
    # warnings
    "speech_not_transcribed": "{minutes} min {seconds} s of speech in this recording produced no words. It may be in "
                              "a language the installed models don't read, very quiet, or people talking over each "
                              "other.",
    "speakers_failed": "Telling the speakers apart failed for this recording ({step}), so it has {fallback}. The log "
                       "is at {log}.",
    "merge_failed": "Putting speaker labels on the words failed, so this transcript has no labels. The log is at "
                    "{log}.",
    "speaker_cap": "audiointake tells apart at most 8 speakers, and this recording reached that, so some people may "
                   "share a label.",
    "speaker_count_differs": "You said {expected} people were speaking; audiointake heard {found}. If two people "
                             "share a label, or one person got two, naming them fixes the transcript: `audiointake "
                             "relabel {id} --label speaker_K --name NAME`.",
    "fill_failed": "The second speech model failed on this recording, so the transcript comes from the first alone "
                   "and may miss a few words.",
    "hungarian_vote_failed": "This recording is Hungarian, but the Hungarian model failed on it, so it was read with "
                             "the core models only.",
    "thai_failed": "Checking this recording for Thai failed, so any Thai in it may come out as English-sounding "
                   "nonsense.",
    "memory_unmeasured": "This Mac has {gb} GB of memory. audiointake hasn't been measured on a Mac this size yet; "
                         "long recordings may be slow.",
    "canary_unavailable": "The test recording couldn't be fetched from {host}, so setup checked the models by their "
                          "fingerprints and a short test only.",
    "canary_failed": "Setup's test recording didn't come out as expected ({found} of {of} words, {speakers} "
                     "speakers), so something may be wrong with the models on this Mac. `audiointake doctor "
                     "--selftest` runs it again.",
    "fluidaudio_tag_moved": "FluidAudio's tag {tag} now points at a different version upstream; audiointake built "
                            "the exact version it was tested with ({commit}) anyway.",
    # notes
    "already_done": "{file} was already transcribed with these settings; nothing was redone.",
    "voices_changed": "{file} was already transcribed with these settings, but the voiceprints have changed since; "
                      "`audiointake apply {id} --choice names` updates the names without transcribing again.",
    "no_speech": "No speech was found in {file}; its transcript is empty.",
    "recorded_at_from_file_date": "The recording date comes from the file's date, which may be when it was copied.",
    "voices_model_mismatch": "{count} voiceprint(s) were made with another speaker model and were left out.",
    "voices_skipped": "{count} file(s) in the voiceprint folder aren't voiceprints and were skipped.",
    "pack_fetched": "Downloaded the {pack} pack ({size} GB).",
    "already_enrolled": "Nothing was added: {name}'s voiceprint already has these recordings.",
}

# Codes whose words depend on params; ``variant_of`` picks one, and ``DEFAULT_VARIANT`` is used when it can't.
VARIANTS: dict[str, dict[str, str]] = {
    "pack_not_installed": {
        "detected_hu": "About {share} % of the words in this recording look Hungarian. It was transcribed with the "
                       "core models; the Hungarian pack reads Hungarian better (word errors on public test speech: "
                       "{core_wer} % with core, {pack_wer} % with the pack). `audiointake apply {id} --lang hu` "
                       "downloads it (+{size} GB) and redoes this recording.",
        "detected_th": "Part of this recording is Thai, which the core models can't transcribe: those turns are "
                       "missing or come out as English-sounding nonsense. `audiointake apply {id} --lang th` "
                       "downloads the Thai pack (+{size} GB) and redoes this recording.",
        "asked": "You asked for {language}, but its pack isn't installed, so the core models read this recording. "
                 "`audiointake apply {id} --lang {code}` downloads it (+{size} GB) and redoes it.",
        "no_room": "{language} was detected, but its pack needs {need} GB and this Mac has {free} GB free, so the "
                   "core models read it.",
    },
    "too_little_speech": {
        "clip": "{file} has {seconds} s of clear speech from one voice, and a voiceprint needs at least 8 s. Record "
                "30 to 60 seconds of {name} talking on their own (reading anything aloud works), or use a longer "
                "recording they're in. Nothing was saved.",
        "label": "{label} has {seconds} s of speech without anyone talking over them in this recording, and a "
                 "voiceprint needs at least 8 s. Nothing was saved.",
    },
    "pack_broken": {
        "core": "A file of the {pack} pack has changed on disk since it was checked ({file}). Run `audiointake setup` "
                "to repair it.",
        "pack": "A file of the {pack} pack has changed on disk since it was checked ({file}). Run `audiointake setup` "
                "to repair it. Until then, {language} is read with the core models.",
    },
    "no_speaker_labels": {
        "record": "This transcript has no speaker labels, because the speaker program isn't set up ({reason}). Once "
                  "it is, `audiointake apply {id} --choice speakers` adds them without transcribing again.",
        "mac": "Transcripts made on this Mac have no speaker labels yet, because the speaker program isn't set up "
               "({reason}). Once it is, `audiointake apply` adds them to those transcripts without transcribing "
               "again.",
    },
    "pack_outdated": {
        "core": "The {pack} pack on this Mac was set up for another version of audiointake ({file}). Run "
                "`audiointake setup` to bring it up to date.",
        "pack": "The {pack} pack on this Mac was set up for another version of audiointake ({file}). Run "
                "`audiointake setup` to bring it up to date. Until then, {language} is read with the core models.",
    },
}
DEFAULT_VARIANT = {"pack_not_installed": "asked", "too_little_speech": "clip", "pack_broken": "core",
                   "pack_outdated": "core", "no_speaker_labels": "record"}

# Word errors on FLEURS (percent) with core and with the pack, per pack whose detected wording quotes them:
# (core_wer, pack_wer), measured through the package itself (never typed in from elsewhere); None until measured.
# hu: FLEURS hu_hu, 50 clips, through the package's own routes: core 14.80 %, the vote route 12.84 % (bench/fleurs_routes.py
# and bench/fleurs.py score; identical, clip for clip, to the same systems measured on the reference implementation).
PACK_WER: dict[str, tuple[float, float] | None] = {"hu": (14.8, 12.84)}

# Offer summaries: the skill relays these words.
OFFERS: dict[str, str] = {
    "language": "{language} is read better with the {language} pack: `audiointake apply {id} --lang {code}` "
                "downloads it (+{size} GB) and redoes this recording.",
    "speakers": "Speaker labels can be added without transcribing again once the speaker program is set up: "
                "`audiointake apply {id} --choice speakers`.",
    "names": "The voiceprints have changed since this transcript was made: `audiointake apply {id} --choice names` "
             "updates the names without transcribing again.",
    "remember": "{label} has {seconds} s of clear speech. With that person's agreement, `audiointake enroll NAME "
                "--from {id} --label {label}` keeps a voiceprint so later recordings name them.",
}


# How render() writes a number in a field: (decimals, rounding). "floor" cuts toward zero, so a value just under a
# threshold ("7.96 s", "a score of 0.439") never prints as the threshold; "ceil" for space a step needs. Other
# number fields print as whole numbers when integral, else to one decimal.
NUMBER_FORMATS: dict[str, tuple[int, str]] = {
    "size": (1, "round"), "need": (1, "ceil"), "free": (1, "floor"), "gb": (0, "round"),
    "score": (2, "floor"), "seconds": (1, "floor"), "minutes": (0, "floor"),
    "percent": (0, "round"), "share": (0, "round"), "core_wer": (1, "round"), "pack_wer": (1, "round"),
}


def format_number(field: str, value: float) -> str:
    """One field's number as every message shows it (``NUMBER_FORMATS``); trailing ".0" is dropped."""
    if not math.isfinite(value):
        return str(value)
    places, how = NUMBER_FORMATS.get(field, (1, "round"))
    mode = {"round": decimal.ROUND_HALF_UP, "floor": decimal.ROUND_DOWN, "ceil": decimal.ROUND_UP}[how]
    q = decimal.Decimal(1).scaleb(-places)
    d = decimal.Decimal(repr(float(value))).quantize(q, rounding=mode)
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _word(field: str, value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return format_number(field, value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_word(field, v) for v in value)
    if isinstance(value, (set, frozenset)):
        return ", ".join(sorted(_word(field, v) for v in value))
    return str(value)


class _Keep(dict):
    """format_map's mapping: a field missing from params stays as written ("{size}")."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def variant_of(code: str, params: Mapping[str, Any] | None = None) -> str | None:
    """Which of a code's worded variants applies, from its params; None for a code with one wording.

    pack_not_installed: reason "detected" with code (the language) hu or th -> detected_hu / detected_th; reason
    "asked" or "no_room" -> that; too_little_speech: params["source"] "clip" or "label" (a "label" param alone means
    label); pack_broken and pack_outdated: pack "core" -> core, any other pack -> pack (the warning form);
    no_speaker_labels: scope "mac" (doctor, about this Mac, with no record to name) -> mac, else record.
    """
    if code not in VARIANTS:
        return None
    p = dict(params or {})
    if code == "pack_not_installed":
        reason = str(p.get("reason", ""))
        pick = f"detected_{p.get('code', '')}" if reason == "detected" else reason
    elif code == "too_little_speech":
        pick = str(p.get("source") or ("label" if "label" in p and "file" not in p else "clip"))
    elif code == "no_speaker_labels":
        pick = "mac" if p.get("scope") == "mac" else "record"
    else:
        pick = "core" if str(p.get("pack", "core")) == "core" else "pack"
    return pick if pick in VARIANTS[code] else DEFAULT_VARIANT[code]


def template(code: str, params: Mapping[str, Any] | None = None) -> str:
    """The message template for a code (an offer kind gives its summary). KeyError for an unknown code."""
    if code not in CODES:
        raise KeyError(code)
    if CODES[code][0] == "offer":
        return OFFERS[code]
    v = variant_of(code, params)
    return VARIANTS[code][v] if v is not None else MESSAGES[code]


def render(code: str, params: Mapping[str, Any] | None = None) -> str:
    """The user-facing words for a code, filled from params (numbers and lists formatted as ``_word`` does)."""
    return template(code, params).format_map(_Keep({k: _word(k, v) for k, v in (params or {}).items()}))


def problem(code: str, /, **params: Any) -> Problem:
    """A Problem for a known code, its params reduced to str, int and float (paths and the like become str, a list
    or tuple ``"a, b"``). The code is positional only, so a ``code=`` param (a language code) is free."""
    if code not in CODES:
        raise KeyError(code)
    return Problem(code, {k: _scalar(k, v) for k, v in params.items()})


def _scalar(key: str, v: Any) -> str | int | float:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple, set, frozenset)):
        return _word(key, v)
    return str(v)


def pack_wer_params(lang: str) -> dict[str, float]:
    """``{"core_wer", "pack_wer"}`` for a pack's detected wording, or {} while it hasn't been measured."""
    m = PACK_WER.get(lang)
    return {} if m is None else {"core_wer": m[0], "pack_wer": m[1]}


CONTEXT_FIELDS = ("id", "log")


def with_context(p: Problem, **context: Any) -> Problem:
    """``p`` with ``id`` / ``log`` added where its template names them and its params don't have them yet (the
    record writer's step; nothing else sets these two)."""
    unknown = set(context) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"not a context field: {sorted(unknown)}")
    names = {f for _, f, _, _ in string.Formatter().parse(template(p.code, p.params)) if f}
    add = {k: _scalar(k, v) for k, v in context.items() if k in names and k not in p.params and v is not None}
    return p if not add else Problem(p.code, {**p.params, **add})


def exit_for(code: str) -> int | None:
    """The exit an error code means (None for warnings, notes and offers)."""
    return CODES[code][1]


def process_exit(exits: Iterable[int]) -> int:
    """The process exit for several records: the first of 2, 3, 1, 4, 5, 0 present; 0 for none."""
    present = set(exits)
    unknown = present - set(EXIT_PRIORITY)
    if unknown:
        raise ValueError(f"not a record exit: {sorted(unknown)}")
    for e in EXIT_PRIORITY:
        if e in present:
            return e
    return EXIT_OK


# --------------------------------------------------------------------------- exceptions
class AudioIntakeError(Exception):
    """Base class. ``exit`` is per class; ``code`` (from CODES) and ``detail`` (the params) per instance.

    Without a message the code's template is rendered from ``detail``.
    """

    exit = EXIT_FAILED

    def __init__(self, code: str, /, message: str = "", **detail: Any) -> None:
        if code not in CODES:
            raise ValueError(f"{code!r} isn't a code in errors.CODES")
        self.code = code
        self.detail: dict[str, Any] = dict(detail)
        self.message = message or render(code, self.detail)
        super().__init__(self.message)

    @property
    def params(self) -> dict[str, Any]:
        return self.detail

    def problem(self) -> Problem:
        return problem(self.code, **self.detail)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.code!r}, {self.message!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (type(self), self.code, self.message, self.detail))


def _rebuild(cls: type[AudioIntakeError], code: str, message: str, detail: dict[str, Any]) -> AudioIntakeError:
    return cls(code, message, **detail)


class UsageError(AudioIntakeError):
    exit = EXIT_USAGE


class SetupError(AudioIntakeError):
    exit = EXIT_SETUP


class Refused(AudioIntakeError):
    exit = EXIT_REFUSED


class Stale(AudioIntakeError):
    exit = EXIT_STALE


class AsrFailed(AudioIntakeError):
    exit = EXIT_FAILED


_CLASS_FOR_EXIT: dict[int, type[AudioIntakeError]] = {
    EXIT_FAILED: AudioIntakeError, EXIT_USAGE: UsageError, EXIT_SETUP: SetupError,
    EXIT_REFUSED: Refused, EXIT_STALE: Stale,
}


def error_for(code: str, /, message: str = "", **detail: Any) -> AudioIntakeError:
    """The exception for an error code, of the class its exit needs (asr_failed is an AsrFailed)."""
    kind, ex = CODES[code]
    if kind != "error" or ex is None:
        raise ValueError(f"{code!r} is a {kind}, not an error")
    cls = AsrFailed if code == "asr_failed" else _CLASS_FOR_EXIT[ex]
    return cls(code, message, **detail)
