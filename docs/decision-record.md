# The decision record

Every audiointake command that changes something writes a **record**: `transcribe` and `apply` one per recording, `relabel`, `notes --into` and `enroll` one each. The record is the interface between the CLI and anything that drives it, the Claude skill included. This page describes record version 1; the code that writes it is `src/audiointake/record.py`.

**A run never waits for an answer.** audiointake asks nothing during a run, and no record holds a question. Every choice comes from the flags the command was given or a default. What a person might still want afterwards (a language pack, speaker labels, updated names, a voiceprint) is listed in the record's `offers`, to mention once after the whole batch.

## Where records live

- Each record is saved to `<data>/records/<id>.json` (mode 0600, written atomically), where `<data>` is `AUDIOINTAKE_HOME` when set, otherwise `~/Library/Application Support/audiointake`. Records and run logs are deleted after 90 days.
- `transcribe --json` and `apply --json` print the same records on stdout as `{"exit": N, "records": [...]}`; `relabel`, `notes --into` and `enroll` print `{"exit": N, "record": {...}}`.
- `audiointake record show REF --json` prints a saved record again; `audiointake record find PATH --json` gives a recording's newest record id with its speaker labels and their displays.
- The run log, `~/Library/Logs/audiointake/<id>.log`, holds step timings, engine tags, codes and the last lines a failed step printed on its error stream. Never transcript text, and never anything a step printed on its output stream.

## Rules

1. **A record never holds transcript text**: no sentence, turn, word or model output. It holds numbers, codes, pack and label ids, hashes, file names and paths, and the names of speakers who were recognised or named by the user. The transcript itself is the `(transcript).json` beside the recording ([transcript-json.md](transcript-json.md)).
2. **It names only who was named.** `speakers` lists each label with its display and seconds of speech, and `name` only for a label a voiceprint named or the user named. Never the list of everyone with a voiceprint, a best guess or a runner-up.
3. **Every vocabulary is closed.** Statuses, offer kinds, warning, note and error codes come from fixed tables in `src/audiointake/errors.py`, and every message is core's own wording filled from the record's params.
4. **An offer is acted on with ids only**: `audiointake apply <record id> --lang <code>`, `--choice speakers` or `--choice names`, and `audiointake enroll NAME --from <record id> --label speaker_K`. The record id matches `^r-[0-9]{8}-[0-9a-f]{8}$` and a label `^speaker_[0-7]$`. Every command that takes a record id also takes the recording's path or its `(transcript).json` path, so a follow-up works after the record has expired. An offer belongs to the transcript, not to one record: every follow-up's record (and the transcript JSON it rewrites) carries the offers still open and leaves out the ones it settled, so any record id of the chain, the recording's path and the JSON's path all find them.
5. **Nothing waits.** A record's exit says how the command went, never that something is waiting.

## Fields

Every field is present; one with nothing to say is `null`, `{}` or `[]`.

| Field | What it holds |
|---|---|
| `record_version` | `1` |
| `id` | `r-YYYYMMDD-xxxxxxxx` |
| `parent` | the record this one follows from (`apply`, `relabel`, `notes`, `enroll --from`, and a `transcribe` that found its work already done), else `null` |
| `tool` | `{"version"}` |
| `command` | `transcribe`, `apply`, `relabel`, `notes` or `enroll` |
| `status` | `done`, `failed`, `refused`, `plan`, `stale` or `usage` |
| `exit` | this record's exit code (below) |
| `created` | local time with its offset |
| `input` | `{path, sha256, size, duration_s, channels, spatial, recorded_at, recorded_at_source}`: `channels` lists each audio stream's channel count; `spatial` is `yes` when a stream has exactly 4 channels; `recorded_at_source` is `filename` (a name starting `YYYYMMDD HHMMSS`), `metadata` (the container's creation time), `file_date` or `null` |
| `options` | `{lang, speakers, names, speakers_off, format, voices, voices_dir}`: `voices` is `default` or `custom` |
| `options_digest` | a hash of the options that change what a transcript says (languages, names, speaker labels) |
| `voices_digest` | a hash of the voiceprint set used: each print's file name, model hash and update date, never its vectors; `""` for none |
| `outputs` | `[{kind, path, sha256}]`, kind `md`, `json`, `txt`, `sidecar` or `notes` |
| `engines` | `{asr_tag, diarizer, speaker_id, packs[{id, version}]}` |
| `signals` | `{hu_word_share, hu_detected, hu_route, thai_screened, thai_windows, thai_turns, thai_seconds, speaker_count}`: numbers and route names only |
| `speakers` | `[{label, display, speech_s, name?, score?, why?}]` |
| `counts` | `{sentences, words, turns}` |
| `timing` | `{seconds, rtf}` |
| `warnings`, `notes` | `[{code, params, message}]` |
| `offers` | `[{kind, options, apply_flag, params, summary}]`; each option is worded with its own figure: a `language` offer's `params.sizes` gives each pack's download size, a `remember` offer's `params.speech_s` each label's seconds of clear speech |
| `error` | `{code, params, message}` or `null` |
| `log` | the run log's path, or `null` |

Some commands add one field: a `transcribe --plan` record has `plan` (`{routes, estimate_s}`: what would run, and the time `speed.json`'s past runs predict); an `apply --choice` record has `choice`; an `apply --lang` record whose pack couldn't be installed has `lang` (and setup's error); an `enroll` record has `voice` (`{name, sources}`), `refusals` (every reason the enrolment check held the voice back, in order, each with its message) and `result` (`{name, added, skipped, seconds, sources, forced?}`: how many sources were added, how many were already in the voiceprint, the seconds added, the voiceprint's source count after, and the check codes `--force` overrode). An enrolment that is refused or fails keeps nothing, not even the clips before the one that was refused. `enroll --json` adds `saved` (the sentence that says how to delete the voiceprint) when something was added, or `nothing_added` when every source was already there.

`hu_route` is `vote`, `gapfill`, `vote_failed`, `no_pack` or `none`.

## Offers

`offers` lists what a person *could* do next. audiointake never stops for them, the skill never asks about them during a batch, and an exit never depends on them. Relay the `summary`, core's words, once after the batch.

| Kind | Apply flag | Option ids | When |
|---|---|---|---|
| `language` | `--lang` | a language pack's code | the recording is in a language a pack reads better and that pack isn't installed: asked for with `--lang`, or detected. `apply --lang CODE` downloads the pack (the person asked) and transcribes again |
| `speakers` | `--choice speakers` | none | speaker labels were unavailable (the speaker program isn't set up). Once it is, `apply --choice speakers` adds them from the kept transcript, without transcribing again |
| `names` | `--choice names` | none | the voiceprints changed since the transcript was made. `apply --choice names` matches the voices again and updates the names, without transcribing or telling the speakers apart again; names the user gave stay, and the transcript then records the voiceprints it matched (`voices_digest`), so running it again says `already_done` |
| `remember` | none (`enroll NAME --from <id> --label LABEL`) | label ids | a label nobody has named by voiceprint, or one the user named with `relabel`, has at least 8 s of clear speech. Acted on only after the user names the person and has their agreement |

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | done; read the warnings, notes and offers. Includes `already_done`, `voices_changed` and `no_speech` |
| 1 | a recording failed and nothing was written for it (`asr_failed`, `decode_failed`, `internal_error`) |
| 2 | usage, for that input only where it applies: bad arguments, a missing, unreadable or unsupported input, an output folder that can't be written, an output already there, an unknown language, record or label, an offer that isn't there, an invalid name |
| 3 | setup: the platform, the models, ffmpeg or the speaker program |
| 4 | refused, nothing written: the voiceprint check held a voice back |
| 5 | stale: the recording, an output or the transcript JSON changed since audiointake wrote it |
| 129, 130, 143 | stopped by SIGHUP, SIGINT (Ctrl-C) or SIGTERM: children stopped, work folder and partial files removed, no records (a signal that arrives while a recording's outputs are being written waits until they and its record are complete) |

With several records, the process exits with the highest-priority code among them, in the order 2, 3, 1, 4, 5, 0. Read every record, whatever the process exit.

## Error codes

**Usage (exit 2):** `bad_arguments` (an argument audiointake doesn't take), `input_missing` (no such file), `input_not_file` (a folder, pipe or device), `input_not_downloaded` (an iCloud file not downloaded to this Mac), `input_not_readable` (macOS didn't let audiointake read it), `unsupported_input` (no audio stream), `output_not_writable` (the output folder takes no file), `output_exists` (an output is already there and wasn't made from this recording with these settings), `unknown_language` (a `--lang` code audiointake doesn't read), `unknown_record` (no such record, and no transcript to rebuild it from), `no_offer` (the record has no offer of that kind), `not_an_option` (the offer doesn't list that option), `no_transcript_json` (the command needs the `(transcript).json`), `unknown_label` (no such speaker label in the transcript), `name_invalid` (1 to 64 characters, no `/`, no line break or other control character, not starting with `.`), `voice_not_found` (no voiceprint by that name), `voice_exists` (`voices rename` to a name another voiceprint has), `voice_other_model` (a file under that name in the voiceprint folder isn't a voiceprint of this speaker model, so `enroll` won't add to it), `core_required` (the core pack can't be removed), `head_invalid` (the notes file already has a `## Transcript` section).

**Setup (exit 3):** `platform_not_macos`, `platform_not_apple_silicon`, `platform_rosetta`, `macos_too_old`, `not_set_up`, `ffmpeg_missing`, `clt_missing`, `xcode_license`, `swift_too_old`, `speakers_build_failed`, `build_timeout`, `no_room`, `download_failed`, `model_changed`, `pack_broken` (a file of the core pack changed on disk; for any other pack it is a warning and that pack's route is off), `pack_outdated` (the core pack was set up for another version of audiointake, whose pinned models differ; for any other pack it is a warning and that pack's route is off), `model_load_failed` (a model matched its fingerprint but didn't load or convert on this Mac, so its pack wasn't set up).

**Failed (exit 1):** `asr_failed` (the first speech model failed), `decode_failed` (ffmpeg couldn't read the audio), `internal_error` (something unexpected went wrong inside audiointake with that recording: the run log names the error's type and where it happened, never its text, and the other recordings still run).

**Refused (exit 4):** `too_little_speech` (under 8 s of clear speech from one voice), `two_voices`, `mixed_voices`, `matches_someone_else`, `not_matching_print`. When the check finds several, the first in the order `not_matching_print`, `matches_someone_else`, `mixed_voices`, `two_voices` is the error and all of them are in `refusals`. `--force` keeps the voice anyway, only on the person's own word after hearing the reason.

**Stale (exit 5):** `input_changed` (the recording changed during the run or since the record), `output_changed` (an output was changed after audiointake wrote it; `--overwrite` replaces it), `transcript_edited` (the transcript JSON isn't what audiointake wrote, so nothing is rebuilt from it).

## Warning codes

A warning is something the user should hear about. Relay every one.

- `pack_not_installed`: a pack that would read this recording better isn't installed; the params' `reason` says whether it was `asked` for, `detected`, or detected without room to install it (`no_room`). Nothing was downloaded.
- `pack_broken`: a file of a non-core pack changed on disk; that pack's route is off until `setup` repairs it.
- `pack_outdated`: a non-core pack was set up for another version of audiointake; that pack's route is off until `setup` brings it up to date.
- `speech_not_transcribed`: at least 30 s, and a quarter, of the speech the speaker step found produced no words.
- `no_speaker_labels`: the speaker program isn't set up, so the transcript has no labels (with a `speakers` offer).
- `speakers_failed`: telling speakers apart or naming them failed; the transcript has no labels or numbered labels.
- `merge_failed`: putting labels on the words failed; the plain transcript stands.
- `speaker_cap`: 8 labels, the most the speaker step tells apart.
- `speaker_count_differs`: `--speakers N` and the number of labels found differ.
- `fill_failed`: the second speech model failed; the first alone wrote the transcript.
- `hungarian_vote_failed`: the language vote's model failed; the core models' reading stands.
- `thai_failed`: checking for, or transcribing, Thai failed.
- `memory_unmeasured`, `canary_unavailable`, `canary_failed`, `fluidaudio_tag_moved`: from `setup` and `doctor`, not from a transcription (`canary_failed`: the test recording ran but didn't find enough of its known words or two speakers).

## Note codes

- `already_done`: this recording was already transcribed with these settings; nothing was redone.
- `voices_changed`: as `already_done`, but the voiceprints changed since (with a `names` offer).
- `no_speech`: no speech found; the transcript is empty but written.
- `recorded_at_from_file_date`: the date comes from the file's date.
- `voices_model_mismatch`: voiceprints made with another speaker model were left out.
- `voices_skipped`: files in the voiceprint folder that aren't voiceprints were skipped.
- `pack_fetched`: `apply --lang` downloaded the pack.
- `already_enrolled`: `enroll` added nothing, because the voiceprint already has those recordings (printed by `enroll`, not stored on a transcript's record).

## Example

```json
{"record_version": 1, "id": "r-20261116-1a2b3c4d", "parent": null, "tool": {"version": "0.1.0"},
 "command": "transcribe", "status": "done", "exit": 0, "created": "2026-11-16T09:31:02.123456+01:00",
 "input": {"path": "/Users/example/Desktop/Team sync.m4a", "sha256": "…", "size": 19783211,
           "duration_s": 2473.2, "channels": [1], "spatial": "no", "recorded_at": "2026-11-16T09:30:00",
           "recorded_at_source": "metadata"},
 "options": {"lang": [], "speakers": null, "names": true, "speakers_off": false, "format": ["md", "json"],
             "voices": "default", "voices_dir": "/Users/example/Library/Application Support/audiointake/voices"},
 "options_digest": "…", "voices_digest": "…",
 "outputs": [{"kind": "md", "path": "/Users/example/Desktop/Team sync (transcript).md", "sha256": "…"},
             {"kind": "json", "path": "/Users/example/Desktop/Team sync (transcript).json", "sha256": "…"}],
 "engines": {"asr_tag": "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill",
             "diarizer": "fluidaudio-v0.18.0-nemotron3-offline",
             "speaker_id": "sherpa-onnx campplus-zh_en-advanced aa3cfc16", "packs": [{"id": "core", "version": 1}]},
 "signals": {"hu_word_share": 0.0121, "hu_detected": false, "hu_route": "none", "thai_screened": false,
             "thai_windows": 0, "thai_turns": 0, "thai_seconds": 0.0, "speaker_count": 2},
 "speakers": [{"label": "speaker_0", "display": "Speaker 1", "speech_s": 412.0, "score": 0.21,
               "why": "below threshold"},
              {"label": "speaker_1", "display": "Jane Doe", "speech_s": 388.5, "name": "Jane Doe", "score": 0.61,
               "why": "named"}],
 "counts": {"sentences": 412, "words": 6120, "turns": 96}, "timing": {"seconds": 131.4, "rtf": 0.0531},
 "warnings": [], "notes": [],
 "offers": [{"kind": "remember", "options": ["speaker_0"], "apply_flag": null, "params": {"speech_s": {"speaker_0": 412.0}},
             "summary": ["speaker_0 has 412 s of clear speech. With that person's agreement, `audiointake enroll NAME --from r-20261116-1a2b3c4d --label speaker_0` keeps a voiceprint so later recordings name them."]}],
 "error": null, "log": "/Users/example/Library/Logs/audiointake/r-20261116-1a2b3c4d.log"}
```

## Compatibility

A field is added only with a new `record_version`. Codes are added in `errors.py` with their message, and this page lists them.
