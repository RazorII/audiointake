# The transcript JSON

`audiointake transcribe` writes `<stem> (transcript).json` beside each recording (or into `--outdir`), unless `--format` leaves `json` out. It is the machine-readable transcript, and the file the follow-up commands work from: `relabel`, `notes`, `apply` and `enroll --from` re-merge or re-render from exactly what the first run used, without transcribing again. This page describes format 1; the code that writes it is `src/audiointake/pipeline.py` (`json_bytes`) and `src/audiointake/outputs.py` (`transcript_bytes`).

**It holds the transcript's text.** It is your file, like the recording: if the folder it's in syncs or is shared, so is the transcript. It never holds the list of everyone with a voiceprint, and names only the speakers a voiceprint or the user named.

## Shape

UTF-8, indented by one space, with a trailing newline. Keys in this order:

| Key | What it holds |
|---|---|
| `format` | `"audiointake-transcript"` |
| `version` | `1` |
| `seal` | the SHA-256 that proves the file is unchanged (below) |
| `record_id` | the record of the run that wrote this file ([decision-record.md](decision-record.md)) |
| `input` | `{name, sha256, duration_s, recorded_at}`: the recording's file name and hash (the follow-up commands check the recording against it), its length in seconds, and when it was recorded, if known |
| `options_digest`, `voices_digest` | the same digests as the record: a later `transcribe` of the same recording with the same options says `already_done`, or `voices_changed` when only the voiceprints changed. `apply --choice names` and `--choice speakers` set `voices_digest` to the voiceprints they matched against, so the next run says `already_done` again |
| `options` | `{lang, names, speakers_off, format, voices}`, the options of the run; `voices` is `"default"` or `"custom"` (`--voices DIR`). No folder path is ever written here, since the file is shared: the record keeps a custom folder's path |
| `engines` | `{asr_tag, diarizer, speaker_id, packs[{id, version}]}` |
| `outputs` | `[{kind, name, sha256}]` for the files written beside this one in the same run, never this file itself |
| `asr` | the speech ensemble's JSON exactly as it wrote it: `{"text", "sentences": [{"text", "start", "end", "tokens"?, …}]}`. Word timings come from `tokens` where the model gave them |
| `rttm` | the speaker step's RTTM text, verbatim (`SPEAKER <file id> 1 <start> <duration> <NA> <NA> speaker_K <NA> <NA>`, one line per turn), or `null` when there were no speaker labels |
| `speakers` | `{label: {name, score, speech_s, why}}` for the labels a voiceprint named, and only those |
| `names_override` | `{label: name}` for the labels the user named with `relabel` |
| `displays` | `{label: display}`, the complete map the transcript was rendered with: a name, or `Speaker N` |
| `offers` | the offers still open on this transcript, without their summaries, so `apply` works after the record has expired. Each follow-up that rewrites this file carries the open offers over and leaves out the ones it settled (`apply --lang` redoes everything and offers afresh) |
| `turns` | `[{label, display, start, end, text}]`, the rendered dialogue's turns. Without speaker labels, `label` is `null`, `display` is `""` and each turn is a paragraph (a pause of 2 s or more starts a new one) |

Speaker labels are the speaker program's own (`speaker_0` to `speaker_7`). Numbers are seconds from the start of the recording.

## The seal

`seal` is the SHA-256 of the file's bytes as written with `seal` set to 64 zeros. To check it, replace the 64-character value after the first `"seal": ` with 64 zeros and hash the result. audiointake refuses to rebuild anything from a file whose seal doesn't match, or whose hash differs from the one its record lists (`transcript_edited`, exit 5): `audiointake transcribe <recording> --overwrite` makes a fresh one. The seal protects against accidental edits; it isn't a signature.

## After the record expires

Records are kept 90 days. After that the transcript JSON alone is enough: `relabel`, `notes`, `apply` and `enroll --from` accept its path, check the recording beside it (by `input.name`) against `input.sha256`, check the files in `outputs` against their hashes before replacing them, and read the offers from `offers`. The same holds while the record is kept but the folder was moved with the transcript and the recording in it: the files are found beside the JSON.

Two things need the record and end with it:

- A `names` offer from a re-run that found the voiceprints changed lives on that re-run's record only (the re-run writes nothing). Once it's gone, running `transcribe` on the recording again makes it again.
- A custom `--voices` folder is named only in the record. After it expires, `apply` can't tell which folder that was, so it names nobody rather than match against the wrong voiceprints.

`notes --check` compares with the hash in the notes record, so after that record expires it always says the notes changed: the safe answer, since nothing then rewrites them.

## Example

```json
{
 "format": "audiointake-transcript",
 "version": 1,
 "seal": "…",
 "record_id": "r-20261116-1a2b3c4d",
 "input": {"name": "Team sync.m4a", "sha256": "…", "duration_s": 2473.2, "recorded_at": "2026-11-16T09:30:00"},
 "options_digest": "…",
 "voices_digest": "…",
 "options": {"lang": [], "names": true, "speakers_off": false, "format": ["md", "json"], "voices": "default"},
 "engines": {"asr_tag": "parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill",
             "diarizer": "fluidaudio-v0.18.0-nemotron3-offline",
             "speaker_id": "sherpa-onnx campplus-zh_en-advanced aa3cfc16", "packs": [{"id": "core", "version": 1}]},
 "outputs": [{"kind": "md", "name": "Team sync (transcript).md", "sha256": "…"}],
 "asr": {"text": "…", "sentences": ["… the ensemble's sentences …"]},
 "rttm": "…",
 "speakers": {"speaker_1": {"name": "Jane Doe", "score": 0.612, "speech_s": 41.0, "why": "named"}},
 "names_override": {"speaker_2": "Ana"},
 "displays": {"speaker_0": "Speaker 1", "speaker_1": "Jane Doe", "speaker_2": "Ana"},
 "offers": [],
 "turns": [{"label": "speaker_0", "display": "Speaker 1", "start": 0.32, "end": 4.42, "text": "…"}]
}
```
