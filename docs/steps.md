# The engine steps

`src/audiointake/steps/` holds the engine: the speech-recogniser ensemble, the Thai worker, speaker identification,
the word-level speaker merge, the slim timestamp sidecar and the RTTM reader. Each module keeps the logic of the
engine it was ported from line for line, so its outputs are byte for byte what that engine wrote; the goldens in
`tests/fixtures/steps/` hold those bytes. A change to anything here changes transcripts: measure before and after,
and say so in the pull request.

Every model runs in its own child process, one after another, so only one model is in memory at a time. The pure
steps (merging, the speaker merge, slimming, counting) run in the calling process. Children are started only through
`proc.run`, and every fixed command line comes from `argv.py`. The pipeline's children get the run's scrubbed
environment; a worker that starts a child of its own (the Thai worker's diarizer, the speaker worker's ffmpeg)
passes on only the run variables from its own environment, and its allowlist is the one program its parent gave it.

Run as a program (the two workers, and the ensemble's command line), a steps module stops its children when it is
stopped. Each child leads its own process group, so a signal to the program's group doesn't reach them by itself:
`ensemble.run_program` turns SIGTERM, SIGHUP and SIGINT into `Killed`, which unwinds through every `finally` (work
folders and temporary files go), then stops every child still running (SIGTERM to its group, SIGKILL a second
later) and exits 128 + the signal. A SIGKILL can't be caught: a program killed that way leaves its children to the
`proc.run` that started it.

| Module | Runs | What it does |
|---|---|---|
| `ensemble.py` | in process; starts the recognisers, Whisper and the Thai worker as children | Parakeet Ultra as the primary model, Parakeet v3 filling its gaps; the vote route for a language Parakeet reads less well; the Thai turns put in place |
| `thai.py` | child (`python -I -B -m audiointake.steps.thai run ...`) | screens for Thai with Whisper's language ID, finds the Thai speaker turns, transcribes them with the Thai model |
| `speaker_id.py` | `match` as a child (`python -I -B -m audiointake.steps.speaker_id match ...`); enrolment and the voiceprint calls in process | names diarizer labels from enrolled voiceprints; enrols voices with a check against mixed or misattributed speech |
| `merge_speakers.py` | in process | puts a speaker label on every word; `Speaker N` numbering; names |
| `slim_asr.py` | in process | the compact `.asr.json` sidecar (sentence and word timestamps) |
| `rttm.py` | in process | the one RTTM reader |

## ensemble

```python
ensemble.transcribe(audio, outdir, template, fmt, rt, *, force_lang=None, log, duration_s=None) -> EnsembleResult
```

Writes `<outdir>/<template>.json` (the recogniser's JSON shape: `{"text", "sentences": [...]}`) and/or
`<outdir>/<template>.txt` (one line), for `fmt` `"json"`, `"txt"` or `"all"`. `outdir` must be under `rt.work_root`:
the wavs a run makes (hundreds of MB for a long recording) are written beside it. Raises `AsrFailed` (`asr_failed`)
when the primary model fails, or when a merge step can't read what a model wrote; nothing is written then. When the
primary model fails, ffmpeg first decodes the recording to nowhere (`argv.decode_check`, stopping at the first
error): if that fails too, the code is `decode_failed` instead.

- **Gap fill.** Ultra's text is kept; v3's words are inserted only where Ultra has nothing, at a mean confidence of
  at least 0.5, each filled run as its own sentence so the diarizer labels it by who spoke at that moment.
- **The vote route** (`VOTE_LANGS`, one entry per language). It runs when `lang()` finds the language's function
  words in Ultra's text (at least 30 words, a share of at least 0.08), or when `force_lang` names that language,
  and only when the route's pack is installed (`hu_vote` in `rt.routes` and the Whisper model in `rt.models`).
  Whisper large-v3 reads the loudness-normalised audio; Ultra and v3 overrule it where they agree; Ultra replaces
  segments Whisper looped on or invented and fills stretches it dropped. Any other `force_lang` changes nothing.
- **Thai.** With `thai_screen` in `rt.routes` and the Whisper model installed, the Thai worker screens every
  recording. With `thai_turns` and the Thai model it also transcribes the Thai turns, and `thai()` drops Parakeet's
  words inside them and puts the Thai in their place. Without the Thai model the worker runs with `--screen-only`:
  it reports Thai it finds and nothing else changes.
- **The tag** (`EnsembleResult.tag`, and `asr=` in the classic sidecar): the models' manifest tags, e.g.
  `parakeet-ultra-mlx+parakeet-tdt-0.6b-v3-gapfill`, `whisper-large-v3+ultra/v3-vote`, `…+typhoon-whisper-large-v3-thai`.
  `+hu-failed` marks a vote that failed (the transcript is Ultra's alone) and `+th-failed` a Thai screen or Thai
  transcription that failed (the transcript is kept without it).

`EnsembleResult` fields beyond the paths and the tag:

| Field | Meaning |
|---|---|
| `hu_share` | the share of Ultra's words that are the vote language's function words |
| `hu_detected` | `lang()` found the vote language (whatever was forced) |
| `hu_route` | `"vote"` the vote ran; `"vote_failed"` it ran and failed (`+hu-failed`); `"no_pack"` the vote language was detected or forced but its pack isn't installed, so gap fill ran; `"gapfill"` gap fill ran for any other recording; `"none"` there was no fill (no fill model, or it failed) |
| `fill_ok` | false only when the fill model ran and failed |
| `thai` | `ThaiSummary`: `screened` (the worker ran and answered), `windows` and `thai_windows` (30 s windows screened, and those at P(th) ≥ 0.5), `turns` and `seconds` (Thai turns put in), `screen_only` (Thai found, no Thai model), `failed` (`+th-failed`) |

`duration_s` sizes the children's timeouts (`proc.child_timeout`); without it each child gets six hours, so a long
recording is never cut off by a floor meant for a short one. `log` gets the stats lines (counts only), the tags
and a failed child's last stderr lines; never transcript text, and a child's stdout is never kept.

The module's command line keeps the engine's subcommands:

```
python -m audiointake.steps.ensemble lang    <primary.json>
python -m audiointake.steps.ensemble gapfill <primary.json> <fill.json> <out.json> [--tau 0.5]
python -m audiointake.steps.ensemble hu      <whisper.json> <ultra.json> <v3.json> <16k.wav> <out.json>
python -m audiointake.steps.ensemble thai    <base.json> <thai.json> <out.json>
python -m audiointake.steps.ensemble transcribe <audio> --output-dir D --output-template T [--output-format txt|json|all]
```

`transcribe` takes the run lock, gets its runtime from `setup.runtime.resolve(names=False)`, works in the run's work
folder, copies `T.json` / `T.txt` into `D`, prints the tag on stdout and exits 0. It exits 3 when the primary model
fails or a merge step fails (nothing is written then); a setup problem exits with its own code (3 when audiointake
isn't set up). A recording whose length ffprobe can't give runs with the six-hour cap on each child. It honours two
switches, and nothing else in the package reads them: `THAI=off` drops the
Thai routes, and `PARAKEET_FILL_MODEL` set to the empty string runs without the fill model. Children never see either
(they get the runtime's environment only). The other variables the reference implementation read (the model names, the
Whisper and diarizer paths, the voiceprint folder) are not read: the package runs its pinned, verified models.

## The Thai worker

```
python -I -B -m audiointake.steps.thai run WAV ULTRA_JSON V3_JSON|"" OUT --lid-model DIR
       [--fluidaudio BIN --nemotron DIR] [--typhoon DIR --typhoon-tag NAME] [--screen-only] [--rttm R]
```

`WAV` is 16 kHz mono; the two JSONs are the recognisers' outputs (`""` without v3). `--lid-model` is the Whisper
large-v3 folder. `--fluidaudio` (an absolute path) and `--nemotron` (both or neither) give speaker turns through
`argv.diarize`; without them the turns come from pauses in the audio's energy (voiced runs split at pauses of 0.3 s
or more). The ensemble passes them whenever the runtime has a diarizer, so only a Mac without the speaker program
takes the Thai turns from energy; `--no-speakers` drops the labels but keeps the diarizer for the Thai turns, so
the Thai text is the same with and without it. `--typhoon` is the converted Thai model and
`--typhoon-tag` the name the output and the tag give it; they go together, and the name never comes from the
folder. Output:

- no Thai: `{"thai": false, "windows", "max_p_th"}`, exit 0;
- `--screen-only` and Thai found: `{"thai": true, "screen_only": true, "windows", "thai_windows", "max_p_th"}`, exit 0;
- Thai found and transcribed: `{"thai": true, "model", "turns_from": "nemotron"|"energy", "windows", "thai_windows",
  "turns": [{start, end, spk, p_thai, p_en, text}], "stats"}`, exit 0;
- Thai found without `--typhoon` or `--screen-only`: exit 1.

Its constants are the engine's: the screen at P(th) 0.5 on voiced 30 s windows; the second pass that also counts the
East and Southeast Asian languages Whisper confuses with Thai (`THAI_LIKE`), only within 10 s of a confirmed Thai
turn (a known limit: someone speaking one of those languages next to Thai speech can be read as Thai); the English
veto from the two recognisers; turns of at most 15 s; a 440-token budget per turn. The mlx and numpy imports are
inside the functions, so the module imports without a Metal device.

## Speaker ID

The worker, as the pipeline runs it after diarization:

```
python -I -B -m audiointake.steps.speaker_id match WAV RTTM --out OUT --voices DIR --model ONNX
```

It prints `label<TAB>name|-<TAB>score<TAB>why` per label and writes the speakers sidecar (`engine`, the thresholds,
`enrolled`, `labels`: per label `name`, `score`, `best_guess`, `runner_up`, `speech_s`, `why`). The sidecar names
every enrolled person, so outside `--format sidecars` only the labels that were named reach a transcript or a record.
A label is named with at least 8 s of clean speech (its turns minus any overlap, 0.25 s trimmed at each cut), a
cosine of at least 0.44 to that person's voiceprint, and a lead of at least 0.08 over the next. The other
subcommands keep their arguments and need `--voices` and `--model` too:

```
... enroll-label NAME WAV RTTM LABEL --source "what this is" [--lang XX] [--force] --voices DIR --model ONNX
... enroll NAME LANG CLIP... [--force] [--ffmpeg PATH] --voices DIR --model ONNX
... list --voices DIR --model ONNX
... check --voices DIR --model ONNX
```

Every failure of the command line prints `speaker_id: <message>` on stderr and exits 1; an invalid name is refused
before the model loads.

Library calls (for the CLI's `enroll` and `voices`): `match`, `enroll_label` (up to 30 s of a label's clean
speech), `enroll_clips` (up to 60 s of each clip's longest 10 s pieces), `list_prints`, `check_prints`,
`remove_print`, `rename_print`, `voices_digest` (per print made with this model: file name, model hash and
`updated`, never vectors; `""` when there is none, the same value as `voices.digest`), `scan` (counts: usable
prints, prints made with another model, other files), `check_name`. A refused source raises `HeldBack` with its
reasons as codes (`not_matching_print`, `matches_someone_else`, `mixed_voices`, `two_voices`, in that order); too
little speech raises `TooLittleSpeech` (`too_little_speech`); a clip ffmpeg can't read raises `ClipUnreadable`
(`decode_failed`, an `AudioIntakeError`). Nothing is written for a refused source. `enroll_clips` adds clips one
after another, each written as it passes, so a clip that fails leaves the earlier ones enrolled: every exception it
raises carries `results`, the clips already added. Each clip's 16 kHz copy goes in `work_dir`, else in the
environment's `TMPDIR` (the run's work folder when the pipeline calls it).

Voiceprints are `<voices>/<name lowercased>.json`: `{name, model_sha256, vectors: [{vec, source, seconds, lang,
added}], updated}`, folder 0700 and files 0600. Every top-level `*.json` not starting with `.` is read as a
voiceprint; a file made with another model file is skipped, never mixed in.

## Merge, slim, RTTM

```python
merge_speakers.merge(asr, rttm_text, names, split=True) -> Merged
merge_speakers.count_speakers(rttm_text) -> int
slim_asr.slim(d) -> str
rttm.segments_from_text(text) -> list[dict]
rttm.segments_from_path(path) -> list[dict]
```

`merge` returns the lines the command line writes (`"Speaker 1: ..."`, or a name) and one `Turn` per line with
its label, display, start and end. `names` maps a label to its display; a label without one (an empty or `None`
display counts as none) is `Speaker N`,
numbered in order of first appearance among the labels without a name, so passing a display for every label keeps
everyone's number. Without diarization there is one plain line and one `Turn` per sentence (label `None`, display
`""`). An ASR JSON with no sentence raises `NoSentences`: the plain text stands, and that isn't a failed merge.

`slim` returns the sidecar's exact text: compact JSON, start and end to 3 decimals, words to 2, `"w"` only where the
words rejoin to the sentence text. The RTTM reader counts fields from the right, because a file id can contain
spaces.

## The goldens

`tests/fixtures/steps/` holds synthetic inputs (made-up words, placeholder names, tone "voices") and the outputs the
reference implementation produced from them; `manifest.json` lists each case. Audio, voiceprints and speaker sidecars are
never stored: the tests rebuild audio from seeds (`fixtures/steps/synth.py`) and compare sidecars and voiceprints by
SHA-256. The goldens are regenerated only by the maintainer, from the reference implementation, and a change to them is a
change to every transcript.
