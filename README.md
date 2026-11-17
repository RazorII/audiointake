# audiointake

[![test](https://github.com/RazorII/audiointake/actions/workflows/test.yml/badge.svg)](https://github.com/RazorII/audiointake/actions/workflows/test.yml)

Transcribe recordings on a Mac with Apple silicon, with speakers named word by word.

```
audiointake transcribe 'Team sync.m4a'
```

**What stays on your Mac.** audiointake transcribes on your Mac: transcribing never goes online, and your audio and voiceprints stay there. Setup and adding a language pack download models, and nothing else does. The transcripts are written next to each recording, so if that folder syncs (iCloud Drive, Dropbox) or is shared, they go wherever it goes.

**What Claude sees.** To write notes, Claude reads the transcript, so its text goes to Anthropic like anything else in your Claude conversation; say "just transcribe" and Claude won't read it. Even then, Claude sees what the command prints: file names, how long each recording is, and the names of speakers it recognised.

**Voiceprints are biometric data.** audiointake makes one only when you ask, from recordings you give it, and keeps it on your Mac (not in iCloud; Time Machine backs it up like the rest of your Library). Only enroll people who have agreed to having a voiceprint kept. `audiointake voices remove NAME` deletes one; transcripts already made keep the name they show, and so does a Time Machine backup.

**It records nothing.** It only transcribes files you give it. Whether you may record a conversation depends on where you are.

## What it does

- **Two speech models, read together.** Parakeet Ultra transcribes each recording, and Parakeet TDT 0.6B v3 fills in words the first one dropped. Both read 25 European languages, and a recording may switch between them.
- **Speakers told apart word by word.** Nemotron 3 diarization finds who spoke when, and every word is labelled with the speaker talking at that moment rather than one speaker per sentence, so a quick interruption gets its own line. On public meeting recordings (AMI) that put 97.7 % of the words with the right speaker, against 91.0 % for one speaker per sentence.
- **Names from enrolled voices.** A speaker is named only when a voiceprint matches with at least 8 seconds of clean speech, a score of at least 0.44 and a lead of at least 0.08 over every other voiceprint; otherwise they're "Speaker 1", "Speaker 2" and so on. You can name a speaker in one transcript yourself (`relabel`), without making a voiceprint.
- **Optional language packs.** Hungarian is a language the core models read less well: its pack adds Whisper large-v3, which transcribes a recording that's mostly Hungarian, and the two Parakeet models correct it where they agree on other words. Thai is written without spaces between words, and the core models can't read it at all: its pack screens every recording for Thai turns and transcribes them with Typhoon Whisper large-v3, a Thai model.
- **A run never stops to ask.** Every choice comes from a flag or a default, so a batch of recordings runs in the background and finishes. What you might want to do afterwards (add a language pack, add speaker labels, update names) is an offer on the run's record, acted on later with `audiointake apply`.
- **Records never hold transcript text.** Each run writes a JSON record with file paths, durations, speaker names, codes and numbers, never a word of what was said. The transcript is in its own files, next to the recording.
- **Nothing is ever uploaded.** `transcribe` makes no network connection: the models are loaded from files already on your Mac, and every program it starts gets an environment with the network features of Hugging Face's libraries switched off.

## Install

You need:

- **A Mac with Apple silicon** (M1 or later) and **macOS 14 Sonoma or newer**. The speech models run on the Mac's GPU with MLX, and the diarizer on Core ML, so Intel Macs, Linux and Windows aren't supported.
- **[uv](https://docs.astral.sh/uv/)**, which also brings the right Python (3.14).
- **ffmpeg** comes with it. audiointake uses an ffmpeg your Mac already has (on your `PATH`, or Homebrew's in `/opt/homebrew/bin` or `/usr/local/bin`), and otherwise the ffmpeg 9.0 build that its `imageio-ffmpeg` dependency installs, checked against a pinned fingerprint by `audiointake setup`. Nothing needs Homebrew.
- **Apple's Command Line Tools** for speaker labels (`xcode-select --install`): setup builds a small speaker program on your Mac from [FluidAudio](https://github.com/FluidInference/FluidAudio)'s source, pinned to one commit. It has been built with Swift 6.4.2, both with Xcode and with the Command Line Tools alone (version 27.2). Without the tools, recordings are transcribed without speaker labels until they're installed.

```
uv tool install --python 3.14 https://github.com/RazorII/audiointake/archive/refs/tags/v0.1.0.tar.gz
audiointake setup
```

The install needs no git: uv fetches the release's source archive. `audiointake setup` checks the Mac, downloads the core models, builds the speaker program and transcribes 30 seconds of a public test recording to prove it all works. `audiointake setup --plan` says first what it would download and how long that takes, and writes nothing. To update later, run the install line for the new version with `--force` (`uv tool upgrade` keeps the old version's pin).

If the shell says `audiointake` isn't found, uv put it in `~/.local/bin`, which isn't on your PATH yet: `uv tool update-shell` fixes that.

### Disk space and downloads

| Pack | What it adds | Downloads | Free space while installing | Kept |
|---|---|---|---|---|
| core | Parakeet Ultra, Parakeet TDT 0.6B v3, Nemotron 3 diarization, the CAM++ speaker model, the speaker program | about 4.1 GB | about 4.7 GB (the speaker program's build folder, deleted once it's built) | about 4.0 GB |
| `hu` | Whisper large-v3 | about 3.1 GB | about 3.1 GB | about 3.1 GB |
| `th` | Whisper large-v3 (shared with `hu`) and Typhoon Whisper large-v3, converted to MLX on your Mac | about 6.2 GB with `hu`, 9.3 GB without | about 9.3 GB with `hu`, 12.3 GB without (Typhoon's download and its conversion sit on the disk together) | about 3.1 GB with `hu`, 6.2 GB without (the download is deleted once converted) |

These figures come from the pinned files' sizes, for a Mac that has none of the models yet: `audiointake setup --plan` works out the real ones for your Mac, counting models already in your Hugging Face cache, which are never downloaded twice. The Python libraries take about 1.2 GB more. Downloaded models live in the standard Hugging Face cache (`~/.cache/huggingface/hub`), pinned to exact revisions, and what audiointake converts or assembles itself lives in its own folder. Every file's fingerprint is checked in full at setup, and every run checks that the files haven't changed since.

```
audiointake setup --pack hu        # add a language pack (or --pack th)
audiointake packs list             # what's installed, and what each pack would add
audiointake packs remove th        # remove one; models another pack needs stay
audiointake uninstall --plan       # what uninstall would remove
audiointake uninstall              # models audiointake downloaded, its data, cache and logs
```

`uninstall` removes only models audiointake downloaded itself, never ones that were in your Hugging Face cache before, and prints the one uv command that removes the tool. `--keep-voices` keeps your voiceprints.

## Use

```
audiointake transcribe 'Team sync.m4a'                    # writes "Team sync (transcript).md" and ".json" beside it
audiointake transcribe 'a.m4a' 'b.mp3' --outdir 'Notes'   # several recordings, one after another
audiointake transcribe 'Kickoff.m4a' --lang de            # the whole recording is in German
audiointake transcribe 'Call.m4a' --lang en,es            # it switches between English and Spanish
audiointake transcribe 'Call.m4a' --speakers 3            # check that three people were heard
audiointake transcribe 'Call.m4a' --no-names              # speaker labels, but no voiceprint matching
audiointake transcribe 'Memo.m4a' --no-speakers           # a plain transcript
audiointake transcribe 'Memo.m4a' --format md,json,txt    # also a plain-text file
audiointake transcribe 'Team sync.m4a' --plan             # what would run, and how long; writes no transcript
audiointake relabel 'Team sync.m4a' --label speaker_1 --name 'Jane Doe'   # name a speaker in this transcript
audiointake enroll 'Jane Doe' 'jane-reading.m4a'          # keep a voiceprint, with Jane's agreement
audiointake enroll 'Jane Doe' --from 'Team sync.m4a' --label speaker_1    # from a transcript's speaker
audiointake voices list                                   # who has a voiceprint
audiointake apply r-20261116-1a2b3c4d --choice speakers   # act on a record's offer later
audiointake record show r-20261116-1a2b3c4d               # a saved record again
audiointake doctor                                        # is everything set up and sound?
audiointake about                                         # versions, models, licences, hosts, folders
```

**Inputs.** Any regular file in which ffmpeg finds an audio stream: m4a, mp3, wav, aac, flac, ogg, opus, aiff, caf, mp4, mov, Voice Memos' qta, webm, mkv and more. A folder, a pipe or a device is refused (`input_not_file`), and so is a file in iCloud Drive that isn't downloaded to the Mac yet (`input_not_downloaded`), without reading it, because reading it would wait for the download. `--paths-from FILE` reads one UTF-8 path per line.

**Outputs.** `<name> (transcript).md`, the readable dialogue with a timestamp on each turn, and `<name> (transcript).json`, the machine transcript later commands work from ([docs/transcript-json.md](docs/transcript-json.md)). `--format md,json,txt` also writes `<name> (transcript).txt`, one `Name: text` line per turn (`--format` lists every file to write, so keep `md` and `json` in it: later commands work from the `.json`). Everything is written at the end of a recording's run, all at once; a failure writes nothing. An existing file is never replaced unless you pass `--overwrite`, and running the same command again on the same recording with the same settings does nothing (`already_done`), so a stopped batch picks up where it left off.

**Languages.** `--lang` with one code says the whole recording is in that language; with several (`--lang en,es`) it only records them. The core models need no language setting and follow a switch between their languages within a recording. The codes: bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk, and th. A language whose pack isn't installed is read with the core models, with a warning and an offer to install the pack; `transcribe` never downloads anything.

**Speakers.** Up to 8 speakers are told apart. `--speakers N` only checks the count (the diarizer takes no count) and warns when it differs. `relabel` names speakers in one transcript, and everyone else keeps their number. `enroll` keeps a voiceprint for later recordings; it needs at least 8 seconds of one person's clear speech, and 30 to 60 seconds of them talking on their own works best (reading anything aloud works), more than one recording welcome. How the seconds enrolled change how often a voice is named, on public meeting recordings (13 AMI speakers, three trials each): with 3 seconds enrolled the speaker was named right in 82 % of decisions, with 5 seconds 90 %, with 8 seconds 92 %, and with 15 or 30 seconds every time. When voiceprints change, `transcribe` on an already-transcribed recording offers to update its names without transcribing again.

### Nothing is asked during a run

audiointake never prompts and never waits for an answer. It writes a JSON **record** for every recording ([docs/decision-record.md](docs/decision-record.md)), and what you might still want is listed in the record's `offers`, to act on later or never:

- **`language`**: the recording is in a language whose pack isn't installed: `audiointake apply <record id> --lang <code>` downloads the pack and redoes it.
- **`speakers`**: the speaker program wasn't set up, so the transcript has no speaker labels: `audiointake apply <record id> --choice speakers` adds them without transcribing again.
- **`names`**: your voiceprints changed since the transcript was made: `audiointake apply <record id> --choice names` updates the names without transcribing again.
- **`remember`**: a speaker said enough to enroll: `audiointake enroll NAME --from <record id> --label <label>`, with that person's agreement.

A record id, the recording's path and its `(transcript).json` path all work wherever a command takes a record, so a transcript stays workable after its record is pruned (records are kept 90 days).

### Exit codes

| Exit | Meaning |
|---|---|
| 0 | Done; read the warnings and offers anyway. Includes `already_done`, `voices_changed` and `no_speech` |
| 1 | A recording failed and nothing was written for it (`asr_failed`, `decode_failed`, `internal_error`) |
| 2 | Usage, for that input only: bad arguments, `input_missing`, `input_not_file` (a folder, a pipe or a device), `input_not_downloaded`, `input_not_readable`, `unsupported_input`, `output_not_writable`, `output_exists`, `unknown_language`; for `apply`, `relabel`, `notes`, `enroll`, `voices` and `record`: `unknown_record`, `no_offer`, `not_an_option`, `no_transcript_json`, `unknown_label`, `name_invalid`, `voice_not_found`, `core_required`, `head_invalid` |
| 3 | Setup: an unsupported Mac (`platform_not_macos`, `platform_not_apple_silicon`, `platform_rosetta`, `macos_too_old`), `not_set_up`, `ffmpeg_missing`, `clt_missing`, `xcode_license`, `swift_too_old`, `speakers_build_failed`, `build_timeout`, `no_room`, `download_failed`, `model_changed`, `model_load_failed`, or a broken or outdated core pack (`pack_broken`, `pack_outdated`) |
| 4 | Refused by `enroll`, nothing saved: `too_little_speech`, `two_voices`, `mixed_voices`, `matches_someone_else`, `not_matching_print` |
| 5 | Stale: `input_changed`, `output_changed`, `transcript_edited` |

With several recordings, each gets its own record, and the process exits with the most important code among them, in the order 2, 3, 1, 4, 5, 0.

Stopped by SIGTERM, SIGHUP or SIGINT (a service manager stopping it, a closed terminal, Ctrl-C), audiointake stops the programs it started, removes its work folder and any half-written output, writes no record, and exits with 128 plus the signal number (143 for SIGTERM, 129 for SIGHUP, 130 for SIGINT). Run the same files again.

## How well it works

Measured through audiointake itself, on public test speech, on a Mac with Apple silicon:

| Test | Result |
|---|---|
| Word error rate, FLEURS German, core | 3.45 % (50 clips) |
| Word error rate, FLEURS Spanish, core | 2.60 % (50 clips, translations of the same sentences as German's) |
| Word error rate, FLEURS Hungarian, core / with the `hu` pack | 14.8 % / 12.8 % (50 clips, the pack's route with `--lang hu`; without it, a whole transcription takes that route only for a recording of at least 30 words, which 5 of these one-sentence clips are: 13.9 %) |
| Character error rate, FLEURS Thai, with the `th` pack | 6.3 % (50 clips; the core models alone: over 99 %) |
| Word error rate, AMI meetings (far-field microphone array) | 41.2 % (four meetings, 92 minutes) |
| Words with the right speaker, AMI meetings | 97.7 % of 9,376 judged words (one speaker per sentence would give 91.0 %) |
| Speed (audio minutes per minute of work) | about 38 on the four AMI meetings with every pack installed (36 to 45 per meeting) |

The models are pinned, so these numbers stay true until a version that changes a pin, which says so and measures again. Memory and speed haven't been measured yet on a Mac with 16 GB of memory or less; on such a Mac, setup and doctor say so (`memory_unmeasured`), and long recordings may be slow.

## Known limits

- **Thai's neighbours.** Within 10 seconds of confirmed Thai, speech that Whisper's language identification hears as Vietnamese, Chinese, Cantonese, Japanese, Korean, Malay, Indonesian, Javanese, Sundanese, Tagalog, Lao, Khmer or Burmese counts as Thai, because Whisper often mistakes short stretches of Thai for one of those. In a recording that mixes Thai with one of these languages, some of that language may be transcribed as Thai. A turn in which both Parakeet models clearly hear the same English words stays English.
- **Eight speakers at most**, and people talking at the same moment get one label.
- **Mostly Hungarian means the whole file.** With the `hu` pack, a recording that's mostly Hungarian is voted on as a whole; English stretches in it may come out as garbled Hungarian.
- **A closed MacBook lid** sleeps the Mac, even in the middle of a batch. audiointake keeps an open Mac awake while it works.
- **iCloud Drive placeholders** aren't downloaded by audiointake itself. The Claude plugin asks macOS to download such a file with `brctl download` (tried on macOS 27.2) and, if it isn't on the Mac after about 9 minutes, says to use Finder's Download Now.
- **Voices without a voiceprint can get someone else's name.** On public meeting recordings made with one room microphone, when a speaker had no voiceprint, the name of someone who had one was put on them in 28 to 59 % of decisions, across enrolment lengths from 3 to 60 seconds. Where not everyone in a recording is enrolled, check the names; `relabel` corrects them in that transcript.

## Use it with Claude

audiointake comes with a Claude plugin. Its skill tells Claude how to run a whole batch without stopping to ask, write meeting notes (a summary, decisions and action items above the speaker-labelled transcript, which audiointake itself appends word for word), and report everything in one summary at the end. It needs Claude Code on the Mac (or your phone reaching it through Remote Control); it doesn't run in claude.ai's cloud sandbox.

```
claude plugin marketplace add RazorII/claude-plugins
claude plugin install audiointake@razorii
```

The plugin installs audiointake on first use if it isn't there yet, pinned to the release the plugin belongs to. What it runs and downloads is listed in [claude-plugin/README.md](claude-plugin/README.md).

## Development

```
git clone https://github.com/RazorII/audiointake && cd audiointake
uv venv --python 3.14 ~/.audiointake-dev/venv
VIRTUAL_ENV=~/.audiointake-dev/venv scripts/dev-install.sh
PYTHONDONTWRITEBYTECODE=1 ~/.audiointake-dev/venv/bin/python -m pytest -p no:cacheprovider -q
```

The tests replace every model with a small fake program, so they need no models and take a few minutes. They download nothing but uv's published checksum, which one test compares with the skill's pin and which it skips without a network (`-m "not network"` leaves it out). The repo holds no audio of any kind: tests generate their audio in code or fetch pinned public clips (`tools/fetch_test_audio.py`). The real models are tested on a Mac before every release (`scripts/models-local.sh`), including a full transcription with the network switched off. Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Questions and bugs

Ask questions in [Discussions](https://github.com/RazorII/audiointake/discussions/categories/q-a). Report bugs in [Issues](https://github.com/RazorII/audiointake/issues), with the `--json` record if you can: records hold file paths, durations, speaker names and codes, never transcript text, so check the names and paths before you post one. Security problems: [SECURITY.md](SECURITY.md).

## Privacy

audiointake collects nothing and sends nothing anywhere. It goes online only to install, to set up, and to add a language pack you asked for. See [PRIVACY.md](PRIVACY.md).

## Licence and credits

MIT. audiointake ships only its own code: the models are downloaded at setup, the libraries are installed from PyPI, and the speaker program is built on your Mac. They keep their own licences ([THIRD-PARTY.md](THIRD-PARTY.md)):

- **Parakeet Ultra**: [moondream's parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), a post-trained version of NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), in [selcukkubur's MLX conversion](https://huggingface.co/selcukkubur/parakeet-ultra-mlx). CC BY 4.0. Credit belongs to moondream for parakeet-ultra and to NVIDIA for parakeet-tdt-0.6b-v3 underneath it.
- **Parakeet TDT 0.6B v3**: by NVIDIA, in [mlx-community's MLX conversion](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3). CC BY 4.0.
- **Nemotron 3 diarization**: NVIDIA's [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), in [Fluid Inference's Core ML conversion](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml), run by [FluidAudio](https://github.com/FluidInference/FluidAudio). OpenMDW 1.1; FluidAudio is Apache-2.0.
- **Whisper large-v3**: by OpenAI, in [mlx-community's MLX conversion](https://huggingface.co/mlx-community/whisper-large-v3-mlx). MIT.
- **Typhoon Whisper large-v3**: by [Typhoon](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3), fine-tuned from Whisper large-v3. MIT; its model card also says that using the model means agreeing to the [OpenTyphoon Terms and Conditions](https://opentyphoon.ai/tac).
- **CAM++ speaker model**: from [3D-Speaker](https://github.com/modelscope/3D-Speaker), as shipped by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Apache-2.0.
- **Test and benchmark audio**: the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) and [FLEURS](https://huggingface.co/datasets/google/fleurs), both CC BY 4.0, fetched when needed and never shipped.

Not legal advice: check the licences yourself for your use.
