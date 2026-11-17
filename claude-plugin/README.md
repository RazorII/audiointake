# audiointake plugin for Claude

Turns recordings into transcripts and meeting notes, on your own Mac. Meetings, calls, interviews and voice memos are transcribed with speakers told apart word by word, and people whose voices you chose to enroll are named. Then Claude writes the notes on top: a summary, the decisions, and action items with owners and dates, followed by the full speaker-labelled transcript, word for word.

It works through a whole batch of recordings without stopping to ask you anything. Claude reads what you asked for, runs the batch, and gives you one short summary at the end: what was written, who spoke, and what it could still do if you want (add a language pack, add speaker labels, put a name to a voice).

The core models read English, Spanish, German and 22 more European languages, and a recording may switch between them. Two optional packs add Hungarian, which the core models read less well, and Thai, which they can't read at all. A pack is downloaded only when you ask for it.

## Requirements

- Claude Code on a Mac with Apple silicon (M1 or later) and macOS 14 Sonoma or newer, used on the Mac or from your phone through Remote Control. It doesn't run in claude.ai's cloud sandbox, on an Intel Mac, on Windows or on Linux, because the speech models run on the Mac's own GPU.
- Nothing else to install for audio: audiointake brings its own ffmpeg, and uses one you already have (Homebrew's, say) first.
- About 4.1 GB of downloads the first time, for the core models, which keep about 4.0 GB on disk; about 1.2 GB more for audiointake's Python libraries. The Hungarian pack adds about 3.1 GB. The Thai pack downloads about 6.2 GB (9.3 GB without the Hungarian pack) and keeps about 3.1 GB of it (6.2 GB without the Hungarian pack).
- Apple's Command Line Tools for speaker labels: a small program is built on your Mac the first time. Without them you still get transcripts, without speaker labels, until they're installed.

## What the plugin runs, installs and connects to

- **The plugin is one skill**: a guide that tells Claude how to use the `audiointake` command. It has no hooks, no MCP servers and no background processes.
- **It runs one command, `audiointake`**, MIT licensed, source at [github.com/RazorII/audiointake](https://github.com/RazorII/audiointake). If it isn't installed, Claude installs the release this plugin belongs to, pinned to its tag, with uv: `uv tool install --python 3.14 https://github.com/RazorII/audiointake/archive/refs/tags/v0.1.0.tar.gz`. If uv itself is missing, Claude first installs uv 0.13.5 from its GitHub release, checked against a SHA-256 fingerprint written into the skill. Both go into `~/.local` (uv keeps its download cache in `~/.cache/uv`). Claude says so in one line before it does either.
- **It may start Apple's installer for the Command Line Tools** (`xcode-select --install`) when speaker labels need them. That opens Apple's own window on your Mac; you install or cancel there.
- **For a recording that's in iCloud Drive but not on your Mac yet**, Claude runs `brctl download` on that one file, so macOS fetches it from your iCloud Drive like Finder's Download Now.
- **Transcribing never goes online.** The models run on your Mac from files already downloaded, and `audiointake transcribe` makes no network connection at all.
- **To write notes, Claude reads the transcript**, so its text goes to Anthropic as part of your conversation. Say "just transcribe" and Claude won't open it. Even then, Claude sees what the command prints: file names, how long each recording is, and the names of speakers it recognised.
- **It connects to these hosts and no others**, only while installing, setting up, or adding a language pack you asked for:
  - `github.com`, and `codeload.github.com`, `objects.githubusercontent.com` and `release-assets.githubusercontent.com`, where GitHub serves its files from: the audiointake source archive, uv, the speaker program's source (FluidAudio) and the prebuilt library it links, and the speaker-recognition model.
  - `pypi.org` and `files.pythonhosted.org`: the Python libraries audiointake needs, installed once by uv.
  - `releases.astral.sh`: Python 3.14 itself, when the Mac has none and uv fetches it (from Astral's mirror; only if that fails, from GitHub).
  - `huggingface.co` and its download hosts under `hf.co`: the speech, language and diarization models.
  - `groups.inf.ed.ac.uk`: 30 seconds of a public test recording from the AMI Meeting Corpus, fetched once during setup to check that everything works.

  No telemetry, no analytics, no account, and no Hugging Face token is ever sent ([privacy policy](https://github.com/RazorII/audiointake/blob/main/PRIVACY.md)).
- **Every download is checked.** Each model file has a fingerprint pinned in audiointake, checked in full when setup downloads it; every run then checks that the files haven't changed since. The speaker program is built from a pinned commit.
- **What it writes:** `<name> (transcript).md` and `<name> (transcript).json` next to each recording, or in the folder you name; `<name> (notes).md` when Claude writes notes. It never replaces a file it didn't make unless you say so. Models go to the standard Hugging Face cache (`~/.cache/huggingface/hub`); audiointake's records, voiceprints, speaker program and the rest of its data go to `~/Library/Application Support/audiointake/`. `audiointake uninstall` removes all of it.
- **Voiceprints are biometric data.** Claude enrolls a voice only when you ask for that person by name, and says before it does that you should only keep one for someone who has agreed. A voiceprint stays on your Mac.

## Examples

- "Transcribe `Team sync.m4a` and write meeting notes." Claude writes `Team sync (notes).md` beside it, with the summary, decisions and action items above the speaker-labelled transcript.
- "Transcribe the five interviews in this folder; they're all in Spanish." Claude runs the whole folder in one batch and reports once at the end, one line per interview.
- "Just transcribe this voice memo, don't read it." Claude writes the transcript and never opens it, so its text stays on your Mac.
- "Speaker 2 in the team sync is Jane." Claude puts Jane's name into that transcript and its notes; everyone else keeps their number.
- "Speaker 1 is me; remember my voice." Claude puts your name into the transcript and enrolls your voice from the parts where you speak, so later transcripts name you.
- "Who said the launch moves to March?" Claude looks it up in the transcript and answers with the speaker and the time.

## Install

In Claude Code:

```
/plugin marketplace add RazorII/claude-plugins
/plugin install audiointake@razorii
```

Or in Terminal:

```
claude plugin marketplace add RazorII/claude-plugins
claude plugin install audiointake@razorii
```

Claude Code asks before each new command it runs. So that a batch doesn't stop on a permission request for every recording, you can allow audiointake's command once, with these two rules in your Claude Code settings: `Bash(audiointake:*)`, and `Bash(~/.local/bin/audiointake:*)` for the way Claude runs it while `~/.local/bin` isn't on your shell's PATH, as it often isn't right after the first install.

Source, issues and the full documentation: [github.com/RazorII/audiointake](https://github.com/RazorII/audiointake). MIT licensed.
