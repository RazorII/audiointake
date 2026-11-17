# CLAUDE.md

You're helping someone contribute to audiointake: a command that transcribes recordings on a Mac with Apple silicon, with speakers named word by word, and a Claude plugin that turns the transcripts into meeting notes. This file sends you to the right guide. Read the guide in full before you change anything.

## Route the request

| The person wants to | Read and follow |
|---|---|
| Support another platform | [docs/add-a-platform.md](docs/add-a-platform.md) |
| Understand the record or the transcript JSON | [docs/decision-record.md](docs/decision-record.md), [docs/transcript-json.md](docs/transcript-json.md) |
| Anything else | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Stop and check with the person

- Any change to a model, a model revision, a package pin or the speaker program's tag: these change every transcript, so they need measurements before and after.
- Any change to a threshold (the speaker-naming score and lead, the 8-second minimum, the language-detection floors).
- Any test or CI failure you can't explain.

## Never

- Commit audio of any kind, under any name: not a clip, not a test file, not a recording of yourself. Tests generate their audio in code or fetch pinned public clips at test time.
- Commit a real transcript, speaker file or voiceprint, or any personal data. Test transcripts are synthetic, and names come from placeholders such as Jane Doe.
- Make `transcribe` open a network connection, or pass a Hugging Face token anywhere.
- Start a process anywhere in `src/` but `src/audiointake/proc.py`, or write a command line the transcripts depend on anywhere but `argv.py`.
- Put transcript text into a record, a log or an error message, or let a model's output become a message: every word comes from `errors.py`.
- Make a run stop for anyone. Nothing may prompt, read standard input for an answer, or wait on a person; what someone might want later is an offer on the record.
- Push, tag, release or publish anything. The person and the maintainer do that.

## Run the tests

```
uv venv --python 3.14 ~/.audiointake-dev/venv                 # once
VIRTUAL_ENV=~/.audiointake-dev/venv scripts/dev-install.sh     # once
PYTHONDONTWRITEBYTECODE=1 ~/.audiointake-dev/venv/bin/python -m pytest -p no:cacheprovider -q
```

The suite runs on fake engines and needs no models. Tests that need the network, the real models or the Mac's GPU skip themselves with a reason when they can't run. A skip isn't a pass: say which tests skipped and why.
