# Contributing

Thanks for helping. Open an issue first for anything bigger than a small fix, so we can agree on the shape before you write it.

| You want to | Start here |
|---|---|
| Fix a bug | an issue with the `--json` record, then a pull request |
| Make transcripts better (a model, a revision, a setting) | an issue with the numbers: word error rates and speaker accuracy before and after, on public test speech |
| Support another platform | [docs/add-a-platform.md](docs/add-a-platform.md) |
| Anything else | an issue |

If you work with Claude, the repo's [CLAUDE.md](CLAUDE.md) points it to the right guide.

## Set up

You need a Mac with Apple silicon, macOS 14 or newer, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```
git clone https://github.com/<you>/audiointake && cd audiointake
uv venv --python 3.14 ~/.audiointake-dev/venv
VIRTUAL_ENV=~/.audiointake-dev/venv scripts/dev-install.sh
PYTHONDONTWRITEBYTECODE=1 ~/.audiointake-dev/venv/bin/python -m pytest -p no:cacheprovider -q
```

Keep the environment outside the source folder (`scripts/dev-install.sh` refuses one inside it): it holds about 1.2 GB, and a source folder in iCloud Drive or Dropbox would upload it. The suite replaces every model with a small fake program (`tests/fakes/`), so it takes a few minutes and needs no models and no GPU. One `network` test reads uv's published checksum from GitHub and skips itself without a network; `-m "not network"` leaves it out.

The real models are tested with `scripts/models-local.sh` on a Mac that has run `audiointake setup` with every pack. It runs `pytest -m models`, which transcribes public test speech end to end and checks the word error rates, the speakers, and that nothing reaches the network.

## Ground rules

These keep the repo publishable and everyone's recordings safe. CI enforces most of them.

- **No audio in the repo, ever**, under any name or extension: not a test clip, not a few seconds, not your own voice. Tests generate audio in code (a sine, noise, silence) or fetch pinned public clips at test time, from the AMI Meeting Corpus and FLEURS (`tools/fetch_test_audio.py`). `tests/test_hygiene.py` looks for audio by its first bytes.
- **No real transcripts, speaker files or voiceprints.** Test transcripts are synthetic, made of made-up words, and names in tests and examples come from the reserved placeholders (Jane Doe, Alice Example, `example.com`).
- **Transcribing never goes online.** Nothing in `transcribe` may open a network connection; the models are passed as local folders and every helper program gets `HF_HUB_OFFLINE=1`.
- **One module starts processes.** Only `src/audiointake/proc.py` imports `subprocess`, from an allowlist of absolute paths, and every command line the transcripts depend on comes from `argv.py`. `tests/test_hygiene.py` checks this statically.
- **No transcript text in records, logs or error messages.** Records hold codes, numbers, paths and speaker names.
- **Core writes every word.** Every warning, note, offer and error is a code in `errors.py` with its message there; nothing a model writes ever becomes a message.
- **A run never stops for anyone.** No prompt, no reading standard input for an answer, no waiting on a person: every choice comes from a flag or a default, and what someone might want later is an offer on the record.
- **Pinned everything.** Models by revision and per-file SHA-256, Python packages by exact version, the speaker program by tag and commit, test audio by URL and SHA-256. A pin bump is its own pull request, with the measurements that justify it.
- **Imports that need the GPU** (`mlx`, `mlx_whisper`, `parakeet_mlx`) go inside the function that needs them, so every module imports anywhere.
- Open text files with `encoding="utf-8"`.

## Pull requests

- Use the **code** template ([.github/PULL_REQUEST_TEMPLATE/code.md](.github/PULL_REQUEST_TEMPLATE/code.md)).
- CI runs the suite on macOS with the oldest and newest supported Python, and the end-to-end tests with the network denied. `ci-ok` is the check that sums them up.
- Every pull request is reviewed by the maintainer before it merges.

By contributing you agree that your contribution is licensed under the MIT licence, like the rest of the code.

## Questions

Ask in [Discussions](https://github.com/RazorII/audiointake/discussions/categories/q-a). Security problems go through [SECURITY.md](SECURITY.md), not an issue.
