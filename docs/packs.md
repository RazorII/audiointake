# Packs

A pack is the set of models one part of audiointake needs, pinned exactly. `core` is always installed: speech
recognition for 25 languages, speaker labels and naming. A language pack adds models for a language the core
models read less well, or don't read at all. `audiointake packs list` shows what this version ships and what is
installed; `audiointake setup --pack ID` installs one, and `audiointake setup --plan --pack ID` says first how much
it downloads and how much room it needs.

Packs are installed only when someone asks: by `setup`, or by `apply REF --lang CODE` when a record offers that
language. `transcribe` never downloads anything. A run that asks for a language whose pack isn't installed is
read with the core models and gets the warning `pack_not_installed` and a `language` offer.

## Where things live

| What | Where |
|---|---|
| Models from Hugging Face | the standard Hugging Face cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`), at their pinned revisions, in huggingface_hub's own layout (`models--<org>--<name>/blobs/<blob>` and the `snapshots/<revision>/<path>` links). A Mac that already has them downloads nothing, and other tools share them. |
| Release assets (the speaker-embedding model) | `<data>/models/<key>/` |
| Folders a manifest assembles (`layout`) | `<data>/models/<dir_name>/`: symlinks into the cache, in the layout a program wants |
| Converted models | `<data>/models/<dir_name>/`; the source is read from the Hugging Face cache when it is there, else downloaded into `<cache>/build/` and deleted once the conversion has checked out |
| The speaker program | `<data>/tools/fluidaudio/<tag>-<commit12>-r<recipe>/` |
| The test recording | `<data>/canary/` |
| Each pack's stamp (the verified ledger) | `<data>/packs/<id>.json` |
| What audiointake itself downloaded or made | `<data>/downloads.json` |
| A build component of the speaker program | SwiftPM's own download cache (`~/Library/Caches/org.swift.swiftpm/artifacts/`); listed in `downloads.json` when audiointake downloaded it |

`<data>` is `AUDIOINTAKE_HOME`, else `~/Library/Application Support/audiointake`; `<cache>` is
`AUDIOINTAKE_CACHE`, else `~/Library/Caches/audiointake`. Both are created private (0700), and so is every
folder setup makes inside `<data>`.

## The manifest (`src/audiointake/setup/packs/<id>.json`, schema 1)

```json
{"schema": 1, "id": "xx", "version": 1, "title": "…", "language": "xx",
 "requires": ["core"], "routes": ["…"],
 "models": [
  {"key": "example", "kind": "hf", "tag": "example-model", "licence": "MIT", "credit": "…",
   "repo": "example-org/example-model", "revision": "<40 hex>",
   "files": [{"path": "config.json", "size": 269, "sha256": "<64 hex>", "blob": "<40 hex git id>"},
             {"path": "weights.safetensors", "size": 1234, "sha256": "<64 hex>", "blob": "<the same 64 hex>"}],
   "layout": {"dir_name": "example-<rev7>", "links": [{"from": "a/b/model.mlmodelc", "to": "model.mlmodelc"}]},
   "smoke": {"kind": "whisper", "seconds": 3}}],
 "canary": {"url": "https://…", "range": [start, end], "sha256": "<64 hex>",
            "wav": {"rate": 16000, "channels": 1, "bits": 16}, "file": "clip.wav",
            "must_find": ["word"], "min_speakers": 2}}
```

- `kind` is `hf` (a Hugging Face repository at a 40-hex `revision`), `github_release` (a release asset: `repo`
  and `release_tag`), or `converted` (made at setup from another model of the same pack: `convert` names it in
  `from`, the `tool`, and every `output` file with its size and SHA-256). A model that is only a conversion's
  source is never kept for runs.
- `blob` is the name the Hugging Face cache stores the file under: for an LFS file, its SHA-256; for a small file
  kept in git, its git blob id. Both come from the Hub's tree API (`/api/models/<repo>/tree/<revision>`: `oid`
  and `lfs.oid`) and are checked against the files themselves before they go into a manifest; nothing is typed
  by hand.
- `tag` is the exact string transcripts and records use for the model.
- `smoke` names how setup checks that the model loads (on 3 s of generated audio, with the program a run uses).
- `layout` builds a folder of symlinks for a program that wants its files under particular names.
- `canary` (core only) is a short public recording fetched once by byte range, kept, and transcribed at the end of
  setup and by `doctor --selftest`; it passes when every `must_find` word appears and the speakers number at
  least `min_speakers`.

The loader is strict: UTF-8 without a byte-order mark, at most 64 KiB, no duplicate keys, no NaN, no unknown
keys, hex where hex belongs, relative paths without `..`, no file name with a code suffix (`.py`, `.so`,
`.dylib`, `.sh` …), models only on `huggingface.co` or GitHub releases. A pack is data: nothing in it is ever
imported or run.

## What setup does

1. Checks the Mac, ffmpeg, and free disk against the plan (`no_room` before anything is downloaded).
2. Downloads each missing file in a child process: HTTPS, resumable (`Range` from the partial file's length), a
   watchdog that restarts a download whose partial stops growing, three attempts. Redirects are followed only to
   the hosts the starting host uses (`huggingface.co` and hosts under `hf.co`; `github.com` and GitHub's release
   hosts). No token or account is ever sent: not `HF_TOKEN`, not a stored login, no `Authorization` or `Cookie`
   header; `HF_ENDPOINT` is ignored. A `https_proxy` you set is used. While a Hugging Face file is written,
   huggingface_hub's own lock for it is held; while the child waits for that lock or hashes a large file it shows
   it is alive, so the watchdog never takes a slow disk for a stalled download. The speaker program's source is
   fetched with git reading none of your git configuration (no URL rewrites, headers or credential helpers),
   over HTTPS only.
3. Checks every file in full (size, then SHA-256 or git blob id). A file that doesn't match is `model_changed`:
   the download is deleted, and that pack stops while the others carry on. What an interrupted setup leaves
   (partial downloads, half-made folders) is resumed or, once a day old, swept.
4. Converts, assembles and smoke-tests the models, builds the speaker program from its pinned commit, runs the
   test recording (with `--no-names`: it never meets your voiceprints), and writes each pack's stamp: the pack's
   version and, for every model, its pinned revision and every file's path, size, inode, modification time,
   blob name and hash.

A pack that fails part-way leaves no stamp, and the next `setup` resumes it. `setup` exits 3 when anything
failed, including a speaker program that couldn't be built, which leaves the packs installed; its JSON gives each
pack's own verdict (`packs[].verified`) and names the pack in each error's `params.pack`. `apply REF --lang CODE`
installs just that pack, without the speaker program and the test recording, so a missing Command Line Tools
never stops it.

## What a run checks

A run compares each file's size, inode, modification time and blob name with the stamp. A file whose stat
changed but whose content didn't (a restore, a migration) is re-hashed once and its entry refreshed, so it isn't
re-hashed again. A file that changed is `pack_broken`: for core, the run stops (exit 3); for a language pack,
that pack's routes are off and the recording is read with the core models. A stamp that no longer matches the
manifest this version ships (a release changed a pin, or the pack's version) counts the same way until `setup`
runs again: a run uses only the models this version pins. `audiointake doctor` makes exactly the same checks, so
it exits 3 whenever a run would, and `doctor --verify` re-hashes every file.

## Removing

`audiointake packs remove ID` deletes the pack's stamp and only the files `downloads.json` lists that no other
installed pack needs; a model that was in the Hugging Face cache before audiointake stays. `core` can't be
removed. `audiointake uninstall --plan` lists everything `uninstall` would delete, and anything it would leave;
`uninstall` then removes what audiointake downloaded, what it made in `<data>` (`packs/`, `models/`, `tools/`,
`canary/`, `records/`, `voices/` unless `--keep-voices`, `downloads.json`, `speed.json`; its lock file stays) and
in `<cache>` (`work/`, `build/`), and the logs, and prints the one command that removes the tool itself. Anything
else in those folders is left alone, so pointing `AUDIOINTAKE_HOME` or `AUDIOINTAKE_CACHE` at a shared folder
loses nothing. After a pin changes and the new revision of a model is installed and checked, setup deletes the
older revisions of that model it had downloaded, unless a pack on this Mac still points at one.

## Adding or changing a pack

A pin change changes transcripts, so it needs measurements before and after (word error rates on public test
speech, and the speaker metrics for a diarization or speaker model) in the pull request. Fill the manifest from
the Hub's tree API and the files themselves, and run the manifest tests.
