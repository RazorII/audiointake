<!--
Open a pull request with this template (add ?template=code.md to the compare URL). A new platform
has its own guide in docs/add-a-platform.md; follow it.
-->

## What and why

## How it was tested

<!-- Which tests you added or ran, on which Mac and macOS version. Say which tests skipped and why: a skip isn't a pass. If you ran the real models (scripts/models-local.sh), say so. -->

## Checklist

- [ ] No audio in the repo, in any form or under any name: tests generate their audio in code or fetch pinned public clips at test time (`tools/fetch_test_audio.py`). No transcript, voiceprint or speaker file of a real person.
- [ ] Tests and examples use made-up values only (example.com, 555-01xx numbers, names like Jane Doe).
- [ ] Only `src/audiointake/proc.py` starts processes, and every command line the transcripts depend on comes from `argv.py`; `tests/test_hygiene.py` passes.
- [ ] `transcribe` still makes no network connection, and a record still holds no transcript text.
- [ ] Every word a user or the skill sees comes from `errors.py`'s templates. A new code is in `errors.CODES` with its message, explained in the skill and in `docs/decision-record.md`; a new exit code is in `errors.py`, the README and the skill.
- [ ] A changed model, revision or package pin says why, and how the output changed (word error rates, speaker accuracy).
- [ ] A run still never stops for anyone: every choice comes from a flag or a default, and what a person might want later is an offer on the record.
