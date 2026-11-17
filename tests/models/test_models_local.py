"""Real models, real children: transcription works with the network denied, leaks no transcript text, and the peak
memory of each model child is recorded.

* Every route runs twice: first under `sandbox-exec` with every network operation denied and answered with SIGKILL
  (so a quiet fallback after a refused connection kills the run instead), on this test's fresh HOME, then normally.
  The inputs cover each kind of model child: the AMI excerpt with a voiceprint folder (the speaker-recognition
  worker, which loads onnxruntime), the vote fixture (Whisper), the turns fixture (the language screen and the turn
  worker) and a German fixture (the core route). A voiceprint is enrolled from the excerpt's own transcript, also
  with the network denied. The `--format sidecars` outputs (which carry no record id) must be byte-identical, and
  onnxruntime's telemetry folder must not appear in HOME.
* No word of the reference transcript, and no long word of the transcript itself, may appear in a record, the run's
  log, stderr or the `--json` output, for every route, including the two whose children print transcript lines on
  their standard output (Whisper in the vote, the turn worker). Words audiointake's own source also uses are left
  out of that word check (a record's field names, the log's fixed phrases), so a second check catches what it
  can't: no run of 4 consecutive words of either transcript may appear there either, unless the package's own
  source has those 4 words in a row.
* The peak physical footprint of every model child (Metal allocations included) is sampled while each route runs
  and written to $AUDIOINTAKE_MODELS_REPORT, one figure per kind of child.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from models_support import (DENY_NETWORK, SANDBOX_EXEC, build_fixtures, check_route, enroll_probe, fleurs_refs,
                            needs_apple_silicon, peak_by_kind, report, transcribe, transcript_text, words_of)

pytestmark = pytest.mark.models
# Every word audiointake itself can print (its source code's text), lower case: the privacy check never counts these.
_OWN_TEXT = [re.findall(r"\w+", f.read_text(encoding="utf-8").lower())
             for f in (Path(__file__).resolve().parents[2] / "src" / "audiointake").rglob("*.py")]
OWN_WORDS = {w for words in _OWN_TEXT for w in words}
RUN = 4                                   # a leaked sentence shows as this many transcript words in a row


def runs_of(words: list[str], n: int = RUN) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


OWN_RUNS = {r for words in _OWN_TEXT for r in runs_of(words)}
SIDECARS = (".txt", ".asr.json", ".diar.rttm", ".speakers.json", ".diarmeta.txt")
DENY = [str(SANDBOX_EXEC), "-p", DENY_NETWORK]
GiB = 1 << 30


@needs_apple_silicon
@pytest.mark.skipif(not SANDBOX_EXEC.exists(), reason="sandbox-exec isn't available on this Mac")
def test_network_denied_runs_match_open_runs(models_env, pinned_audio, tmp_path):
    fx = build_fixtures(pinned_audio, tmp_path / "fixtures")
    ami = pinned_audio("ami-es2004a-825-915")
    voices = enroll_probe(models_env, ami, tmp_path / "enroll", wrap=DENY)
    cases = [(ami, ["--voices", str(voices)], None), (fx / "vote.wav", ["--no-names"], "vote"),
             (fx / "turns.wav", ["--no-names"], "thai_turns"), (fx / "gapfill-de.wav", ["--no-names"], None)]
    for side, wrap in (("denied", DENY), ("open", None)):
        for audio, flags, route in cases:
            _, rec = transcribe(models_env, audio, tmp_path / side, *flags, fmt="sidecars", wrap=wrap)
            check_route(rec, route, f"{audio.name} ({side})")
    compared = 0
    for audio, _, _ in cases:
        stem = audio.name[: audio.name.rfind(".")]
        for suffix in SIDECARS:
            a, b = tmp_path / "denied" / f"{stem}{suffix}", tmp_path / "open" / f"{stem}{suffix}"
            assert a.exists() == b.exists(), f"{audio.name}{suffix}: written on one side only"
            if a.exists():
                assert a.read_bytes() == b.read_bytes(), f"{audio.name}: {suffix} differs with the network denied"
                compared += 1
    ami_stem = ami.name[: ami.name.rfind(".")]
    assert (tmp_path / "denied" / f"{ami_stem}.speakers.json").is_file(), "the speaker-recognition worker didn't run"
    assert compared >= 4 * len(cases) - 3      # every input's .txt, .asr.json, .diar.rttm and .diarmeta.txt at least
    assert not (Path(models_env["HOME"]) / "Library/Application Support/Microsoft").exists()


@needs_apple_silicon
def test_no_transcript_text_in_records_logs_or_output(models_env, pinned_audio, tmp_path):
    fx = build_fixtures(pinned_audio, tmp_path / "fixtures")
    cases = [(pinned_audio("fleurs-de_de-1670"), pinned_audio("fleurs-de_de-1670-ref").read_text(encoding="utf-8")),
             (pinned_audio("ami-es2004a-825-915"), ""),
             (fx / "vote.wav", fleurs_refs(pinned_audio, "hu_hu")),           # Whisper prints transcript lines
             (fx / "turns.wav", fleurs_refs(pinned_audio, "th_th"))]          # so does the turn worker
    logs_dir = Path(models_env["HOME"]) / "Library/Logs/audiointake"
    records_dir = Path(models_env["AUDIOINTAKE_HOME"]) / "records"
    for audio, ref in cases:
        stem = audio.name[: audio.name.rfind(".")]
        out = tmp_path / "out" / stem
        r, rec = transcribe(models_env, audio, out, "--no-names", fmt="md,json")
        said = transcript_text(out, stem)
        words = words_of(ref) | words_of(said, 7)
        assert words, f"{audio.name}: nothing to look for"
        runs = (runs_of(re.findall(r"\w+", ref.lower())) | runs_of(re.findall(r"\w+", said.lower()))) - OWN_RUNS
        assert runs, f"{audio.name}: no run of {RUN} words to look for"
        record = records_dir / f"{rec['id']}.json"
        log = logs_dir / f"{rec['id']}.log"
        assert record.is_file(), f"{audio.name}: the record wasn't written"
        assert log.is_file(), f"{audio.name}: the run log wasn't written"
        haystacks = {"stdout": r.stdout, "stderr": r.stderr,
                     "record": record.read_text(encoding="utf-8", errors="replace"),
                     "log": log.read_text(encoding="utf-8", errors="replace")}
        for where, text in haystacks.items():
            # Whole words only, and not audiointake's own words: a record's field names (thai_windows) are one \w+
            # token each, and the log's fixed phrases ("3 windows") come from the package's source. A transcript
            # word that is also one of those isn't a leak; a leaked sentence still shows through its other words.
            tokens = set(re.findall(r"\w+", text.lower()))
            leaked = sorted(w for w in words if w in tokens and w not in OWN_WORDS and w not in audio.name.lower())
            assert not leaked, f"{audio.name}: {len(leaked)} transcript word(s) in the {where}"
            sequences = runs & runs_of(re.findall(r"\w+", text.lower()))
            assert not sequences, f"{audio.name}: {len(sequences)} run(s) of {RUN} transcript words in the {where}"


@needs_apple_silicon
def test_peak_footprint_of_every_model_child_recorded(models_env, pinned_audio, tmp_path):
    fx = build_fixtures(pinned_audio, tmp_path / "fixtures")
    ami = pinned_audio("ami-es2004a-825-915")
    voices = enroll_probe(models_env, ami, tmp_path / "enroll")
    routes = [  # route, input, flags, {kind of child: least plausible peak footprint}
        ("core", fx / "gapfill-de.wav", ["--no-names"], {"parakeet-mlx": GiB // 2, "fluidaudiocli": 1}),
        ("vote", fx / "vote.wav", ["--no-names"], {"mlx_whisper": GiB}),
        ("thai_turns", fx / "turns.wav", ["--no-names"], {"audiointake.steps.thai": GiB}),
        ("speakers", ami, ["--voices", str(voices)], {"audiointake.steps.speaker_id": 1}),
    ]
    for route, audio, flags, floors in routes:
        _, rec, children = transcribe(models_env, audio, tmp_path / route, *flags, fmt="json", sample=True)
        peaks = peak_by_kind(children)
        report({"test": "peak_footprint", "route": route, "children": peaks, "seconds": rec["timing"]["seconds"],
                "audio_s": rec["input"]["duration_s"]})
        for kind, floor in floors.items():
            assert kind in peaks, f"{route}: no {kind} child was seen ({sorted(peaks)})"
            assert peaks[kind] >= floor, f"{route}: {kind} peaked at {peaks[kind]} bytes, below {floor}"
