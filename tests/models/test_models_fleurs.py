"""Real models on FLEURS test clips: each clip takes its language's route, and the corpus error rate of each language
stays under its ceiling.

The ceilings in models_support.FLEURS_CHECKS are the rates the reference engine measured on exactly these clips
(bench/fleurs_routes.py) plus a margin for other Apple-silicon chips. The language whose pack adds the Whisper vote
runs with its `--lang` code, which forces the vote even on clips too short for the language detector; every one of
its clips must take the vote, and its ceiling sits below the core route's rate on the same clips, so a silent fall
back to core fails twice. The language scored by character is found by the screen, and every one of its clips must
go through the turn route. FLEURS: CC BY 4.0.
"""
from __future__ import annotations

import pytest

from models_support import (FLEURS_CHECKS, bench_module, check_route, needs_apple_silicon, report, transcribe,
                            transcript_text)

pytestmark = pytest.mark.models


@needs_apple_silicon
@pytest.mark.parametrize("lang", sorted(FLEURS_CHECKS))
def test_fleurs_route_and_error_rate_under_ceiling(lang, models_env, pinned_audio, tmp_path):
    ids, flags, measured, ceiling, route = FLEURS_CHECKS[lang]
    fleurs = bench_module("fleurs")
    ref_units = edits = 0
    for i in ids:
        wav = pinned_audio(f"fleurs-{lang}-{i}")
        ref = pinned_audio(f"fleurs-{lang}-{i}-ref").read_text(encoding="utf-8")
        out = tmp_path / i
        _, rec = transcribe(models_env, wav, out, "--no-names", *flags, fmt="json")
        check_route(rec, route, f"{lang} {i}")
        r, h = fleurs.units(ref, lang), fleurs.units(transcript_text(out, i), lang)
        ref_units += len(r)
        edits += fleurs.edit_distance(r, h)
    rate = 100.0 * edits / ref_units
    report({"test": "fleurs", "lang": lang, "clips": len(ids), "rate": round(rate, 2), "measured": measured,
            "ceiling": ceiling})
    assert ceiling is not None, "no ceiling recorded for this language"
    assert rate <= ceiling, f"{lang}: {rate:.2f} % against a ceiling of {ceiling} %"
