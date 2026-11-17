"""The step functions bench/ calls exist with the arguments it passes (no marker: runs in the ordinary suite, no models).

bench/ reaches into the step modules (the routes, the speaker merge, the speaker-recognition constants), and nothing
else in the ordinary suite runs bench/, so a change to one of these names would break the bench silently. Importing
the step modules needs no model and no Metal.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

USED_BY_BENCH = {  # module -> {name: the positional parameters bench/ passes, or None for a constant or a class}
    "audiointake.steps.ensemble": {"load": ["p"], "gapfill": ["primary", "fill", "out"],
                                    "hu": ["whisper", "ultra", "v3", "wav", "out"]},
    "audiointake.steps.merge_speakers": {"asr_segments": ["d"], "speaker_for": ["seg", "diars"],
                                          "units": ["asr", "diars", "split"]},
    "audiointake.steps.rttm": {"segments_from_path": ["path"]},
    "audiointake.steps.speaker_id": {"clean_pieces": ["segs", "label"],
                                      "enroll_label": ["name", "wav", "rttm_text", "label", "source", "lang",
                                                       "voices_dir", "model", "force"],
                                      "HeldBack": None, "THRESHOLD": None, "MARGIN": None, "MIN_SPEECH": None,
                                      "MATCH_CAP": None},
}


@pytest.mark.parametrize("module", sorted(USED_BY_BENCH))
def test_step_names_bench_uses_exist(module):
    mod = importlib.import_module(module)
    for name, params in USED_BY_BENCH[module].items():
        assert hasattr(mod, name), f"{module}.{name} is gone; bench/ uses it"
        if params is None:
            continue
        sig = inspect.signature(getattr(mod, name))
        positional = [p.name for p in sig.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert positional[: len(params)] == params, f"{module}.{name}{sig}: bench/ passes {params}"
        required = [p for p in sig.parameters.values() if p.default is p.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        assert len(required) <= len(params), f"{module}.{name}{sig} now needs more than bench/ passes"


def test_speaker_constants_are_numbers():
    sid = importlib.import_module("audiointake.steps.speaker_id")
    for name in ("THRESHOLD", "MARGIN", "MIN_SPEECH", "MATCH_CAP"):
        assert isinstance(getattr(sid, name), (int, float)), name
    assert issubclass(sid.HeldBack, Exception)
