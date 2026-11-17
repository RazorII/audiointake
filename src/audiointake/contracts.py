"""The shapes every part of audiointake codes against: models, tools, the runtime, results, problems.

A change here changes every caller, so it's made for all of them at once. ``setup.runtime.resolve`` has the
``Resolve`` signature; callers outside setup import it inside a function, never at module level.

What ``resolve`` does with its arguments (the one meaning every caller and the implementation share):

* ``need``: the routes this run asks for, a subset of ``ROUTES`` (``ROUTE_OF_LANG`` maps each ``--lang`` code
  whose pack isn't core to its route; the caller passes the routes of every code it was given). ``need`` never
  switches a route on or off: ``routes`` is always every route whose pack verifies. A needed route that isn't
  available is never an error and never downloads anything: the route stays off, and resolve adds the warning
  ``pack_not_installed`` (``reason="asked"``, ``code`` the language, ``language`` its English name, ``size`` the
  gigabytes ``store.plan`` would fetch) and a ``language`` offer with the same params. A name outside ``ROUTES``
  is a ValueError (a caller's bug, not a user's).
* ``speakers=False`` (``--no-speakers``): ``voices_dir`` is None, and neither ``no_speaker_labels`` nor a
  ``speakers`` offer is added (nobody asked for labels). ``tools.fluidaudiocli`` is still filled when the built
  speaker program passes its check, because the Thai worker takes its turns from it: dropping the labels must not
  change the Thai text. The pipeline, not the runtime, skips diarization under ``--no-speakers``. With
  ``speakers=True`` a missing or failing FluidAudio gives ``no_speaker_labels`` (``reason``) and a ``speakers``
  offer.
* ``names=False`` (``--no-names``), a folder that doesn't exist, or one with no usable voiceprint (none made with
  this speaker model, with vectors): ``voices_dir`` is None.
* ``work_root`` is the run's own folder, ``<cache>/work/<run-id>/``, created 0700 by resolve, which first deletes
  any ``<cache>/work/*`` folder older than a day. The caller makes one subfolder per input under it and deletes
  ``work_root`` when the run ends.
* Problems resolve returns carry no ``id`` and no ``log``: the record writer adds them (``errors.with_context``).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

ROUTES = ("hu_vote", "thai_screen", "thai_turns", "es_vote")          # es_vote: only once Spanish is measured to need it
LABEL_RE = r"^speaker_[0-7]$"                                          # FluidAudio's labels; Nemotron caps at 8
RECORD_ID_RE = r"^r-[0-9]{8}-[0-9a-f]{8}$"

@dataclass(frozen=True)
class Model:
    key: str            # "ultra" | "v3" | "whisper" | "typhoon" | "nemotron" | "campplus"
    tag: str            # the exact string tags and sidecars use, e.g. "parakeet-ultra-mlx", "typhoon-whisper-large-v3"
    path: Path          # verified local path: HF snapshot dir, converted dir, Nemotron dir, or the .onnx file
    revision: str       # 40-hex (HF) or the release tag / sha256 (CAM++)

@dataclass(frozen=True)
class Tools:
    ffmpeg: Path
    ffprobe: Path | None         # None with the fallback ffmpeg (no ffprobe): probe.py parses `ffmpeg -i` instead
    python: Path                 # the tool venv's interpreter, unresolved (sys.executable)
    parakeet_cli: Path           # sysconfig scripts dir / "parakeet-mlx"
    whisper_cli: Path            # sysconfig scripts dir / "mlx_whisper"
    fluidaudiocli: Path | None   # the keyed build; None = speaker labels unavailable
    fluidaudio_tag: str          # "v0.18.0": diarmeta's engine= and the record's diarizer field
    caffeinate: Path = Path("/usr/bin/caffeinate")

@dataclass(frozen=True)
class Runtime:
    tools: Tools
    models: Mapping[str, Model]          # only verified models appear
    routes: frozenset[str]               # subset of ROUTES
    env: Mapping[str, str]               # the scrubbed child environment (proc.run passes exactly this)
    work_root: Path                      # this run's folder, <cache>/work/<run-id>/ (0700, outside synced folders); every outdir a step gets is under it
    voices_dir: Path | None              # None = speaker ID off for this run (no folder, no usable print, --no-names)

class Resolve(Protocol):                 # the one signature of setup.runtime.resolve (other modules import it inside functions)
    def __call__(self, *, need: frozenset[str] = frozenset(), voices_dir: Path | None = None,
                 names: bool = True, speakers: bool = True) -> tuple[Runtime, list[Problem]]: ...   # raises SetupError

@dataclass
class ThaiSummary:
    screened: bool                       # the screen ran
    windows: int = 0
    thai_windows: int = 0
    turns: int = 0
    seconds: float = 0.0
    screen_only: bool = False            # Thai found but the th pack isn't installed
    failed: bool = False                 # the screen or the transcription failed (the tag gets +th-failed)

@dataclass
class EnsembleResult:
    tag: str                             # exactly what ensemble.py printed
    json_path: Path | None               # <outdir>/<template>.json (merged sentences, tokens where they exist)
    txt_path: Path | None                # <outdir>/<template>.txt (one line)
    hu_share: float                      # lang()'s Hungarian function-word share
    hu_detected: bool
    hu_route: str                        # "vote" | "gapfill" | "vote_failed" | "no_pack" | "none"
    fill_ok: bool
    thai: ThaiSummary = field(default_factory=lambda: ThaiSummary(False))

@dataclass(frozen=True)
class Turn:
    label: str | None                    # speaker_k or None
    display: str                         # name, "Speaker N" or "Speaker ?"
    start: float
    end: float
    text: str

@dataclass
class Merged:
    lines: list[str]                     # exactly merge_speakers.py's lines
    text: str                            # "\n".join(lines) + "\n": the classic .txt sidecar's bytes, decoded
    turns: list[Turn]

@dataclass(frozen=True)
class Problem:
    code: str                            # a closed code from errors.CODES
    params: Mapping[str, str | int | float] = field(default_factory=dict)

    def __hash__(self) -> int:           # equal problems hash equal, so a set can drop repeats
        return hash((self.code, tuple(sorted(self.params.items()))))

Log = Callable[[str], None]              # the run log; never receives transcript text


# --------------------------------------------------------------------------- shared constants (Step 0 additions)
# The languages `--lang` accepts: the 25 the core models read, plus Thai (the th pack). Anything else is
# `unknown_language`. LANGUAGE_NAMES gives the English name messages use for {language}.
CORE_LANGS: tuple[str, ...] = ("bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
                               "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk")
LANGS: tuple[str, ...] = CORE_LANGS + ("th",)
LANGUAGE_NAMES: Mapping[str, str] = {
    "bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "fi": "Finnish", "fr": "French", "hr": "Croatian", "hu": "Hungarian",
    "it": "Italian", "lt": "Lithuanian", "lv": "Latvian", "mt": "Maltese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish",
    "uk": "Ukrainian", "th": "Thai",
}
# Language packs beyond core, by the language code `--lang` takes; `core` is always installed.
PACKS: tuple[str, ...] = ("core", "hu", "th")
PACK_OF_LANG: Mapping[str, str] = {"hu": "hu", "th": "th"}
# Terms a pack's models come with beyond their licence: setup's plan lists them, and the terminal prints them
# under a language offer for that pack (never in a record or the JSON).
PACK_TERMS: Mapping[str, str] = {"th": "The Thai model comes with OpenTyphoon's terms (opentyphoon.ai/tac)."}
# The route a `--lang` code asks for when its pack isn't core (what a caller passes to resolve as `need`).
ROUTE_OF_LANG: Mapping[str, str] = {"hu": "hu_vote", "th": "thai_turns"}
SPEAKER_CAP = 8                          # Nemotron's offline profile tells apart at most this many speakers

__all__: Sequence[str] = (
    "ROUTES", "LABEL_RE", "RECORD_ID_RE", "Model", "Tools", "Runtime", "Resolve", "ThaiSummary", "EnsembleResult",
    "Turn", "Merged", "Problem", "Log", "CORE_LANGS", "LANGS", "LANGUAGE_NAMES", "PACKS", "PACK_OF_LANG",
    "ROUTE_OF_LANG", "SPEAKER_CAP",
)
