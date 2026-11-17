"""What a run writes beside a recording, and under which names: pure functions, no file access.

Formats (``--format``, comma-separated, default ``md,json``):

* ``md``: ``<stem> (transcript).md``, the readable dialogue (``render.md_file``).
* ``json``: ``<stem> (transcript).json``, the machine transcript (format 1, ``docs/transcript-json.md``);
  ``relabel``, ``notes``, ``apply`` and ``enroll --from`` work from it.
* ``txt``: ``<stem> (transcript).txt``, the classic plain-text transcript, byte for byte.
* ``sidecars``: the classic set under the classic names: ``<stem>.txt`` (always), ``<stem>.asr.json``,
  ``<stem>.diar.rttm`` and ``<stem>.speakers.json`` (each only when there is something in it), and
  ``<stem>.diarmeta.txt`` (always).

The transcript JSON seals itself: ``seal`` is the SHA-256 of the file's bytes with ``seal`` set to 64 zeros, so
``read_transcript`` can tell an edited file from the one audiointake wrote even after its record has expired.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .errors import UsageError

FORMATS = ("md", "json", "txt", "sidecars")
DEFAULT_FORMATS = ("md", "json")
TRANSCRIPT_FORMAT = "audiointake-transcript"
TRANSCRIPT_VERSION = 1
SUFFIX = " (transcript)"
NAMED_KINDS = ("md", "json", "txt")
SIDECARS = (("txt", ".txt"), ("asr", ".asr.json"), ("rttm", ".diar.rttm"), ("speakers", ".speakers.json"),
            ("diarmeta", ".diarmeta.txt"))
ZERO_SEAL = "0" * 64
DIARMETA_THRESHOLD = "0.4"


def parse_formats(value: str | None) -> tuple[str, ...]:
    """``"md,json,txt"`` -> ("md", "json", "txt") in a fixed order; an unknown or empty value is bad_arguments."""
    if value is None:
        return DEFAULT_FORMATS
    parts = [p.strip() for p in value.split(",") if p.strip()]
    bad = [p for p in parts if p not in FORMATS]
    if not parts or bad:
        raise UsageError("bad_arguments", detail=f"--format takes {', '.join(FORMATS)} (comma-separated)"
                         + (f", not {', '.join(bad)}" if bad else ""), command="transcribe")
    return tuple(f for f in FORMATS if f in parts)


def name_for(stem: str, kind: str) -> str:
    """``<stem> (transcript).<md|json|txt>``."""
    if kind not in NAMED_KINDS:
        raise ValueError(kind)
    return f"{stem}{SUFFIX}.{kind}"


def sidecar_name(stem: str, part: str) -> str:
    return stem + dict(SIDECARS)[part]


def names(stem: str, formats: Iterable[str]) -> list[str]:
    """Every file name the formats can write (all five sidecar names for ``sidecars``)."""
    fs = tuple(formats)
    out = [name_for(stem, k) for k in NAMED_KINDS if k in fs]
    if "sidecars" in fs:
        out += [stem + suffix for _, suffix in SIDECARS]
    return out


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def options_digest(*, langs: Sequence[str], names: bool, speakers_off: bool) -> str:
    """The digest of the options that change what a transcript says (resume compares it)."""
    return hashlib.sha256(canonical({"lang": sorted(set(langs)), "names": bool(names),
                                     "speakers_off": bool(speakers_off)})).hexdigest()


def diarmeta(*, speakers: int | None, runs: int | None, fluidaudio_tag: str, spatial: str, asr_tag: str,
             named: str) -> str:
    """The provenance line the sidecar set ends with, exactly as the classic pipeline writes it."""
    return (f"speakers={'?' if speakers is None else speakers} runs={'-' if runs is None else runs} "
            f"threshold={DIARMETA_THRESHOLD} engine=fluidaudio-{fluidaudio_tag}-nemotron3-offline "
            f"spatial={spatial} asr={asr_tag} named={named}\n")


def diarizer_name(fluidaudio_tag: str) -> str:
    return f"fluidaudio-{fluidaudio_tag}-nemotron3-offline"


# --------------------------------------------------------------------------- the sealed transcript JSON
_SEAL_RE = re.compile(rb'"seal": "([0-9a-f]{64})"')


def transcript_bytes(doc: Mapping[str, Any]) -> bytes:
    """The transcript JSON's bytes (UTF-8, indent 1, trailing newline) with its seal filled in.

    ``seal`` comes right after ``format`` and ``version``, so its key is the first match in the file.
    """
    d = {"format": TRANSCRIPT_FORMAT, "version": TRANSCRIPT_VERSION, "seal": ZERO_SEAL}
    d.update({k: v for k, v in doc.items() if k not in ("format", "version", "seal")})
    raw = (json.dumps(d, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    seal = hashlib.sha256(raw).hexdigest()
    return raw.replace(f'"seal": "{ZERO_SEAL}"'.encode(), f'"seal": "{seal}"'.encode(), 1)


def seal_ok(data: bytes) -> bool:
    """Whether a transcript JSON's bytes are exactly what ``transcript_bytes`` wrote."""
    m = _SEAL_RE.search(data)
    if not m:
        return False
    zeroed = data[:m.start(1)] + ZERO_SEAL.encode() + data[m.end(1):]
    return hashlib.sha256(zeroed).hexdigest() == m.group(1).decode()


def parse_transcript(data: bytes) -> dict[str, Any] | None:
    """The transcript JSON as a dict when it is format 1, else None (no seal check here)."""
    try:
        d = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("format") != TRANSCRIPT_FORMAT or d.get("version") != TRANSCRIPT_VERSION:
        return None
    return d


def plain_text(asr: Mapping[str, Any] | None) -> str:
    """The ensemble's one-line text, as it writes it beside its JSON."""
    sents = (asr or {}).get("sentences") if isinstance(asr, Mapping) else None
    parts = [str(s.get("text") or "").strip() for s in (sents or []) if isinstance(s, Mapping)]
    return " ".join(parts).strip() + "\n"


def word_count(asr: Mapping[str, Any] | None) -> tuple[int, int]:
    """(sentences with text, words) in the ensemble's JSON: numbers only."""
    sents = (asr or {}).get("sentences") if isinstance(asr, Mapping) else None
    n = w = 0
    for s in sents or []:
        if isinstance(s, Mapping):
            t = str(s.get("text") or "").strip()
            if t:
                n += 1
                w += len(t.split())
    return n, w
