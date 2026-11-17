"""Voiceprints: the folder, its names, and the commands that change it (``enroll``, ``voices remove|rename``).

Voiceprints are biometric data: only keep one for someone who has agreed to having a voiceprint kept.

The folder is ``<data>/voices/`` (0700, files 0600, outside iCloud) unless ``--voices DIR`` names another; one
file per person, ``name.lower() + ".json"``: ``{name, model_sha256, vectors[{vec, source, seconds, lang, added}],
updated}``. Only top-level ``*.json`` files not starting with ``.`` are read. A file that isn't a voiceprint is
skipped (note ``voices_skipped``); a print made with another speaker model is left out (note
``voices_model_mismatch``), never mixed in. Audio is never stored.

Embedding a voice is the engine's job (``steps.speaker_id``, imported when needed), and so are the model hash and
the digest of a voiceprint set (each print's file name, ``model_sha256`` and ``updated``, never its vectors),
so a transcript can tell that the prints changed since it was made; everything else here is plain file work. An
enroll that is refused or fails puts the print back as it was (``snapshot``, ``restore``).
"""
from __future__ import annotations

import json
import os
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import fsutil, lock, platforms
from .errors import UsageError

NAME_MAX = 64
CONSENT = "Voiceprints are biometric data: only keep one for someone who has agreed to having a voiceprint kept."


def saved_line(name: str) -> str:
    return (f"Saved. `audiointake voices remove '{name}'` deletes the voiceprint; transcripts and records already "
            "made keep the name they show.")


def nothing_added_line(name: str) -> str:
    """An enroll whose every source was already in the voiceprint: nothing new was kept (note
    ``already_enrolled``)."""
    from .errors import render
    return render("already_enrolled", {"name": name})


def model_sha256() -> str:
    """The speaker model's file hash (CAM++, 3D-Speaker zh+en "advanced"), from the engine: a voiceprint names the
    model it was made with, and prints made with any other file are left out."""
    from .steps.speaker_id import MODEL_SHA256
    return MODEL_SHA256


def default_dir() -> Path:
    return platforms.data_dir() / "voices"


def folder(voices: Path | str | None) -> Path:
    """The voiceprint folder a command uses: ``--voices DIR`` or the default."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(voices)))) if voices else default_dir()


def check_name(name: str) -> str:
    """A name the user gave: 1–64 characters, not only spaces, no '/', not starting with '.', and no control or
    format character (NUL, a line break or a tab would split the ``Name: text`` lines and diarmeta's one line)."""
    if (not isinstance(name, str) or not 1 <= len(name) <= NAME_MAX or not name.strip() or "/" in name
            or name.startswith(".") or any(unicodedata.category(c) in ("Cc", "Cf") for c in name)):
        raise UsageError("name_invalid")
    return name


def file_for(voices_dir: Path, name: str) -> Path:
    return Path(voices_dir) / f"{name.lower()}.json"


# --------------------------------------------------------------------------- reading the folder
@dataclass
class Scan:
    prints: list[tuple[Path, dict[str, Any]]] = field(default_factory=list)      # usable (this model)
    other_model: list[tuple[Path, dict[str, Any]]] = field(default_factory=list)
    skipped: int = 0

    @property
    def all(self) -> list[tuple[Path, dict[str, Any]]]:
        return sorted(self.prints + self.other_model, key=lambda x: x[0].name)


def candidates(voices_dir: Path | None) -> list[Path]:
    """The folder's own JSON files, in name order; dotfiles and subfolders are never read."""
    if voices_dir is None or not Path(voices_dir).is_dir():
        return []
    return sorted(p for p in Path(voices_dir).glob("*.json") if not p.name.startswith(".") and p.is_file())


def scan(voices_dir: Path | None, *, model: str | None = None) -> Scan:
    model = model or model_sha256()
    s = Scan()
    for p in candidates(voices_dir):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s.skipped += 1
            continue
        if not (isinstance(d, dict) and "vectors" in d and "name" in d):
            s.skipped += 1
            continue
        (s.prints if d.get("model_sha256") == model else s.other_model).append((p, d))
    return s


def digest(voices_dir: Path | None) -> str:
    """The voiceprint set's digest, the engine's one function (``steps.speaker_id.voices_digest``): file
    name, model_sha256 and updated of each usable print, never vectors; "" for none."""
    if voices_dir is None:
        return ""
    from .steps.speaker_id import voices_digest
    return voices_digest(voices_dir)


def scan_notes(voices_dir: Path | None) -> list[tuple[str, int]]:
    """(note code, count) for ``voices_model_mismatch`` and ``voices_skipped`` (empty when neither applies)."""
    s = scan(voices_dir)
    out = []
    if s.other_model:
        out.append(("voices_model_mismatch", len(s.other_model)))
    if s.skipped:
        out.append(("voices_skipped", s.skipped))
    return out


def listing(voices_dir: Path) -> list[dict[str, Any]]:
    """``voices list``: name, sources (descriptions), seconds, whether the model matches, updated. No vectors."""
    out = []
    for p, d in scan(voices_dir).all:
        vecs = [v for v in d.get("vectors") or [] if isinstance(v, Mapping)]
        out.append({"name": d.get("name"), "file": p.name, "sources": [str(v.get("source")) for v in vecs],
                    "seconds": round(sum(float(v.get("seconds") or 0) for v in vecs), 1),
                    "model_ok": d.get("model_sha256") == model_sha256(), "updated": d.get("updated")})
    return out


def find_print(voices_dir: Path, name: str) -> tuple[Path, dict[str, Any]]:
    """The voiceprint whose name matches ``name`` (case-insensitively); UsageError ``voice_not_found``."""
    for p, d in scan(voices_dir).all:
        if str(d.get("name", "")).lower() == name.lower():
            return p, d
    raise UsageError("voice_not_found", name=name)


# --------------------------------------------------------------------------- changing the folder (lock held)
def remove(voices_dir: Path, name: str) -> dict[str, Any]:
    """Delete one person's voiceprint file. Transcripts and records already made keep the name they show.

    A file under that name that isn't a usable voiceprint at all (broken JSON, no vectors) is what ``enroll``'s
    ``voice_other_model`` refusal says this command deletes, so it is deleted too."""
    lock.require_held()
    try:
        p, d = find_print(voices_dir, name)
    except UsageError:
        f = file_for(voices_dir, name)
        if f.is_file() and not f.is_symlink() and not addable(voices_dir, name):
            os.unlink(f)
            return {"removed": name, "file": f.name}
        raise
    os.unlink(p)
    return {"removed": d.get("name"), "file": p.name}


def rename(voices_dir: Path, old: str, new: str) -> dict[str, Any]:
    """Rewrite the print's ``name`` and its file name (``new.lower() + ".json"``)."""
    lock.require_held()
    check_name(new)
    p, d = find_print(voices_dir, old)
    dest = file_for(voices_dir, new)
    same_file = dest.name == p.name
    if not same_file and os.path.lexists(dest):
        raise UsageError("voice_exists", name=new)
    d = dict(d)
    d["name"] = new
    data = json.dumps(d, indent=1, ensure_ascii=False).encode("utf-8")
    fsutil.atomic_write(dest, data, overwrite=same_file, mode=0o600)
    if not same_file:
        os.unlink(p)
    return {"renamed": old, "to": new, "file": dest.name}


def addable(voices_dir: Path, name: str) -> bool:
    """Whether a voiceprint under this name can take a new source: no file yet, or a voiceprint of this model."""
    f = file_for(voices_dir, name)
    if not os.path.lexists(f):
        return True
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and "vectors" in d and d.get("model_sha256") == model_sha256()


def snapshot(voices_dir: Path, name: str) -> bytes | None:
    """The voiceprint file's bytes before an enroll changes it (None: there was none)."""
    try:
        return file_for(voices_dir, name).read_bytes()
    except OSError:
        return None


def restore(voices_dir: Path, name: str, before: bytes | None) -> None:
    """Put the voiceprint file back as ``snapshot`` found it (an enroll that was refused keeps nothing)."""
    lock.require_held()
    f = file_for(voices_dir, name)
    if before is None:
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
        return
    try:
        if f.read_bytes() == before:
            return
    except OSError:
        pass
    fsutil.atomic_write(f, before, overwrite=True, mode=0o600)


def ensure_folder(voices_dir: Path) -> Path:
    """The folder, created 0700 when it's the default; another folder must already exist."""
    if Path(voices_dir) == default_dir():
        return fsutil.private_dir(voices_dir)
    if not Path(voices_dir).is_dir():
        raise UsageError("input_missing", file=str(voices_dir))
    return Path(voices_dir)


def has_prints(voices_dir: Path | None) -> bool:
    return bool(candidates(voices_dir))


def labels_named(speakers: Mapping[str, Any]) -> dict[str, str]:
    """{label: name} for labels a voiceprint named (from the engine's speakers object)."""
    out = {}
    for label, v in (speakers.get("labels") or {}).items():
        if isinstance(v, Mapping) and v.get("name"):
            out[str(label)] = str(v["name"])
    return out


def reduce_speakers(speakers: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The labels a voiceprint named, each ``{name, score, speech_s, why}``: never the roster of everyone
    enrolled, a best guess or a runner-up."""
    out: dict[str, dict[str, Any]] = {}
    for label, v in ((speakers or {}).get("labels") or {}).items():
        if isinstance(v, Mapping) and v.get("name"):
            out[str(label)] = {"name": str(v["name"]), "score": v.get("score"), "speech_s": v.get("speech_s"),
                               "why": v.get("why")}
    return out


def source_for_label(input_name: str, label: str, start: float | None, end: float | None) -> str:
    """A label enrolment's source description: the file name, the label and the time it spans."""
    span = f" {start:.0f}-{end:.0f} s" if start is not None and end is not None else ""
    return f"{input_name} {label}{span}"


def problems_in_order(problems: Sequence[Any]) -> list[Any]:
    """Refusal problems in the order the enrolment check gives them precedence."""
    order = ("not_matching_print", "matches_someone_else", "mixed_voices", "two_voices", "too_little_speech")
    return sorted(problems, key=lambda p: order.index(p.code) if p.code in order else len(order))
