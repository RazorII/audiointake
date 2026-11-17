"""``audiointake about``: what this install is made of, who made it, and where it talks to and writes.

The version; every pinned model (repository, revision or release, size, licence and credit, from the pack
manifests); FluidAudio's tag and commit and where its licences are; the mlx and mlx-metal wheels' tags and the
ffmpeg in use; every host audiointake contacts and when; every folder it writes, with its size; and the line every
user should be able to find. It writes nothing and opens no connection.
"""
from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__, platforms

PRIVACY_LINE = ("Transcribing makes no network connection; your transcripts reach Anthropic only if you ask Claude "
                "to read them.")
HOSTS: tuple[dict[str, str], ...] = (
    {"host": "huggingface.co", "when": "setup, and apply --lang for a pack that isn't installed: the models"},
    {"host": "hosts under hf.co", "when": "the same downloads, where huggingface.co redirects them"},
    {"host": "github.com", "when": "setup: FluidAudio's source, the speaker-embedding model, a build component"},
    {"host": "objects.githubusercontent.com", "when": "setup: where github.com redirects release downloads"},
    {"host": "release-assets.githubusercontent.com", "when": "setup: where github.com redirects release downloads"},
    {"host": "groups.inf.ed.ac.uk", "when": "setup, once: a 30-second public test recording (AMI corpus)"},
    {"host": "pypi.org", "when": "installing or updating audiointake (uv)"},
    {"host": "files.pythonhosted.org", "when": "installing or updating audiointake (uv)"},
    {"host": "codeload.github.com", "when": "installing or updating audiointake (the release archive)"},
)
FLUIDAUDIO_LICENCE = {"spdx": "Apache-2.0", "files": ["LICENSE", "ThirdPartyLicenses/"],
                      "note": "built on this Mac from its pinned commit and never redistributed"}
DATA_CREDITS = ("AMI Meeting Corpus, collected by the AMI project consortium (CC BY 4.0, "
                "https://groups.inf.ed.ac.uk/ami/corpus/): the setup test recording, and the public meetings the "
                "speaker figures come from",
                "FLEURS, by Google (CC BY 4.0, https://huggingface.co/datasets/google/fleurs): the public test speech "
                "the accuracy figures come from")


def wheel_tags(dist: str) -> list[str]:
    """The ``Tag:`` lines of an installed distribution's WHEEL file ([] when it isn't installed)."""
    try:
        text = metadata.distribution(dist).read_text("WHEEL") or ""
    except metadata.PackageNotFoundError:
        return []
    return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("Tag:")]


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def folder_bytes(p: Path) -> int:
    """Bytes of the regular files under ``p`` (symlinks not followed)."""
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(p):
        for f in files:
            fp = Path(dirpath) / f
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


def about() -> dict[str, Any]:
    from .setup import ffmpeg as ffm
    from .setup import fluidaudio, manifest, store

    models = []
    seen: dict[str, dict] = {}
    for pk in manifest.all_packs():
        for m in pk.models:
            if m.key in pk.sources:
                continue
            if m.key in seen:
                seen[m.key]["packs"].append(pk.id)
                continue
            if m.convert is not None:
                src = pk.model(m.convert.from_key)
                row = {"key": m.key, "tag": m.tag, "repo": src.repo, "revision": src.revision,
                       "converted": True, "bytes": sum(o.size for o in m.convert.output)}
            else:
                row = {"key": m.key, "tag": m.tag, "repo": m.repo,
                       "revision": m.revision if m.kind == "hf" else m.release_tag,
                       "converted": False, "bytes": m.fetch_bytes}
            row.update(licence=m.licence, credit=m.credit, packs=[pk.id])
            seen[m.key] = row
            models.append(row)
    ff, fp = ffm.find()
    info = fluidaudio.check()
    hf_bytes = 0
    for row in models:
        m = next(x for pk in manifest.all_packs() for x in pk.models if x.key == row["key"])
        if m.kind == "hf":
            for f in m.files:
                link = store.snapshot_dir(m.repo, m.revision) / f.path
                try:
                    hf_bytes += os.stat(link).st_size
                except OSError:
                    pass
    folders = [
        {"path": str(platforms.data_dir()), "what": "converted and assembled models, the speaker program, "
         "voiceprints, records, the packs' stamps, the test recording", "bytes": folder_bytes(platforms.data_dir())},
        {"path": str(platforms.cache_dir()), "what": "work folders (deleted after each run) and builds",
         "bytes": folder_bytes(platforms.cache_dir())},
        {"path": str(platforms.logs_dir()), "what": "run logs (no transcript text), kept 90 days",
         "bytes": folder_bytes(platforms.logs_dir())},
        {"path": str(platforms.hf_cache()), "what": "the Hugging Face cache: audiointake's pinned models in it",
         "bytes": hf_bytes},
    ]
    from .setup import swiftpm

    for a in swiftpm.pinned_artifacts():
        p = swiftpm.artifacts_dir() / swiftpm.cache_name(a["url"])
        if p.is_file():
            folders.append({"path": str(p), "what": "SwiftPM's download cache: a component of the speaker "
                            "program's build (uninstall removes it when audiointake downloaded it)",
                            "bytes": p.stat().st_size})
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "models": models,
        "fluidaudio": {"tag": fluidaudio.TAG, "commit": fluidaudio.COMMIT, "recipe": fluidaudio.RECIPE,
                       "repo": fluidaudio.REPO_URL, "built": info is not None,
                       "path": info["path"] if info else None, "licence": FLUIDAUDIO_LICENCE},
        "packages": {d: {"version": _version(d), "wheel_tags": wheel_tags(d)} for d in ("mlx", "mlx-metal")},
        "ffmpeg": {"path": str(ff) if ff else None, "version": ffm.version(ff) if ff else "",
                   "fallback": ff is not None and ff == ffm.fallback_link(), "ffprobe": str(fp) if fp else None},
        "hosts": list(HOSTS),
        "folders": folders,
        "data_credits": list(DATA_CREDITS),
        "network": PRIVACY_LINE,
    }
