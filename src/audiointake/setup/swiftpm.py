"""Seed SwiftPM's binary-artifact cache before ``swift build`` runs.

SwiftPM's own downloader can stall on a ``binaryTarget`` zip with no timeout and no error (seen for hours on
FluidAudio's NemoTextProcessing.xcframework.zip, which a plain HTTPS client fetched in seconds). SwiftPM reuses a
zip it finds in ``~/Library/Caches/org.swift.swiftpm/artifacts/<url with every non-alphanumeric character as _>``
once its checksum matches, so ``prefetch`` reads every ``.binaryTarget(url:, checksum:)`` from ``Package*.swift``
in the package folder and in its resolved checkouts, fetches each zip that isn't cached through setup's own
downloader (resumable, redirects only to GitHub's hosts, SHA-256 checked) and places it there. The downloader
needs each zip's size, so only the zips ``PINNED`` lists are prefetched (the pinned FluidAudio commit names one);
any other is left to SwiftPM, which checks its checksum itself. Best effort: a zip that can't be fetched is left
for SwiftPM to try.
"""
from __future__ import annotations

import re
from pathlib import Path

_TARGET = re.compile(r"\.binaryTarget\s*\((.*?)\)", re.S)
_URL = re.compile(r'url:\s*"([^"]+)"', re.S)
_SUM = re.compile(r'checksum:\s*"([0-9a-f]{64})"', re.S)
# What the pinned FluidAudio commit's Package.swift names (for setup --plan, before any source is fetched).
PINNED = ({"url": "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.1/"
                  "NemoTextProcessing.xcframework.zip",
           "checksum": "5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa", "size": 87_267_919},)


def artifacts_dir() -> Path:
    return Path.home() / "Library" / "Caches" / "org.swift.swiftpm" / "artifacts"


def cache_name(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", url)


def binary_targets(text: str) -> list[tuple[str, str]]:
    """(url, checksum) of every ``.binaryTarget`` in a manifest's text that has both."""
    out = []
    for m in _TARGET.finditer(text):
        body = m.group(1)
        u, c = _URL.search(body), _SUM.search(body)
        if u and c:
            out.append((u.group(1), c.group(1)))
    return out


def manifests(src: Path) -> list[Path]:
    src = Path(src)
    found = sorted(src.glob("Package*.swift")) + sorted((src / ".build" / "checkouts").glob("*/Package*.swift"))
    return [p for p in found if p.is_file()]


def targets(src: Path) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for p in manifests(src):
        for t in binary_targets(p.read_text(encoding="utf-8", errors="replace")):
            seen.setdefault(t, None)
    return sorted(seen)


def cached(url: str, checksum: str) -> bool:
    from .download import matches

    p = artifacts_dir() / cache_name(url)
    try:
        return matches(p, p.stat().st_size, checksum, False)
    except OSError:
        return False


def pinned_artifacts() -> tuple[dict, ...]:
    return PINNED


def prefetch(src: Path, *, progress=None) -> list[dict]:
    """Fetch every uncached binary artifact of the package at ``src``; one ``{url, status}`` per artifact, and for
    one fetched now its ``path``, the ``hosts`` contacted and whether audiointake ``downloaded`` it (uninstall
    removes only what it downloaded)."""
    from . import store

    out = []
    for url, checksum in targets(src):
        if not url.startswith("https://github.com/"):
            out.append({"url": url, "status": "skipped"})
            continue
        if cached(url, checksum):
            out.append({"url": url, "status": "cached"})
            continue
        size = next((a["size"] for a in PINNED if a["url"] == url and a["checksum"] == checksum), None)
        if size is None:
            out.append({"url": url, "status": "unpinned"})       # SwiftPM fetches and checks it itself
            continue
        dest = artifacts_dir() / cache_name(url)
        try:
            got = store.fetch_file(url, dest, size, checksum, label=url.rsplit("/", 1)[-1], progress=progress,
                                   lock_file=dest.with_name(f".{dest.name}.lock"))
            out.append({"url": url, "status": "fetched", "path": str(dest), "hosts": list(got.get("hosts") or []),
                        "downloaded": got.get("status") == "downloaded"})
        except store.DownloadFailed as exc:
            out.append({"url": url, "status": "failed", "hosts": [exc.params.get("host", "")]})
    return out

