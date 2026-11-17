#!/usr/bin/env python3
"""Run audiointake's transcription routes on the FLEURS clips that bench/fleurs.py prepared.

Every system writes WORKDIR/fo/<system>/<lang>/<id>.txt, which `fleurs.py score` reads. Run the steps in this order:

  fleurs_routes.py asr WORKDIR LANG... [--parakeet BIN] [--model-ref]
      parakeet-mlx in batch with each Parakeet model (ultra, v3): JSON with token confidences in fj/<model>/<lang>/,
      text in fo/<model>/<lang>/.
  fleurs_routes.py whisper WORKDIR LANG... [--whisper BIN] [--variant route|plain] [--ffmpeg BIN] [--model-ref]
      Whisper large-v3. "route" (the default) is how the vote route calls it: a loudness-normalised 16 kHz wav,
      `--language <code> --condition-on-previous-text False --temperature 0 --compression-ratio-threshold 2.0`;
      writes fj/whisper/ and fo/whisper/. "plain" is the clip as it is with only
      `--condition-on-previous-text False --temperature 0`; writes fj/whisper-plain/ and fo/whisper-plain/. Also
      writes the plain 16 kHz wav the vote route's energy check reads (wav16/<lang>/<id>.wav).
  fleurs_routes.py routes WORKDIR LANG... [--ensemble FILE] [--whisper-variant route|plain]
      Per clip, from the outputs above: "core" = ensemble.gapfill(ultra, v3), the route every language without a
      pack takes; "vote" = ensemble.hu(whisper, ultra, v3, wav16), the Whisper vote exactly as built; "vote3" = the
      text-only three-way vote of the first ensemble study, for comparison with old numbers. Text is joined the way
      ensemble.transcribe joins it. Also writes outro.json: per language, how many Whisper outputs carry a
      subtitle-outro phrase of each language that the clip's reference doesn't contain. With --whisper-variant plain
      the two votes read the plain call's Whisper output instead and are written as "vote-plain" and "vote3-plain"
      (core and outro.json are left as they are), to tell how much of a vote's number is the Whisper call's.
  fleurs_routes.py e2e WORKDIR LANG... [--ensemble FILE --python PY]
      The whole `ensemble transcribe` per clip, Thai screen included, into fo/e2e/; tags.json records each clip's
      route tag. By default `python -m audiointake.steps.ensemble`; with --ensemble, PY runs that file.

--ensemble FILE loads the ensemble module from a file instead of audiointake.steps.ensemble, so a copy of the
module can be measured against the package's own. Models are the revisions audiointake pins: asr and whisper pass the
local snapshot folder of each pinned revision (from the package's pack manifests and the Hugging Face cache, found as
huggingface_hub finds it), and refuse when it isn't there (`audiointake setup` puts it there). --model-ref passes the
repository name instead, so the child reads whatever the cache's main points to: that is how a caller resolving models
by name runs them, for measuring such a caller. Every child runs with HF_HUB_OFFLINE=1 and without any PARAKEET_*,
THAI*, WHISPER_*, SPEAKER_ID_*, Hugging Face token or endpoint variable. Nothing here writes outside WORKDIR.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True   # no __pycache__ beside the bench scripts, wherever the clone lives
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleurs  # noqa: E402

MODEL_KEYS = {"ultra": "core", "v3": "core", "whisper": "hu"}   # model key -> the pack manifest that pins it
LANG_CODE = {"hu_hu": "hu", "de_de": "de", "es_419": "es", "th_th": "th"}
OUTRO = {  # subtitle-credit phrases Whisper invents on silence, per language (case-insensitive)
    "hu_hu": r"megnézt|feliratoz|iratkozz|köszönöm a figyelm",
    "de_de": r"untertitel|vielen dank",
    "es_419": r"amara|subtítulos|suscr|gracias por ver",
}
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


def scripts_dir() -> Path:
    return Path(sysconfig.get_path("scripts"))


def hf_cache() -> Path:
    """The Hugging Face hub cache, found as huggingface_hub finds it."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return Path(os.path.expanduser(os.environ[name])).resolve()
    if os.environ.get("HF_HOME"):
        return Path(os.path.expanduser(os.environ["HF_HOME"])).resolve() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(os.path.expanduser(xdg)) if xdg else Path.home() / ".cache") / "huggingface" / "hub"


def pinned(key: str) -> tuple[str, str]:
    """(repository, 40-hex revision) of a model, from the package's own pack manifest."""
    from importlib import resources
    pack = json.loads(resources.files("audiointake.setup").joinpath(f"packs/{MODEL_KEYS[key]}.json")
                      .read_text(encoding="utf-8"))
    m = next(m for m in pack["models"] if m["key"] == key)
    return m["repo"], m["revision"]


def model_arg(key: str, by_ref: bool) -> str:
    """What to pass as --model: the pinned revision's local snapshot folder, or with by_ref the repository name."""
    repo, rev = pinned(key)
    if by_ref:
        return repo
    snap = hf_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if not (snap / "config.json").is_file():
        raise SystemExit(f"fleurs_routes: the pinned revision {rev[:12]} of {repo} isn't in the Hugging Face cache "
                         f"({hf_cache()}); `audiointake setup` (or `--pack` for Whisper) puts it there")
    return str(snap)


DROPPED = ("PARAKEET_", "WHISPER_", "SPEAKER_ID_", "THAI", "HF_TOKEN", "HUGGING_FACE", "HF_ENDPOINT", "HF_HOME",
           "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROPPED)}
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", ORT_DISABLE_TELEMETRY="1", PYTHONUTF8="1",
               HF_HUB_CACHE=str(hf_cache()))
    return env


def run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    r = subprocess.run([str(a) for a in argv], env=kw.pop("env", child_env()), stdin=subprocess.DEVNULL, **kw)
    if r.returncode:
        raise SystemExit(f"fleurs_routes: {Path(str(argv[0])).name} exited {r.returncode}")
    return r


def load_ensemble(path: str | None):
    if not path:
        return importlib.import_module("audiointake.steps.ensemble")
    spec = importlib.util.spec_from_file_location("ensemble_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def joined(d: dict) -> str:
    """The one-line text ensemble.transcribe writes for a sentence list."""
    return " ".join(x["text"].strip() for x in d["sentences"]).strip() + "\n"


def clip_ids(wd: Path, lang: str) -> list[str]:
    ids = fleurs.IDS[lang]
    missing = [i for i in ids if not (wd / "fl" / lang / f"{i}.wav").is_file()]
    if missing:
        raise SystemExit(f"fleurs_routes: {lang} has {len(missing)} clips missing; run fleurs.py prep first")
    return ids


# --------------------------------------------------------------------------- asr
def cmd_asr(wd: Path, langs: list[str], parakeet: Path, by_ref: bool) -> None:
    models = {key: model_arg(key, by_ref) for key in ("ultra", "v3")}
    for lang in langs:
        ids = clip_ids(wd, lang)
        for key, repo in models.items():
            fj, fo = wd / "fj" / key / lang, wd / "fo" / key / lang
            fj.mkdir(parents=True, exist_ok=True); fo.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=wd) as tmp:
                run([parakeet, *[wd / "fl" / lang / f"{i}.wav" for i in ids], "--model", repo,
                     "--output-dir", tmp, "--output-format", "all"], stdout=subprocess.DEVNULL)
                for i in ids:
                    shutil.move(Path(tmp) / f"{i}.json", fj / f"{i}.json")
                    shutil.move(Path(tmp) / f"{i}.txt", fo / f"{i}.txt")
            print(f"asr {key} {lang}: {len(ids)} clips")


# --------------------------------------------------------------------------- whisper
def cmd_whisper(wd: Path, langs: list[str], whisper: Path, variant: str, ffmpeg: str, by_ref: bool) -> None:
    model = model_arg("whisper", by_ref)
    for lang in langs:
        ids = clip_ids(wd, lang)
        w16, wln = wd / "wav16" / lang, wd / "wavln" / lang
        w16.mkdir(parents=True, exist_ok=True); wln.mkdir(parents=True, exist_ok=True)
        for i in ids:
            src = wd / "fl" / lang / f"{i}.wav"
            run([ffmpeg, "-nostdin", "-y", "-v", "error", "-vn", "-i", src, "-ar", "16000", "-ac", "1", w16 / f"{i}.wav"])
            if variant == "route":
                run([ffmpeg, "-nostdin", "-y", "-v", "error", "-vn", "-i", src, "-ar", "16000", "-ac", "1",
                     "-af", LOUDNORM, wln / f"{i}.wav"])
        name = "whisper" if variant == "route" else "whisper-plain"
        fo = wd / "fo" / name / lang
        fo.mkdir(parents=True, exist_ok=True)
        if variant == "route":
            inputs = [wln / f"{i}.wav" for i in ids]
            flags = ["--language", LANG_CODE[lang], "--condition-on-previous-text", "False", "--temperature", "0",
                     "--compression-ratio-threshold", "2.0"]
        else:
            inputs = [wd / "fl" / lang / f"{i}.wav" for i in ids]
            flags = ["--condition-on-previous-text", "False", "--temperature", "0"]
        # One process per clip, as the route runs it. mlx_whisper 0.4.3 given several files names every output after
        # the first file (its output name is set once and reused), so a batch would overwrite one result 50 times.
        with tempfile.TemporaryDirectory(dir=wd) as tmp:
            for i, src in zip(ids, inputs):
                run([whisper, src, "--model", model, *flags, "--output-format", "all", "--output-dir", tmp,
                     "--output-name", i, "--verbose", "False"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            fj = wd / "fj" / name / lang
            fj.mkdir(parents=True, exist_ok=True)
            for i in ids:
                shutil.move(Path(tmp) / f"{i}.txt", fo / f"{i}.txt")
                shutil.move(Path(tmp) / f"{i}.json", fj / f"{i}.json")
        print(f"whisper {variant} {lang}: {len(ids)} clips")


# --------------------------------------------------------------------------- routes
def vote3_words(backbone: list[str], u: list[str], v: list[str]) -> list[str]:
    """Text-only three-way vote: keep the backbone unless the other two agree against it at the same position."""
    def mapper(x, y):
        ops = difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
        def m(i):
            for tg, i1, i2, j1, j2 in ops:
                if i1 <= i < i2 or (i == i2 and i == len(x) and i2 == len(x)):
                    return j1 + (i - i1 if tg == "equal" else 0)
            return len(y)
        return m
    mv = mapper(backbone, v)
    out: list[str] = []
    for tg, i1, i2, j1, j2 in difflib.SequenceMatcher(None, backbone, u, autojunk=False).get_opcodes():
        if tg == "equal":
            out += backbone[i1:i2]; continue
        vv = v[mv(i1):mv(i2)] if i2 > i1 else v[mv(i1):mv(i1)]
        out += u[j1:j2] if vv == u[j1:j2] else backbone[i1:i2]
    return out


def cmd_routes(wd: Path, langs: list[str], ensemble_file: str | None, variant: str = "route") -> None:
    ens = load_ensemble(ensemble_file)
    wsys, suffix = ("whisper", "") if variant == "route" else ("whisper-plain", "-plain")
    outro = {}
    for lang in langs:
        ids = clip_ids(wd, lang)
        for s in ("core", "vote", "vote3"):
            (wd / "fo" / (s + suffix if s != "core" else s) / lang).mkdir(parents=True, exist_ok=True)
        hits = {name: 0 for name in OUTRO}
        for i in ids:
            u, v, w = (wd / "fj" / m / lang / f"{i}.json" for m in ("ultra", "v3", wsys))
            with tempfile.TemporaryDirectory(dir=wd) as tmp:
                if not suffix:
                    ens.gapfill(str(u), str(v), f"{tmp}/core.json")
                    (wd / "fo" / "core" / lang / f"{i}.txt").write_text(joined(ens.load(f"{tmp}/core.json")),
                                                                         encoding="utf-8")
                ens.hu(str(w), str(u), str(v), str(wd / "wav16" / lang / f"{i}.wav"), f"{tmp}/vote.json")
                (wd / "fo" / f"vote{suffix}" / lang / f"{i}.txt").write_text(joined(ens.load(f"{tmp}/vote.json")),
                                                                            encoding="utf-8")
            t = {m: fleurs.norm((wd / "fo" / m / lang / f"{i}.txt").read_text(encoding="utf-8"))
                 for m in (wsys, "ultra", "v3")}
            (wd / "fo" / f"vote3{suffix}" / lang / f"{i}.txt").write_text(
                " ".join(vote3_words(t[wsys], t["ultra"], t["v3"])) + "\n", encoding="utf-8")
            wtext = (wd / "fo" / wsys / lang / f"{i}.txt").read_text(encoding="utf-8")
            rtext = (wd / "fl" / lang / f"{i}.ref").read_text(encoding="utf-8")
            for name, pat in OUTRO.items():   # counted only where the reference doesn't say it too
                hits[name] += bool(re.search(pat, wtext, re.I)) and not re.search(pat, rtext, re.I)
        outro[lang] = hits
        print(f"routes{suffix} {lang}: {len(ids)} clips; Whisper outputs with an outro phrase the reference lacks: "
              f"{hits}")
    if suffix:
        return
    p = wd / "outro.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(outro)
    p.write_text(json.dumps(old, indent=1) + "\n")


# --------------------------------------------------------------------------- e2e
def cmd_e2e(wd: Path, langs: list[str], ensemble_file: str | None, python: str) -> None:
    tags_p = wd / "tags.json"
    tags = json.loads(tags_p.read_text()) if tags_p.exists() else {}
    for lang in langs:
        ids = clip_ids(wd, lang)
        fo = wd / "fo" / "e2e" / lang
        fo.mkdir(parents=True, exist_ok=True)
        tags[lang] = {}
        for i in ids:
            cmd = [python, ensemble_file] if ensemble_file else [python, "-m", "audiointake.steps.ensemble"]
            with tempfile.TemporaryDirectory(dir=wd) as tmp:
                r = run([*cmd, "transcribe", wd / "fl" / lang / f"{i}.wav", "--output-dir", tmp,
                         "--output-template", i, "--output-format", "all"],
                        capture_output=True, text=True)
                shutil.move(Path(tmp) / f"{i}.txt", fo / f"{i}.txt")
            tags[lang][i] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"e2e {lang}: {len(ids)} clips; tags {sorted(set(tags[lang].values()))}")
    tags_p.write_text(json.dumps(tags, indent=1) + "\n")
    # A route that failed (a "+...-failed" tag) fell back to another route's text, so the numbers would measure
    # the fallback, not the system: refuse them rather than score them.
    failed = sorted({t for per in tags.values() for t in per.values() if "failed" in t})
    if failed:
        raise SystemExit(f"fleurs_routes: e2e routes failed ({', '.join(failed)}); see the clips in {tags_p.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fleurs_routes.py", description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("asr", "whisper", "routes", "e2e"):
        p = sp.add_parser(name)
        p.add_argument("workdir", type=Path)
        p.add_argument("langs", nargs="+")
        if name == "asr":
            p.add_argument("--parakeet", type=Path, default=scripts_dir() / "parakeet-mlx")
        if name == "whisper":
            p.add_argument("--whisper", type=Path, default=scripts_dir() / "mlx_whisper")
            p.add_argument("--variant", choices=("route", "plain"), default="route")
            p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
        if name in ("asr", "whisper"):
            p.add_argument("--model-ref", action="store_true",
                           help="pass the repository name, not the pinned revision's snapshot folder")
        if name in ("routes", "e2e"):
            p.add_argument("--ensemble")
        if name == "routes":
            p.add_argument("--whisper-variant", choices=("route", "plain"), default="route")
        if name == "e2e":
            p.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    bad = [x for x in a.langs if x not in fleurs.IDS]
    if bad:
        ap.error(f"unknown language {bad[0]}; known: {', '.join(sorted(fleurs.IDS))}")
    wd = a.workdir.resolve()
    if a.cmd == "asr":
        cmd_asr(wd, a.langs, a.parakeet, a.model_ref)
    elif a.cmd == "whisper":
        cmd_whisper(wd, a.langs, a.whisper, a.variant, a.ffmpeg, a.model_ref)
    elif a.cmd == "routes":
        cmd_routes(wd, a.langs, a.ensemble, a.whisper_variant)
    else:
        cmd_e2e(wd, a.langs, a.ensemble, a.python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
