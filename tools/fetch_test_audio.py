"""Fetch the public test audio the models tests and bench fixtures use, pinned by size and SHA-256.

Everything lands under::

    <root>/ami/ES2004a-825-915.wav            AMI ES2004a, 825-915 s (fetched by HTTP byte range)
    <root>/fleurs/fl/<lang>/<id>.wav, .ref     FLEURS test clips and their human transcripts

where ``<root>`` is ``--root``, else ``$AUDIOINTAKE_TEST_AUDIO``, else ``~/.audiointake-dev/audio``.
``tests/conftest.py``'s ``pinned_audio(key)`` reads the same layout through ``AUDIO`` below, and
``bench/fixtures.py build OUT --fleurs <root>/fleurs --ami <root>/ami/ES2004a-825-915.wav --ami-start 825`` builds the
path fixtures from it.

Sources:
* AMI: one byte range of the meeting's Array1-01 wav on the AMI mirror (16 kHz mono 16-bit; the 44-byte header was
  checked), given a fresh 44-byte WAV header here.
* FLEURS: each language's test.tsv (the transcripts) and test audio archive at a fixed google/fleurs commit. The
  archive (about 550 MB) is streamed and read only up to the last wanted clip; its members come in name order. Only
  the pinned clips are written, nothing else of the archive is kept.
Every written file is checked against its pinned size and SHA-256 before it gets its name; a mismatch keeps nothing.
A file already present with the pinned size and SHA-256 is kept and not fetched again. No token or account is used.

Standard library only, so it runs from a fresh clone before audiointake is installed::

    python3 tools/fetch_test_audio.py              # fetch whatever is missing or wrong
    python3 tools/fetch_test_audio.py --check      # only check; never touches the network
    python3 tools/fetch_test_audio.py --only ami   # ami, fleurs, or a key (repeatable)

Exit 0 when every requested file is present and correct afterwards, 1 when one is missing or wrong (--check) or
couldn't be fetched, 2 for bad arguments.

Credits. AMI Meeting Corpus, CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/corpus/; Carletta et al., "The AMI Meeting
Corpus", 2005). FLEURS, CC BY 4.0 (google/fleurs; Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech", 2022).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

ENV = "AUDIOINTAKE_TEST_AUDIO"
DEFAULT_ROOT = Path.home() / ".audiointake-dev" / "audio"
UA = {"User-Agent": "audiointake-test-audio"}

AMI_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Array1-01.wav"
AMI_HEADER = 44                    # bytes of the source file's WAV header; samples follow at 16 kHz, mono, 16-bit
FLEURS_COMMIT = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/" + FLEURS_COMMIT + "/data/{lang}/{path}"
# lang -> (test.tsv sha256, test.tar.gz bytes, test.tar.gz sha256). The archive's size is checked against the length
# the server announces before anything is read; its SHA-256 is kept for reference only, because the stream stops at
# the last wanted clip (every clip written is checked against its own SHA-256).
FLEURS = {
    "de_de": ("82fab72c58a347345675c238f2492bb997105a4ab45d69e2a2b34ed29082fa97", 568734559,
              "e86b42dfcdef749926cd92135045f87c25966c09e50d01c401adb04ee7d8628f"),
    "es_419": ("d107a93a4f54a18ac25cd470bb4cdadce14fb075b0c1d1542258e274d209ec09", 582112372,
               "981802f6c828fd214fcf8bfc1036d80c9184b6eeb5650b3f7882f8affec046c9"),
    "hu_hu": ("939accd545532c3b0eb34891742829314267efd6c67a4a5ac5e15aef23dec82f", 542250479,
              "d2335311350b4c9a6a1811b3ca3faff772a97befe0a7106dbda1edf675b0cb8a"),
    "th_th": ("8db09d12ac384654f6677e141f6e55bf5bc42028160cbc0d2d9f7fac4fe25271", 593463305,
              "a3c8bbfab5a124fb6b6475c238a9ec09957be5727aeacefe00eddbcf4bc0a82e"),
}

# key -> {"file", "bytes", "sha256"} plus where it comes from. The FLEURS clips are the ones bench/fleurs.py selects
# for these ids (same bytes), so the fixtures and ceilings here match the bench.
AUDIO: dict[str, dict] = {
    "ami-es2004a-825-915": {"file": "ami/ES2004a-825-915.wav", "bytes": 2880044,
                            "sha256": "c706cb6b6c4d96039a8b3462563328b2bda73a7e43cf1e3bb0972b43b254af3a", "ami": (825, 915)},
}
FLEURS_CLIPS = {  # lang -> the ids pinned here: the bench fixtures' clips (bench/fixtures.py)
    "de_de": "1662 1665 1670 1687 1688".split(),
    "es_419": "1662 1665 1670 1687 1688".split(),
    "hu_hu": "1660 1670 1676 1677 1680 1683 1684 1694 1713".split(),
    "th_th": "1663 1672 1673 1675".split(),
}
for _lang, _pins in {
    "de_de": {
        "1662": ("11871610055270234526.wav", 1501498, "cb4108041c7b9ea6d0ae9109b810646b954646476225c632cf0fb5bd6467ac53", 137, "a10b37ba32ca9ce6523f4aab4525c0f22582132c846a79cf206b106da237c642"),
        "1665": ("10534115037762380445.wav", 683578, "7e5caa4feee3f52b1e071df3baed6aa9a09a0b220bebdf3825931b98a9a143f0", 175, "6ac9f25615a1c3e95653f79b2698694a22512cd0efbc995f2678c550392e4f58"),
        "1670": ("10844590730633377222.wav", 1351738, "a2671212059e0e1be57eb72678f056c613f40ceda0441c38527d14a5f9302799", 367, "116a3aa22f961bdd635ad84286a1f01a23360761e78a1e9478fe6cfb0bcb01c1"),
        "1687": ("18306148757630670005.wav", 449338, "d8bfbf179e5654a53b2486a3f753d623b5c2421fe543c1ce324551c0418f76d6", 111, "1bb08f2d8372e8c7a4dca686fd6c91962b69296ccc300a9bb46149d134721038"),
        "1688": ("4484278680542083067.wav", 1040698, "701c815b7708d46e5e8a9bbc7cda939430bb995776cea673215b40b9d6cfbed7", 225, "58f0e0c52faef768ba8bb809e6dbde63b38ad4abf0256c7326ce35155411ffc4"),
    },
    "es_419": {
        "1662": ("8616615381295825121.wav", 610618, "97a321fc05f1eb5c6e641abd33778d3a3b33d3c977e8fa410d0aa47e3b93f1cf", 130, "b0bef3ac78417ae7aa1d4caab6ef294a90ed37cb8940315a6227e07c88092f7e"),
        "1665": ("13170025242269863091.wav", 1006138, "8c7fd7bf1da7dce96cad02479e57fd61ecc07e92c286b234ee1e1b9629d74002", 214, "6dff23a7a61a3423f4717d0ff472a62fc4464cca59d8b7c22c557e78805bbefe"),
        "1670": ("13066583163751268675.wav", 1885498, "fe737a838ba3a2f89a47c4c4af2e43d29f6c59daf0a4e5f7603cb1cf01442208", 372, "e8a5712077a4f45df15abe0cc5e76a6e6cfdd7ac5846e87071a0e481fe4cad37"),
        "1687": ("1528362042872231265.wav", 656698, "1f281872611e6dba80dc85860c4d065fc03e7a36425b63391d037bb8ff289c38", 104, "e44199d637ddb9cb55d479dacf6eb9ea1374b966ff948c7c6d1ae5b58183cecf"),
        "1688": ("1934738271656370362.wav", 979258, "0f61639c273aa19693ffafab740e57a44ae2cd002b96146c9e021c1d112fa94c", 212, "ae151d120a2eaca42daefae084d23539d0435886e4da3afce132fba23d76d5d3"),
    },
    "hu_hu": {
        "1660": ("15687421911359186268.wav", 852538, "784de500114074153422e306a15000f063330f937ca8ec6ab0d9a536101ef61a", 140, "abf47f48b63bf25a9a5c646e766c96596e5311d58df4abf9d069f2aec469d31d"),
        "1670": ("8997960483099423968.wav", 1559098, "d55806d53d7f4ffbc9702d5f84208b19be84f955fd79fd6d6295647f7afe9cf3", 410, "1493cecd6ed290e84e00858a16cbc0e074c8c247cf328b5553c00a7752758f9d"),
        "1676": ("12655698477449380128.wav", 595258, "4654bcd89b6207cf27b3e73a89080cc8b3f4fa6ee75c76d14401c96d5ecfae4b", 117, "530f5b9910edfb45b6fc6bbfc4f13b246ecc7bd873599c0b1116677f097b9281"),
        "1677": ("10821670015947767881.wav", 522298, "7057d0bcbc9783960294135cc7fd076f824148f19f446ccdbe281646c7a0eac7", 91, "013acb48682a02bcfd55de8d3c211a28821c71a64814d92d6db1219f354b3f76"),
        "1680": ("1904220574709946773.wav", 829498, "86fed1ea6dfedc86f0e841e959dd82284a94d1572f190d39b6548de801921822", 170, "172194eba263246da37efac8d6037d6202bb3342145fb4d92323d41d9601f82b"),
        "1683": ("17519930839937527286.wav", 1063738, "d70eb0f078d72123191ee34661ec0b993e1134fa0e2a6f5bc8ce0f5d1149ebfd", 249, "fc5a9f9a8c0e152d5335bbbaa04d943965fbce82579fb0993f6cf53e0c43076a"),
        "1684": ("14239310076151391667.wav", 675898, "9fb0946b05a62514809f084eb062c0315e772e7fd1306b0e91fa0821a589e79b", 107, "981261bc2a393fab3e13f500653a7c4be0c02e3133c2234bc9ef0af35ac1b12e"),
        "1694": ("1922818108344798965.wav", 867898, "03887290a0d5240a3d1b31286d67a32a76b4b2263607cd12ba9345da0e212575", 137, "713504ec5671d379544a8bf73f6af59d4e9919e055f47e6e1d2aea46196e90b1"),
        "1713": ("11753403436568941118.wav", 841018, "33e0ae0117dc7d57ab6ddf6d7288740a6636e090c5cb3ba2d529deda58cc2484", 162, "7808bd4f9fde4a8c63d085afd9c8fdf6906f74022d86b982355cb4b885a5369b"),
    },
    "th_th": {
        "1663": ("2840208424796413508.wav", 1006138, "abe758e402fdd5b703adec7b5f79df19216eaa0aaa063bf6a33c43ce477569c1", 324, "d293df7b5c828415169c69f8ab9b299fc3b61dc3013b16a665739c519b1fe446"),
        "1672": ("18003499988277993422.wav", 595258, "bb98811737bac5dd112dbafa68ba72c4534202f81f1c3bfaad7bc5d200aa2e1e", 237, "6388efe10c2ee8e8b1f654c2f3f554959a4dea4671f554aedf88401bd91febf8"),
        "1673": ("17613567360463218581.wav", 937018, "e06b17834fcb6010e6c7cce5ea0e641f1e567fc93e5cb094b6675f9daa011086", 422, "eb477fcfe303b3d17eb61ca89aa164611978ed44eb7275acaf61f3063aa8faff"),
        "1675": ("6081159288778122130.wav", 368698, "6072a231250887f849095ead41dac79249b37e383a76bf6d82396033febfbda2", 271, "66c62a33f0485135cd3e01486aebb2fabc9e6f68a1dca8f1e8796a0b747d31c9"),
    },
}.items():
    for _i, (_member, _wb, _ws, _rb, _rs) in _pins.items():
        AUDIO[f"fleurs-{_lang}-{_i}"] = {"file": f"fleurs/fl/{_lang}/{_i}.wav", "bytes": _wb, "sha256": _ws,
                                         "lang": _lang, "member": _member}
        AUDIO[f"fleurs-{_lang}-{_i}-ref"] = {"file": f"fleurs/fl/{_lang}/{_i}.ref", "bytes": _rb, "sha256": _rs,
                                             "lang": _lang, "member": _member}


# --------------------------------------------------------------------------- helpers
def default_root() -> Path:
    env = os.environ.get(ENV)
    return Path(env).expanduser() if env else DEFAULT_ROOT


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def present(root: Path, key: str) -> bool:
    item = AUDIO[key]
    p = root / item["file"]
    try:
        if not p.is_file() or p.stat().st_size != item["bytes"]:
            return False
        return sha256_bytes(p.read_bytes()) == item["sha256"]
    except OSError:
        return False


def place(root: Path, key: str, data: bytes) -> None:
    item = AUDIO[key]
    if len(data) != item["bytes"] or sha256_bytes(data) != item["sha256"]:
        raise RuntimeError(f"{key}: fetched data doesn't match its pin; nothing kept")
    dest = root / item["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.partial-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


def ami_excerpt(start: float, end: float) -> bytes:
    a, b = AMI_HEADER + int(start * 32000), AMI_HEADER + int(end * 32000)
    req = urllib.request.Request(AMI_URL, headers={**UA, "Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            raise RuntimeError(f"the AMI mirror answered {r.status} to a byte-range request")
        pcm = r.read()
    if len(pcm) != b - a:
        raise RuntimeError("the AMI mirror sent a short range")
    return wav_bytes(pcm)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def fleurs_lang(root: Path, lang: str, keys: list[str]) -> None:
    tsv_sha, tar_bytes, _ = FLEURS[lang]
    tsv = fetch(FLEURS_BASE.format(lang=lang, path="test.tsv"))
    if sha256_bytes(tsv) != tsv_sha:
        raise RuntimeError(f"{lang} test.tsv doesn't match its pin")
    raw = {}
    for line in tsv.decode("utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 3:
            raw[f[1]] = f[2]
    for k in [k for k in keys if k.endswith("-ref")]:
        place(root, k, raw[AUDIO[k]["member"]].encode("utf-8"))
    want = {AUDIO[k]["member"]: k for k in keys if not k.endswith("-ref")}
    if not want:
        return
    req = urllib.request.Request(FLEURS_BASE.format(lang=lang, path="audio/test.tar.gz"), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        announced = r.headers.get("Content-Length")
        if announced is not None and int(announced) != tar_bytes:
            raise RuntimeError(f"{lang} test.tar.gz is {announced} bytes; the pin is {tar_bytes}")
        with tarfile.open(fileobj=r, mode="r|gz") as tf:
            for m in tf:
                name = m.name.rsplit("/", 1)[-1]
                if m.isfile() and name in want:
                    place(root, want.pop(name), tf.extractfile(m).read())
                    if not want:
                        break
    if want:
        raise RuntimeError(f"{lang}: {len(want)} clips weren't in the archive")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_test_audio.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path)
    ap.add_argument("--check", action="store_true", help="only check; never touch the network")
    ap.add_argument("--only", action="append", default=[], help="ami, fleurs, or a key (repeatable)")
    a = ap.parse_args(argv)
    root = (a.root or default_root()).expanduser()
    keys = list(AUDIO)
    if a.only:
        sel = []
        for o in a.only:
            if o == "ami":
                sel += [k for k in keys if "ami" in AUDIO[k]]
            elif o == "fleurs":
                sel += [k for k in keys if "lang" in AUDIO[k]]
            elif o in AUDIO:
                sel.append(o)
            else:
                ap.error(f"unknown selection {o}")
        keys = [k for k in keys if k in sel]
    missing = [k for k in keys if not present(root, k)]
    if a.check or not missing:
        for k in missing:
            print(f"missing or wrong: {k} ({root / AUDIO[k]['file']})")
        print(f"{len(keys) - len(missing)} of {len(keys)} present and correct under {root}")
        return 1 if missing else 0
    failed = 0
    try:
        for k in [k for k in missing if "ami" in AUDIO[k]]:
            place(root, k, ami_excerpt(*AUDIO[k]["ami"]))
        by_lang: dict[str, list[str]] = {}
        for k in missing:
            if "lang" in AUDIO[k]:
                by_lang.setdefault(AUDIO[k]["lang"], []).append(k)
        for lang, ks in sorted(by_lang.items()):
            fleurs_lang(root, lang, ks)
    except (OSError, RuntimeError, tarfile.TarError) as e:
        print(f"fetch_test_audio: {e}", file=sys.stderr)
        failed = 1
    still = [k for k in keys if not present(root, k)]
    print(f"{len(keys) - len(still)} of {len(keys)} present and correct under {root}")
    return 1 if (still or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
