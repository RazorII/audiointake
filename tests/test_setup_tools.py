"""Setup's smaller parts: SwiftPM's artifact prefetch, the converter's key map, finding ffmpeg (and the fallback),
the smoke helpers, the setup children's environment."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from audiointake.setup import convert, store, swiftpm
from audiointake.setup import ffmpeg as ffm
from audiointake.setup import smoke

PKG = '''
let package = Package(
    targets: [
        .binaryTarget(
            name: "NemoTextProcessing",
            url:
                "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.1/NemoTextProcessing.xcframework.zip",
            checksum: "5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa"
        ),
        .binaryTarget(name: "Local", path: "x.xcframework"),
    ]
)
'''


def test_binary_targets_are_parsed_across_lines():
    assert swiftpm.binary_targets(PKG) == [(swiftpm.PINNED[0]["url"], swiftpm.PINNED[0]["checksum"])]


def test_cache_names_match_swiftpms():
    assert swiftpm.cache_name(swiftpm.PINNED[0]["url"]) == (
        "https___github_com_FluidInference_text_processing_rs_releases_download_v0_3_1_NemoTextProcessing_xcframework_zip")


def test_prefetch_uses_the_cache_and_skips_unpinned_artifacts(tmp_path, monkeypatch, network_attempts):
    src = tmp_path / "pkg"
    (src / ".build" / "checkouts" / "dep").mkdir(parents=True)
    (src / "Package.swift").write_text(PKG)
    (src / ".build" / "checkouts" / "dep" / "Package@swift-6.2.swift").write_text(
        '.binaryTarget(name: "O", url: "https://github.com/o/r/releases/download/v1/o.zip", '
        f'checksum: "{"a" * 64}")')
    assert len(swiftpm.targets(src)) == 2
    monkeypatch.setattr(swiftpm, "cached", lambda url, checksum: url == swiftpm.PINNED[0]["url"])
    fetched = []
    monkeypatch.setattr(store, "fetch_file", lambda *a, **k: fetched.append(a))
    out = swiftpm.prefetch(src)
    assert sorted(o["status"] for o in out) == ["cached", "unpinned"]     # the unpinned zip is left to SwiftPM
    assert not fetched and not network_attempts


def test_an_uncached_pinned_zip_is_fetched_into_swiftpms_cache(tmp_path, monkeypatch):
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "Package.swift").write_text(PKG)
    calls = []

    def fetch(url, dest, size, digest, **k):
        calls.append((url, dest, size))
        return {"status": "downloaded", "fetched": size, "hosts": ["github.com", "release-assets.githubusercontent.com"]}

    monkeypatch.setattr(store, "fetch_file", fetch)
    out = swiftpm.prefetch(src)
    url, dest, size = calls[0]
    assert out == [{"url": swiftpm.PINNED[0]["url"], "status": "fetched", "path": str(dest), "downloaded": True,
                    "hosts": ["github.com", "release-assets.githubusercontent.com"]}]
    assert dest == swiftpm.artifacts_dir() / swiftpm.cache_name(url) and size == 87_267_919


def test_converter_key_map():
    assert convert.map_key("model.encoder.layers.0.self_attn.q_proj.weight") == "encoder.blocks.0.attn.query.weight"
    assert convert.map_key("model.decoder.layers.3.encoder_attn.out_proj.bias") == "decoder.blocks.3.cross_attn.out.bias"
    assert convert.map_key("model.decoder.embed_tokens.weight") == "decoder.token_embedding.weight"
    assert convert.map_key("model.encoder.layer_norm.weight") == "encoder.ln_post.weight"
    dims = convert.dims_from_config({"num_mel_bins": 128, "max_source_positions": 1500, "d_model": 1280,
                                     "encoder_attention_heads": 20, "encoder_layers": 32, "vocab_size": 51866,
                                     "max_target_positions": 448, "decoder_attention_heads": 20,
                                     "decoder_layers": 32})
    assert dims["n_mels"] == 128 and dims["n_text_layer"] == 32 and dims["n_vocab"] == 51866


def test_converter_module_imports_without_mlx(tmp_path):
    import subprocess

    stub = tmp_path / "stub"
    (stub / "mlx").mkdir(parents=True)
    (stub / "mlx" / "__init__.py").write_text("raise ImportError('no Metal here')\n")
    env = {**os.environ, "PYTHONPATH": str(stub), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-B", "-c", "import audiointake.setup.convert, audiointake.setup.smoke, "
                        "audiointake.setup.store, audiointake.setup.runtime, audiointake.doctor, audiointake.about"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _exe(p: Path, body: str = "exit 0") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"#!/bin/sh\n{body}\n")
    p.chmod(0o755)
    return p


def test_ffmpeg_on_path_comes_first_with_its_own_ffprobe(tmp_path, monkeypatch):
    a = _exe(tmp_path / "a" / "ffmpeg")
    _exe(tmp_path / "a" / "ffprobe")
    monkeypatch.setenv("PATH", f"{tmp_path / 'a'}:/usr/bin:/bin")
    assert ffm.find() == (a, tmp_path / "a" / "ffprobe")


def test_no_ffmpeg_anywhere_and_no_fallback_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(ffm.platforms, "FFMPEG_DIRS", (tmp_path / "nope",))
    monkeypatch.setattr(ffm, "fallback_binary", lambda: None)
    assert ffm.find(ensure_fallback=True) == (None, None)


def test_the_fallback_is_linked_only_when_its_pin_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOINTAKE_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(ffm.platforms, "FFMPEG_DIRS", (tmp_path / "nope",))
    binary = _exe(tmp_path / "pkg" / "binaries" / ffm.FALLBACK_NAME, "echo ffmpeg version 9.0")
    monkeypatch.setattr(ffm, "fallback_binary", lambda: binary)
    assert ffm.find(ensure_fallback=True) == (None, None)            # wrong size and hash: refused
    monkeypatch.setattr(ffm, "FALLBACK_SIZE", binary.stat().st_size)
    monkeypatch.setattr(ffm, "FALLBACK_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())
    ff, fp = ffm.find(ensure_fallback=True)
    assert ff == ffm.fallback_link() and fp is None and os.readlink(ff) == str(binary)
    assert ff.name == "ffmpeg"                                        # children find it on PATH by that name
    assert ffm.find() == (ff, None)                                   # a run uses the link setup made


def test_the_fallback_pin_is_the_wheel_check():
    assert (ffm.FALLBACK_VERSION, ffm.FALLBACK_NAME, ffm.FALLBACK_SIZE) == ("0.8.1", "ffmpeg-macos-aarch64-v9.0",
                                                                             52_070_376)
    assert ffm.FALLBACK_SHA256.startswith("591260c9")


def test_transcript_words_and_rttm_speakers():
    doc = {"asr": {"sentences": [{"text": " Zorvel brimdal, quessa."}]}}
    assert smoke.transcript_words(doc) == {"zorvel", "brimdal", "quessa"}
    doc["turns"] = [{"text": "Toraph's velmine"}]
    assert smoke.transcript_words(doc) == {"toraph's", "velmine"}
    assert smoke.rttm_speakers("SPEAKER a 1 0 1 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER a 1 1 1 <NA> <NA> speaker_0 "
                               "<NA> <NA>\nSPEAKER a 1 2 1 <NA> <NA> speaker_3 <NA> <NA>\n") == 2
    assert smoke.rttm_speakers(None) == 0


def test_setup_children_get_no_token_and_no_endpoint(monkeypatch, tmp_path):
    for k, v in (("HF_TOKEN", "x"), ("HF_ENDPOINT", "https://evil.invalid"), ("GITHUB_TOKEN", "y"),
                 ("AWS_SECRET_ACCESS_KEY", "z"), ("https_proxy", "http://127.0.0.1:9"), ("DEVELOPER_DIR", "/x")):
        monkeypatch.setenv(k, v)
    env = store.setup_env(tmp_path)
    assert not any(("TOKEN" in k and k != "HF_HUB_DISABLE_IMPLICIT_TOKEN") or k.endswith("_KEY") or k == "HF_ENDPOINT"
                   or "SECRET" in k for k in env)
    assert env["https_proxy"] == "http://127.0.0.1:9" and env["DEVELOPER_DIR"] == "/x"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null" and env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["TMPDIR"] == str(tmp_path)


def test_the_canary_child_is_the_cli_and_gets_the_same_folders(monkeypatch, tmp_path):
    cmd = store._canary_command(tmp_path / "c.wav", tmp_path / "out")
    assert cmd[1:5] == ["-I", "-B", "-m", "audiointake"] and cmd[5] == "transcribe"
    assert "--no-names" in cmd                                    # never matched against the user's voiceprints
    monkeypatch.setenv("PATH", "/somewhere/odd:/usr/bin:/bin")
    monkeypatch.setenv("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    ff = tmp_path / "ffbin" / "ffmpeg"
    env = store._canary_env(tmp_path, ff)
    assert env["HF_HUB_OFFLINE"] == "1" and env["AUDIOINTAKE_HOME"] == os.environ["AUDIOINTAKE_HOME"]
    assert "AUDIOINTAKE_RUN_LOCK_HOLDER" not in env
    from audiointake.setup import runtime

    assert env["PATH"] == f"{runtime.scripts_dir()}:{ff.parent}:/usr/bin:/bin"     # the run children's PATH
    assert "DEVELOPER_DIR" not in env and "/somewhere/odd" not in env["PATH"]


@pytest.mark.parametrize("kind", ["parakeet", "whisper", "typhoon", "campplus"])
def test_smoke_commands_run_the_programs_a_run_uses(kind, tmp_path):
    cmd, expect = smoke._command(kind, tmp_path / "m", tmp_path / "t.wav", tmp_path, tmp_path / "bin")
    assert expect == tmp_path / "smoke.json"
    if kind == "parakeet":
        assert cmd[0].endswith("parakeet-mlx") and "--model" in cmd
    elif kind == "campplus":
        assert cmd[1:5] == ["-I", "-B", "-m", "audiointake.setup.smoke"]
    else:
        assert cmd[0].endswith("mlx_whisper") and cmd[cmd.index("--language") + 1] == ("th" if kind == "typhoon"
                                                                                         else "hu")


def _verified_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOINTAKE_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(ffm.platforms, "FFMPEG_DIRS", (tmp_path / "nope",))
    binary = _exe(tmp_path / "pkg" / "binaries" / ffm.FALLBACK_NAME, "echo ffmpeg version 9.0")
    monkeypatch.setattr(ffm, "fallback_binary", lambda: binary)
    monkeypatch.setattr(ffm, "FALLBACK_SIZE", binary.stat().st_size)
    monkeypatch.setattr(ffm, "FALLBACK_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())
    return binary


def test_before_setup_the_bundled_ffmpeg_counts_without_anything_written(tmp_path, monkeypatch):
    """A fresh install on a Mac without Homebrew: doctor must not say ffmpeg_missing (setup links the bundled one)."""
    binary = _verified_fallback(tmp_path, monkeypatch)
    assert ffm.find() == (None, None)
    assert ffm.find_without_writing() == (binary, None)
    assert not ffm.fallback_link().exists() and not (tmp_path / "data").exists()
    (pre,) = [p for p in ffm.prerequisites(*ffm.find_without_writing()) if p["name"] == "ffmpeg"]
    assert pre["ok"] and pre["fallback"] and "code" not in pre


def test_a_dangling_link_is_made_again_by_a_run(tmp_path, monkeypatch):
    binary = _verified_fallback(tmp_path, monkeypatch)
    link = ffm.fallback_link()
    link.parent.mkdir(parents=True)
    os.symlink(str(tmp_path / "moved" / ffm.FALLBACK_NAME), link)     # the tool was reinstalled elsewhere
    assert ffm.find() == (None, None)
    ff, fp = ffm.find(ensure_fallback=True)
    assert ff == link and fp is None and os.readlink(link) == str(binary)


def test_no_usable_ffmpeg_at_all_is_still_none_without_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(ffm.platforms, "FFMPEG_DIRS", (tmp_path / "nope",))
    bad = _exe(tmp_path / "pkg" / "binaries" / ffm.FALLBACK_NAME, "echo not the pinned build")
    monkeypatch.setattr(ffm, "fallback_binary", lambda: bad)
    assert ffm.find_without_writing() == (None, None)
