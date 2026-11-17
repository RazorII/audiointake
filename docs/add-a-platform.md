# Add a platform

audiointake runs on one platform today: a Mac with Apple silicon (M1 or later) and macOS 14 or newer, with an Apple-silicon Python. Every other machine is refused by name before anything is read or installed (`src/audiointake/platforms.py`, `require_supported()`):

| Machine | Code | Why |
|---|---|---|
| Linux, Windows, anything not macOS | `platform_not_macos` | the speech models run on MLX and the diarizer on Core ML, both macOS-only |
| a Mac with an Intel processor | `platform_not_apple_silicon` | MLX runs only on Apple silicon |
| an Intel Python under Rosetta on an Apple-silicon Mac | `platform_rosetta` | a translated process can't use the GPU; reinstalling with an Apple-silicon Python fixes it |
| macOS 13 or older | `macos_too_old` | FluidAudio declares macOS 14, and MLX's oldest wheels are for macOS 14 |

The platform is read from the running process, not the machine underneath: `platform.machine()` says `arm64`, and `sysctl.proc_translated` (read through ctypes, no process) says it isn't running under Rosetta.

## What a new platform needs

Each piece of the pipeline has to run there, and has to be measured there:

1. **Speech recognition.** Parakeet Ultra and Parakeet TDT 0.6B v3 run through `parakeet-mlx`, and Whisper through `mlx-whisper`. Another platform needs another runtime for the same weights (NVIDIA's NeMo or an ONNX export, for example) with the same word timings and confidences, because the ensemble in `steps/ensemble.py` merges the two models word by word on exactly those.
2. **Diarization.** `fluidaudiocli` runs Nemotron 3 diarization on Core ML, and is built at setup from FluidAudio's Swift source. Another platform needs the same model through another runtime, writing the same RTTM.
3. **Speaker recognition** already runs anywhere sherpa-onnx does; it needs no change.
4. **Setup.** `setup/` downloads, checks and assembles per pack. A new runtime means new manifest entries, with every file pinned by revision and SHA-256, and a new smoke test.
5. **Paths.** `platforms.data_dir()`, `cache_dir()` and `logs_dir()` follow macOS's folders; another OS needs its own, and `tests/test_platforms.py` gets cases for them.

Outputs are compared by **metrics, never by bytes**, across machines: word error rates on FLEURS and AMI, and the share of words with the right speaker on AMI, within the margins `tests/models/` sets. Byte-for-byte output is only expected on one machine with one set of pinned versions.

## Turning it on

1. Add the platform's check to `platforms.require_supported()`, with its own refusal code for what it can't do, and the code and its message in `errors.py`.
2. Make the whole suite pass there, then the real-model tests (`scripts/models-local.sh`, or its equivalent for that platform), with nothing skipped.
3. Add a CI job for it in `.github/workflows/test.yml` and a line under *Install* in [README.md](../README.md) and in [claude-plugin/README.md](../claude-plugin/README.md). The skill's first step checks the platform before it installs anything; update it in the same pull request.
4. Open the pull request with the **code** template, with the suite's output and the measurements from the new platform.
