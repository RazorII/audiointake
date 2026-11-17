# Third-party software and data

audiointake's own code is MIT licensed ("Copyright (c) 2026 RazorII"). Nothing below is copied into this repository except the code listed under *Code ported from elsewhere*: the libraries are installed from PyPI when you install audiointake, the models are downloaded from their publishers at setup, and the speaker program is built from its source on your Mac. Licence ids are SPDX identifiers, read from each package's metadata and licence files (2026-11-17). Not legal advice.

## Runtime dependencies

| Package | Licence | What audiointake uses it for |
|---|---|---|
| `parakeet-mlx` | Apache-2.0 | runs the two Parakeet speech models on the Mac's GPU |
| `mlx-whisper` | MIT | runs Whisper large-v3 and the Thai model, and Whisper's language identification |
| `mlx` | MIT | Apple's machine-learning framework the models run on |
| `mlx-metal` | MIT | MLX's GPU library for macOS |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | arrays for audio, timings and voice embeddings |
| `scipy` | BSD-3-Clause | signal filtering in mlx-whisper's word timing; its wheels bundle further libraries under their own licences, listed in the wheel |
| `librosa` | ISC | the mel filters parakeet-mlx computes its audio features with |
| `numba` | BSD-2-Clause | compiles mlx-whisper's word-timing code |
| `soxr` | LGPL-2.1-or-later | resampling inside librosa; bundles libsoxr (LGPL-2.1-or-later) and PFFFT. Installed by uv from PyPI on your Mac, unmodified; never redistributed by this project |
| `tiktoken` | MIT | Whisper's tokenizer |
| `huggingface-hub` | Apache-2.0 | imported by parakeet-mlx and mlx-whisper, which use it to fetch a model by name; audiointake always hands them a model folder already on the Mac, with `HF_HUB_OFFLINE=1`, and setup downloads with its own fetcher |
| `sherpa-onnx` | Apache-2.0 | runs the CAM++ speaker-recognition model |
| `sherpa-onnx-core` | Apache-2.0 | sherpa-onnx's native library; bundles onnxruntime (MIT), whose telemetry audiointake switches off |
| `soundfile` | BSD-3-Clause | reads WAV files for speaker recognition; bundles libsndfile (LGPL-2.1-or-later), installed from PyPI, never redistributed by this project |
| `imageio-ffmpeg` | BSD-2-Clause; the ffmpeg 9.0 binary it carries is built with `--enable-gpl` (GPL-2.0-or-later) | the fallback ffmpeg, used only on a Mac with no ffmpeg of its own: run as a separate program, checked against its pinned fingerprint, installed by uv from PyPI on your Mac and never redistributed by this project |

## Installed with them

These come in with the packages above; uv picks their versions when it installs.

| Package | Licence | Pulled in by |
|---|---|---|
| `annotated-doc` | MIT | typer |
| `annotated-types` | MIT | pydantic |
| `anyio` | MIT | httpx2 |
| `certifi` | MPL-2.0 | requests |
| `cffi` | MIT-0 | soundfile |
| `charset-normalizer` | MIT | requests |
| `click` | BSD-3-Clause | huggingface-hub |
| `cloudpickle` | BSD-3-Clause | joblib |
| `dacite` | MIT | parakeet-mlx |
| `decorator` | BSD-2-Clause | librosa |
| `filelock` | MIT | huggingface-hub, torch |
| `fsspec` | BSD-3-Clause | huggingface-hub, torch |
| `h11` | MIT | httpcore2 |
| `hf-xet` | Apache-2.0 | huggingface-hub (installed, never used: setup doesn't use Xet) |
| `httpcore2` | BSD-3-Clause | httpx2 |
| `httpx2` | BSD-3-Clause | huggingface-hub |
| `idna` | BSD-3-Clause | anyio, httpx2, requests |
| `jinja2` | BSD-3-Clause | torch |
| `joblib` | BSD-3-Clause | librosa, scikit-learn |
| `lazy-loader` | BSD-3-Clause | librosa |
| `llvmlite` | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | numba |
| `markdown-it-py` | MIT | rich |
| `markupsafe` | BSD-3-Clause | jinja2 |
| `mdurl` | MIT | markdown-it-py |
| `more-itertools` | MIT | mlx-whisper |
| `mpmath` | BSD-3-Clause | sympy |
| `msgpack` | Apache-2.0 | librosa |
| `narwhals` | MIT | scikit-learn |
| `networkx` | BSD-3-Clause | torch |
| `packaging` | Apache-2.0 OR BSD-2-Clause | huggingface-hub, lazy-loader, pooch |
| `platformdirs` | MIT | pooch |
| `pooch` | BSD-3-Clause | librosa |
| `pycparser` | BSD-3-Clause | cffi |
| `pydantic` | MIT | typer |
| `pydantic-core` | MIT | pydantic |
| `pygments` | BSD-2-Clause | rich |
| `pyyaml` | MIT | huggingface-hub |
| `regex` | Apache-2.0 AND CNRI-Python | tiktoken |
| `requests` | Apache-2.0 | pooch, tiktoken |
| `rich` | MIT | typer |
| `scikit-learn` | BSD-3-Clause | librosa |
| `setuptools` | MIT | torch |
| `shellingham` | ISC | typer |
| `sympy` | BSD-3-Clause | torch |
| `threadpoolctl` | BSD-3-Clause | scikit-learn |
| `torch` | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | mlx-whisper (installed because mlx-whisper declares it; audiointake never imports it) |
| `tqdm` | MPL-2.0 AND MIT | huggingface-hub, mlx-whisper |
| `truststore` | MIT | httpcore2, httpx2 |
| `typer` | MIT | parakeet-mlx |
| `typing-extensions` | PSF-2.0 | anyio, huggingface-hub, pydantic, soundfile, torch |
| `typing-inspection` | MIT | pydantic |
| `urllib3` | MIT | requests |

## Models

Downloaded at setup from Hugging Face or GitHub, each file at a pinned revision and checked against its pinned SHA-256. None is redistributed by this project.

| Model | Pack | Licence | Credit and source |
|---|---|---|---|
| Parakeet Ultra, MLX conversion | core | CC-BY-4.0 | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), a post-trained version of NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), in the MLX conversion [selcukkubur/parakeet-ultra-mlx](https://huggingface.co/selcukkubur/parakeet-ultra-mlx). Credit belongs to moondream for parakeet-ultra and to NVIDIA for parakeet-tdt-0.6b-v3 underneath it (the conversion's model card). |
| Parakeet TDT 0.6B v3, MLX conversion | core | CC-BY-4.0 | NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), in the MLX conversion [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3). |
| Nemotron 3 diarization, Core ML conversion | core | OpenMDW-1.1 | NVIDIA's [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), released under the OpenMDW License v1.1, converted to Core ML by Fluid Inference ([FluidInference/nemotron-3-diarization-coreml](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml)); the converted weights carry the same licence (its model card). |
| CAM++ speaker recognition (`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`) | core | Apache-2.0 | From [3D-Speaker](https://github.com/modelscope/3D-Speaker) (Apache-2.0), model `iic/speech_campplus_sv_zh_en_16k-common_advanced` on ModelScope, whose card says Apache License 2.0; the ONNX file comes from sherpa-onnx's [speaker-recongition-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) release, whose notes say each model has its own licence, that of its repository. Checked at all three sources on 2026-11-17. |
| Whisper large-v3, MLX conversion | `hu`, `th` | MIT | OpenAI's Whisper large-v3, in the MLX conversion [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx). |
| Typhoon Whisper large-v3 | `th` | MIT | [typhoon-ai/typhoon-whisper-large-v3](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3), a Thai fine-tune of Whisper large-v3 by Typhoon, MIT "inherited from OpenAI Whisper" (its model card). The same card says that by using the model you agree to the [OpenTyphoon Terms and Conditions](https://opentyphoon.ai/tac) and acknowledge its [Privacy Notice](https://opentyphoon.ai/privacy). audiointake downloads it from Hugging Face only when you install the `th` pack, and converts it to MLX on your Mac with the key map below. |

## Built at setup on your Mac

The speaker program, `fluidaudiocli`, is built by `audiointake setup` from [FluidAudio](https://github.com/FluidInference/FluidAudio) at tag v0.18.0, commit `e15287985625dbf544a343df679436789ffb03fb`, with the Swift toolchain of Apple's Command Line Tools or Xcode. It is built on your Mac and never redistributed by this project, and no prebuilt copy is ever shipped. Its licences are in FluidAudio's `LICENSE` and `ThirdPartyLicenses/` folder at that commit; `audiointake about` points to them.

| Part | Licence | Note |
|---|---|---|
| FluidAudio | Apache-2.0 | FluidAudio v0.18.0 has no top-level NOTICE file |
| NemoTextProcessing (FluidInference/text-processing-rs v0.3.1, prebuilt xcframework, linked by default) | Apache-2.0, with permissive (Apache-2.0 or MIT) dependencies: NVIDIA NeMo Text Processing's grammars (Apache-2.0) and rustfst (MIT OR Apache-2.0) among them | Pinned by FluidAudio's `Package.swift` checksum. FluidAudio's `ThirdPartyLicenses/NemoTextProcessing-LICENSE.md` names v0.3.0, while that commit's `Package.swift` downloads v0.3.1. |
| fastcluster | BSD-2-Clause | part of FluidAudio's diarization code |
| VBx | Apache-2.0 | part of FluidAudio's diarization code |
| LuxTTS English pronunciation data, in FluidAudio's default build | derived from espeak-ng (GPL-3.0-or-later) and FrequencyWords (CC-BY-SA-4.0) | Text-to-speech data that diarization never uses. It is built on your Mac and is not copied into audiointake's tool folder, which holds only `fluidaudiocli` and the resource bundles diarization needs. FluidAudio's text-to-speech data for other languages is likewise left out. |

## Code ported from elsewhere

`src/audiointake/setup/convert.py` converts the Thai model to MLX with the key map from [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) `whisper/convert.py`, commit `e52c128d113f10546f0fa391f87edcc58d3880cb`, under this licence:

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Test and bench data

Fetched when a test, the bench or setup needs it, checked against its pinned SHA-256, and never committed to this repository.

| Data | Licence | Used for |
|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), audio and annotations | CC-BY-4.0 ([licence](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)) | setup's 30-second test recording (part of meeting ES2004a), the tests, and the speaker benchmarks |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | CC-BY-4.0 | the word-error-rate benchmarks and the language tests |
| [BUT AMI diarization setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) reference RTTMs | Apache-2.0 (the repository's LICENSE, re-read on 2026-11-17); the annotations they're made from are AMI's (CC-BY-4.0) | the speaker benchmarks' reference of who spoke when |

Credit: the AMI Meeting Corpus, collected by the AMI project consortium, and FLEURS, by Google.

## Never used

- No audio, transcript or voiceprint is committed to this repository. The tests and the bench fetch public AMI and FLEURS clips when they run, and keep anything made from them, voiceprints of AMI's speakers included, in a work folder outside the repository.
- No prebuilt speaker program is ever shipped, so FluidAudio's text-to-speech data, derived from GPL-3.0-or-later and CC-BY-SA-4.0 sources, is never distributed by this project.
- No model is downloaded through `huggingface_hub`'s own downloader or Xet, and no Hugging Face token is ever read or sent.
- No PyPI data wheels: the models are too large for them, and come from their publishers instead.
