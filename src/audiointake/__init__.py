"""audiointake: transcribe recordings on a Mac with Apple silicon, with speakers named word by word."""
import os as _os
__version__ = "0.1.0"          # the only version source; release.sh greps this exact line
# onnxruntime (inside sherpa-onnx) and Hugging Face send telemetry by default; off before anything imports them.
_os.environ["ORT_DISABLE_TELEMETRY"] = "1"
_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
