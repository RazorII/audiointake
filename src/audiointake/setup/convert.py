"""Convert a Hugging Face (transformers) Whisper fine-tune to the MLX layout mlx_whisper loads.

Run by setup as a child, never imported by a run::

    python -I -m audiointake.setup.convert --source DIR --out DIR [--weights NAME]...

``--source`` holds the fine-tune's ``config.json`` and its ``*.safetensors`` shards (the Hugging Face cache's
snapshot folder, or the folder setup downloaded them into); ``--weights`` names the shards to read (default: every
``*.safetensors`` in the folder, sorted). ``--out`` receives ``weights.safetensors`` (fp16) and ``config.json``.
Two differences from the upstream converter, on purpose: every shard is read (upstream reads only
``model.safetensors``), and the run fails unless every weight of the MLX model is supplied (upstream loads with
``strict=False``, so a mis-mapped checkpoint would load silently incomplete). Exit 0 and one JSON line on stdout
(``{"weights": n, "extra": n}``) on success; exit 1 with the reason on stderr otherwise.

The key map below is adapted from ml-explore/mlx-examples, whisper/convert.py, commit e52c128d11:

    Copyright © 2023 Apple Inc.

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without
    limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    Software, and to permit persons to whom the Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions
    of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

``mlx`` and ``mlx_whisper`` are imported inside ``convert``, so the module imports on a machine without Metal.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

KEYS = [("model.", ""), (".layers", ".blocks"), (".self_attn", ".attn"), (".attn_layer_norm", ".attn_ln"),
        (".encoder_attn.", ".cross_attn."), (".encoder_attn_layer_norm", ".cross_attn_ln"),
        (".final_layer_norm", ".mlp_ln"), (".q_proj", ".query"), (".k_proj", ".key"), (".v_proj", ".value"),
        (".out_proj", ".out"), (".fc1", ".mlp1"), (".fc2", ".mlp2"), ("embed_positions.weight", "positional_embedding"),
        ("decoder.embed_tokens", "decoder.token_embedding"), ("encoder.layer_norm", "encoder.ln_post"),
        ("decoder.layer_norm", "decoder.ln")]

OUTPUT_FILES = ("weights.safetensors", "config.json")


class ConvertError(Exception):
    pass


def map_key(k: str) -> str:
    """A transformers weight name as the MLX Whisper model names it."""
    for a, b in KEYS:
        k = k.replace(a, b)
    return k


def dims_from_config(hf: dict) -> dict:
    """mlx_whisper's ModelDimensions from a transformers Whisper config.json."""
    return dict(n_mels=hf["num_mel_bins"], n_audio_ctx=hf["max_source_positions"], n_audio_state=hf["d_model"],
                n_audio_head=hf["encoder_attention_heads"], n_audio_layer=hf["encoder_layers"],
                n_vocab=hf["vocab_size"], n_text_ctx=hf["max_target_positions"], n_text_state=hf["d_model"],
                n_text_head=hf["decoder_attention_heads"], n_text_layer=hf["decoder_layers"])


def convert(source: Path, out: Path, weights: list[str] | None = None) -> dict:
    """Convert ``source`` into ``out`` (created). Returns ``{"weights": n, "extra": n}``; ConvertError if any
    weight of the MLX model is missing."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_whisper.whisper import ModelDimensions, Whisper

    src = Path(source)
    hf = json.loads((src / "config.json").read_text(encoding="utf-8"))
    dims = dims_from_config(hf)
    shards = [src / w for w in weights] if weights else sorted(src.glob("*.safetensors"))
    if not shards:
        raise ConvertError("no *.safetensors in the source folder")
    w = {}
    for f in sorted(shards):
        w.update(mx.load(str(f)))
    w.pop("proj_out.weight", None)  # tied to the token embedding
    converted = {}
    for k, v in w.items():
        k = map_key(k)
        if k == "encoder.positional_embedding":  # sinusoidal, rebuilt by the model
            continue
        if "conv" in k and v.ndim == 3:
            v = v.swapaxes(1, 2)
        converted[k] = v.astype(mx.float16)
    model = Whisper(ModelDimensions(**dims), mx.float16)
    want = {k for k, _ in tree_flatten(model.parameters())} - {"alignment_heads"}  # a buffer, not a weight
    missing, extra = want - set(converted), set(converted) - want
    if missing:
        raise ConvertError(f"{len(missing)} weights missing, e.g. {sorted(missing)[:5]}: not converted")
    model.load_weights([(k, converted[k]) for k in want], strict=False)  # coverage checked above
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(o / "weights.safetensors"), dict(tree_flatten(model.parameters())))
    cfg = asdict(model.dims)
    cfg["model_type"] = "whisper"
    (o / "config.json").write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return {"weights": len(want), "extra": len(extra)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m audiointake.setup.convert", allow_abbrev=False)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights", action="append", default=None)
    a = ap.parse_args(argv)
    try:
        result = convert(a.source, a.out, a.weights)
    except (ConvertError, OSError, KeyError, ValueError) as exc:
        print(f"convert: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
