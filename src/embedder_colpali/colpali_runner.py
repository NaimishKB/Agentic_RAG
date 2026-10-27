"""Clean runner for loading the quantized ColPali model and running a small sample.

Usage:
    python src/embedder_colpali/colpali_runner.py [--debug]

- Keeps instrumentation off by default.
- Loads the model with the same BitsAndBytesConfig used for 4-bit NF4.
- Prints minimal output unless --debug is provided.
"""
from __future__ import annotations

import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor
import torch.nn.functional as F


SAMPLE_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg",
]

SAMPLE_QUERIES = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]


def download_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))


def setup_sdpa_instrumentation(debug: bool = False) -> None:
    """Install a lightweight SDPA wrapper that ensures mask/bias dtypes match query dtype.

    The wrapper is installed by default (to avoid model failures). When debug=True it
    will print cast events; otherwise it silently casts.
    """
    _orig = getattr(F, "scaled_dot_product_attention", None)
    if _orig is None:
        if debug:
            print("scaled_dot_product_attention not found; skipping instrumentation")
        return

    def _sdpa_safe_cast(*args, **kwargs):
        try:
            q = kwargs.get("query", args[0] if len(args) > 0 else None)
            # attn_mask may be positional arg index 3 in some callsites
            mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
            qd = getattr(q, "dtype", None)
            if qd is not None and mask is not None:
                md = getattr(mask, "dtype", None)
                if md is not None and md != qd:
                    try:
                        new_mask = mask.to(qd)
                        if "attn_mask" in kwargs:
                            kwargs["attn_mask"] = new_mask
                        else:
                            args = list(args)
                            args[3] = new_mask
                            args = tuple(args)
                        if debug:
                            print(f"[sdpa] cast attn_mask {md} -> {qd}")
                    except Exception:
                        # best-effort: if cast fails, continue and let original raise
                        pass
        except Exception:
            pass
        return _orig(*args, **kwargs)

    setattr(F, "scaled_dot_product_attention", _sdpa_safe_cast)
    if debug:
        print("SDPA safe-cast wrapper installed")


def load_model(model_name: str, device: torch.device, debug: bool = False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Try from_pretrained with device_map auto first (preferred)
    try:
        model = ColPaliForRetrieval.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    except Exception:
        model = ColPaliForRetrieval.from_pretrained(model_name, quantization_config=bnb_config, device_map=None)

    model = model.eval()
    # Move top-level model to device to let bnb init internally
    try:
        model = model.to(device)
    except Exception:
        pass

    if debug:
        # best-effort call .to(device) on 4-bit-like modules
        for mod in model.modules():
            name = mod.__class__.__name__
            if name.startswith("Linear4bit") or name.startswith("LinearFP4") or "Params4bit" in name:
                try:
                    mod.to(device)
                except Exception:
                    pass

    return model


def prepare_inputs(processor: ColPaliProcessor, images, texts, device: torch.device):
    inputs_images = processor(images=images, return_tensors="pt")
    inputs_text = processor(text=texts, return_tensors="pt")

    # Move tensors to device, resilient to non-tensor fields
    for k, v in list(inputs_images.items()):
        try:
            inputs_images[k] = v.to(device)
        except Exception:
            inputs_images[k] = v
    for k, v in list(inputs_text.items()):
        try:
            inputs_text[k] = v.to(device)
        except Exception:
            inputs_text[k] = v

    return inputs_images, inputs_text


def print_input_summary(input_dict: dict, name: str) -> None:
    print(f"--- {name} ---")
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: device={v.device} dtype={v.dtype} shape={tuple(v.shape)}")
        else:
            print(f"{k}: type={type(v)}")
    print("--- end")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug prints and SDPA instrumentation")
    parser.add_argument("--model", default="vidore/colpali-v1.3-hf", help="Model name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_sdpa_instrumentation(args.debug)

    model = load_model(args.model, device, debug=args.debug)
    processor = ColPaliProcessor.from_pretrained(args.model, use_fast=True)

    # download sample images
    images = [download_image(u) for u in SAMPLE_IMAGES]
    texts = SAMPLE_QUERIES

    inputs_images, inputs_text = prepare_inputs(processor, images, texts, device)
    if args.debug:
        print_input_summary(inputs_images, "inputs_images")
        print_input_summary(inputs_text, "inputs_text")

    with torch.no_grad():
        out_im = model(**inputs_images).embeddings
        out_q = model(**inputs_text).embeddings

    scores = processor.score_retrieval(out_q, out_im)
    print(scores)


if __name__ == "__main__":
    main()
