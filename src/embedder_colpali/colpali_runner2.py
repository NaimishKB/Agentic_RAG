import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor
import torch.nn.functional as F


class ColPaliRunner:
    SAMPLE_IMAGES = [
        "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg",
    ]

    SAMPLE_QUERIES = [
        "When was the United States Declaration of Independence proclaimed?",
        "Who printed the edition of Romeo and Juliet?",
    ]

    def __init__(self, model_name: str, device: torch.device, debug: bool = False):
        self.device = device
        self.debug = debug
        self.model_name = model_name
        self.processor = None
        self.model = None

    def download_image(self, url: str):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

    def setup_sdpa_instrumentation(self):
        _orig = getattr(F, "scaled_dot_product_attention", None)
        if _orig is None:
            if self.debug:
                print("scaled_dot_product_attention not found; skipping instrumentation")
            return

        def _sdpa_safe_cast(*args, **kwargs):
            try:
                q = kwargs.get("query", args[0] if len(args) > 0 else None)
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
                            if self.debug:
                                print(f"[sdpa] cast attn_mask {md} -> {qd}")
                        except Exception:
                            pass
            except Exception:
                pass
            return _orig(*args, **kwargs)

        setattr(F, "scaled_dot_product_attention", _sdpa_safe_cast)
        if self.debug:
            print("SDPA safe-cast wrapper installed")

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        try:
            self.model = ColPaliForRetrieval.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        except Exception:
            self.model = ColPaliForRetrieval.from_pretrained(
                self.model_name, quantization_config=bnb_config, device_map=None
            )
        self.model = self.model.eval()

        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass

        if self.debug:
            for mod in self.model.modules():
                name = mod.__class__.__name__
                if name.startswith("Linear4bit") or name.startswith(
                    "LinearFP4"
                ) or "Params4bit" in name:
                    try:
                        mod.to(self.device)
                    except Exception:
                        pass

        self.processor = ColPaliProcessor.from_pretrained(self.model_name, use_fast=True)

    def prepare_inputs(self, images, texts):
        inputs_images = self.processor(images=images, return_tensors="pt")
        inputs_text = self.processor(text=texts, return_tensors="pt")

        for k, v in list(inputs_images.items()):
            try:
                inputs_images[k] = v.to(self.device)
            except Exception:
                inputs_images[k] = v
        for k, v in list(inputs_text.items()):
            try:
                inputs_text[k] = v.to(self.device)
            except Exception:
                inputs_text[k] = v

        return inputs_images, inputs_text

    def print_input_summary(self, input_dict: dict, name: str):
        print(f"--- {name} ---")
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: device={v.device} dtype={v.dtype} shape={tuple(v.shape)}")
            else:
                print(f"{k}: type={type(v)}")
        print("--- end")

    def run(self):
        self.setup_sdpa_instrumentation()
        self.load_model()
        images = [self.download_image(u) for u in self.SAMPLE_IMAGES]
        texts = self.SAMPLE_QUERIES
        inputs_images, inputs_text = self.prepare_inputs(images, texts)

        if self.debug:
            self.print_input_summary(inputs_images, "inputs_images")
            self.print_input_summary(inputs_text, "inputs_text")

        with torch.no_grad():
            out_im = self.model(**inputs_images).embeddings
            out_q = self.model(**inputs_text).embeddings
        scores = self.processor.score_retrieval(out_q, out_im)
        print(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug prints and SDPA instrumentation")
    parser.add_argument("--model", default="vidore/colpali-v1.3-hf", help="Model name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = ColPaliRunner(args.model, device, debug=args.debug)
    runner.run()


if __name__ == "__main__":
    main()
