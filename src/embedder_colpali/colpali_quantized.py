import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor
import torch
import torch.nn.functional as F
import traceback
import argparse
try:
    # bitsandbytes internal helper (not always present depending on bnb version)
    from bitsandbytes.nn import modules as _bnb_nn_modules
    _bnb_fix_4bit = getattr(_bnb_nn_modules, "fix_4bit_weight_quant_state_from_module", None)
except Exception:
    _bnb_fix_4bit = None
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug prints and runtime instrumentation")
args = parser.parse_args()

if torch.cuda.is_available() and args.debug:
    print("GPU detected:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated(0))
    print("Memory Cached:", torch.cuda.memory_reserved(0))
elif args.debug:
    print("No GPU detected, using CPU")

model_name = "vidore/colpali-v1.3-hf"

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

try:
    # Let the library place modules automatically (preferred when CUDA is available)
    model = ColPaliForRetrieval.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
except Exception:
    # Fallback: explicit load and move
    model = ColPaliForRetrieval.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=None,
    )
# Ensure we place the quantized model and its 4-bit submodules on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.debug:
    print("Using device:", device)

# Move model to device so bitsandbytes 4-bit modules can initialize their quant state
model = model.to(device)
model.eval()

# Some bitsandbytes 4-bit modules require an explicit .to(device) call to set their
# internal quantization state. Try to call .to(device) on modules that look like
# 4-bit/bnb layers to be safe (non-fatal if it fails).
for module in model.modules():
    cls_name = module.__class__.__name__
    if cls_name.startswith("Linear4bit") or cls_name.startswith("LinearFP4") or "Params4bit" in cls_name:
        try:
            module.to(device)
        except Exception:
            # best-effort only
            pass

# Diagnostic inspection: list candidate bitsandbytes 4-bit modules and their weight shapes/devices
if args.debug:
    print("Inspecting modules for bitsandbytes 4-bit layers...")
for name, mod in model.named_modules():
    cls = mod.__class__.__name__
    if hasattr(mod, "weight"):
        try:
            w = mod.weight
            shape = tuple(w.shape)
            dev = w.device
            dtype = w.dtype
            w_type = type(w).__name__
        except Exception as ex:
            shape = f"<error: {ex}>"
            dev = "<error>"
            dtype = "<error>"
            w_type = "<error>"
    else:
        shape = None
        dev = None
        dtype = None
# Only print module lines when debugging
    if ("Linear4bit" in cls) or ("LinearFP4" in cls) or ("Params4bit" in cls) or ("bnb" in cls.lower()):
        if args.debug:
            print(f"{name} | {cls} | weight.shape={shape} device={dev} dtype={dtype} weight_type={w_type}")
if args.debug:
    print("Finished inspection of 4-bit-like modules")
# If bitsandbytes exposes the fix helper, try calling it per-module to surface which one fails
if _bnb_fix_4bit is not None and args.debug:
    print("Attempting per-module bitsandbytes 4-bit fix to identify failing module...")
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if ("Linear4bit" in cls) or ("LinearFP4" in cls) or ("Params4bit" in cls) or ("bnb" in cls.lower()):
            try:
                # call the helper directly on the module
                _bnb_fix_4bit(mod)
            except Exception as ex:
                print(f"bitsandbytes init failed for module: {name} | class={cls} | error={ex}")
                # also print repr of the module.weight if present
                if hasattr(mod, "weight"):
                    try:
                        w = mod.weight
                        print(f"  -> weight type={type(w).__name__} shape={getattr(w,'shape',None)} device={getattr(w,'device',None)} dtype={getattr(w,'dtype',None)}")
                    except Exception as ex2:
                        print(f"  -> weight inspect error: {ex2}")
    print("Done per-module bnb fix attempts")
elif args.debug:
    print("bitsandbytes fix helper not found in this bnb installation; skipping per-module attempts")
# Instrument torch's scaled_dot_product_attention to log dtypes/shapes for debugging
try:
    _orig_sdpa = getattr(F, "scaled_dot_product_attention", None)
    def _sdpa_instrumented(*args, **kwargs):
        # Extract query,key,value,attn_mask from args/kwargs defensively
        try:
            query = kwargs.get('query', args[0] if len(args) > 0 else None)
            key = kwargs.get('key', args[1] if len(args) > 1 else None)
            value = kwargs.get('value', args[2] if len(args) > 2 else None)
            attn_mask = kwargs.get('attn_mask', args[3] if len(args) > 3 else None)
            qd = getattr(query, 'dtype', None)
            kd = getattr(key, 'dtype', None)
            vd = getattr(value, 'dtype', None)
            md = getattr(attn_mask, 'dtype', None) if attn_mask is not None else None
            print(f"[sdpa-instrument] query.dtype={qd} key.dtype={kd} value.dtype={vd} mask.dtype={md}")
        except Exception as _e:
            print(f"[sdpa-instrument] introspect failed: {_e}")
        if _orig_sdpa is None:
            raise RuntimeError("scaled_dot_product_attention not available in this torch version")
        # If attn_mask/bias dtype doesn't match query dtype, attempt to cast it to query dtype
        try:
            q_dtype = getattr(query, 'dtype', None)
            if q_dtype is not None and 'attn_mask' in kwargs and kwargs['attn_mask'] is not None:
                mask = kwargs['attn_mask']
                if getattr(mask, 'dtype', None) != q_dtype:
                    try:
                        # Dump a short stack to find where the mask was created
                        print("[sdpa-instrument] ATTENTION MASK DTYPE MISMATCH - dumping stack to locate origin:")
                        traceback.print_stack(limit=10)
                        kwargs['attn_mask'] = mask.to(q_dtype)
                        print(f"[sdpa-instrument] cast attn_mask from {getattr(mask,'dtype',None)} to {q_dtype}")
                    except Exception as _c:
                        print(f"[sdpa-instrument] failed to cast attn_mask: {_c}")
            # Also handle positional attn_mask if provided as 4th arg
            elif q_dtype is not None and len(args) > 3 and args[3] is not None:
                mask = args[3]
                if getattr(mask, 'dtype', None) != q_dtype:
                    try:
                        print("[sdpa-instrument] ATTENTION MASK DTYPE MISMATCH (positional) - dumping stack to locate origin:")
                        traceback.print_stack(limit=10)
                        new_mask = mask.to(q_dtype)
                        args = list(args)
                        args[3] = new_mask
                        args = tuple(args)
                        print(f"[sdpa-instrument] cast positional attn_mask from {getattr(mask,'dtype',None)} to {q_dtype}")
                    except Exception as _c:
                        print(f"[sdpa-instrument] failed to cast positional attn_mask: {_c}")
        except Exception:
            pass
        return _orig_sdpa(*args, **kwargs)
    if _orig_sdpa is not None and args.debug:
        setattr(F, 'scaled_dot_product_attention', _sdpa_instrumented)
        print("Installed sdpa instrumentation wrapper")
    elif args.debug:
        print("scaled_dot_product_attention not found; cannot instrument")
except Exception as _exc:
    print("Failed to install sdpa instrumentation:", _exc)
processor = ColPaliProcessor.from_pretrained(model_name,use_fast=True)

url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

def load_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # raise error if request failed
    return Image.open(BytesIO(response.content))

images = [
    load_image(url1),
    load_image(url2),
]


queries = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]




# Process the inputs
inputs_images = processor(images=images, return_tensors="pt")
inputs_text = processor(text=queries, return_tensors="pt")

# Move inputs to the same device as the model (be resilient to non-tensor fields)
for k, v in inputs_images.items():
    try:
        inputs_images[k] = v.to(device)
    except Exception:
        inputs_images[k] = v
for k, v in inputs_text.items():
    try:
        inputs_text[k] = v.to(device)
    except Exception:
        inputs_text[k] = v


# (removed commented cruft)


# Forward pass
with torch.no_grad():
    def _print_input_devices(input_dict, dict_name: str):
        try:
            print(f"--- devices for {dict_name} ---")
            for kk, vv in input_dict.items():
                try:
                    # Some processor outputs may be lists/ints; handle safely
                    if isinstance(vv, torch.Tensor):
                        print(f"{kk}: device={vv.device} dtype={vv.dtype} shape={tuple(vv.shape)}")
                    else:
                        # attempt to show type and device if it's a list/tuple of tensors
                        if isinstance(vv, (list, tuple)) and len(vv) > 0 and isinstance(vv[0], torch.Tensor):
                            print(f"{kk}: list(len={len(vv)}) example[0].device={vv[0].device} dtype={vv[0].dtype} shape={tuple(vv[0].shape)}")
                        else:
                            print(f"{kk}: type={type(vv)} (non-tensor)")
                except Exception as _e:
                    print(f"{kk}: <error inspecting: {_e}>")
            print(f"--- end {dict_name} ---")
        except Exception as _e:
            print(f"failed to print devices for {dict_name}: {_e}")

    # Print devices before forward to help debug any CPU/CUDA mismatches
    _print_input_devices(inputs_images, "inputs_images")
    _print_input_devices(inputs_text, "inputs_text")

    try:
        image_embeddings = model(**inputs_images).embeddings
        query_embeddings = model(**inputs_text).embeddings
    except AssertionError as e:
        # Provide a clearer hint for the common bitsandbytes FP4 initialization problem
        print("AssertionError during forward pass:", e)
        print("This usually means some bitsandbytes 4-bit modules weren't moved to CUDA before the forward.")
        print("Model device:", device)
        print("Attempted to call .to(device) on known 4-bit modules; if the error persists, try upgrading bitsandbytes/transformers or explicitly moving submodules to CUDA.")
        raise
    except RuntimeError as e:
        # Likely a device mismatch (e.g., index on CPU, weights on CUDA). Dump devices for inputs to locate offending tensor.
        print("RuntimeError during forward pass:", e)
        print("Dumping all input tensor devices/dtypes to help locate the CPU tensor causing the error:")
        _print_input_devices(inputs_images, "inputs_images (exception)")
        _print_input_devices(inputs_text, "inputs_text (exception)")
        # If the model has parameters, print a couple of representative parameter devices/dtypes
        try:
            print("--- sample model parameter devices ---")
            for i, (n, p) in enumerate(model.named_parameters()):
                if i >= 8:
                    break
                print(f"{n}: device={p.device} dtype={p.dtype} shape={tuple(p.shape)}")
            print("--- end sample model parameters ---")
        except Exception as _e:
            print("failed to inspect model parameters:", _e)
        # Re-raise so the stack trace is still visible to the user
        raise

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)

if args.debug:
    print("Retrieval scores (query x image):")
    print(scores)
else:
    # minimal output when not debugging
    try:
        print(scores.numpy())
    except Exception:
        print(scores)