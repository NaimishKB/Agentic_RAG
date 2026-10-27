from typing import cast

import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor

model_name = "vidore/colpali-v1.3"

# from pathlib import Path
# path=Path(__file__).parent
# offload_dir = path/"colpali_offload"
# offload_dir.mkdir(exist_ok=True)

# Choose device dynamically
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device='cpu'
model = ColPali.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    torch_dtype=torch.float32,
    device_map=None       # automatically distributes layers   # required when some layers are offloaded
).eval()

model.to(device)



processor = ColPaliProcessor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)














