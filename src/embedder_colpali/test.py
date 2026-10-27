import os
import sys
from PIL import Image
import io
import fitz
import torch
from pathlib import Path
import torch.nn.functional as F
import base64
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_google_genai import ChatGoogleGenerativeAI
google_api_key=os.environ.get("GEMINI_API_KEY")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash" ,api_key=google_api_key)
import numpy as np
from langchain_groq import ChatGroq

api_key=os.environ.get("Groq_key")
model_name="qwen/qwen3-32b"
llm = ChatGroq(temperature=0,api_key=api_key ,model=model_name)

# Add project root to sys.path to import the runner
sys.path.append(r"D:\GenAi_Code\Agentic_RAG")
from src.embedder_colpali.colpali_runner2 import ColPaliRunner


def pdf_to_images_pymupdf(pdf_path, save_folder):
    doc = fitz.open(pdf_path)
    images = []
    os.makedirs(save_folder, exist_ok=True)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        save_path = os.path.join(save_folder, f"page_{page_num + 1}.png")
        img.save(save_path)
        images.append(img)
    return images

# def prepare_inputs(self, images=None, texts=None):
#     inputs_images = None
#     inputs_text = None

#     if images:
#         inputs_images = self.processor(images=images, return_tensors="pt")
#         for k, v in list(inputs_images.items()):
#             try:
#                 inputs_images[k] = v.to(self.device)
#             except Exception:
#                 inputs_images[k] = v

#     if texts:
#         inputs_text = self.processor(text=texts, return_tensors="pt")
#         for k, v in list(inputs_text.items()):
#             try:
#                 inputs_text[k] = v.to(self.device)
#             except Exception:
#                 inputs_text[k] = v

#     return inputs_images, inputs_text

# def embed_images_in_batches(runner, images, batch_size=8, device="cpu"):
#     all_embeddings = []
#     for i in range(0, len(images), batch_size):
#         batch_images = images[i : i + batch_size]
        
#         # Prepare image inputs only (no text query)
#         inputs_images, _ = runner.prepare_inputs(batch_images, None)
#         inputs_images = {k: v.to(device) for k, v in inputs_images.items()} if inputs_images else None
        
#         with torch.no_grad():
#             embedding_output = runner.model(**inputs_images)
        
#         # Extract tensor from output object (adjust attribute as per your model)
#         batch_embeddings = embedding_output.embeddings.cpu()
        
#         all_embeddings.append(batch_embeddings)

#     # Concatenate all batch embeddings into single tensor
#     all_embeddings = torch.cat(all_embeddings, dim=0)
#     return all_embeddings.numpy()  # Convert to numpy for FAISS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "vidore/colpali-v1.3-hf"
    pdf_path = r"D:\GenAi_Code\Agentic_RAG\data\raw\2.2_2.3_Hydraulic_and_structure_design.pdf"
    save_folder = r"D:\GenAi_Code\Agentic_RAG\data\processed"

    # Convert PDF pages to images
    pages_images = pdf_to_images_pymupdf(pdf_path, save_folder)

    # Instantiate and prepare runner
    runner = ColPaliRunner(model_name, device, debug=False)
    runner.setup_sdpa_instrumentation()
    runner.load_model()

    # Select pages 21-26 (0-indexed 20:26)
    selected_pages = pages_images[20:26]

    # Prepare inputs with images and query text
    query = "What is equation of designing the trench weir"
    inputs_images, inputs_text = runner.prepare_inputs(pages_images, [query])

    # inputs_images = {k: v.to(device) for k, v in inputs_images.items()} if inputs_images else None
    inputs_text = {k: v.to(device) for k, v in inputs_text.items()} if inputs_text else None
    import faiss
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(pages_images), batch_size):
        batch_images = pages_images[i : i + batch_size]
        
        # Prepare image inputs only (no text query)
        inputs_images, inputs_text = runner.prepare_inputs(batch_images, [query])
        inputs_images = {k: v.to(device) for k, v in inputs_images.items()} if inputs_images else None
        
        with torch.no_grad():
            embedding_output = runner.model(**inputs_images)
        
        # Extract tensor from output object (adjust attribute as per your model)
        batch_embeddings = embedding_output.embeddings.cpu()
        
        all_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    embeddings_3d=all_embeddings.numpy()
    embeddings_2d = embeddings_3d.reshape(-1, embeddings_3d.shape[-1]) 
    metadata = []
    for sample_idx in range(embeddings_3d.shape[0]):
        for chunk_idx in range(embeddings_3d.shape[1]):
            metadata.append({
                "sample": sample_idx,
                "chunk": chunk_idx,
                "image_name": pages_images[sample_idx],  # add image filename here
            })
    # Create FAISS index manually
    dimension = embeddings_2d.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_2d)
    # Wrap LangChain FAISS vectorstore around raw FAISS index

    # Build a simple docstore with your metadata or textual content (here metadata only)
    docstore = {}
    index_to_docstore_id = []

    for i, meta in enumerate(metadata):
        doc_id = str(i)  # or another unique key
        docstore[doc_id] = meta
        index_to_docstore_id.append(doc_id)



    vectorstore = FAISS(
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=None,  # No embedding function needed as embeddings are precomputed
    )

    vectorstore.save_local("faiss_index_folder")

    with torch.no_grad():
        # embeddings = runner.model(**inputs_images)
        query_embedding = runner.model(**inputs_text)
    # Load saved FAISS index from disk
    # embedding_function = None  
    vectorstore = FAISS.load_local("faiss_index_folder", embedding_function,allow_dangerous_deserialization=True)
    query_embedding = query_embedding.embeddings
    # query_embedding is torch tensor
    query_tensor = query_embedding.squeeze(0)  # remove first dimension if it's 1; shape (22,128) or (128,)

    # If query_tensor is 2D (22,128), average chunks:
    if query_tensor.ndim == 2:
        query_tensor = query_tensor.mean(dim=0, keepdim=True)  # shape (1, 128)

    # Convert to contiguous numpy float32
    query_vector = query_tensor.cpu().numpy().astype('float32')
    query_vector = np.ascontiguousarray(query_vector)
    print(query_vector.shape)  # must be (1, 128)
    print(query_vector.dtype)  # must be float32
    print(query_vector.flags['C_CONTIGUOUS'])

# D, I = vectorstore.index.search(query_vector, k)
# print("Distances:", D)
# print("Indices:", I)


    # Perform FAISS raw search
    distances, indices = vectorstore.index.search(query_vector, k=5)
    print("Distances:", distances)
    print("Indices:", indices)

    # Map indices to your docstore items
    docs = []
    for idx in indices[0]:
        doc = vectorstore.docstore.get(vectorstore.index_to_docstore_id[int(idx)])
        docs.append(doc)

    print("Top documents/metadata:", docs)

    top_image_names = [
    vectorstore.docstore[vectorstore.index_to_docstore_id[idx]].metadata["image_name"]
    for idx in indices[0]
    ]
    print("Top matched image names:", top_image_names)














    results = vectorstore.similarity_search_by_vector(query_vector, k=2)










    top_indices = [int(doc.metadata["sample"]) for doc in results]
    top_images = [pages_images[i] for i in top_indices]
    print("Top matched pages (samples):", top_indices)


    scores = runner.processor.score_retrieval(embeddings, query_embedding).squeeze()# shape = (6,)

    # scores = runner.processor.score_multi_vector(query_embedding, embeddings).squeeze(0)
    top_scores, top_indices = torch.topk(scores, k=1, largest=True)
    top_images = [selected_pages[int(i)] for i in top_indices]

    # === 7️⃣ Prepare prompt with Base64 Images for Gemini ===
    prompt = f"Answer the question strictly based on the following document images. **Return NA if nothing is found out from following content** \n\nQuery: {query}\n"
    
    for idx, img in enumerate(top_images):
        buffered = io.BytesIO()
        # img = img.resize((512, 512))
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        prompt += f"\n[Image {idx+1} Base64]: {img_str}\n"
    print(prompt)

    models={
        "a": "qwen/qwen3-32b",
        "b": "gemma2-9b-it",
        "c": "llama-3.3-70b-versatile",
        "d": "llama-3.1-8b-instant",
        "e": "groq/compound",
        "f": "openai/gpt-oss-120b",
        "g": "moonshotai/kimi-k2-instruct",
        "h":"openai/gpt-oss-120b",
        "i": "deepseek-r1-distill-llama-70b",
        "j":"meta-llama/llama-4-maverick-17b-128e-instruct",
        "k":"meta-llama/llama-guard-4-12b"#no tool calling support

        }
    # === 8️⃣ Send to Gemini via LangChain ===
    for key,value in models.items():
        try:
            llm = ChatGroq(temperature=0,api_key=api_key ,model=value)
            response = llm.invoke(prompt)
            print(f"\n\n💡Model {value}  Response:\n {response.content}")
        except Exception as e:
            print(f"Error with model {key} ({value}): {e}")



if __name__ == "__main__":
    main()
