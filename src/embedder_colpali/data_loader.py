from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
import fitz  # PyMuPDF
from PIL import Image
import io,os
import torch
from colpali_engine.models.paligemma.colpali import ColPaliProcessor, ColPali
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
google_api_key=os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash" ,api_key=google_api_key)
from tqdm import tqdm
import base64
model_name = "vidore/colpali-v1.3"
device = "cpu"  # or "cuda:0" if GPU available and configured
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model with 4-bit quantization using bitsandbytes
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
import google.generativeai as genai
# genai.configure(api_key=google_api_key)
# llm = genai.GenerativeModel("gemini-2.5-flash")
processor = ColPaliProcessor.from_pretrained(model_name,use_fast=True)
model = ColPali.from_pretrained(model_name, device_map=None).to(device).eval()



def pdf_to_images_pymupdf(pdf_path,save_folder):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render page to a pixmap (image)
        pix = page.get_pixmap()
        # Convert pixmap to PIL Image
        img = Image.open(io.BytesIO(pix.tobytes()))
                # Save image in the specified folder
        save_path = os.path.join(save_folder, f"page_{page_num + 1}.png")
        img.save(save_path)
        images.append(img)
    return images

# # Usage
pdf_path = r"D:\GenAi_Code\Agentic_RAG\data\raw\2.2_2.3_Hydraulic_and_structure_design.pdf"
save_folder = r"D:\GenAi_Code\Agentic_RAG\data\processed"
pages_as_images = pdf_to_images_pymupdf(pdf_path, save_folder)
pages_images=pages_as_images[20:25]
# # Now pages_as_images is a list of PIL Image objects ready for ColPali processing
pages_as_images[21]
batch_images = processor.process_images(pages_images).to(device)
# # Step 4: Embed pages to get image embeddings
embeddings_list = []
with torch.no_grad():
    page_embeddings = model(**batch_images)


# # Step 5: Query example
query = "What is equation of designing the trench weir"
batch_query = processor.process_queries([query]).to(device)

with torch.no_grad():
    query_embedding = model(**batch_query)

# === 6️⃣ Compute Similarity Scores ===
scores = processor.score_multi_vector(query_embedding, page_embeddings).squeeze(0)
topk_scores, topk_indices = torch.topk(scores, k=2, largest=True)
top_images = [pages_images[int(i)] for i in topk_indices]
print("✅ Retrieved top relevant pages")
# === 6️⃣ Send Top Images to Gemini (no base64 needed) ===


# === 7️⃣ Prepare prompt with Base64 Images for Gemini ===
prompt = f"Answer the question strictly based on the following document images. **Return NA if nothing is found out from following content** \n\nQuery: {query}\n"

for idx, img in enumerate(top_images):
    buffered = io.BytesIO()
    img = img.resize((512, 512))
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    prompt += f"\n[Image {idx+1} Base64]: {img_str}\n"


# === 8️⃣ Send to Gemini via LangChain ===
response = llm.invoke(prompt)
print("\n💡 Gemini Response:\n", response.content)






























# Function to get top-k relevant pages from ColPali

# def get_topk_pages(query, pages_images, top_k=3, batch_size=8):

embeddings_list = []
num_batches = (len(pages_images) + batch_size - 1) // batch_size

for i in tqdm(range(num_batches), desc="Embedding PDF pages"):
    batch_chunk = pages_images[i * batch_size : (i + 1) * batch_size]
    batch_images = processor.process_images(batch_chunk).to(device)
    with torch.no_grad():
        embeddings = model(**batch_images)
    embeddings_list.append(embeddings.cpu())

page_embeddings = torch.cat(embeddings_list, dim=0).to(device)

batch_query = processor.process_queries([query]).to(device)
with torch.no_grad():
    query_embedding = model(**batch_query)

scores = processor.score_multi_vector(query_embedding, page_embeddings)  # (1, num_pages)
scores = scores.squeeze(0)

topk_scores, topk_indices = torch.topk(scores, k=top_k, largest=True)
topk_pages = [(pages_images[int(i)], topk_scores[idx].item()) for idx, i in enumerate(topk_indices)]
    return topk_pages





# Function to send images + query to Gemini-2.5-flash LLM and get answer
def ask_gemini_with_images(query, images, google_api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key)

    # Convert images to base64 strings to send as part of prompt (or upload and provide URLs)
    import base64
    buffered_images = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        buffered_images.append(img_str)

    # Prepare a prompt with query + embedded images
    prompt = f"Answer the question based on the following images:\nQuery: {query}\n"
    for idx, img_str in enumerate(buffered_images):
        prompt += f"\n[Image {idx+1} base64]: {img_str}\n"

    # Send prompt to Gemini
    response = llm.chat(prompt)
    return response

# Example usage:

pdf_path = r"D:\GenAi_Code\Agentic_RAG\data\raw\2.2_2.3_Hydraulic_and_structure_design.pdf"
save_folder = r"D:\GenAi_Code\Agentic_RAG\data\processed"
pages_images = pdf_to_images_pymupdf(pdf_path, save_folder)

query = "Explain the design of gabion weir"
top_pages = get_topk_pages(query, pages_images, top_k=2)
top_images = [img for img, score in top_pages]

answer = ask_gemini_with_images(query, top_images, google_api_key)

print("Gemini Model Answer:")
print(answer)