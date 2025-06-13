import os
from dotenv import load_dotenv
from pdf_extraction import extract_text_as_documents
from chunking import chunk_texts
from weaviate_handler import WeaviateHandler
from rag import RAGRetriever
from hf_embedder import HFEmbedderAPI 
from weaviate import connect_to_local
import weaviate.classes as wvc 
from generator import generate_answer_hf_api
from strategy import choose_strategy
from summarizer import get_or_create_summary
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*swigvarlink.*")


# CONFIGS 
# These are just environmental variables, but put here. You could put them in .env file too. Though there is no sensitive data for these, we didn't put them.   
COLLECTION_NAME = "LectureSlides"  # input file
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_TOKENS = 50  # ⏳ Minimum tokens to keep a chunk
MODEL_CONTEXT = 2048 # Zephyr context


pdf_path= "test.pdf"

file_name = os.path.basename(pdf_path) #would work for both with pdf_path being either a full path or just something like "sample.pdf"


print(f"📄 Processing: {file_name}")

# 1. Extract full text from PDF
docs = extract_text_as_documents(pdf_path)

# Step 1.5: Clean line breaks and excess spacing in each page. This increases the chunk qualty considerably. 
cleaned_docs = [doc.replace("\n", " ").replace("  ", " ").strip() for doc in docs]

# Step 2: Chunk
chunked_docs = chunk_texts(cleaned_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Step 2.5: Extract chunks and filter by length
# keep only docs that pass the MIN_TOKENS filter
filtered_docs = [
    doc for doc in chunked_docs
    if len(doc.page_content.split()) >= MIN_TOKENS
    ]

# texts you will embed
chunks = [doc.page_content for doc in filtered_docs]

print(f"✂️ Split into {len(chunks)} chunks after filtering short ones.")

for i in range(5): 
    print(chunks[i][:400])