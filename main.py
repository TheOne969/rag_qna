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


# Load environment variables
load_dotenv() # Used such that any file in the environment could access it, that is other files or functions that are being imported.

# CONFIGS 
# These are just environmental variables, but put here. You could put them in .env file too. Though there is no sensitive data for these, we didn't put them.   
COLLECTION_NAME = "LectureSlides"
PDF_PATH = "test.pdf"  # input file
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_TOKENS = 50  # ⏳ Minimum tokens to keep a chunk
MODEL_CONTEXT = 2048 # Zephyr context

# Load embedding model
embedding_model = HFEmbedderAPI() 

def generate_metadata(chunk_index, file_name, page=None, section="N/A"):
    return {
        "chunk_index": chunk_index,
        "file_name": file_name,
        "page": page or 0,
        "section": section,
    }

def ingest_pdf(pdf_path):
    file_name = os.path.basename(pdf_path) #would work for both with pdf_path being either a full path or just something like "sample.pdf"
    with connect_to_local(port=8080, grpc_port=50051) as client:
        handler = WeaviateHandler(COLLECTION_NAME, client)

        # ---- duplication check ----
        if handler.document_already_exists(file_name):
            print(f"⚠️  {file_name} is already in Weaviate — skipping ingestion.")
            return
        # ---------------------------
    
    
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

    # 3. Generate embeddings
    embeddings = embedding_model.encode(chunks)  #  returns list[list[float]]

    # 4. Generate metadata
    metadatas = [
        generate_metadata(i, file_name, page=doc.metadata["page"])
        for i, doc in enumerate(filtered_docs)
        ]
    

    # 5. Store into Weaviate

    with connect_to_local(
            port=8080,
            grpc_port=50051,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=10))
        ) as client:                           # ← client auto-closes on exit.
        
        handler = WeaviateHandler(COLLECTION_NAME,client)
        handler.insert_chunks(chunks, embeddings, metadatas)


def run_rag_query_and_generate(query,k):
    print(f"\n💬 Query: {query}")

    with connect_to_local(port=8080, grpc_port=50051) as client:
        retriever = RAGRetriever(COLLECTION_NAME, embedding_model, client)
        hits = retriever.retrieve(query, k=k)# [{'text', 'summary', ...}]

        collection = retriever.weaviate_handler.collection

    # Decide which tactic to use
    strategy = choose_strategy(len(hits), CHUNK_SIZE, MODEL_CONTEXT)
    print("🔧 Chosen strategy →", strategy)

    if strategy == "full_text":
        context  = [h["text"] for h in hits]

    
    elif strategy == "sliding_window":
        window_tokens = 200           # first 200-token slice of each hit
        context = [h["text"].split()[:window_tokens] for h in hits]
        context = [" ".join(t) for t in context]

    elif strategy == "summarize":
        context = [get_or_create_summary(h, collection) for h in hits]

    sources = sorted({f"{h['file_name']} page {h['page']}" for h in hits}) #removes duplicates

    answer = generate_answer_hf_api(query, context)

    print("\n🤖 Answer:\n", answer)
    print("\n📚 Sources:", ", ".join(sources))
    

if __name__ == "__main__":

    ingest_pdf(PDF_PATH)
    test_query = "Who is Mr. Higgins?"
    run_rag_query_and_generate(test_query,3)
    