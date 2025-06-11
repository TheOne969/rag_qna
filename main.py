import os
from dotenv import load_dotenv
from pdf_extraction import extract_text_as_documents
from chunking import chunk_texts
from weaviate_handler import WeaviateHandler
from rag import RAGRetriever
from hf_embedder import HFEmbedderAPI 
from weaviate import connect_to_local
import weaviate.classes as wvc 

# Load environment variables
load_dotenv() # Used such that any file in the environment could access it, that is other files or functions that are being imported.

# CONFIGS 
# These are just environmental variables, but put here. You could put them in .env file too. Though there is no sensitive data for these, we didn't put them.   
COLLECTION_NAME = "LectureSlides"
PDF_PATH = "dagon.pdf"  # input file
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_TOKENS = 50  # ⏳ Minimum tokens to keep a chunk

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
    chunks = [
        doc.page_content for doc in chunked_docs
        if len(doc.page_content.split()) >= MIN_TOKENS
    ]

    print(f"✂️ Split into {len(chunks)} chunks after filtering short ones.")

    # 3. Generate embeddings
    embeddings = embedding_model.encode(chunks)  #  returns list[list[float]]

    # 4. Generate metadata
    metadatas = [generate_metadata(i, file_name) for i in range(len(chunks))]

    # 5. Store into Weaviate

    with connect_to_local(
            port=8080,
            grpc_port=50051,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=10))
        ) as client:                           # ← client auto-closes on exit.
        
        handler = WeaviateHandler(COLLECTION_NAME,client)
        handler.insert_chunks(chunks, embeddings, metadatas)



def run_rag_query(query,k):
    print(f"\n💬 Query: {query}")

    with connect_to_local(port=8080, grpc_port=50051) as client:
        retriever = RAGRetriever(COLLECTION_NAME, embedding_model, client)
        results = retriever.retrieve(query, k=k)

    print("\n🔍 Top Retrieved Chunks:")
    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] {chunk[:300]}...\n")


if __name__ == "__main__":
    ingest_pdf(PDF_PATH)

    # Test RAG query
    test_query = "Describe the monolith the narrator finds on the island."
    run_rag_query(test_query,3)
