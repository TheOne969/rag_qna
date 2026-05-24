import os
import getpass
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

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# CONFIGS 
COLLECTION_NAME = "LectureSlides"
PDF_PATH = "test.pdf"  # input file
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_TOKENS = 50  
MODEL_CONTEXT = 2048 

def generate_metadata(chunk_index, file_name, page=None, section="N/A"):
    return {
        "chunk_index": chunk_index,
        "file_name": file_name,
        "page": page or 0,
        "section": section,
    }

# Pass embedding_model explicitly
def ingest_pdf(pdf_path, embedding_model):
    file_name = os.path.basename(pdf_path) 
    with connect_to_local(port=8080, grpc_port=50051) as client:
        handler = WeaviateHandler(COLLECTION_NAME, client)

        if handler.document_already_exists(file_name):
            print(f"⚠️  {file_name} is already in Weaviate — skipping ingestion.")
            return
            
    print(f"📄 Processing: {file_name}")
    docs = extract_text_as_documents(pdf_path)
    cleaned_docs = [doc.replace("\n", " ").replace("  ", " ").strip() for doc in docs]
    chunked_docs = chunk_texts(cleaned_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    filtered_docs = [doc for doc in chunked_docs if len(doc.page_content.split()) >= MIN_TOKENS]
    chunks = [doc.page_content for doc in filtered_docs]

    print(f"✂️ Split into {len(chunks)} chunks after filtering short ones.")

    # Use the model passed into the function
    embeddings = embedding_model.encode(chunks)  

    metadatas = [generate_metadata(i, file_name, page=doc.metadata["page"]) for i, doc in enumerate(filtered_docs)]

    with connect_to_local(
            port=8080,
            grpc_port=50051,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=10))
        ) as client:                           
        handler = WeaviateHandler(COLLECTION_NAME,client)
        handler.insert_chunks(chunks, embeddings, metadatas)

# Pass both embedding_model and hf_key explicitly
def run_rag_query_and_generate(query, k, embedding_model, hf_key):
    print(f"\n💬 Query: {query}")

    with connect_to_local(port=8080, grpc_port=50051) as client:
        retriever = RAGRetriever(COLLECTION_NAME, embedding_model, client)
        hits = retriever.retrieve(query, k=k)
        collection = retriever.weaviate_handler.collection

    strategy = choose_strategy(len(hits), CHUNK_SIZE, MODEL_CONTEXT)
    print("🔧 Chosen strategy →", strategy)

    if strategy == "full_text":
        context  = [h["text"] for h in hits]
    
    elif strategy == "sliding_window":
        window_tokens = 200           
        context = [h["text"].split()[:window_tokens] for h in hits]
        context = [" ".join(t) for t in context]

    elif strategy == "summarize":
        # Assumes summarizer needs the key
        context = [get_or_create_summary(h, collection, api_key=hf_key) for h in hits]

    sources = sorted({f"{h['file_name']} page {h['page']}" for h in hits}) 

    # Assumes generator needs the key
    answer = generate_answer_hf_api(query, context, api_key=hf_key)

    print("\n Answer:\n", answer)
    print("\n Sources:", ", ".join(sources))


if __name__ == "__main__":
    # Securely ask for the key in the terminal (input is hidden like a password)
    print("\n--- Local Testing Mode ---")
    hf_key = getpass.getpass("Enter your Hugging Face API Key: ")
    
    embedding_model = HFEmbedderAPI(api_key=hf_key)
    
    ingest_pdf(PDF_PATH, embedding_model)
    test_query = "Who is Mr. Higgins?"
    run_rag_query_and_generate(test_query, 3, embedding_model, hf_key)