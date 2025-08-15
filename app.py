# app.py
import os, tempfile, shutil, warnings,atexit
import streamlit as st
from dotenv import load_dotenv
from weaviate import connect_to_local
import weaviate.classes as wvc

from pdf_extraction import extract_text_as_documents
from chunking          import chunk_texts
from weaviate_handler  import WeaviateHandler
from rag               import RAGRetriever
from hf_embedder       import HFEmbedderAPI
from generator         import generate_answer_hf_api
from strategy          import choose_strategy
from summarizer        import get_or_create_summary   
from main import generate_metadata       

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*swigvarlink.*")
load_dotenv()

# ------------------------------------------------------------------
# ----- CONFIG ------------------------------------------------------
COLLECTION_NAME   = "LectureSlides"
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
MIN_TOKENS        = 50
MODEL_CONTEXT     = 2048            
PORT_HTTP, PORT_GRPC = 8080, 50051
TMP_DIR = tempfile.mkdtemp(prefix="rag_upload_") #temporary director where pdfs would be stored. 

@st.cache_resource(show_spinner=False)
def get_client():
    # One Weaviate connection reused across reruns
    return connect_to_local(port=PORT_HTTP, grpc_port=PORT_GRPC)

@st.cache_resource(show_spinner=False)
def get_embedder():
    # Embedder is also the same across reruns. 
    return HFEmbedderAPI()

client    = get_client()
embedder  = get_embedder()
handler   = WeaviateHandler(COLLECTION_NAME, client)

# ------------------------------------------------------------------
# ----- INGESTION ---------------------------------------------------
def ingest_pdf_file(file_path: str):
    file_name = os.path.basename(file_path)
    if handler.document_already_exists(file_name):
        st.info(f"{file_name} already indexed – skipping.")
        return

    st.write(f"Extracting **{file_name}** …")
    docs        = extract_text_as_documents(file_path)
    cleaned_docs      = [doc.replace("\n", " ").replace("  ", " ").strip() for doc in docs]
    chunked_docs  = chunk_texts(cleaned_docs, CHUNK_SIZE, CHUNK_OVERLAP)

    # Filter & split meta / text
    filtered_docs = [d for d in chunked_docs
                     if len(d.page_content.split()) >= MIN_TOKENS]
    chunks = [doc.page_content for doc in filtered_docs]
    metas   = [ generate_metadata(i, file_name, page=doc.metadata["page"])
               for i, doc in enumerate(filtered_docs)]

    st.write(f" {len(chunks)} chunks after filtering; embedding …")
    vectors = embedder.encode(chunks)

    handler.insert_chunks(chunks, vectors, metas)
    st.success(f"✅ Ingested {file_name}")

# ------------------------------------------------------------------
# ----- RAG QUERY ---------------------------------------------------
def answer_query(question: str, k: int):
    retriever  = RAGRetriever(COLLECTION_NAME, embedder, client)
    hits       = retriever.retrieve(question, k=k)
    strategy   = choose_strategy(len(hits), CHUNK_SIZE, MODEL_CONTEXT)

    if strategy == "full_text":
        context = [h["text"] for h in hits]
    elif strategy == "sliding_window":
        context = [" ".join(h["text"].split()[:200]) for h in hits]
    else:                       # summarize
        context = [get_or_create_summary(h, handler.collection) for h in hits]

    sources = sorted({f"{h['file_name']} page {h['page']}" for h in hits})
    answer  = generate_answer_hf_api(question, context,max_tokens=400,
        temperature=0.2)

    return answer, sources, strategy

# ------------------------------------------------------------------
# ----- STREAMLIT LAYOUT -------------------------------------------
st.title("RAG Knowledge Assistant")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader(
        label="Add one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True)

    if uploaded:
        if st.button("Ingest"):
            for file in uploaded:
                dest = os.path.join(TMP_DIR, file.name)
                with open(dest, "wb") as f:
                    f.write(file.getbuffer())
                ingest_pdf_file(dest)
            st.success("All selected files processed.")

st.subheader("Ask a question")
col_q, col_k = st.columns([3, 1])
question = col_q.text_input("Enter your question")
top_k     = col_k.slider("k", 1, 8, 3)

if st.button("Get answer") and question:
    with st.spinner("Retrieving …"):
        answer, sources, strat = answer_query(question, top_k)

    st.markdown("#### Answer")
    st.write(answer)
    st.markdown("**Sources:** " + ", ".join(sources))
    st.caption(f"Strategy used: {strat}")

# ------------------------------------------------------------------
# ----- CLEAN-UP WHEN SERVER STOPS ---------------------------------
atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))   # cleans up on server exit
