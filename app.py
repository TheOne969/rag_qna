# app.py
import os, tempfile, shutil, warnings, atexit, uuid
import streamlit as st
from dotenv import load_dotenv
import weaviate.classes as wvc

from weaviate import connect_to_weaviate_cloud
from weaviate.classes.init import Auth, AdditionalConfig, Timeout

from pdf_extraction import extract_text_as_documents
from chunking          import chunk_texts
from weaviate_handler  import WeaviateHandler
from rag               import RAGRetriever
from hf_embedder       import HFEmbedderAPI
from generator         import generate_answer_hf_api
from strategy          import choose_strategy
from summarizer        import get_or_create_summary   
from main              import generate_metadata       

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")
load_dotenv()

# ------------------------------------------------------------------
# ----- CONFIG & SESSION STATE --------------------------------------
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
MIN_TOKENS        = 50
MODEL_CONTEXT     = 2048            
TMP_DIR = tempfile.mkdtemp(prefix="rag_upload_") 

# Generate a unique Weaviate Collection Name for this specific browser session
if "session_id" not in st.session_state:
    # Weaviate collections must start with a capital letter
    st.session_state.session_id = "Session_" + str(uuid.uuid4()).replace("-", "")

COLLECTION_NAME = st.session_state.session_id

# ------------------------------------------------------------------
# ----- UI: SIDEBAR (SECRETS & UPLOADS) -----------------------------
def get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass 
    return os.environ.get(key, "")

with st.sidebar:
    st.header("🔑 API Settings")
    user_hf_key = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        help="Required to generate embeddings and answers."
    )
    
    st.divider()
    st.subheader("Database Overrides")
    st.caption("Leave blank to use the default community database.")
    input_wcd_url = st.text_input("Weaviate REST Endpoint (Optional)", placeholder="https://your-cluster...")
    input_wcd_key = st.text_input("Weaviate API Key (Optional)", type="password", placeholder="Admin API Key")

    st.divider()
    
    # MOVED BACK TO SIDEBAR!
    st.header("📄 Upload PDFs")
    uploaded = st.file_uploader(
        label="Add one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True)

    if uploaded:
        if st.button("Ingest"):
            if not user_hf_key:
                st.error("Please enter a Hugging Face API key first.")
            else:
                for file in uploaded:
                    dest = os.path.join(TMP_DIR, file.name)
                    with open(dest, "wb") as f:
                        f.write(file.getbuffer())
                    # The function call happens below after clients are initialized
                    st.session_state.pending_uploads = True 

    st.divider()
    # THE CLEANUP BUTTON
    if st.button("🗑️ End Session & Clear Data", type="primary"):
        st.session_state.clear_requested = True

if not user_hf_key:
    st.info("👈 Please provide your Hugging Face API Key in the sidebar to start!")
    st.stop()

# ------------------------------------------------------------------
# ----- RESOLVE CREDENTIALS & CACHED CLIENTS ------------------------
user_provided_db = bool(input_wcd_url.strip() and input_wcd_key.strip())
final_wcd_url = input_wcd_url if input_wcd_url.strip() else get_secret("WEAVIATE_URL")
final_wcd_key = input_wcd_key if input_wcd_key.strip() else get_secret("WEAVIATE_API_KEY")

@st.cache_resource(show_spinner=False)
def get_client(url, api_key, is_custom_user_db):
    if not url or not api_key:
        st.error("Missing Weaviate credentials. Check sidebar overrides or secrets.")
        st.stop()
        
    try:
        client = connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120) 
            )
        )
        return client
    except Exception as e:
        if not is_custom_user_db:
            st.error("⚠️ **The default community database has expired or is unavailable.**")
            st.info("Please create a free Sandbox cluster at **console.weaviate.cloud** and enter your URL and API Key in the sidebar!")
        else:
            st.error("❌ **Failed to connect to your custom Weaviate database.**")
        st.stop()

@st.cache_resource(show_spinner=False)
def get_embedder(hf_key):
    return HFEmbedderAPI(api_key=hf_key)

client    = get_client(final_wcd_url, final_wcd_key, user_provided_db)
embedder  = get_embedder(user_hf_key)
handler   = WeaviateHandler(COLLECTION_NAME, client)

# ------------------------------------------------------------------
# ----- SESSION CLEANUP TRIGGER -------------------------------------
# If they clicked the red button, delete their unique collection and reset Streamlit
if st.session_state.get("clear_requested"):
    try:
        client.collections.delete(COLLECTION_NAME)
    except Exception:
        pass # Fails safely if the collection didn't exist yet
    st.session_state.clear()
    st.rerun()

# ------------------------------------------------------------------
# ----- INGESTION LOGIC ---------------------------------------------
def ingest_pdf_file(file_path: str):
    file_name = os.path.basename(file_path)
    if handler.document_already_exists(file_name):
        st.sidebar.info(f"{file_name} already indexed – skipping.")
        return

    st.sidebar.write(f"Extracting **{file_name}** …")
    docs          = extract_text_as_documents(file_path)
    cleaned_docs  = [doc.strip() for doc in docs]
    chunked_docs  = chunk_texts(cleaned_docs, CHUNK_SIZE, CHUNK_OVERLAP)

    filtered_docs = [d for d in chunked_docs if len(d.page_content.split()) >= MIN_TOKENS]
    chunks  = [doc.page_content for doc in filtered_docs]
    metas   = [generate_metadata(i, file_name, page=doc.metadata["page"]) for i, doc in enumerate(filtered_docs)]

    st.sidebar.write(f" {len(chunks)} chunks filtered; embedding …")
    vectors = embedder.encode(chunks)
    handler.insert_chunks(chunks, vectors, metas)
    st.sidebar.success(f"✅ Ingested {file_name}")

if st.session_state.get("pending_uploads"):
    with st.spinner("Processing documents..."):
        for file in uploaded:
            dest = os.path.join(TMP_DIR, file.name)
            ingest_pdf_file(dest)
    st.session_state.pending_uploads = False

# ------------------------------------------------------------------
# ----- RAG QUERY ---------------------------------------------------
def answer_query(question: str, k: int, hf_key: str):
    retriever  = RAGRetriever(COLLECTION_NAME, embedder, client)
    hits       = retriever.retrieve(question, k=k)
    
    if not hits:
        return "No documents found. Please ingest a PDF first through the sidebar!", [], "none"
        
    strategy   = choose_strategy(len(hits), CHUNK_SIZE, MODEL_CONTEXT)

    if strategy == "full_text":
        context = [h["text"] for h in hits]
    elif strategy == "sliding_window":
        context = [" ".join(h["text"].split()[:200]) for h in hits]
    else:  
        context = [get_or_create_summary(h, handler.collection, api_key=hf_key) for h in hits]

    cleaned_context = [text.replace("<br>", "\n") for text in context]
    sources = sorted({f"{h['file_name']} page {h['page']}" for h in hits})
    
    prompt_instruction = (
        f"{question}\n\nInstruction: Provide a concise answer based on the context. "
        "Monitor your length and ensure your response finishes cleanly. "
        "Do not stop mid-sentence or leave thoughts incomplete."
    )
    
    answer  = generate_answer_hf_api(prompt_instruction, cleaned_context, max_tokens=800, temperature=0.2, api_key=hf_key)
    return answer, sources, strategy

# ------------------------------------------------------------------
# ----- MAIN PANEL UI ----------------------------------------------
st.title("RAG Knowledge Assistant")
st.markdown(f"*Session Data ID: `{COLLECTION_NAME}`*")

st.subheader("Ask a question")
col_q, col_k = st.columns([3, 1])
question = col_q.text_input("Enter your question")
top_k     = col_k.slider("k", 1, 8, 3)

if st.button("Get answer") and question:
    with st.spinner("Retrieving …"):
        answer, sources, strat = answer_query(question, top_k, user_hf_key)

    st.markdown("#### Answer")
    st.write(answer)
    if sources:
        st.markdown("**Sources:** " + ", ".join(sources))
        st.caption(f"Strategy used: {strat}")

# ------------------------------------------------------------------
# ----- SERVER SHUTDOWN CLEAN-UP -----------------------------------
atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))