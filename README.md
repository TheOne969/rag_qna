# 📚 RAG Knowledge Assistant
 
The app ingests documents, stores chunk embeddings in Weaviate, and answers natural-language questions with Retrieval-Augmented Generation (RAG).

---

##  Features
- **Drag-and-drop uploads** – add one or more PDFs in the Streamlit sidebar.  
- **Automatic chunking & embeddings** – 512-token chunks with overlap, embedded via Hugging Face API.  
- **Adaptive context** – full-text, sliding-window or lazy summary chosen on the fly.  
- **Lazy summary cache** – first retrieval triggers a one-off summary call; result is stored back in Weaviate.  
- **Cited answers** – filename + page number shown with every response.  
- **Local privacy** – raw vectors and PDFs never leave your machine; only summaries hit the HF endpoint.

---

## High-level architecture

User ──► Streamlit UI
│
│ upload
▼
Ingestion (chunk → embed) ─► Weaviate ◄─ Vector search ─ RAG Retriever
│
│ top-k context
▼
(optional) HF Summariser ─► Zephyr 7B LLM ─► Answer

---

## 🏁 Quick start

### 1. Prerequisites
| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | tested on 3.11 |
| Docker Desktop | latest | used to run Weaviate |
| (optional) WSL 2 | | recommended on Windows |

### 2. Clone & install

git clone https://github.com/TheOne969/rag_qna.git
cd rag_qna
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### 3. Start Weaviate

starts Weaviate + dependencies on ports 8080 / 50051

bash start_weaviate.sh # or: docker compose up -d weaviate

### 4. Set environment variables

Create a `.env` file in the repo root:
HUGGINGFACE_API_KEY=<your-hf-token>


### 5. Launch the frontend

streamlit run app.py


Open `http://localhost:8501` in your browser.

---


## 🛠️ Usage guide

| Action | Where | Result |
|--------|-------|--------|
| **Ingest PDF** | Sidebar → *Ingest* | File chunked → embedded → stored; duplicates skipped. |
| **Ask question** | Main panel | Retrieves top-k chunks, selects strategy, generates answer. |
| **Delete collection** | `python weaviate_delete_collection.py` | Drops *all* vectors for a fresh start. |

---

## 🧩 Repository layout

rag_qna/
├── app.py                      # Streamlit frontend
├── main.py                     # CLI pipeline test
├── chunking.py                 # Recursive splitter
├── pdf_extraction.py           # PDF loader
├── summarizer.py               # HF summariser + lazy cache
├── generator.py                # Zephyr-7B answer generator
├── rag.py                      # retrieval logic
├── weaviate_handler.py         # collection helpers
├── start_weaviate.sh           # convenience launcher
├── requirements.txt            
└── README.md
