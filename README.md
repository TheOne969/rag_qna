# RAG Knowledge Assistant

The app ingests documents, stores chunk embeddings in Weaviate, and answers natural-language questions with Retrieval-Augmented Generation (RAG).

---

## Features
- **Drag-and-drop uploads** – add one or more PDFs in the Streamlit sidebar. 
- **Layout-Aware Parsing & Local OCR** – natively handles multi-column layouts, Markdown tables, and automatically runs local Hybrid OCR on image-only PDFs.
- **Automatic chunking & embeddings** – 512-token chunks with overlap, embedded via Hugging Face API. 
- **Adaptive context** – full-text, sliding-window, or lazy summary chosen on the fly based on retrieval size. 
- **Lazy summary cache** – first retrieval triggers a one-off summary call; the result is stored back in Weaviate to save API costs. 
- **Cited answers** – filename + page number shown with every response. 
- **Local privacy** – raw vectors, OCR processing, and PDFs never leave your machine; only the final context hits the HF endpoint.

---

## High-level architecture

```mermaid
graph TD
    User([User]) -->|1. Upload PDF| Streamlit[Streamlit UI]
    Streamlit -->|2. Page-by-Page Extraction| PyMuPDF[pymupdf4llm <br/> Layout Engine & Local OCR]
    PyMuPDF -->|3. Markdown Output| Chunking[Recursive Character Splitter]
    Chunking -->|4. Text & Metadata| Embedder[HF Embedder API]
    Embedder -->|5. Store Chunks & Vectors| Weaviate[(Weaviate Local DB)]

    User -->|6. Ask Question| Streamlit
    Streamlit -->|7. Vector Search| Retriever[RAG Retriever]
    Retriever -->|8. Fetch Top-K Hits| Weaviate
    Retriever -->|9. Inspect Context Size| Strategy{Strategy Picker}

    Strategy -->|Size < Limit| FullText[full_text: Raw Chunks]
    Strategy -->|Size > Limit| SlidingWindow[sliding_window: 200-Word Cutoff]
    Strategy -->|Critical Size| Summarizer[summarize: HF Lazy Cache]

    FullText --> Generator[Generator LLM]
    SlidingWindow --> Generator
    Summarizer -->|Save Summary Back| Weaviate
    Summarizer --> Generator
    
    Generator -->|10. Final Cited Answer| User

    style PyMuPDF fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    style Strategy fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff
    style Summarizer fill:#34A853,stroke:#333,stroke-width:2px,color:#fff
---

## Setting Up

### 1. Prerequisites
| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | tested on 3.12 |
| Docker Desktop | latest | used to run Weaviate |
| (optional) WSL 2 | | recommended on Windows |

### 2. Clone & install

```
git clone [https://github.com/TheOne969/rag_qna.git](https://github.com/TheOne969/rag_qna.git)
cd rag_qna
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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


## Usage guide

<mark>NEW!</mark> The extraction pipeline now features an AI layout engine and local OCR. It can successfully ingest complex multi-column documents, tables, and scanned/image-only PDFs.

Pdfs like the "test.pdf" in the repo would work. 

| Action | Where | Result |
|--------|-------|--------|
| **Ingest PDF** | Sidebar → *Ingest* | File chunked → embedded → stored; duplicates skipped. |
| **Ask question** | Main panel | Retrieves top-k chunks, selects strategy, generates answer. |
| **Delete collection** | `python weaviate_delete_collection.py` | Drops *all* vectors for a fresh start. |

---

## Repository layout
```

rag_qna/
├── app.py                      # Streamlit frontend
├── main.py                     # CLI pipeline test
├── chunking.py                 # Recursive splitter
├── pdf_extraction.py           # PDF loader
├── summarizer.py               # HF summariser + lazy cache
├── generator.py                # Using a model from Hugging face as generator
├── rag.py                      # retrieval logic
├── weaviate_handler.py         # collection helpers
├── start_weaviate.sh           # convenience launcher
├── requirements.txt            
└── README.md

```

## Future Upgrades 

- Advanced Chunking Strategies: Explore Semantic Chunking (splitting based on sentence meaning rather than character limits) to further improve retrieval context.

- UI Streaming: Implement token-by-token streaming in the Streamlit UI to reduce perceived latency during generation.