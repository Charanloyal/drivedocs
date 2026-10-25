# DriveDocs: Multimodal Retrieval-Augmented Generation (RAG) Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU_1.8%2B-orange.svg)](https://github.com/facebookresearch/faiss)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 32 Passed](https://img.shields.io/badge/Tests-32%20Passed-brightgreen.svg)](tests/)

> **DriveDocs** is a high-performance multimodal Retrieval-Augmented Generation (RAG) assistant designed to ingest multi-page documents (PDF, Markdown, TXT) and technical diagrams (PNG, JPEG), index them in dual FAISS vector spaces using Hugging Face sentence transformers and CLIP vision models, and synthesize evidence-grounded answers with verifiable citations.

---

## 1. System Architecture

DriveDocs implements a dual-track vector retrieval architecture. Text and images are extracted from ingested files, projected into normalized embedding spaces, and indexed in separate FAISS stores (`IndexFlatIP`) for exact cosine similarity matching. During query time, text and visual evidence are retrieved, ranked, and passed to a verified answer generator.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Content Extraction & Ingestion"]
        Doc[Uploaded Documents<br/>PDF, TXT, MD, PNG, JPG] --> Parse[Document Parser & Security Validator]
        Parse -->|Page-Segmented Text| Chunker[Metadata-Preserving Chunker]
        Parse -->|Embedded / Standalone Images| ImgProc[Image Normalizer & Storage]
        Chunker --> Texts[Text Chunks<br/>doc_id, page_no, chunk_id]
        ImgProc --> Images[Visual Images<br/>doc_id, page_no, format]
    end

    subgraph Embedding ["2. Embedding in Vector Space"]
        Texts -->|all-MiniLM-L6-v2| TextEmb[Text Embeddings<br/>384-d L2-Normalized]
        Images -->|CLIP Vision Encoder| ImageEmb[Visual Embeddings<br/>512-d L2-Normalized]
        TextEmb --> TextStore[(FAISS Text Index<br/>IndexFlatIP)]
        ImageEmb --> ImageStore[(FAISS Image Index<br/>IndexFlatIP)]
    end

    subgraph Retrieval ["3. Multimodal Semantic Retrieval"]
        UserQ[User Question / Query] --> QEmb[Text Embedding Model]
        UserQ --> CLIPText[CLIP Text Encoder]
        QEmb -->|Cosine Similarity| TextStore
        CLIPText -->|Cross-Modal Matching| ImageStore
        TextStore --> RetText[Retrieved Text Evidence]
        ImageStore --> RetImg[Retrieved Visual Evidence]
    end

    subgraph Synthesis ["4. Grounded Answer Generation"]
        RetText --> Unifier[Multimodal Evidence Ranker]
        RetImg --> Unifier
        Unifier --> GroundCheck{Confidence >= Threshold?}
        GroundCheck -->|No| Fallback["I cannot find sufficient evidence<br/>in the provided documents."]
        GroundCheck -->|Yes| Generator[Grounded Synthesizer<br/>Extractive / Gemini / OpenAI]
        Generator --> Output[Grounded Answer<br/>+ Exact [Source: doc, Page X] Citations<br/>+ Visual Image References]
    end
```

---

## 2. Text vs Image Retrieval

| Feature | Text Retrieval Track | Multimodal Image Retrieval Track |
| :--- | :--- | :--- |
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` | `openai/clip-vit-base-patch32` |
| **Embedding Dimension** | 384 dimensions | 512 dimensions |
| **Distance Metric** | Cosine Similarity (FAISS `IndexFlatIP` with L2-normalized vectors) | Cosine Similarity (FAISS `IndexFlatIP` with L2-normalized vectors) |
| **Query Modalities** | Text query $\to$ Text chunks | 1. Text description $\to$ Visual images (CLIP text encoder)<br/>2. Image $\to$ Visual images (CLIP vision encoder) |
| **Metadata Preserved** | `doc_id`, `filename`, `page_number`, `chunk_index`, `char_count` | `image_id`, `doc_id`, `filename`, `page_number`, `dimensions`, `url_path` |
| **Index Isolation** | Stored in dedicated `TextFAISSIndex` | Stored in dedicated `ImageFAISSIndex` |

> [!NOTE]
> Text and image indexes are logically separate and strongly typed. Text embeddings (384-d) and CLIP vision/text embeddings (512-d) are never mixed or compared directly against each other, preventing dimensionality mismatch and cross-modal semantic drift.

---

## 3. Supported File Formats

- **PDF (`.pdf`)**: Page-by-page text extraction with source page preservation; automated extraction and deduplication of embedded images/diagrams with their source page number.
- **Markdown (`.md`)**: Full structural text parsing and chunking with heading preservation.
- **Plain Text (`.txt`)**: Raw text chunking with configurable overlap.
- **Standalone Images (`.png`, `.jpg`, `.jpeg`)**: Direct ingestion, dimensions extraction, and CLIP visual indexing.

---

## 4. Benchmark & Evaluation Results

A benchmark evaluation suite was executed against a public multi-page technical corpus (`data/sample/`), evaluating both factual in-corpus queries, visual diagram queries, and out-of-corpus negative queries.

Evaluation results are logged under [`reports/evaluation_report.json`](reports/evaluation_report.json) and [`reports/EVALUATION.md`](reports/EVALUATION.md):

| Metric | Measured Score | Benchmark Target | Evaluation Rubric |
| :--- | :---: | :---: | :--- |
| **Retrieval Recall@5** | **`100.0%`** | $\ge 80\%$ | Target ground-truth document retrieved in top 5 |
| **Mean Reciprocal Rank (MRR)** | **`1.000`** | $\ge 0.75$ | Target evidence ranked #1 across all test queries |
| **Retrieval Precision@5** | **`63.3%`** | $\ge 60\%$ | Signal-to-noise ratio in top 5 retrieved candidates |
| **Citation Correctness** | **`87.5%`** | $\ge 85\%$ | Exact match of cited filename and page number |
| **Groundedness Score** | **`91.7%`** | $\ge 85\%$ | Statement-level factual alignment with retrieved evidence |
| **Negative Query Handling** | **`100.0%`** | $100\%$ | Correctly triggers "insufficient evidence" for out-of-corpus queries |
| **Mean Query Latency** | **`35.0 ms`** | $< 500\text{ ms}$ | Average end-to-end CPU search and extraction time |

### Evaluation Methodology
1. **Recall@5 & MRR**: Computed across text and visual target evidence.
2. **Citation Verification**: Every citation `[Source: filename, Page X]` is validated against true source annotations.
3. **Negative Query Rejection**: Out-of-corpus questions (e.g. baking recipes, historical stock prices) must trigger the canonical refusal without fabricating citations or hallucinating facts.

---

## 5. Getting Started

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- CPU-only environment fully supported (CUDA automatically utilized if present)

### Installation (Windows PowerShell)

```powershell
# Clone or navigate to DriveDocs directory
cd DriveDocs

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Installation (Linux / macOS)

```bash
cd DriveDocs
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 6. Running the Application

### 1. Seed Sample Documents & Benchmark Data
```bash
python scripts/seed_sample_data.py
```

### 2. Run Evaluation Benchmark
```bash
python scripts/run_eval.py
```

### 3. Run Automated Tests
```bash
pytest
```
*Output: 32 passed in ~29s.*

### 4. Start FastAPI Server & Web Dashboard
```bash
uvicorn drivedocs.api:app --host 0.0.0.0 --port 8000 --reload
```
- **Web UI & Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Reference**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 7. API Reference & Examples

### `GET /health`
Returns system status, device info, and vector index chunk counts.
```bash
curl -X GET http://localhost:8000/health
```

### `POST /documents/upload`
Uploads a document, extracts text and embedded images, and updates FAISS indexes.
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/sample/quantum_photonics_spec.pdf"
```

### `GET /documents`
Lists all ingested documents and their chunk/image counts.
```bash
curl -X GET http://localhost:8000/documents
```

### `POST /search/text`
Performs semantic text search using sentence embeddings.
```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "silicon nitride waveguide transmission loss", "top_k": 2}'
```

### `POST /search/images`
Performs multimodal image search via text description (CLIP text encoder) or uploaded query image.
```bash
# Text-to-Image Search
curl -X POST http://localhost:8000/search/images \
  -F "query=quantum efficiency curve chart" \
  -F "top_k=1"

# Image-to-Image Search
curl -X POST http://localhost:8000/search/images \
  -F "image_file=@data/sample/architecture_diagram.png" \
  -F "top_k=2"
```

### `POST /chat`
Executes multimodal RAG retrieval and produces an evidence-grounded answer with citations.
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the peak quantum efficiency of the SNSPD detector?", "include_debug": true}'
```

---

## 8. Generation Providers & Configuration

DriveDocs is configured via `configs/config.yaml` and environment variables (`.env`):

| Provider | Setting | Requirements | Behavior |
| :--- | :--- | :--- | :--- |
| **`extractive`** *(Default)* | `LLM_PROVIDER=extractive` | Zero API keys required; 100% offline & deterministic | Extracts and synthesizes verbatim statements with page citations. |
| **`gemini`** | `LLM_PROVIDER=gemini` | `GEMINI_API_KEY` set in environment | Uses Google Gemini 1.5 Flash with strict grounding instructions. |
| **`openai`** | `LLM_PROVIDER=openai` | `OPENAI_API_KEY` set in environment | Uses GPT-4o-mini with verified attribution prompts. |

> [!TIP]
> If an external provider (`gemini` or `openai`) is configured but credentials are missing or the API call fails, DriveDocs automatically falls back to the deterministic local extractive synthesizer.

---

## 9. Security, Privacy & Limitations

1. **Untrusted Data Isolation**: Uploaded documents are treated as passive, untrusted input. Document text is enclosed within strict data delimiters (`--- UNTRUSTED RETRIEVED DOCUMENT EVIDENCE ---`) to defend against prompt injection.
2. **File Sanitization**: Upload filenames are scrubbed to eliminate directory traversal sequences (`../`, `..\`).
3. **Upload Limits**: File uploads are capped at 15 MB by default to prevent resource exhaustion.
4. **Duplicate Detection**: SHA-256 hashes are verified prior to ingestion to block redundant indexing.
5. **OCR Limitation**: Scanned image-only PDFs without an embedded text layer require optional OCR (`pytesseract` and system `tesseract-ocr` binary). Visual diagrams are still retrieved via CLIP even without OCR.
6. **CLIP Fine Details**: CLIP associates broad visual concepts with text representations. It is not an optical text reader or fine-grained OCR engine.

*Notice: DriveDocs is a research and engineering portfolio prototype, not a certified compliance repository.*

---

## 10. Docker Deployment

### Build the Image
```bash
docker build -t drivedocs:latest .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name drivedocs-app drivedocs:latest
```

Verify health check:
```bash
curl http://localhost:8000/health
```

---

## 11. Project Structure

```
DriveDocs/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── config.yaml               # Application, model, and storage parameters
├── data/
│   ├── sample/                   # Synthetic PDFs, diagrams, and benchmark set
│   │   ├── architecture_diagram.png
│   │   ├── quantum_photonics_spec.pdf
│   │   ├── agent_safety_protocol.md
│   │   ├── distributed_systems_notes.txt
│   │   └── evaluation_set.json
│   └── storage/                  # Ingested images and local FAISS indexes
├── reports/
│   ├── EVALUATION.md             # Rendered evaluation report
│   └── evaluation_report.json    # Machine-readable benchmark metrics
├── scripts/
│   ├── run_eval.py               # Benchmark execution script
│   └── seed_sample_data.py       # Sample data generation script
├── src/
│   └── drivedocs/
│       ├── __init__.py
│       ├── api.py                # FastAPI server & web interface
│       ├── chunking.py           # Text segmentation & page preservation
│       ├── config.py             # Settings manager
│       ├── evaluation.py         # Automated evaluation harness
│       ├── generation.py         # Grounded synthesis & citations
│       ├── image_index.py        # CLIP image FAISS vector index
│       ├── ingestion.py          # Document parser, PDF extractor & security
│       ├── retrieval.py          # Multimodal hybrid retriever
│       ├── schemas.py            # Pydantic models & API schemas
│       └── text_index.py         # SentenceTransformer FAISS vector index
└── tests/
    ├── test_api.py               # FastAPI integration & endpoint tests
    ├── test_chunking.py          # Text chunking unit tests
    ├── test_generation.py        # Grounded QA & citation tests
    ├── test_image_index.py       # CLIP embedding & search tests
    ├── test_ingestion.py         # Document parsing & security tests
    ├── test_retrieval.py         # Multimodal retrieval tests
    └── test_text_index.py        # FAISS text search tests
```

---

## 12. License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
