"""FastAPI application for DriveDocs multimodal ingestion, retrieval, and chat."""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from drivedocs.config import settings
from drivedocs.generation import GroundedMultimodalGenerator
from drivedocs.ingestion import (
    DocumentIngestionManager,
    DuplicateDocumentError,
    FileSizeLimitExceededError,
    MalformedDocumentError,
    UnsupportedFileTypeError,
)
from drivedocs.retrieval import MultimodalRetriever
from drivedocs.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    DocumentMetadata,
    HealthResponse,
    ImageSearchResult,
    SearchImageRequest,
    SearchTextRequest,
    TextSearchResult,
    UploadResponse,
)
from drivedocs.text_index import TextFAISSIndex
from drivedocs.image_index import ImageFAISSIndex


# Global singletons initialized once during lifespan
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring embedding models and vector indexes are loaded once."""
    print(f"[*] Initializing DriveDocs on device '{settings.device}'...")

    # 1. Initialize Vector Indexes & Storage
    text_index = TextFAISSIndex()
    image_index = ImageFAISSIndex()
    retriever = MultimodalRetriever(text_index=text_index, image_index=image_index)
    ingestion_mgr = DocumentIngestionManager()
    generator = GroundedMultimodalGenerator()

    # 2. Try loading any previously saved indexes
    loaded_ok = retriever.load(settings.indexes_dir)
    if loaded_ok and (text_index.count() > 0 or image_index.count() > 0):
        print(f"[+] Loaded existing vector indexes: {text_index.count()} text chunks, {image_index.count()} images.")
    else:
        # Auto-seed demo corpus if index is empty (e.g. fresh Render cloud deployment)
        sample_dir = settings.project_root / "data" / "sample"
        if sample_dir.exists():
            demo_files = [
                sample_dir / "quantum_photonics_spec.pdf",
                sample_dir / "agent_safety_protocol.md",
                sample_dir / "distributed_systems_notes.txt",
                sample_dir / "architecture_diagram.png",
            ]
            for f in demo_files:
                if f.exists():
                    try:
                        parsed = ingestion_mgr.ingest_file(f, allow_duplicates=True)
                        retriever.index_parsed_document(parsed)
                        print(f"[+] Auto-seeded demo corpus: {f.name} ({len(parsed.chunks)} chunks, {len(parsed.images)} images)")
                    except Exception as e:
                        print(f"[!] Note on seeding {f.name}: {e}")

    state["text_index"] = text_index
    state["image_index"] = image_index
    state["retriever"] = retriever
    state["ingestion_mgr"] = ingestion_mgr
    state["generator"] = generator

    yield

    # Clean shutdown / optional index persistence
    try:
        retriever.save(settings.indexes_dir)
        print("[*] DriveDocs state saved successfully.")
    except Exception as e:
        print(f"[!] Warning saving state: {e}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multimodal Retrieval-Augmented Generation (RAG) assistant for documents and images",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files for extracted / uploaded images
app.mount(
    "/static/images",
    StaticFiles(directory=str(settings.images_dir)),
    name="images",
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check and vector index statistics."""
    retriever: MultimodalRetriever = state.get("retriever")
    ingestion_mgr: DocumentIngestionManager = state.get("ingestion_mgr")

    doc_count = len(ingestion_mgr.list_documents()) if ingestion_mgr else 0
    text_count = retriever.text_index.count() if retriever else 0
    image_count = retriever.image_index.count() if retriever else 0

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        device=settings.device,
        indexed_documents=doc_count,
        indexed_text_chunks=text_count,
        indexed_images=image_count,
        models_loaded={
            "text_embedding": retriever.text_index.embedding_model._model is not None if retriever else False,
            "image_clip": retriever.image_index.embedding_model._model is not None if retriever else False,
        },
    )


@app.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    allow_duplicates: bool = Form(False),
) -> UploadResponse:
    """Upload and ingest a document (PDF, TXT, MD, PNG, JPG). Extracts text & images and indexes in FAISS."""
    ingestion_mgr: DocumentIngestionManager = state["ingestion_mgr"]
    retriever: MultimodalRetriever = state["retriever"]

    content = await file.read()

    try:
        parsed_doc = ingestion_mgr.ingest_bytes(
            filename=file.filename or "uploaded_document",
            content=content,
            allow_duplicates=allow_duplicates,
        )
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileSizeLimitExceededError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except DuplicateDocumentError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(e),
                "existing_document": e.existing_doc.model_dump(),
            },
        )
    except MalformedDocumentError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion error: {str(e)}")

    # Add to FAISS text and image indexes
    retriever.index_parsed_document(parsed_doc)

    return UploadResponse(
        message=f"Document '{parsed_doc.metadata.filename}' ingested successfully.",
        document=parsed_doc.metadata,
        chunks_indexed=len(parsed_doc.chunks),
        images_indexed=len(parsed_doc.images),
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents and their metadata."""
    ingestion_mgr: DocumentIngestionManager = state["ingestion_mgr"]
    retriever: MultimodalRetriever = state["retriever"]

    docs = ingestion_mgr.list_documents()
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=retriever.text_index.count(),
        total_images=retriever.image_index.count(),
    )


@app.post("/search/text", response_model=List[TextSearchResult])
async def search_text(req: SearchTextRequest) -> List[TextSearchResult]:
    """Search text chunks using FAISS cosine similarity."""
    retriever: MultimodalRetriever = state["retriever"]
    return retriever.text_index.search(query=req.query, top_k=req.top_k)


@app.post("/search/images", response_model=List[ImageSearchResult])
async def search_images(
    query: Optional[str] = Form(None),
    top_k: int = Form(2),
    image_file: Optional[UploadFile] = File(None),
) -> List[ImageSearchResult]:
    """Multimodal image search: query images via text description (CLIP text encoder) or uploaded query image."""
    retriever: MultimodalRetriever = state["retriever"]

    if image_file is not None:
        img_bytes = await image_file.read()
        from PIL import Image
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return retriever.image_index.search_by_image(query_image=pil_img, top_k=top_k)

    if query and query.strip():
        return retriever.image_index.search_by_text(text_query=query, top_k=top_k)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Must provide either a text 'query' or an 'image_file' for image retrieval.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_multimodal(req: ChatRequest) -> ChatResponse:
    """Multimodal RAG endpoint: retrieves text & visual evidence and returns a grounded answer with citations."""
    retriever: MultimodalRetriever = state["retriever"]
    generator: GroundedMultimodalGenerator = state["generator"]

    t0 = time.perf_counter()

    evidence = retriever.retrieve(
        query=req.query,
        top_k_text=req.top_k_text,
        top_k_images=req.top_k_images,
    )

    resp = generator.generate(
        query=req.query,
        evidence=evidence,
        include_debug=bool(req.include_debug),
    )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if req.include_debug and resp.debug_info is not None:
        resp.debug_info["total_latency_ms"] = latency_ms

    return resp


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> str:
    """Serve modern, interactive DriveDocs Web Interface."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DriveDocs | Multimodal RAG Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(99, 102, 241, 0.2);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --accent: #06b6d4;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: 'Plus Jakarta Sans', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
      color: var(--text-main);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(9, 13, 22, 0.8);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .logo-container {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .logo-badge {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      width: 36px;
      height: 36px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.2rem;
      color: #fff;
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    .title-box h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
    .title-box p { font-size: 0.8rem; color: var(--text-muted); }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-size: 0.78rem;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-weight: 600;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    main {
      flex: 1;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 2rem;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    .glass-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .card-title {
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #e2e8f0;
    }
    .dropzone {
      border: 2px dashed rgba(99, 102, 241, 0.4);
      border-radius: 12px;
      padding: 1.75rem 1rem;
      text-align: center;
      background: rgba(99, 102, 241, 0.04);
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .dropzone:hover {
      border-color: var(--primary);
      background: rgba(99, 102, 241, 0.08);
    }
    .file-input { display: none; }
    .btn {
      background: linear-gradient(135deg, var(--primary), var(--primary-hover));
      color: #fff;
      border: none;
      padding: 0.65rem 1.25rem;
      border-radius: 8px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      width: 100%;
      margin-top: 0.75rem;
    }
    .btn:hover {
      opacity: 0.95;
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .doc-list {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      max-height: 280px;
      overflow-y: auto;
      margin-top: 0.5rem;
    }
    .doc-item {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 0.7rem 0.85rem;
      font-size: 0.82rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .doc-name { font-weight: 600; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
    .doc-badge { font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 4px; background: rgba(99, 102, 241, 0.2); color: #818cf8; }
    .chat-section {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      height: calc(100vh - 140px);
    }
    .chat-history {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      padding-right: 0.5rem;
    }
    .message {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-width: 85%;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .message.user { align-self: flex-end; }
    .message.assistant { align-self: flex-start; max-width: 90%; }
    .bubble {
      padding: 1rem 1.25rem;
      border-radius: 14px;
      font-size: 0.92rem;
      line-height: 1.55;
    }
    .message.user .bubble {
      background: linear-gradient(135deg, var(--primary), #4338ca);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .message.assistant .bubble {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: #e2e8f0;
      border-bottom-left-radius: 4px;
    }
    .citations-container {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }
    .citation-chip {
      background: rgba(6, 182, 212, 0.12);
      border: 1px solid rgba(6, 182, 212, 0.3);
      color: #38bdf8;
      font-size: 0.75rem;
      padding: 0.3rem 0.65rem;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-family: var(--font-mono);
    }
    .images-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 0.75rem;
      margin-top: 0.75rem;
    }
    .image-card {
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      background: rgba(15, 23, 42, 0.8);
      cursor: pointer;
      transition: transform 0.2s ease;
    }
    .image-card:hover { transform: scale(1.03); }
    .image-card img { width: 100%; height: 120px; object-fit: cover; display: block; }
    .image-caption { font-size: 0.72rem; color: var(--text-muted); padding: 0.4rem 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .chat-input-bar {
      display: flex;
      gap: 0.75rem;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 0.6rem 0.8rem;
    }
    .chat-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: #fff;
      font-size: 0.95rem;
      font-family: inherit;
    }
    .chat-send-btn {
      width: auto;
      margin: 0;
      padding: 0.6rem 1.25rem;
    }
    .demo-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem;
      padding: 0.1rem 0.2rem;
    }
    .demo-label {
      font-size: 0.75rem;
      font-weight: 700;
      color: #94a3b8;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }
    .demo-pill {
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      font-size: 0.74rem;
      padding: 0.3rem 0.7rem;
      border-radius: 9999px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .demo-pill:hover {
      background: rgba(99, 102, 241, 0.28);
      border-color: #818cf8;
      color: #fff;
      transform: translateY(-1px);
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-container">
      <div class="logo-badge">D</div>
      <div class="title-box">
        <h1>DriveDocs</h1>
        <p>Multimodal RAG Assistant &bull; FAISS + CLIP + Transformers</p>
      </div>
    </div>
    <div class="status-badge" id="systemStatus">
      <span class="status-dot"></span>
      <span id="statusText">Connecting...</span>
    </div>
  </header>

  <main>
    <div class="sidebar">
      <div class="glass-card">
        <div class="card-title">
          <span>📄</span> Ingest Document
        </div>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileUpload').click()">
          <p style="font-size: 0.85rem; color: #94a3b8;">Drag & drop or click to upload</p>
          <p style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">PDF, TXT, MD, PNG, JPG (Max 15MB)</p>
          <input type="file" id="fileUpload" class="file-input" accept=".pdf,.txt,.md,.png,.jpg,.jpeg" onchange="handleFileSelect(this)">
        </div>
        <p id="selectedFileName" style="font-size: 0.78rem; color: #38bdf8; margin-top: 0.5rem;"></p>
        <button class="btn" id="uploadBtn" onclick="uploadSelectedFile()" disabled>Upload & Index</button>
      </div>

      <div class="glass-card">
        <div class="card-title">
          <span>📚</span> Indexed Corpus (<span id="docCount">0</span>)
        </div>
        <div class="doc-list" id="docList">
          <p style="font-size: 0.8rem; color: #64748b; text-align: center; padding: 1rem;">No documents ingested yet.</p>
        </div>
      </div>
    </div>

    <div class="chat-section">
      <div class="chat-history" id="chatHistory">
        <div class="message assistant">
          <div class="bubble">
            <strong>Welcome to DriveDocs!</strong><br>
            I am a Multimodal Retrieval-Augmented Generation assistant. You can upload multi-page documents (PDF, TXT, Markdown) or diagrams (PNG, JPEG), and ask questions. I retrieve relevant text and visual evidence and answer with verifiable citations.<br><br>
            <em>👉 Click any of the pre-loaded demo questions below to test instant multimodal retrieval and grounded answering!</em>
          </div>
        </div>
      </div>

      <div class="demo-bar">
        <span class="demo-label">⚡ Live Demo Questions:</span>
        <button class="demo-pill" onclick="askPreset('What is the optical transmission loss of the silicon nitride waveguide bus?')">🔬 Waveguide Loss (Page 1)</button>
        <button class="demo-pill" onclick="askPreset('What is the peak quantum efficiency of the SNSPD single-photon detector?')">📊 SNSPD Efficiency (Page 2 + Chart)</button>
        <button class="demo-pill" onclick="askPreset('Find the architecture diagram showing Content Parser, Vector Store, and Grounded QA.')">📐 System Architecture Diagram</button>
        <button class="demo-pill" onclick="askPreset('What is the periodic heartbeat interval dispatched by the Raft leader?')">⏱️ Raft Heartbeat</button>
        <button class="demo-pill" onclick="askPreset('What is the capital city of Mars in the year 3000?')">🛡️ Refusal / Negative Query</button>
      </div>

      <div class="chat-input-bar">
        <input type="text" id="chatInput" class="chat-input" placeholder="Ask a question about your documents or visual diagrams..." onkeydown="if(event.key==='Enter') sendChatMessage()">
        <button class="btn chat-send-btn" onclick="sendChatMessage()">Ask</button>
      </div>
    </div>
  </main>

  <script>
    let selectedFile = null;

    function askPreset(query) {
      document.getElementById('chatInput').value = query;
      sendChatMessage();
    }

    async function checkHealth() {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          document.getElementById('statusText').innerText = 
            `Device: ${data.device.toUpperCase()} | ${data.indexed_text_chunks} Chunks | ${data.indexed_images} Images`;
          document.getElementById('docCount').innerText = data.indexed_documents;
        }
      } catch (err) {
        document.getElementById('statusText').innerText = 'Offline';
      }
    }

    async function refreshDocList() {
      try {
        const res = await fetch('/documents');
        if (res.ok) {
          const data = await res.json();
          document.getElementById('docCount').innerText = data.total_documents;
          const container = document.getElementById('docList');
          if (data.documents.length === 0) {
            container.innerHTML = '<p style="font-size: 0.8rem; color: #64748b; text-align: center; padding: 1rem;">No documents ingested yet.</p>';
            return;
          }
          container.innerHTML = data.documents.map(d => `
            <div class="doc-item">
              <span class="doc-name" title="${d.filename}">${d.filename}</span>
              <span class="doc-badge">${d.chunk_count} ch / ${d.image_count} img</span>
            </div>
          `).join('');
        }
      } catch (err) {
        console.error(err);
      }
    }

    function handleFileSelect(input) {
      if (input.files && input.files[0]) {
        selectedFile = input.files[0];
        document.getElementById('selectedFileName').innerText = `Selected: ${selectedFile.name} (${(selectedFile.size/1024).toFixed(1)} KB)`;
        document.getElementById('uploadBtn').disabled = false;
      }
    }

    async function uploadSelectedFile() {
      if (!selectedFile) return;
      const btn = document.getElementById('uploadBtn');
      btn.disabled = true;
      btn.innerText = 'Indexing...';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const res = await fetch('/documents/upload', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        if (res.ok) {
          alert(`Successfully ingested '${data.document.filename}' (${data.chunks_indexed} chunks, ${data.images_indexed} images).`);
          selectedFile = null;
          document.getElementById('fileUpload').value = '';
          document.getElementById('selectedFileName').innerText = '';
          await refreshDocList();
          await checkHealth();
        } else {
          alert(`Error: ${data.detail || JSON.stringify(data)}`);
        }
      } catch (err) {
        alert(`Upload error: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.innerText = 'Upload & Index';
      }
    }

    async function sendChatMessage() {
      const input = document.getElementById('chatInput');
      const query = input.value.trim();
      if (!query) return;

      input.value = '';
      appendMessage('user', query);

      const history = document.getElementById('chatHistory');
      const loadingDiv = document.createElement('div');
      loadingDiv.className = 'message assistant';
      loadingDiv.innerHTML = '<div class="bubble"><em>Searching multimodal vector indexes & synthesizing answer...</em></div>';
      history.appendChild(loadingDiv);
      history.scrollTop = history.scrollHeight;

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query, include_debug: true })
        });
        const data = await res.json();
        loadingDiv.remove();

        if (res.ok) {
          appendAssistantResponse(data);
        } else {
          appendMessage('assistant', `Error: ${data.detail || 'Failed to generate answer'}`);
        }
      } catch (err) {
        loadingDiv.remove();
        appendMessage('assistant', `Error: ${err.message}`);
      }
    }

    function appendMessage(role, text) {
      const history = document.getElementById('chatHistory');
      const msg = document.createElement('div');
      msg.className = `message ${role}`;
      msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
      history.appendChild(msg);
      history.scrollTop = history.scrollHeight;
    }

    function appendAssistantResponse(data) {
      const history = document.getElementById('chatHistory');
      const msg = document.createElement('div');
      msg.className = 'message assistant';

      let html = `<div class="bubble">${data.answer.replace(/\\n/g, '<br>')}</div>`;

      // Citations
      if (data.citations && data.citations.length > 0) {
        html += '<div class="citations-container">';
        data.citations.forEach(c => {
          html += `<span class="citation-chip" title="${escapeHtml(c.snippet)}">📌 ${c.filename} (p.${c.page_number || 1})</span>`;
        });
        html += '</div>';
      }

      // Visual Evidence
      if (data.images && data.images.length > 0) {
        html += '<div class="images-grid">';
        data.images.forEach(img => {
          html += `
            <div class="image-card" onclick="window.open('${img.image_url}', '_blank')">
              <img src="${img.image_url}" alt="${img.filename}">
              <div class="image-caption">📷 ${img.filename} (Score: ${img.score})</div>
            </div>
          `;
        });
        html += '</div>';
      }

      msg.innerHTML = html;
      history.appendChild(msg);
      history.scrollTop = history.scrollHeight;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.innerText = text;
      return div.innerHTML;
    }

    // Initialize
    checkHealth();
    refreshDocList();
    setInterval(checkHealth, 15000);
  </script>
</body>
</html>
"""
