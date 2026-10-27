# DriveDocs Implementation Audit

**Audit date:** 2026-10-26
**Audited revision:** `fc15031` (`main`)
**Method:** I read every source file, script, test, config and deployment file. I ran the existing test suite, the
evaluation script and targeted reproduction scripts locally (Windows, Python 3.14.7, CPU, torch 2.14.1+cpu,
sentence-transformers 6.2.0, transformers 5.20.2, faiss-cpu 1.15.1). Each finding below is labelled with how it was
established:

- **[verified]**: reproduced or measured by running code.
- **[code]**: established by reading the code; not exercised at runtime.

Nothing is marked as working just because the README says so.

> This document records the state of the repository **before** the upgrade. Section 10 lists the status of each item
> after the upgrade.

---

## 1. Current architecture (as implemented)

```
FastAPI app (src/drivedocs/api.py, single module, ~825 lines incl. inline HTML dashboard)
 ├─ lifespan(): builds TextFAISSIndex, ImageFAISSIndex, MultimodalRetriever,
 │              DocumentIngestionManager, GroundedMultimodalGenerator into a global dict
 │              → loads indexes from data/storage/indexes, else auto-seeds data/sample/*
 │              → saves indexes on shutdown only
 ├─ GET  /health              index counts + "models loaded" flags
 ├─ POST /documents/upload    ingest → index (in-memory)
 ├─ GET  /documents           list in-memory registry
 ├─ POST /search/text         MiniLM + FAISS IndexFlatIP
 ├─ POST /search/images       CLIP text→image or image→image
 ├─ POST /chat                retrieve → extractive / Gemini / OpenAI answer
 ├─ GET  /                    inline HTML/JS dashboard
 └─ /static/images            StaticFiles mount of the whole images directory

ingestion.py   validation (size, extension), SHA-256 dedupe, pypdf page text, embedded images (PIL → PNG)
chunking.py    LangChain RecursiveCharacterTextSplitter if importable, else a native fallback
text_index.py  SentenceTransformer(all-MiniLM-L6-v2, 384-d) + faiss.IndexFlatIP, JSON metadata
image_index.py CLIP ViT-B/32 (512-d) image + text towers + faiss.IndexFlatIP, JSON metadata
retrieval.py   runs both indexes; sufficiency = max(text score, image score) ≥ 0.25
generation.py  extractive sentence picker with [Source: file, Page N] tags; optional Gemini/OpenAI over httpx
evaluation.py  Recall@5, MRR, Precision@5, "citation correctness", "groundedness", negative handling, latency
```

Text and image embeddings are stored in two separate FAISS indexes with different dimensions (384 and 512). They are
never compared vector-to-vector. **[code]**

## 2. Verified working features

| Feature | Evidence |
| --- | --- |
| Existing test suite | `pytest` → **32 passed in 12.5 s** **[verified]** (needs locally cached models; see §4.11) |
| TXT / MD / PDF / PNG / JPG ingestion | Sample corpus ingested: PDF 4 chunks + 1 embedded image, MD 3 chunks, TXT 2 chunks, PNG 1 image **[verified]** |
| PDF page numbers on chunks | Page-2 facts cited as Page 2 in evaluation output **[verified]** |
| Embedded PDF image extraction | SNSPD chart extracted from page 2 of the sample PDF **[verified]** |
| MiniLM text retrieval | Correct document ranked first for all 6 in-corpus queries **[verified]** |
| CLIP text→image and image→image | Existing CLIP tests pass with real weights **[verified]** |
| Extractive answering without API keys | Answers contain the expected facts for q1–q4 **[verified]** |
| Negative-query refusal on the sample set | Both negatives refused, but with a narrow margin (top scores 0.217 and 0.210 vs. a 0.25 threshold) **[verified]** |
| Evaluation script | Reproduces the committed numbers (Recall@5 100%, MRR 1.0, P@5 63.3%, citation 87.5%, "groundedness" 91.7%); latency 22.4 ms on this machine **[verified]** |

## 3. Existing bugs

| # | Severity | Bug | Evidence |
| --- | --- | --- | --- |
| B1 | High | **The document registry is not persisted.** After a restart, `/documents` returns 0 documents while the FAISS indexes still hold the chunks. | Restart probe: `health docs=0 chunks=11` **[verified]** |
| B2 | High | **Duplicate detection resets on restart.** Re-uploading the same file after a restart returns 201 and adds duplicate index entries (11 → 12 → 13 chunks). `allow_duplicates=true` on the API also inserts duplicates. | Restart probe **[verified]** |
| B3 | High | **Indexes are saved only on clean shutdown.** A crash, OOM kill or SIGKILL loses every upload since boot. | `lifespan` **[code]** |
| B4 | High | **The test suite writes into the real `data/storage/indexes`.** `TestClient` shutdown persists test documents into the developer's index. | Files appeared in `data/storage/` after `pytest` **[verified]** |
| B5 | High | **LLM fallback paths crash.** `_generate_gemini` and `_generate_openai` call `_generate_extractive(query, evidence, citations, images)` with 4 arguments; the method accepts 3. Any API error raises `TypeError` → HTTP 500. | **[code]** |
| B6 | High | **OpenAI auth header is the literal string `"******"`**, so every OpenAI call fails with 401 (and then hits B5). | **[code]** |
| B7 | Medium | Gemini API key is sent as a URL query parameter, where proxies and HTTP logs can record it. | **[code]** |
| B8 | Medium | Text from Gemini or OpenAI is returned unchecked; its citations are never verified. | **[code]** |
| B9 | Medium | `DRIVEDOCS_STORAGE_DIR` does not move `images_dir` or `indexes_dir`; they always resolve to `data/storage/...`. | Eval with the env var set still wrote images to `data/storage/images` **[verified]** |
| B10 | Medium | Images are attached to every answer when any image exists, even if CLIP similarity is low. A text-only question ("zebra telemetry") got an unrelated "Visual Reference" note. | Restart probe **[verified]** |
| B11 | Medium | q5 (architecture diagram) is answered with unrelated photonics text, yet evaluation scores it 100% "groundedness" because the metric only checks for the words "architecture" and "visual". | Eval output **[verified]** |
| B12 | Medium | Evidence sufficiency compares `max(MiniLM score, CLIP score)` against a single threshold. The two score distributions are not comparable: CLIP text↔image cosine is typically 0.2–0.35 even for weak matches. | **[code]** + eval scores |
| B13 | Medium | Evaluation ranks text and image hits together by raw score, mixing incompatible similarity spaces. "Citation correctness" checks only the filename, not the page, despite claiming page alignment. | **[code]** |
| B14 | Low | The native chunker fallback ignores `chunk_overlap`, except when hard-splitting a single over-long token run. Behavior depends on which LangChain version is installed. | **[code]** |
| B15 | Low | `sanitize_filename` uses `os.path.basename`, which does not split on `\` on Linux. `..\..\x.exe` becomes a random name on Linux but `x.exe` on Windows, so the existing test passes only on Windows. | **[code]** |
| B16 | Low | The `.dockerignore` rule `*.md` excludes `data/sample/agent_safety_protocol.md`, so the Docker image seeds only 3 of 4 sample files. | **[code]** |
| B17 | Low | The Dockerfile hardcodes port 8000 and ignores `$PORT` (Render's default is 10000). | **[code]** |
| B18 | Low | `render.yaml` regenerates sample data at build time (`seed_sample_data.py`). This overwrites the committed `evaluation_set.json` and requires `reportlab` in production. | **[code]** |
| B19 | Medium | **Windows checkouts corrupt the sample PDF.** With `core.autocrlf=true` (Git for Windows default) and no `.gitattributes`, git converts the PDF's LF line endings to CRLF. The file grows from 11,020 to 11,130 bytes, its xref offsets break, and pypdf logs "incorrect startxref pointer". | `git ls-files --eol` showed `w/crlf`; byte sizes compared **[verified]** (found during implementation) |
| B20 | Medium | `httpx` (used at runtime by the Gemini/OpenAI providers) was only listed as a test dependency in `pyproject.toml`. | **[code]** (found during implementation) |

## 4. Missing functionality

1. No document details endpoint, no delete endpoint, no per-document processing status.
2. No readiness endpoint. `/health` mixes liveness with index statistics.
3. No citation verification layer: `[Source: …]` tags are never checked against the registry or page counts.
4. Hybrid ranking is not implemented. `text_weight` and `image_weight` are read from config but never used.
5. Markdown heading preservation is claimed in the README but not implemented.
6. Embedded image de-duplication is claimed in the README but not implemented.
7. No content (magic-byte) validation. A `.txt` of arbitrary binary is accepted through the `latin-1` fallback.
8. A PDF with no extractable text and no images is "ingested" with 0 chunks and no warning.
9. Failed ingestions are not recorded anywhere. The error is returned once and then lost.
10. The UI has no library management, passage inspection, visual search, evaluation view or history.
11. The test suite downloads ~700 MB of models on first run and has no mock embedding backend.
12. There is no logging, only `print`.

## 5. Performance limitations

- Models load lazily on first use, which is good. The first chat request after boot pays the full load time, and
  start-up seeding loads both models synchronously inside `lifespan`. Render cannot bind the port until seeding
  finishes. **[code]**
- All endpoints are `async def` but run blocking CPU inference, so one embedding call stalls the event loop for every
  other request, including `/health`. **[code]**
- `pip install torch` from PyPI on Linux pulls the CUDA build (several GB) even though deployment is CPU-only. **[code]**
- Model weights on disk: MiniLM 91 MB (safetensors) and CLIP ViT-B/32 605 MB (`pytorch_model.bin`). With torch,
  loading both exceeds a 512 MB instance. Measured figures are in `docs/DEPLOYMENT.md`. **[verified sizes]**
- `IndexFlatIP` is exact, brute-force search. That is fine for thousands of chunks but grows linearly. Acceptable for
  this scale.

## 6. Security issues

| # | Severity | Issue |
| --- | --- | --- |
| S1 | **Critical** | **Stored XSS in the dashboard.** `appendAssistantResponse` inserts `data.answer` (verbatim document text) through `innerHTML`. Any uploaded document containing `<img src=x onerror=…>` executes script in every visitor's browser. Citation filenames are also unescaped. |
| S2 | High | `CORS allow_origins=["*"]` together with `allow_credentials=True`. |
| S3 | High | The upload handler reads the entire body into memory before checking size (`await file.read()`), allowing memory exhaustion. There is no decompression-bomb guard for images and no page or image caps for PDFs. |
| S4 | Medium | API responses include absolute server filesystem paths (`image_path`). |
| S5 | Medium | `/static/images` serves every file in the images directory, not just registered images. |
| S6 | Medium | The unhandled-exception path returns `str(e)` to clients (`"Ingestion error: …"`). Image search on a corrupt upload raises an uncaught PIL error (HTTP 500). |
| S7 | Medium | Gemini key in the URL (B7). LLM output is not post-validated (B8). |
| S8 | Low | No security headers (CSP, nosniff, frame-ancestors). The dashboard loads Google Fonts, a third-party request on every visit. |
| S9 | Info | No authentication. Every visitor can see and search every uploaded document. This must be documented for any public demo. |

## 7. Deployment risks

1. **Memory:** CLIP + MiniLM + torch on Render Free/Starter (512 MB) is very likely to be OOM-killed.
2. **Ephemeral disk:** Render free services have no persistent disk, so all uploads are lost on every restart or
   deploy. The UI never says so.
3. **Model downloads at runtime:** without a pre-fetched cache, each cold boot downloads ~700 MB from Hugging Face.
4. **Build size and time:** the CUDA torch wheel (see the torch item in §5).
5. The Docker image ignores `$PORT` and is missing one sample file (B16, B17).
6. The README advertises "Tests: 32 Passed", benchmark tables and a 35 ms latency as static claims. A Python
   3.10–3.14 support claim is unverified, and "CUDA automatically utilized if present" is false: CUDA is only used
   when `DEVICE=cuda` is set.

## 8. README accuracy

| README claim | Reality |
| --- | --- |
| Markdown "heading preservation" | Not implemented |
| "Automated extraction and deduplication of embedded images" | Extraction yes; de-duplication no |
| "Citation Verifier" (diagram) | No verifier exists |
| "Citation correctness: exact match of cited filename and page number" | Only the filename is checked |
| "Groundedness: statement-level factual alignment" | Substring check for expected keywords in the answer |
| "CUDA automatically utilized if present" | Only if `DEVICE=cuda` |
| "Multimodal Evidence Ranker" | Not implemented; scores from both spaces are compared with `max()` |
| "falls back to extractive if API call fails" | Crashes (B5) |
| Persistent document list | Lost on restart (B1) |

## 9. Prioritized implementation plan

**P0: correctness and security**
1. Fix the dashboard XSS by rendering all user/document data through `textContent` (S1).
2. Fix the LLM fallbacks, the OpenAI header and the Gemini key placement (B5–B7).
3. Persist the document registry together with the indexes, written atomically after every mutation (B1–B3).
4. Make duplicate uploads idempotent: never add duplicate index entries (B2).
5. Isolate tests from real storage and add a deterministic mock embedding backend (B4).
6. Derive all storage paths from one configurable root (B9).

**P1: ingestion and RAG quality**
7. Stream uploads with a hard byte cap; add magic-byte checks, decompression-bomb guard, page/image caps,
   status tracking and recorded failures (S3, §4.7–4.9).
8. Use a deterministic chunker with real overlap and Markdown heading context; remove the LangChain dependency (B14).
9. Use separate per-modality thresholds; attach images only when CLIP similarity passes the image threshold (B10, B12).
10. Use transparent hybrid ranking (weighted Reciprocal Rank Fusion), so raw scores from the two spaces are never
    compared.
11. Add a citation verification layer covering existence, page range, passage membership and lexical support (§4.3).

**P2: API, UI, deployment**
12. Add a service layer with thin routes, `/ready`, document details and delete, a safe image route, consistent errors,
    configurable CORS and security headers.
13. Build a premium UI with workspace, library, chat, visual search and evaluation views.
14. Fix the evaluation (page-aware citations, separated modalities, honest metric names, hard negatives).
15. Render deployment: CPU torch wheel, model pre-fetch at build time, `$PORT`, a single worker, documented memory
    and storage modes.
16. Rewrite the README so it matches the implementation.

## 10. Status after the upgrade

| Item | Status | Where / evidence |
| --- | --- | --- |
| S1 stored XSS | Fixed | UI renders with DOM APIs/`textContent` only; strict CSP without inline scripts. An E2E browser run with XSS payloads in a file name, document text and a question did not execute script |
| S2 CORS | Fixed | Disabled by default; `DRIVEDOCS_CORS_ORIGINS` opt-in, no credentials |
| S3 resource exhaustion | Fixed | Streaming body limit (Content-Length + counted bytes), per-file cap, PDF page cap, image pixel cap, per-document image cap, document-count cap |
| S4/S5 paths & static files | Fixed | Public schemas omit `image_path`; images served only by registered ID through a regex-validated route |
| S6 error leakage | Fixed | Uniform `{"detail": {"code", "message"}}`; unhandled errors return a generic 500 and are logged server-side |
| S7 LLM key & output | Fixed | Gemini key in `x-goog-api-key` header; LLM output verified, falls back to extractive |
| S8 headers | Fixed | CSP, nosniff, frame-ancestors/X-Frame-Options, no-referrer; Google Fonts removed |
| S9 no auth | Documented | Banner in the UI, `/system/info.authentication=false`, README and DEPLOYMENT.md |
| B1–B4 persistence & tests | Fixed | Registry + indexes persisted atomically after each mutation, reconciled on boot; tests use temp storage |
| B5–B8 generation | Fixed | `generation.py` rewrite; tests use a mock HTTP transport |
| B9 storage dir | Fixed | All paths derived from one root |
| B10–B13 retrieval & eval | Fixed | Per-modality rules, RRF, page-aware evaluation (see `docs/RAG_DESIGN.md`) |
| B14 chunker | Fixed | Deterministic splitter with real overlap; LangChain dependency removed |
| B15 filenames | Fixed | Splits on `/` and `\` on every OS |
| B16–B18 deployment | Fixed | New `.dockerignore`, `$PORT`, build no longer regenerates samples |
| B19 CRLF PDF | Fixed | `.gitattributes` marks binaries |
| B20 httpx | Fixed | Runtime dependency |
| Missing features §4.1–4.12 | Implemented | Details/delete endpoints, statuses, `/ready`, verifier, hybrid ranking, heading context, image dedupe, content sniffing, failure records, new UI, mock embeddings, logging |
| Deployment footprint | Improved | Embeddings moved to ONNX Runtime (no PyTorch): ~0.44 GB RAM instead of ~1 GB, so the full demo fits Render's free 512 MB plan; evaluation re-verified (19/19). See `docs/RAG_DESIGN.md` §9 |
| Not verified | Open | `docker build` was not run (Docker unavailable) and memory was not measured on Linux. The model download script, a clean install from `requirements.txt`, Linux/CPython 3.14 wheel resolution and a Render-equivalent runtime were verified later (see `docs/DEPLOYMENT.md` §1). No live Render deployment exists yet |
