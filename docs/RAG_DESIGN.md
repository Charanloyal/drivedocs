# Retrieval & Grounding Design

This document explains how DriveDocs retrieves evidence and builds answers, and records the measurements behind
each design choice.

## 1. Two embedding spaces, two indexes

| | Text track | Image track |
| --- | --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` | `openai/clip-vit-base-patch32` |
| Vectors | 384-d, L2-normalized | 512-d, L2-normalized (image tower and text tower share this space) |
| Index | `faiss.IndexFlatIP` (exact cosine) | `faiss.IndexFlatIP` (exact cosine) |
| Query encoder | MiniLM | CLIP text tower (text→image) or CLIP image tower (image→image) |
| What is embedded | chunk text, prefixed with its Markdown heading path | normalized PNG (EXIF-rotated, transparency on white, ≤1600 px) |

MiniLM and CLIP vectors are never compared with each other, and neither are their **scores**: a CLIP text→image
cosine of 0.28 can be a good match while a MiniLM cosine of 0.28 is weak. Each modality has its own relevance rule
(below), and the combined view uses rank fusion, not score mixing.

## 2. Calibration data

Top scores per query, measured separately per modality before any threshold was chosen (ONNX models).

**Vehicle-manual demo corpus (current):**

| Query kind | Top MiniLM score | Top CLIP text→image score | Top caption similarity |
| --- | --- | --- | --- |
| Answerable text questions (14) | ≥ 0.28 | n/a | n/a |
| Diagram requests (5) | varies | **0.21 to 0.32 for right and wrong diagrams alike** | **≥ 0.40 for the right diagram, ≤ 0.32 for wrong ones** |
| Out-of-corpus (World Cup, Mars, cookie recipe, stock price) | ≤ 0.21 | low | low |
| Near-domain unanswerable (engine oil, spark plugs) | ~0.32 | n/a | n/a |

Conclusions:

* The text gate was recalibrated from 0.30 to **0.27**. Answerable questions are ≥ 0.28 and negatives ≤ 0.21.
* Near-domain questions about an EV (engine oil, spark plugs) score *above* the gate, because the manual talks about
  maintenance and fluids. They are handled by a **corpus-vocabulary check** in the generator: a question is refused
  when at least 2 of its content terms, and at least a third of them, occur nowhere in the indexed documents.
* **CLIP alone cannot rank drawn diagrams.** The fuse box, tire placard and warning-light panel are flat line
  drawings with small labels, and their CLIP scores are bunched within 0.11 of each other. **Caption similarity**
  (MiniLM cosine between the query and the figure caption) separates them cleanly, so image retrieval fuses the two
  (§3a).

**Earlier technical corpus** (photonics PDF, Raft notes; now `tests/fixtures/legacy_corpus/`): answerable questions
scored 0.43–0.68 and out-of-corpus ≤ 0.19, which led to the original 0.30 gate. There, an irrelevant chart scored
0.284 for an API-rate-limit question, above the relevant architecture diagram (0.277). That was the first evidence
that no single CLIP threshold works.

## 3. Relevance rules

**Text** uses a two-level threshold:
1. *Gate*: at least one passage must reach `text_min_score` (0.27), otherwise the answer is "insufficient evidence".
2. *Context floor*: once the gate passes, passages ≥ `text_context_min_score` (0.20) may contribute sentences.

The floor exists because the most relevant chunk is not always the top-scoring one.

**Images.** An image is accepted when one of these applies, checked in order:

| Rule | Condition |
| --- | --- |
| `strong_clip_match` | CLIP score ≥ 0.30 |
| `caption_match` | the query asks for a visual (diagram, chart, layout, picture, …) **and** caption similarity ≥ 0.40 |
| `visual_query_match` | the query asks for a visual **and** CLIP ≥ 0.27 **and** (no caption, or caption similarity ≥ 0.30) |
| `same_page_as_text_evidence` | the image is on the same page as a passage that passed the gate, CLIP ≥ 0.25, **and** that page is cited in the final answer |

Rejected images are still returned by `/search/images` with "below the relevance thresholds", so users can browse them.

### 3a. Caption-fused image retrieval

Captions come from the PDF itself: a `Figure N.N: …` line on the image's page, or the image's own file name
(`front_compartment_diagram.png` → "front compartment diagram"). For text→image search:

1. CLIP ranks all images by text→image cosine.
2. MiniLM ranks all captioned images by caption similarity. Caption vectors are cached and refreshed when images
   change. Both sides are stripped of filler such as "show me", "diagram" and "figure N.N" (`caption_text`), so
   similarity reflects the subject.
3. The ranks are combined with Reciprocal Rank Fusion (`1/(60+rank_clip) + 1/(60+rank_caption)`). The scores
   themselves are never mixed.
4. Images that pass a confident rule (CLIP ≥ 0.30 or caption ≥ 0.40) are listed first.

Result on the 5 diagram queries: **Image Recall@3 100%, MRR 1.000**. With CLIP alone, the right diagram was often
not ranked first. Both scores are returned (`score`, `caption_score`), and the UI shows the caption and the
acceptance reason.

## 4. Hybrid ranking

The combined evidence list uses weighted Reciprocal Rank Fusion:
`fused = weight_modality / (60 + rank_within_modality)` with weights 0.7 (text) and 0.3 (image). Accepted items
come before rejected ones. Raw scores are kept on each item for transparency.

Measured against raw-score ordering (sorting MiniLM and CLIP scores together):

| Ordering | Document MRR, vehicle corpus | Document MRR, earlier corpus |
| --- | --- | --- |
| Weighted RRF | 0.956 | 0.964 |
| Raw-score mixing (baseline) | 0.961 | 0.964 |

**No measured benefit on these datasets** (raw mixing is even marginally ahead on the vehicle set). RRF is kept
because it does not depend on the two score scales being comparable, which raw mixing silently assumes. No
reranking model was added: it would cost memory without a demonstrated gain here.

## 5. Grounded answer generation

The answer pipeline in [`generation.py`](../src/drivedocs/generation.py) runs in this order:

1. **Refuse early.** No passage passes the text gate and no image is accepted → "insufficient evidence". The
   question's content terms are mostly absent from the corpus vocabulary (§2) → refused with the missing terms
   named. Neither case calls the LLM, which saves quota.
2. **LangChain agent** (`LLM_PROVIDER=agent` with a `GEMINI_API_KEY`). [`agent.py`](../src/drivedocs/agent.py)
   binds Gemini (`langchain-google-genai`) to two `StructuredTool`s:
   * `search_manual_text(query)`: MiniLM + FAISS, returning file, page, score and passage;
   * `search_diagrams(query)`: CLIP + caption fusion, returning file, page, caption and scores.

   The model plans its own searches (rephrasing, or adding a diagram search) for at most
   `DRIVEDOCS_AGENT_MAX_STEPS` = 4 tool rounds. Tool output is PII-redacted and wrapped as
   `UNTRUSTED DOCUMENT TEXT` before the model sees it, and the system prompt forbids following instructions found
   in it. Every tool call is recorded in `agent_trace`, which the UI shows. The verification pool is the
   retriever's accepted passages plus every passage the agent's tools returned.
3. **Verify the agent's answer.** If no citation survives verification, or the agent raises (missing or invalid
   key, quota, timeout, malformed output), the extractive answer is used and a warning explains why.
4. **Extractive answerer** (no API key, also the fallback):
   * re-flows PDF line breaks;
   * converts Markdown tables into sentences ("Cabin air filter: Replace every 12 months or 20,000 km.");
   * keeps labelled SYMPTOM/CAUSE/ACTION blocks together as one unit;
   * drops headings (short lines without sentence punctuation, including those ending in ")") and figure captions;
   * selects sentences by **coverage** of the query terms, ignoring terms that occur in most passages and so do
     not discriminate, and quotes up to 3 with `[Source: file, Page N]`;
   * never quotes a sentence that matches an injection pattern;
   * if salient query terms appear nowhere in the evidence, prefixes *"Note: the indexed documents do not mention
     …"*.
5. **Mask PII** in the answer, citations' snippets and the passages returned for inspection (§6a).

Document text cannot forge citations: `[Source:` inside documents is neutralized before quoting or prompting.

## 6a. Security: PII and prompt injection

[`security.py`](../src/drivedocs/security.py) provides detectors that run at answer time. Stored documents are
unchanged, so an authorized pipeline could still use the original text.

| Detector | Rule |
| --- | --- |
| `EMAIL` | Address pattern with word boundaries |
| `PHONE` | 10-digit numbers with an area code and separators (`(555) 010-2345`, `555-010-2345`), optional `+country` code; bare digit runs such as part numbers do not match |
| `SSN` | `ddd-dd-dddd`, excluding invalid prefixes (000, 666, 9xx) |
| `CARD` | 13–19 digits with optional separators **and** a valid Luhn checksum |
| `VIN` | 17 letters and digits without I/O/Q, with a valid ISO 3779 check digit **or** "VIN"/"vehicle identification" just before |
| `LABELLED_NAME` | 1–3 capitalised words after a label (`Customer:`, `Owner:`, `Name:`, `Driver:`, `Contact:`, `Client:`, …) on the same line |

Redaction replaces values with `[REDACTED EMAIL]` and similar tags, counts them in `pii_redactions`, and is applied
to `/chat`, `/search/text`, `/documents/{id}` and everything sent to an LLM.

`injection_signals()` looks for instruction-override patterns:
- "ignore/disregard … previous … instructions";
- "reveal/print … system prompt / API key / secret";
- "you are now", "act as a …", "new instructions:";
- fake "system:" notices;
- "answer every question …";
- attempts to close the untrusted-data delimiters.

Matching passages get `flags: ["prompt_injection"]` and a warning. Their sentences are never quoted by the
extractive answerer, and the agent sees them only as labelled untrusted data. Results of the adversarial suite are in
[EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md#security-suite).

## 6. Citation verification

Every `[Source: file, Page N]` tag is checked against the document registry and the retrieved evidence: the document
exists, the page exists, and a retrieved passage from that page exists. Failing tags are removed (replaced by
`[unverified citation removed]`) and reported in `warnings`. Claims whose words mostly do not appear in the cited
passage are flagged as weakly supported. This verifies **provenance**; it does not prove that a paraphrase is
correct.

## 7. Chunking

The chunker is a dependency-free recursive splitter (paragraph → line → sentence → word → character) with real
overlap carried between chunks. Markdown is split per section, and the heading path (e.g.
`Northwind Aurora Maintenance Schedule > Fluids and capacities`) is stored on each chunk and prepended to the
embedded text. Markdown tables are rewritten as one sentence per row before chunking, so a row stays intact and is
quotable.

Sweep on the earlier 19-query evaluation set (all other settings default):

| chunk_size / overlap | Text MRR | Text P@5 | Fact coverage | Answer rate | Pass rate |
| --- | --- | --- | --- | --- | --- |
| 250 / 50 | 1.000 | 78.2% | 100% | 100% | 100% |
| 500 / 0 | 1.000 | 52.7% | 100% | 100% | 100% |
| **500 / 100 (default)** | 1.000 | 52.7% | 100% | 100% | 100% |
| 800 / 150 | 1.000 | 41.8% | 100% | 100% | 100% |
| 1200 / 200 | 1.000 | 30.9% | 90.9% | 92.9% | 94.7% |

Very large chunks start to hurt (one fact was lost at 1200). Between 250 and 800 the end-to-end results are identical
on this small corpus. Higher P@5 for small chunks mostly reflects more chunks per relevant page. The default stays
at 500/100.

## 8. Iteration log

**Earlier technical corpus (19 queries):**

| Step | Pass rate | Fact coverage | Answer rate | Notes |
| --- | --- | --- | --- | --- |
| Original code + original 8-query evaluation | n/a | "91.7% groundedness" | n/a | The old metric credited an unrelated answer to the diagram query |
| New harness, before RAG fixes | 78.9% (15/19) | 63.6% | 85.7% | PDF line breaks cut answers mid-sentence; audit-trail and edge-coupler questions refused |
| + PDF re-flow, two-level text threshold | 100% | 100% | 100% | |
| + heading/caption filtering, page-cited image rule | 100% | 100% | 100% | Answer text cleaner; no off-page charts attached |

**Vehicle-manual corpus (26 queries + 10 security queries).** Switching to drawn diagrams and manual-style text
exposed failures that the earlier corpus had hidden:

| Failure observed | Change |
| --- | --- |
| Pure CLIP ranking was unreliable on the drawn diagrams (right and wrong diagrams scored within 0.21–0.32) | Caption extraction + caption-fused RRF ranking + `caption_match` rule (§3a) |
| Maintenance-table rows did not form quotable sentences | Markdown tables converted to sentences |
| Troubleshooting answers could separate a symptom from its cause and action | SYMPTOM/CAUSE/ACTION blocks kept together; coverage-based sentence selection |
| Heading lines ending in ")" were treated as sentences | Heading detection extended |
| Answerable questions just under 0.30 | Text gate recalibrated to 0.27 (negatives ≤ 0.21) |
| "Engine oil" / "spark plugs" for an EV passed the similarity gate | Corpus-vocabulary refusal (§2) |
| Injection suite 4/5: the question-borne injection was counted as leaked because the scan included the evidence passages | Scoped the check to answer, citations and images; passages are displayed data (see EVALUATION_METHODOLOGY) |

The final figures (26/26, groundedness 100%, security 10/10) are **in-sample**, because the rules above were
changed while looking at these queries. Add your own queries to `data/sample/evaluation_set.json` (or use
`--dataset`) to measure generalization.

## 9. Embedding runtime: ONNX instead of PyTorch

The original stack (PyTorch + Transformers + Sentence-Transformers) needed ~1 GB of RAM, which forced a paid 2 GB
Render instance. The default backend now runs the same two models with **ONNX Runtime**, using pre-exported graphs
from the Hugging Face Hub (`Xenova/all-MiniLM-L6-v2`, `Xenova/clip-vit-base-patch32`). Tokenization uses
`tokenizers`, and CLIP image preprocessing (shortest-edge 224 bicubic resize, center crop, CLIP mean/std) is
implemented with Pillow + NumPy. PyTorch remains available as `DRIVEDOCS_EMBEDDING_BACKEND=transformers`.

Variants measured against the PyTorch reference (sample corpus, 19 evaluation queries, 2 images; memory = extra RSS
of the loaded sessions, Windows):

| Model | Variant | Cosine to PyTorch (min) | Ranking agreement | Notes |
| --- | --- | --- | --- | --- |
| MiniLM | fp32 | 1.000 | 100% top-1 | 2 ms/query |
| MiniLM | int8 (full dynamic quantization) | 0.988 | 100% top-1 | −65 MB, but one answerable query fell from 0.300 to 0.295, under the then 0.30 gate |
| MiniLM | **fp32 + int8 embedding table only (default)** | ≥ 0.9998 | 100% | Only the token-embedding Gather table is quantized (per-row scale), so the transformer layers stay fp32: file 86 → 53 MB, RAM 116 → 64 MB |
| CLIP | fp32 text + fp32 vision | 1.000 / 0.986 | 100% | +660 MB: does not fit 512 MB |
| CLIP | fp16 | same as fp32 | 100% | ONNX Runtime CPU upcasts: no memory saving |
| CLIP | int8 text + int8 vision | 0.767 / 0.905 | 89% | Too lossy (text tower) |
| CLIP | bnb4 text + bnb4 vision | 0.978 / 0.939 | 95% | 76 ms per text query (slow kernel) |
| CLIP | **q4 text + q4 vision (default)** | 0.961 / 0.955 | 95% top-1 on the raw 19×2 comparison; **100% in the end-to-end evaluation** | +207 MB, 7 ms per text query. The text tower's embedding table is also stored as int8 (file 120 → 48 MB, cosine to q4 ≥ 0.9998) |

Since the vehicle-corpus update, the **vision tower is loaded lazily**. Text→image search only needs the text tower,
because image vectors are computed at ingestion time and stored in FAISS. The vision session (~69 MB) is therefore
created only for image uploads or image→image queries, and dropped after `DRIVEDOCS_VISION_IDLE_UNLOAD_SECONDS`.
With the demo index built at build time, a normal visitor session never loads it.

End-to-end result with the defaults on the earlier corpus: the full evaluation still passed 19/19, with the same
Recall/MRR/citation figures as PyTorch and p50 latency of 9.9 ms (was ~15 ms). Measured process memory before the
int8 tables and the lazy vision tower (`/ready` → `memory_rss_mb`, uvicorn, Windows):

| | PyTorch backend | ONNX backend (default) |
| --- | --- | --- |
| Text + image search, after start-up | 864 MB | **428 MB** |
| After a stress run (30 concurrent chats, 9 uploads incl. 60 PDF pages, 5 image searches) | n/a | **436 MB** |
| Text-only (`DRIVEDOCS_ENABLE_IMAGE_SEARCH=false`) | 506 MB | **~250 MB** |
| Cold start (models cached) | 6–10 s | **~1–2.5 s** |
| Installed Python packages | ~1.2 GB | **270 MB** |

One performance detail mattered: ONNX Runtime worker threads busy-wait ("spin") by default. With three sessions plus
FAISS in one process, the threads contended with each other and requests were 4–10× slower (93 ms vs 10 ms per
chat). Spinning is disabled and the thread count is capped (`DRIVEDOCS_ONNX_THREADS`).
