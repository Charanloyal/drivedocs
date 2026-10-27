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

## 2. Calibration data (sample corpus)

Top scores per query, measured separately per modality before any threshold was chosen:

| Query kind | Top MiniLM score | Top CLIP score |
| --- | --- | --- |
| Answerable text questions (9) | 0.43 to 0.68 | 0.22 to 0.29 (irrelevant images) |
| Visual requests for the architecture diagram | 0.20 to 0.21 | 0.27 to 0.28 (relevant) |
| Visual requests for the efficiency chart | 0.24 to 0.39 | 0.35 to 0.36 (relevant) |
| Out-of-corpus (stock price, cookies, Mars, World Cup) | 0.09 to 0.19 | 0.18 to 0.22 |
| Near-domain unanswerable (Paxos / lithium niobate) | 0.42 / 0.17 | 0.22 / 0.24 |

Conclusions:

* A **MiniLM threshold of 0.30** separates answerable questions (≥ 0.43) from out-of-corpus questions (≤ 0.19) with
  a margin on both sides.
* **No single CLIP threshold works**: an irrelevant chart scored 0.284 for an API-rate-limit question, above the
  relevant architecture diagram (0.277). CLIP acceptance therefore uses context (rules below).
* Near-domain questions (Paxos) pass any reasonable similarity threshold because the corpus describes a closely
  related topic (Raft). They are handled by a term-coverage check in the generator instead.

## 3. Relevance rules

**Text** uses a two-level threshold:
1. *Gate*: at least one passage must reach `text_min_score` (0.30), otherwise the answer is "insufficient evidence".
2. *Context floor*: once the gate passes, passages ≥ `text_context_min_score` (0.20) may contribute sentences.

The floor exists because the most relevant chunk is not always the top-scoring one. For "How are audit trail records
protected against tampering?" the chunk containing the answer scored 0.288 while a less relevant chunk scored 0.305.

**Images.** An image is accepted when one of these applies, checked in order:

| Rule | Condition |
| --- | --- |
| `strong_clip_match` | CLIP score ≥ 0.30 |
| `visual_query_match` | query asks for a visual (diagram, chart, figure, image, plot, …) **and** score ≥ 0.25 |
| `same_page_as_text_evidence` | image is on the same document page as a passage that passed the gate, score ≥ 0.25, **and** that page is cited in the final answer |

Rejected images are still returned by `/search/images` with "below the relevance thresholds" so users can browse them.

## 4. Hybrid ranking

The combined evidence list uses weighted Reciprocal Rank Fusion:
`fused = weight_modality / (60 + rank_within_modality)` with weights 0.7 (text) and 0.3 (image). Accepted items
come before rejected ones. Raw scores are kept on each item for transparency.

Measured against the previous evaluation's ordering (sorting MiniLM and CLIP scores together):

| Ordering | Document MRR (answerable queries) |
| --- | --- |
| Weighted RRF | 0.964 |
| Raw-score mixing (baseline) | 0.964 |

**No measured benefit on this dataset.** RRF is kept because it does not depend on the two score scales being
comparable, which raw mixing silently assumes. No reranking model was added: it would cost memory without a
demonstrated gain here.

## 5. Grounded answer generation

The default `extractive` provider:

1. Re-flows PDF line breaks so sentences are not cut mid-line (pypdf returns hard-wrapped lines).
2. Splits accepted passages into sentences and drops headings, titles and reference lines.
3. Scores sentences by query-term overlap plus passage similarity. Figure/table captions are down-weighted
   because they label a visual rather than state a fact.
4. Quotes up to 3 sentences scoring ≥ 60% of the best, each with `[Source: file, Page N]`.
5. If salient query terms (names, acronyms, alphanumeric identifiers) appear nowhere in the evidence, it prefixes
   *"Note: the indexed documents do not mention 'paxos'"*.
6. If no sentence matches, it returns the insufficient-evidence message (or a visual-only answer when an image was
   accepted).

Document text cannot forge citations: `[Source:` inside documents is neutralized before quoting or prompting.
Optional Gemini/OpenAI providers receive the evidence inside a delimited block marked as untrusted. Their output
goes through the same verifier, and the extractive answer replaces it if no citation verifies or the call fails.

## 6. Citation verification

Every `[Source: file, Page N]` tag is checked against the document registry and the retrieved evidence: the document
exists, the page exists, and a retrieved passage from that page exists. Failing tags are removed (replaced by
`[unverified citation removed]`) and reported in `warnings`. Claims whose words mostly do not appear in the cited
passage are flagged as weakly supported. This verifies **provenance**; it does not prove that a paraphrase is
correct.

## 7. Chunking

The chunker is a dependency-free recursive splitter (paragraph → line → sentence → word → character) with real
overlap carried between chunks. Markdown is split per section, and the heading path (e.g.
`Guide > Setup`) is stored on each chunk and prepended to the embedded text.

Sweep on the evaluation set (all other settings default):

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

## 8. Iteration log (evaluation set, real models)

| Step | Pass rate | Fact coverage | Answer rate | Notes |
| --- | --- | --- | --- | --- |
| Original code + original 8-query evaluation | n/a | "91.7% groundedness" | n/a | The old metric credited an unrelated answer to the diagram query |
| New harness, before RAG fixes | 78.9% (15/19) | 63.6% | 85.7% | PDF line breaks cut answers mid-sentence; audit-trail and edge-coupler questions refused |
| + PDF re-flow, two-level text threshold | 100% | 100% | 100% | |
| + heading/caption filtering, page-cited image rule | 100% | 100% | 100% | Answer text cleaner; no off-page charts attached |

Because these fixes were driven by failures on the same 19 queries, the final figures are in-sample. Add your own
queries to `data/sample/evaluation_set.json` (or a separate file via `--dataset`) to measure generalization.

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
| MiniLM | **fp32 (default)** | 1.000 | 100% top-1 | 2 ms/query |
| MiniLM | int8 | 0.988 | 100% top-1 | −65 MB, but one answerable query fell from 0.300 to 0.295, under the 0.30 gate |
| CLIP | fp32 text + fp32 vision | 1.000 / 0.986 | 100% | +660 MB: does not fit 512 MB |
| CLIP | fp16 | same as fp32 | 100% | ONNX Runtime CPU upcasts: no memory saving |
| CLIP | int8 text + int8 vision | 0.767 / 0.905 | 89% | Too lossy (text tower) |
| CLIP | bnb4 text + bnb4 vision | 0.978 / 0.939 | 95% | 76 ms per text query (slow kernel) |
| CLIP | **q4 text + q4 vision (default)** | 0.961 / 0.955 | 95% top-1 on the raw 19×2 comparison; **100% in the end-to-end evaluation** | +207 MB, 7 ms per text query |

End-to-end result with the defaults: the full evaluation still passes 19/19, with the same Recall/MRR/citation
figures as PyTorch and p50 latency of 9.9 ms (was ~15 ms). Measured process memory (`/ready` → `memory_rss_mb`,
uvicorn, Windows):

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
