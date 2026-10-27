# Retrieval, Agent & Grounding Design

This document explains how DriveDocs plans, retrieves evidence and builds answers for vehicle-manual questions, and
records the measurements behind each design choice. Numbers labelled *v0.2* were measured on the previous generic demo
corpus and are kept because the decision they support did not change.

## 1. Two embedding spaces, two indexes

| | Text track | Image track |
| --- | --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` (int8 ONNX) | `openai/clip-vit-base-patch32` (4-bit ONNX) |
| Vectors | 384-d, L2-normalized | 512-d, L2-normalized (image tower and text tower share this space) |
| Index | `faiss.IndexFlatIP` (exact cosine) | `faiss.IndexFlatIP` (exact cosine) |
| Query encoder | MiniLM | CLIP text tower (text→image) or CLIP image tower (image→image) |
| What is embedded | chunk text, prefixed with its Markdown heading path; personal data already redacted | normalized PNG (EXIF-rotated, transparency on white, ≤1600 px) |

MiniLM and CLIP vectors are never compared with each other, and neither are their **scores**: a CLIP text→image
cosine of 0.28 can be a good match while a MiniLM cosine of 0.28 is weak. Each modality has its own relevance rule
(§3), and the combined view uses rank fusion, not score mixing.

## 2. The LangChain agent

`POST /chat` runs [`DriveDocsAgent`](../src/drivedocs/agent.py), a LangChain tool-calling loop:

```
messages = [SystemMessage(rules), HumanMessage(screened question)]
model = planner.bind_tools([search_manual_text, search_manual_diagrams])
repeat ≤ 4 turns:
    ai = model.invoke(messages)            # AIMessage with .tool_calls
    if no tool_calls: stop
    for call in ai.tool_calls: messages.append(tool.invoke(call))   # ToolMessage(content, artifact)
evidence = assemble(artifacts)  →  grounded generator  →  citation verifier  →  output guardrails
```

* **Tools** are `StructuredTool`s with `response_format="content_and_artifact"`: the planner sees a compact,
  PII-masked, injection-stripped rendering of the results; the structured `TextSearchResult`/`ImageSearchResult`
  objects travel as the artifact, so the rest of the pipeline works on exact scores and page metadata. Both indexes are
  also exposed as LangChain `BaseRetriever`s (`ManualTextRetriever`, `ManualDiagramRetriever`).
* **Planners.** With `LLM_PROVIDER=openai|gemini`, a key and `langchain-openai`/`langchain-google-genai` installed, an
  LLM (`ChatOpenAI` / `ChatGoogleGenerativeAI`) chooses the tool calls. Otherwise `ManualRouterChatModel`, a
  deterministic `BaseChatModel` subclass, plans them offline, so the free demo needs no key and the evaluation is
  reproducible. A failing LLM planner falls back to the router for that request (noted in the trace).
* **Grounding policy.** The planner's own prose is never the answer. If it stops without a text search, the agent
  searches with the question anyway; the answer always comes from the grounded generator over retrieved evidence.

### Router planner

1. **Decomposition.** "X and what/how/where … Y?" is split into up to 3 sub-questions. A later part that lost its
   subject inherits the first part's topic words: "How often should brake fluid be replaced and what type of fluid
   should I use?" → "what type of fluid should I use (brake fluid)?". A part with ≥ 2 topic words of its own is left
   alone ("… and what is the oil capacity?").
2. **Routing.** Every sub-question gets a text search. A diagram search is added for visual requests and for
   vehicle-manual cues where a figure helps: warning lights/symbols/indicators, dashboard, locations ("where",
   "located"), fuses, jump-starting/cables/terminals, labels/placards, engine bay, reservoirs, dipstick, battery, tire
   pressure.
3. **Diagram-query rewriting.** CLIP matches a description of the picture better than a question. For location
   questions about under-hood components the diagram query becomes "engine compartment diagram showing the …".
   Measured CLIP scores for the engine-bay diagram:

   | Query sent to CLIP | Engine-bay diagram | Best other image |
   | --- | --- | --- |
   | "Where is the coolant reservoir?" | 0.250 | 0.241 |
   | "engine compartment diagram showing the coolant reservoir" | **0.286** | 0.215 |
   | "Where is the brake fluid reservoir?" | 0.235 (rank 2) | 0.250 |
   | "engine compartment diagram showing the brake fluid reservoir" | **0.290** | 0.248 |
   | "Where is the oil dipstick?" | 0.202 (rank 2) | 0.223 |
   | "engine compartment diagram showing the oil dipstick" | **0.293** | 0.236 |

   The diagram labels were also enlarged (22 px font) after measuring that CLIP could not associate the 14 px labels
   with component names. Because the planner explicitly asked for a diagram, the rewritten query counts as a visual
   request for the acceptance rules (§3).
4. **Per-part answers.** Each sub-question is answered from its own evidence; verified parts are concatenated and
   sentences already quoted by an earlier part are dropped.

The direct pipeline (`mode: "direct"`, one retrieval pass with the raw question) passes 38/39 QA questions; the agent
passes 39/39, the difference being the coolant-reservoir location question that only the rewritten diagram query
answers.

## 3. Calibration data and relevance rules

Top scores per question type on the vehicle corpus (top-5 retrieval, int8 MiniLM, 4-bit CLIP):

| Question type | Top MiniLM score | Top CLIP score |
| --- | --- | --- |
| Answerable text (27) | 0.34 to 0.81 | 0.19 to 0.31 (mostly unrelated images) |
| Visual requests (6) | 0.31 to 0.78 | 0.25 to 0.34 (relevant diagram) |
| Out-of-corpus (World Cup, Australia, cookies) | 0.08 to 0.21 | 0.16 to 0.19 |
| Near-domain unanswerable (CarPlay, AWD Sport trim, diesel DPF) | 0.23 to 0.55 | 0.28 |

* A **MiniLM gate of 0.30** separates answerable questions (≥ 0.34) from out-of-corpus questions (≤ 0.21).
* Near-domain questions can pass the gate (the AWD question shares "tire pressure" with page 2). They are handled by
  the generator: salient terms missing from the evidence ("awd", "sport") produce an explicit *"the indexed documents
  do not mention …"* flag, and questions without any matching sentence are refused.
* **No single CLIP threshold works** (v0.2 and again here: "How do I pair Apple CarPlay?" scores 0.268 against the
  fuse-box diagram), so CLIP acceptance uses context.

**Text** uses a two-level threshold: a *gate* (at least one passage ≥ 0.30, otherwise "insufficient evidence") and a
*context floor* (once the gate passes, passages ≥ 0.20 may contribute sentences), because the passage with the answer
is not always the top-scoring one.

**Images** are accepted by the first rule that applies:

| Rule | Condition |
| --- | --- |
| `strong_clip_match` | CLIP score ≥ 0.30 |
| `visual_query_match` | a diagram was requested (by the user's wording or the agent's diagram query), the image is the **top-ranked** result, and score ≥ 0.25 |
| `same_page_as_text_evidence` | the image is on the same page as a passage that passed the gate, score ≥ 0.25, **and** that page is cited in the final answer |

The visual-query rule admits only the best-ranked image because different diagrams often score within 0.01–0.03 of
each other: "Show me the fuse box diagram" scored the fuse box 0.294 and the jump-start diagram 0.267, and both were
attached before this rule. Runners-up still qualify through the strong-match or co-location rules. Rejected images are
still returned by `/search/images`, labelled "below the relevance thresholds".

## 4. Hybrid ranking

The combined evidence list uses weighted Reciprocal Rank Fusion, `fused = weight / (60 + rank_within_modality)` with
weights 0.7 (text) and 0.3 (image); accepted items come first and raw scores are kept for transparency.

| Ordering | Document MRR (answerable questions) |
| --- | --- |
| Weighted RRF | 0.934 |
| Raw-score mixing (baseline) | 0.942 |

**No measured benefit on this dataset** (v0.2: 0.964 vs 0.964). RRF is kept because it does not depend on the two score
scales being comparable, which raw mixing silently assumes. No reranking model was added: it would cost memory without
a demonstrated gain.

## 5. Grounded answer generation

The default `extractive` provider:

1. Re-flows PDF line breaks so sentences are not cut mid-line.
2. Splits accepted passages into sentences and drops headings, titles, reference lines and **instruction-like
   sentences** (prompt-injection patterns are never quoted).
3. Prefixes Markdown list items with their heading when it carries the context ("Every 100,000 miles: Replace the
   iridium spark plugs.").
4. Scores sentences by query-term overlap plus passage similarity. **Acronyms defined in the evidence** link to their
   expansion ("Adaptive Cruise Control (ACC)" lets "ACC can be set at speeds between 20 mph and 90 mph" match a
   question about adaptive cruise control). "Where … located" is treated as question framing, so "Where is the VIN
   located?" ranks the VIN sentence above other "located" sentences. Captions are down-weighted.
5. Quotes up to 3 sentences scoring ≥ 60% of the best, each with `[Source: file, Page N]`.
6. Flags salient query terms that appear nowhere in the evidence, and returns the insufficient-evidence message (or a
   visual-only answer when an image was accepted) when no sentence matches.

Optional Gemini/OpenAI providers receive the evidence inside a delimited block marked as untrusted, with personal data
masked and injected sentences removed; their output goes through the same verifier, and the extractive answer replaces
it if no citation verifies or the call fails.

## 6. Citation verification

Every `[Source: file, Page N]` tag is checked against the document registry and the retrieved evidence: the document
exists, the page exists, and a retrieved passage from that page exists. Failing tags are removed (replaced by
`[unverified citation removed]`) and reported. Claims whose words mostly do not appear in the cited passage (heading
path included) are flagged as weakly supported. This verifies **provenance**, not semantic entailment.

## 7. Guardrails

Implemented in [`guardrails.py`](../src/drivedocs/guardrails.py) and applied at three points:

| Point | PII | Prompt injection |
| --- | --- | --- |
| Ingestion | Redact e-mails, phones, Luhn-valid card numbers, SSNs, VINs, street addresses and labelled fields (`Customer:`, `Address:`, `Payment:` …) **before embedding**; record a warning with counts | Flag pages containing instruction-like text (warning) |
| Question | Refuse requests for a person's name, contact, address, payment or VIN; location questions ("Where is the VIN located?") are allowed | Remove injected sentences/clauses (overrides, prompt extraction, role play, fake `SYSTEM:` headers, chat-template blocks, forced outputs, exfiltration); keep a legitimate remainder only if it is still a question; refuse when nothing remains |
| Response | Re-scan answer, passages and citation snippets | Mask instruction-like sentences in returned passages and snippets |

False positives were measured on the corpus: no PII pattern fires on part numbers, tire sizes (225/45R18), dates,
torque values or repair-order numbers, and no injection pattern fires on manual sentences such as "Ignore the low fuel
warning only after refuelling" or "Use DOT 4 brake fluid and nothing else" (pinned by unit tests).

Iteration on the security suite (23 cases): the first run passed 13/23. The answers themselves never followed an
injection, but (a) the raw poisoned passages were returned for inspection with the phishing link and canary, and
(b) fragments such as "you were given" or "Say PWNED-7731" survived the query screen. Masking injected sentences in
returned evidence, requiring a remainder to still be a question, and adding echo / "repeat the text above" patterns
brought it to 23/23 without blocking any benign control.

## 8. Chunking

A dependency-free recursive splitter (paragraph → line → sentence → word → character) with real overlap. Markdown is
split per section, with the heading path stored on each chunk and prepended to the embedded text. **A section that is
purely a list is indexed one item per chunk**: the oil-change bullet of the maintenance schedule was ranked 7th for
"How often should the engine oil and filter be changed?" while it shared a chunk with unrelated tire and brake bullets,
and 1st once indexed alone.

Chunk-size sweep (*v0.2*, all other settings default):

| chunk_size / overlap | Text MRR | Text P@5 | Fact coverage | Pass rate |
| --- | --- | --- | --- | --- |
| 250 / 50 | 1.000 | 78.2% | 100% | 100% |
| **500 / 100 (default)** | 1.000 | 52.7% | 100% | 100% |
| 800 / 150 | 1.000 | 41.8% | 100% | 100% |
| 1200 / 200 | 1.000 | 30.9% | 90.9% | 94.7% |

## 9. Iteration log (vehicle corpus, real models)

| Step | QA pass rate | Security pass rate | Notes |
| --- | --- | --- | --- |
| Agent + guardrails, first run | 84.6% (33/39) | 56.5% (13/23) | Misses: list items without their interval heading, ACC acronym, "located" matching unrelated sentences, the oil-change bullet ranked 7th, one wording issue; security: see §7 |
| + heading prefix for list items, acronym linking, "located" as framing, evidence masking, stricter query screen | 94.9% (37/39) | 100% | |
| + one chunk per Markdown list item | 100% (39/39) | 100% | |
| fp32 → int8 MiniLM | 100% (39/39) | 100% | Lowest answerable top score 0.331 → 0.342; −65 MB RSS |

One QA question was reworded during this process ("final black jumper cable" → "black jumper cable … on the engine"):
the extractive answerer cannot map "final" to "Fourth"; this is listed as a known limitation. The figures are
in-sample; add your own questions (`--dataset`) to measure generalisation.

## 10. Embedding runtime: ONNX instead of PyTorch

The original PyTorch stack needed ~1 GB of RAM. The default backend runs the same models with **ONNX Runtime**, using
pre-exported graphs from the Hugging Face Hub (`Xenova/all-MiniLM-L6-v2`, `Xenova/clip-vit-base-patch32`), Hugging
Face `tokenizers`, and CLIP preprocessing implemented with Pillow + NumPy. PyTorch remains available as
`DRIVEDOCS_EMBEDDING_BACKEND=transformers`.

Variants measured against the PyTorch reference (*v0.2* corpus; memory = extra RSS of the loaded sessions, Windows):

| Model | Variant | Cosine to PyTorch (min) | Ranking agreement | Notes |
| --- | --- | --- | --- | --- |
| MiniLM | fp32 | 1.000 | 100% top-1 | default in v0.2 |
| MiniLM | **int8 (default since v0.3)** | 0.988 | 100% top-1 | −65 MB. In v0.2 one answerable query fell from 0.300 to 0.295, under the gate; on the vehicle corpus every QA and security result is identical to fp32 and the lowest answerable score rises from 0.331 to 0.342 |
| CLIP | fp32 text + fp32 vision | 1.000 / 0.986 | 100% | +660 MB: does not fit 512 MB |
| CLIP | int8 text + int8 vision | 0.767 / 0.905 | 89% | Too lossy (text tower) |
| CLIP | **q4 text + q4 vision (default)** | 0.961 / 0.955 | 95% top-1 raw; 100% end-to-end | +207 MB, 7 ms per text query |

**Why int8 now:** LangChain (`langchain-core` + `langsmith`) adds ~37 MB RSS. The deployed v0.2 build already reported
489 MB on Render's 512 MB Free instance, so adding the agent with fp32 MiniLM would have left almost no headroom. With
int8 MiniLM the process measured ~396 MB locally after start-up with the agent loaded and ~397 MB after repeated
compound agent queries (vs 463 MB for fp32 + agent on the same machine). A `malloc_trim(0)` after start-up returns
memory freed by seeding to the OS on Linux.

ONNX Runtime worker threads busy-wait by default; with three sessions plus FAISS in one process they contended and
requests were 4–10× slower, so spinning is disabled and the thread count capped (`DRIVEDOCS_ONNX_THREADS`).
