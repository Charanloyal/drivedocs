# Evaluation Methodology

`scripts/run_eval.py` measures DriveDocs end-to-end: real embedding models, the real retriever, the real answer
generator and the real citation verifier. It writes:

* `reports/evaluation_report.json`: machine-readable results (served by `GET /evaluation` and shown in the UI);
* `reports/EVALUATION.md`: a human-readable summary with per-query rows and failing queries.

No number in the README or UI is typed by hand; all are copied from these generated files.

## Running it

```bash
python scripts/run_eval.py                 # K=5 text results, K=3 image results, extractive provider
python scripts/run_eval.py --k 3 --k-images 2
python scripts/run_eval.py --provider openai   # requires OPENAI_API_KEY; evaluates the LLM path + verifier
```

The script ingests the four bundled sample files into a **temporary** storage directory (your library is never
touched), runs a warm-up query (excluded from latency), then evaluates every query in
[`data/sample/evaluation_set.json`](../data/sample/evaluation_set.json). It refuses to run with the mock embedding
backend.

## Dataset

19 hand-written queries over the 4 sample files (a 2-page PDF with an embedded chart, a Markdown file, a text file
and a standalone PNG diagram):

| Type | Count | What a correct result looks like |
| --- | --- | --- |
| Text | 11 | Relevant (file, page) retrieved; answer contains the expected fact; citations point at that page |
| Visual | 3 | The relevant image (file, page) is retrieved by CLIP and attached to the answer |
| Negative | 3 | Clearly out-of-corpus; must be refused with no citations |
| Hard negative | 2 | Near-domain but unanswerable (e.g. "Paxos" when only Raft is described); must be refused **or** explicitly flagged as "not mentioned" |

Each query lists `relevant_filenames`, `relevant_pages` and (for text queries) `expected_facts`.

## Metrics

| Metric | Definition |
| --- | --- |
| Text Recall@K | Text queries: share of the relevant files that appear in the top-K text chunks on a relevant page (`relevant_pages` apply to every listed file); averaged |
| Text Precision@K | Share of the top-K text chunks with a relevant (file, page). Single-page answers in a small corpus cap this well below 1, and it rises mechanically with smaller chunks |
| Text MRR | Mean of 1 / rank of the first relevant text chunk |
| Image Recall@K / MRR | Same, over the top-K CLIP results, for visual queries |
| Hybrid document MRR | 1 / rank of the first relevant item in the fused (weighted RRF) evidence list; compared with the previous evaluation's raw-score ordering |
| Citation precision | Answers that cite something: share of cited (file, page) pairs that are relevant (for visual-only answers, the returned images) |
| Citation verification rate | Citation tags that passed `CitationVerifier` (document exists, page exists, passage retrieved from that page) |
| Cited-claim lexical support | Mean content-word overlap between each cited claim and its passage. Extractive answers quote passages, so this is ~1.0 by construction; it is informative for LLM providers |
| Fact coverage | Share of `expected_facts` strings found in the answer (0 if refused). A recall proxy, **not** a groundedness proof |
| Answer rate | Answerable queries that were not refused |
| Negative rejection | Out-of-corpus queries refused with no citations |
| Hard-negative handling | Hard negatives refused, or answered with the explicit "do not mention" flag |
| Pass rate | Per query: answered, relevant evidence retrieved, citation precision ≥ 0.5, and fact coverage ≥ 0.5 (when facts are defined); negatives pass when handled as above |
| Latency mean / p50 / p95 | Wall-clock retrieve + generate per query, warm models, single process, the machine that ran the script |

## Limitations (read before quoting numbers)

* **Small, author-written dataset.** 19 queries over 4 synthetic files written by the same project. Several
  retrieval/generation fixes were made after observing failures on this set (see [RAG_DESIGN.md](RAG_DESIGN.md)), so
  the current 100% pass rate is an in-sample result. It shows the system works on the demo corpus. It is not a
  general benchmark of retrieval quality.
* **No human or model-graded groundedness.** Groundedness is approximated by citation verification, lexical support
  and expected-fact matching. Paraphrased but wrong statements from an LLM provider could pass lexical checks.
* **Thresholds were calibrated on the same corpus.** MiniLM/CLIP score distributions differ across domains;
  recalibrate `retrieval.*` thresholds for your documents (the `/search/text` endpoint returns raw scores).
* **Latency** excludes network transfer and model loading and depends on hardware.
