# DriveDocs Evaluation & Benchmark Report

**Generated**: 2026-10-25T07:09:27.761209 UTC  
**Environment**: CPU-compatible execution (`all-MiniLM-L6-v2` & `openai/clip-vit-base-patch32`)  
**Generation Provider**: Extractive Grounded Synthesizer (Zero-API-key deterministic fallback)

---

## 1. Executive Summary

| Metric | Measured Value | Target / Benchmark Standard | Interpretation |
| :--- | :---: | :---: | :--- |
| **Total Test Queries** | `8` | Full test battery | Covers text, multi-page, visual, and negative queries |
| **Retrieval Recall@5** | `100.0%` | >= 80% | High hit-rate for true relevant source documents |
| **Mean Reciprocal Rank (MRR)** | `1.000` | >= 0.75 | Relevant evidence ranks #1 or #2 in retrieved list |
| **Retrieval Precision@5** | `63.3%` | >= 60% | Signal-to-noise ratio in top 5 retrieved items |
| **Citation Correctness** | `87.5%` | >= 90% | Exact document filename and page number alignment |
| **Groundedness Score** | `91.7%` | >= 85% | Faithfulness to source facts without hallucination |
| **Negative Query Handling** | `100.0%` | 100% | Correctly triggers insufficient evidence fallback |
| **Mean Query Latency** | `35.0 ms` | < 500 ms (CPU) | Rapid FAISS cosine search & extraction |

---

## 2. Granular Query Results

| Query ID | Question | Query Type | Top Score | Recall@5 | MRR | Citation Acc | Latency | Result |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `q1_waveguide_loss` | What is the optical transmission loss of the silicon nitride waveguide bus? | In-Corpus Factual | 0.53 | 1.00 | 1.00 | 100% | 67.2ms | PASS |
| `q2_detector_efficiency` | What is the peak quantum efficiency of the SNSPD single-photon detector and at what wavelength? | In-Corpus Factual | 0.54 | 1.00 | 1.00 | 100% | 31.8ms | PASS |
| `q3_agent_rate_limit` | What is the maximum API throughput rate limit for agent workers? | In-Corpus Factual | 0.68 | 1.00 | 1.00 | 100% | 28.4ms | PASS |
| `q4_raft_heartbeat` | What is the periodic heartbeat interval dispatched by the Raft leader? | In-Corpus Factual | 0.53 | 1.00 | 1.00 | 100% | 28.6ms | PASS |
| `q5_visual_architecture` | Find the architecture diagram showing Content Parser, Vector Store, and Grounded QA. | In-Corpus Factual | 0.28 | 1.00 | 1.00 | 0% | 33.8ms | PASS |
| `q6_visual_efficiency_curve` | Show the calibrated quantum efficiency curve versus wavelength chart. | In-Corpus Factual | 0.39 | 1.00 | 1.00 | 100% | 28.1ms | PASS |
| `q7_unsupported_apple_stock` | What was the stock price of Apple Computer Inc on January 15, 1999? | Negative / Out-of-Corpus | 0.22 | 1.00 | 1.00 | 100% | 29.3ms | PASS |
| `q8_unsupported_baking_recipe` | What is the traditional culinary recipe and oven temperature for baking chocolate chip cookies? | Negative / Out-of-Corpus | 0.21 | 1.00 | 1.00 | 100% | 32.7ms | PASS |

---

## 3. Evaluation Methodology

1. **Retrieval Recall@5**: Evaluates whether the ground-truth document containing the target answer was retrieved within the top 5 candidates.
2. **Mean Reciprocal Rank (MRR)**: Evaluates the rank position of the first correct evidence item ($1 / \text{rank}$).
3. **Citation Correctness**: Compares every cited `[Source: filename, Page X]` against true source metadata to verify that citations are never fabricated.
4. **Groundedness & Completeness Rubric**: Checks that generated statements contain explicit factual tokens from source evidence.
5. **Negative Query Rejection**: Proves that unsupported questions produce an explicit:  
   `"I cannot find sufficient evidence in the provided documents to answer this question."`
