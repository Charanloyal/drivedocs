"""Script to run the full multimodal retrieval and groundedness evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from drivedocs.config import settings
from drivedocs.evaluation import DriveDocsEvaluator, EvaluationQuery
from drivedocs.generation import GroundedMultimodalGenerator
from drivedocs.ingestion import DocumentIngestionManager
from drivedocs.retrieval import MultimodalRetriever
from drivedocs.text_index import TextFAISSIndex
from drivedocs.image_index import ImageFAISSIndex


def main():
    print("=====================================================================")
    print("           DRIVEDOCS MULTIMODAL RETRIEVAL & QA BENCHMARK             ")
    print("=====================================================================")

    sample_dir = Path("data/sample")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Components
    print("\n[*] Initializing Vector Indexes and Models...")
    text_index = TextFAISSIndex()
    image_index = ImageFAISSIndex()
    retriever = MultimodalRetriever(text_index=text_index, image_index=image_index)
    ingestion_mgr = DocumentIngestionManager()
    generator = GroundedMultimodalGenerator(provider="extractive")

    # 2. Ingest Sample Documents
    sample_files = [
        sample_dir / "quantum_photonics_spec.pdf",
        sample_dir / "agent_safety_protocol.md",
        sample_dir / "distributed_systems_notes.txt",
        sample_dir / "architecture_diagram.png",
    ]

    print("\n[*] Ingesting sample documents into DriveDocs...")
    for file_path in sample_files:
        if file_path.exists():
            parsed = ingestion_mgr.ingest_file(file_path, allow_duplicates=True)
            retriever.index_parsed_document(parsed)
            print(f"  [+] Ingested '{file_path.name}': {len(parsed.chunks)} chunks, {len(parsed.images)} images")
        else:
            print(f"  [!] Missing file: {file_path}")

    print(f"\n[*] Total Indexed Corpus: {text_index.count()} text chunks, {image_index.count()} visual items.")

    # 3. Load Evaluation Set
    eval_json_path = sample_dir / "evaluation_set.json"
    with open(eval_json_path, "r", encoding="utf-8") as f:
        raw_eval_data = json.load(f)

    eval_queries = [EvaluationQuery(**item) for item in raw_eval_data]
    print(f"[*] Loaded {len(eval_queries)} benchmark queries from {eval_json_path.name}.")

    # 4. Run Benchmark
    print("\n[*] Running evaluation queries...")
    evaluator = DriveDocsEvaluator(retriever=retriever, generator=generator)
    report = evaluator.run_benchmark(eval_queries)

    # 5. Save Reports
    json_out = reports_dir / "evaluation_report.json"
    md_out = reports_dir / "EVALUATION.md"
    evaluator.write_reports(report, json_out, md_out)

    # 6. Display Summary
    print("\n=====================================================================")
    print("                       BENCHMARK RESULTS                             ")
    print("=====================================================================")
    print(f" Total Benchmark Queries      : {report.total_queries}")
    print(f" Retrieval Recall@5           : {report.mean_recall_at_5 * 100:.1f}%")
    print(f" Mean Reciprocal Rank (MRR)   : {report.mean_mrr:.4f}")
    print(f" Retrieval Precision@5        : {report.mean_precision_at_5 * 100:.1f}%")
    print(f" Citation Correctness         : {report.mean_citation_correctness * 100:.1f}%")
    print(f" Groundedness Score           : {report.mean_groundedness_score * 100:.1f}%")
    print(f" Negative Query Handling      : {report.unsupported_detection_accuracy * 100:.1f}%")
    print(f" Mean Query Latency           : {report.avg_latency_ms:.1f} ms")
    print("=====================================================================")
    print(f"[+] Output written to: {json_out} and {md_out}")


if __name__ == "__main__":
    main()
