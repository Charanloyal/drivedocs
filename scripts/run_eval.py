"""Run the DriveDocs evaluation: Recall@5 and groundedness on the curated QA set, plus the security suite.

Uses temporary storage (never the application's data/storage) and the real embedding models configured in
configs/config.yaml. Writes reports/evaluation_report.json and reports/EVALUATION.md.

    python scripts/run_eval.py [--k 5] [--k-images 5] [--mode agent|direct] [--provider extractive]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drivedocs.agent import DriveDocsAgent  # noqa: E402
from drivedocs.config import settings  # noqa: E402
from drivedocs.evaluation import (  # noqa: E402
    DriveDocsEvaluator,
    EvaluationQuery,
    SecurityCase,
    run_security_suite,
)
from drivedocs.generation import GroundedMultimodalGenerator  # noqa: E402
from drivedocs.service import DriveDocsService  # noqa: E402
from drivedocs.storage import LocalStorage  # noqa: E402


def _label(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def _service(root: Path, extra: list[Path]) -> DriveDocsService:
    service = DriveDocsService(storage=LocalStorage(root=root))
    service.initialize()
    for path in [settings.sample_dir / n for n in settings.sample_files] + extra:
        doc = service.ingest(path.name, path.read_bytes(), source="sample").document
        print(f"    {path.name}: {doc.chunk_count} chunks, {doc.image_count} images, {doc.page_count} page(s)"
              + (f"  [{'; '.join(w[:70] for w in doc.warnings)}]" if doc.warnings else ""))
    return service


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="top-K text results for retrieval metrics (Recall@K)")
    parser.add_argument("--k-images", type=int, default=5, help="top-K image results for retrieval metrics")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "sample" / "evaluation_set.json")
    parser.add_argument("--security-dataset", type=Path,
                        default=ROOT / "data" / "security" / "security_eval_set.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--provider", default="extractive", help="generation provider to evaluate")
    parser.add_argument("--mode", choices=["agent", "direct"], default="agent", help="answer path to evaluate")
    parser.add_argument("--skip-security", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if settings.embedding_backend == "mock":
        print("Refusing to evaluate with the mock embedding backend; unset DRIVEDOCS_EMBEDDING_BACKEND.")
        return 2
    settings.seed_sample_data = False

    with tempfile.TemporaryDirectory(prefix="drivedocs_eval_") as tmp:
        print(f"[*] Ingesting {len(settings.sample_files)} sample files...")
        service = _service(Path(tmp) / "qa", [])
        generator = GroundedMultimodalGenerator(provider=args.provider, verifier=service.verifier)
        agent = DriveDocsAgent(service.retriever, generator) if args.mode == "agent" else None
        evaluator = DriveDocsEvaluator(service.retriever, generator, k=args.k, k_images=args.k_images, agent=agent)
        queries = [EvaluationQuery(**item) for item in json.loads(args.dataset.read_text(encoding="utf-8"))]
        print(f"[*] Running {len(queries)} QA queries ({args.mode} answer path)...")
        report = evaluator.run_benchmark(queries, dataset_path=_label(args.dataset))

        if not args.skip_security:
            spec = json.loads(args.security_dataset.read_text(encoding="utf-8"))
            extra = [args.security_dataset.parent / name for name in spec.get("extra_documents", [])]
            print(f"[*] Ingesting the corpus plus {len(extra)} poisoned document(s) for the security suite...")
            sec_service = _service(Path(tmp) / "security", extra)
            sec_agent = DriveDocsAgent(
                sec_service.retriever,
                GroundedMultimodalGenerator(provider=args.provider, verifier=sec_service.verifier),
            )
            cases = [SecurityCase(**c) for c in spec["cases"]]
            print(f"[*] Running {len(cases)} security cases...")
            report.security = run_security_suite(
                sec_agent, cases, pii_values=spec.get("pii_values"), canaries=spec.get("canaries"),
                dataset_path=_label(args.security_dataset),
            )

    json_out = args.output_dir / "evaluation_report.json"
    md_out = args.output_dir / "EVALUATION.md"
    evaluator.write_reports(report, json_out, md_out)

    print("\n=========================== RESULTS ===========================")
    for key, value in report.summary.items():
        print(f"  {key:32s} {value}")
    for key, value in report.comparison.items():
        print(f"  {key:32s} {value}")
    print(f"  {'failing queries':32s} {[r.query_id for r in report.query_results if not r.passed]}")
    if report.security:
        print("--------------------------- SECURITY ---------------------------")
        for key, value in report.security.summary.items():
            print(f"  {key:32s} {value}")
        print(f"  {'failing cases':32s} {[r.case_id for r in report.security.results if not r.passed]}")
    print(f"[+] Wrote {json_out.relative_to(ROOT)} and {md_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
