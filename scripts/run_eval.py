"""Run the DriveDocs evaluation benchmark against the bundled sample corpus.

Uses a temporary storage directory (never the application's data/storage) and the real embedding models configured
in configs/config.yaml. Writes reports/evaluation_report.json and reports/EVALUATION.md.

    python scripts/run_eval.py [--k 5] [--k-images 3] [--output-dir reports]
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

from drivedocs.config import settings  # noqa: E402
from drivedocs.evaluation import DriveDocsEvaluator, EvaluationQuery  # noqa: E402
from drivedocs.generation import GroundedMultimodalGenerator  # noqa: E402
from drivedocs.service import DriveDocsService  # noqa: E402
from drivedocs.storage import LocalStorage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="top-K text results")
    parser.add_argument("--k-images", type=int, default=3, help="top-K image results")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "sample" / "evaluation_set.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--provider", default="extractive", help="generation provider to evaluate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if settings.embedding_backend == "mock":
        print("Refusing to evaluate with the mock embedding backend; unset DRIVEDOCS_EMBEDDING_BACKEND.")
        return 2

    with tempfile.TemporaryDirectory(prefix="drivedocs_eval_") as tmp:
        settings.seed_sample_data = False
        service = DriveDocsService(storage=LocalStorage(root=Path(tmp)))
        service.initialize()
        print(f"[*] Ingesting {len(settings.sample_files)} sample files...")
        for name in settings.sample_files:
            result = service.ingest(name, (settings.sample_dir / name).read_bytes(), source="sample")
            doc = result.document
            print(f"    {name}: {doc.chunk_count} chunks, {doc.image_count} images, {doc.page_count} page(s)")

        queries = [EvaluationQuery(**item) for item in json.loads(args.dataset.read_text(encoding="utf-8"))]
        generator = GroundedMultimodalGenerator(provider=args.provider, verifier=service.verifier)
        evaluator = DriveDocsEvaluator(service.retriever, generator, k=args.k, k_images=args.k_images)
        print(f"[*] Running {len(queries)} queries...")
        dataset = args.dataset.resolve()
        dataset_label = dataset.relative_to(ROOT).as_posix() if dataset.is_relative_to(ROOT) else dataset.name
        report = evaluator.run_benchmark(queries, dataset_path=dataset_label)

    json_out = args.output_dir / "evaluation_report.json"
    md_out = args.output_dir / "EVALUATION.md"
    evaluator.write_reports(report, json_out, md_out)

    print("\n=========================== RESULTS ===========================")
    for key, value in report.summary.items():
        print(f"  {key:32s} {value}")
    for key, value in report.comparison.items():
        print(f"  {key:32s} {value}")
    print(f"  {'failing queries':32s} {[r.query_id for r in report.query_results if not r.passed]}")
    print(f"[+] Wrote {json_out.relative_to(ROOT)} and {md_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
