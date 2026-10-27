"""Run the DriveDocs evaluation: retrieval/answer benchmark + security suite.

Local (default): builds an in-process service on a temporary storage directory, ingests the demo corpus, runs the
benchmark, then ingests the security fixtures and runs the security suite.

Remote: evaluates a deployed instance over HTTP, which is how the LLM agent (whose API key exists only on the
server) is measured. The security fixtures are uploaded temporarily and deleted afterwards (--no-security skips this).

    python scripts/run_eval.py
    python scripts/run_eval.py --provider extractive
    python scripts/run_eval.py --base-url https://drivedocs-3xr4.onrender.com --output-dir reports/live
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drivedocs.config import settings  # noqa: E402
from drivedocs.evaluation import DriveDocsEvaluator, EvaluationQuery, SecurityQuery  # noqa: E402
from drivedocs.generation import GroundedMultimodalGenerator  # noqa: E402
from drivedocs.schemas import ChatResponse  # noqa: E402

SECURITY_DIR = ROOT / "data" / "security"


def _label(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


# Remote --------------------------------------------------------------------------------------------------------------


class RemoteClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: Optional[bytes] = None, ctype: Optional[str] = None) -> Any:
        req = urllib.request.Request(self.base + path, data=body, method=method,
                                     headers={"Content-Type": ctype} if ctype else {})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read() or b"null")
            except urllib.error.HTTPError as err:
                if err.code in (502, 503) and attempt < 3:
                    time.sleep(10)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < 3:
                    time.sleep(10)
                    continue
                raise

    def chat(self, query: str, k_text: int, k_images: int) -> ChatResponse:
        payload = {"query": query, "top_k_text": k_text, "top_k_images": k_images, "include_debug": True}
        data = self.request("POST", "/chat", json.dumps(payload).encode(), "application/json")
        return ChatResponse(**data)

    def upload(self, path: Path) -> Optional[str]:
        boundary = uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        try:
            data = self.request("POST", "/documents/upload", body, f"multipart/form-data; boundary={boundary}")
            return data["document"]["doc_id"]
        except urllib.error.HTTPError as err:
            if err.code == 409:
                detail = json.loads(err.read() or b"{}").get("detail", {})
                return (detail.get("existing_document") or {}).get("doc_id")
            raise

    def delete(self, doc_id: str) -> None:
        try:
            self.request("DELETE", f"/documents/{doc_id}")
        except urllib.error.HTTPError:
            pass


def remote_evaluator(args) -> tuple[DriveDocsEvaluator, RemoteClient]:
    client = RemoteClient(args.base_url)
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if client.request("GET", "/ready").get("ready"):
                break
        except Exception:
            pass
        time.sleep(5)
    info = client.request("GET", "/system/info")
    env = {
        "target": args.base_url,
        "text_model": info.get("text_model"),
        "image_model": info.get("image_model"),
        "image_search_enabled": info.get("image_search_enabled"),
        "provider": info.get("llm_provider"),
        "thresholds": info.get("thresholds"),
    }
    return DriveDocsEvaluator(client.chat, k=args.k, k_images=args.k_images, environment=env), client


# Main ----------------------------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="top-K text results")
    parser.add_argument("--k-images", type=int, default=3, help="top-K image results")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "sample" / "evaluation_set.json")
    parser.add_argument("--security-dataset", type=Path, default=SECURITY_DIR / "security_eval_set.json")
    parser.add_argument("--no-security", action="store_true", help="skip the security suite")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--provider", default=None, help="local only: answer provider (default: configured)")
    parser.add_argument("--base-url", default=None, help="evaluate a deployed instance instead of a local one")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    queries = [EvaluationQuery(**item) for item in json.loads(args.dataset.read_text(encoding="utf-8"))]
    security_spec = json.loads(args.security_dataset.read_text(encoding="utf-8"))
    security_queries = [SecurityQuery(**q) for q in security_spec["queries"]]

    if args.base_url:
        evaluator, client = remote_evaluator(args)
        print(f"[*] Evaluating {args.base_url} (provider: {evaluator.environment.get('provider')})")
        report = evaluator.run_benchmark(queries, dataset_path=_label(args.dataset))
        if not args.no_security:
            uploaded: List[str] = []
            try:
                for name in security_spec["documents"]:
                    doc_id = client.upload(SECURITY_DIR / name)
                    if doc_id:
                        uploaded.append(doc_id)
                report.security = evaluator.run_security(security_queries, dataset_path=_label(args.security_dataset))
            finally:
                for doc_id in uploaded:
                    client.delete(doc_id)
                print(f"[*] Removed {len(uploaded)} temporary security fixture(s) from the deployment")
    else:
        if settings.embedding_backend == "mock":
            print("Refusing to evaluate with the mock embedding backend; unset DRIVEDOCS_EMBEDDING_BACKEND.")
            return 2
        from drivedocs.service import DriveDocsService
        from drivedocs.storage import LocalStorage

        with tempfile.TemporaryDirectory(prefix="drivedocs_eval_") as tmp:
            settings.seed_sample_data = False
            service = DriveDocsService(storage=LocalStorage(root=Path(tmp)))
            service.initialize()
            print(f"[*] Ingesting {len(settings.sample_files)} sample files...")
            for name in settings.sample_files:
                doc = service.ingest(name, (settings.sample_dir / name).read_bytes(), source="sample").document
                print(f"    {name}: {doc.chunk_count} chunks, {doc.image_count} images, {doc.page_count} page(s)")
            generator = None
            if args.provider:
                generator = GroundedMultimodalGenerator(
                    provider=args.provider, verifier=service.verifier, agent=service.agent,
                    vocabulary=service.corpus_vocabulary,
                )
            evaluator = DriveDocsEvaluator.for_service(service, k=args.k, k_images=args.k_images, generator=generator)
            print(f"[*] Running {len(queries)} benchmark queries (provider: {evaluator.environment['provider']})...")
            report = evaluator.run_benchmark(queries, dataset_path=_label(args.dataset))
            if not args.no_security:
                for name in security_spec["documents"]:
                    service.ingest(name, (SECURITY_DIR / name).read_bytes())
                print(f"[*] Running {len(security_queries)} security queries...")
                report.security = evaluator.run_security(security_queries, dataset_path=_label(args.security_dataset))

    json_out = args.output_dir / "evaluation_report.json"
    md_out = args.output_dir / "EVALUATION.md"
    evaluator.write_reports(report, json_out, md_out)

    print("\n=========================== RESULTS ===========================")
    for key, value in {**report.summary, **report.comparison}.items():
        print(f"  {key:32s} {value}")
    if report.security:
        for key, value in report.security["summary"].items():
            print(f"  {key:32s} {value}")
        print(f"  {'security failures':32s} {[r['query_id'] for r in report.security['results'] if not r['passed']]}")
    print(f"  {'failing queries':32s} {[r.query_id for r in report.query_results if not r.passed]}")
    print(f"[+] Wrote {_label(json_out)} and {_label(md_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
