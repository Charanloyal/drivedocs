"""Smoke-test a running DriveDocs AI instance (local or deployed). Standard library only.

    python scripts/smoke_test.py http://127.0.0.1:8000
    python scripts/smoke_test.py https://<your-service>.onrender.com

Read-only except for the invalid-upload checks, which are all rejected by the server (nothing is stored).
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
READY_TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
results: List[Tuple[bool, str, str]] = []


def request(method: str, path: str, body: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None,
            timeout: float = 120.0) -> Tuple[int, Dict[str, str], bytes]:
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read()


def get_json(path: str) -> Tuple[int, Any]:
    status, _, raw = request("GET", path)
    return status, json.loads(raw or b"null")


def post_json(path: str, payload: dict) -> Tuple[int, Any]:
    status, _, raw = request("POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return status, json.loads(raw or b"null")


def multipart(fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, ctype) in files.items():
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                  f"Content-Type: {ctype}\r\n\r\n".encode())
        buf.write(data + b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def post_form(path: str, fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[int, Any]:
    body, ctype = multipart(fields, files)
    status, _, raw = request("POST", path, body, {"Content-Type": ctype})
    try:
        return status, json.loads(raw or b"null")
    except json.JSONDecodeError:
        return status, raw[:200]


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((bool(ok), name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return bool(ok)


def main() -> int:
    print(f"Smoke-testing {BASE}\n")
    # Free Render instances can take ~1 minute to spin up; retry until the port answers.
    deadline = time.time() + READY_TIMEOUT
    while True:
        try:
            t0 = time.perf_counter()
            status, health = get_json("/health")
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if time.time() > deadline:
                check("GET /health reachable", False, str(exc))
                return 1
            time.sleep(3)
    check("GET /health returns 200", status == 200 and health.get("status") == "healthy",
          f"{(time.perf_counter() - t0) * 1000:.0f} ms, ready={health.get('ready') if isinstance(health, dict) else '?'}")

    deadline = time.time() + READY_TIMEOUT
    ready: Any = None
    while time.time() < deadline:
        status, ready = get_json("/ready")
        if status == 200 and ready.get("ready"):
            break
        time.sleep(5)
    check("GET /ready reports ready", status == 200 and ready and ready.get("ready"),
          f"phase={ready.get('phase') if isinstance(ready, dict) else '?'}, rss={ready.get('memory_rss_mb') if isinstance(ready, dict) else '?'} MB")

    status, headers, html = request("GET", "/")
    check("GET / serves the web UI", status == 200 and b"DriveDocs" in html and "text/html" in headers.get("content-type", ""))
    check("UI has a strict Content-Security-Policy", "script-src 'self'" in headers.get("content-security-policy", ""))
    status, _, js = request("GET", "/assets/app.js")
    check("UI script asset loads", status == 200 and b"renderAnswerText" in js)

    status, info = get_json("/system/info")
    check("GET /system/info", status == 200, f"storage={info.get('storage_mode')}, image_search={info.get('image_search_enabled')}")

    status, docs = get_json("/documents")
    names = {d["filename"]: d for d in docs.get("documents", [])} if status == 200 else {}
    check("GET /documents lists the sample corpus", status == 200 and "aurora_owners_manual.pdf" in names,
          f"{docs.get('total_documents')} documents, {docs.get('total_chunks')} chunks, {docs.get('total_images')} images")

    status, hits = post_json("/search/text", {"query": "rear window defroster fuse", "top_k": 3})
    check("POST /search/text finds the right page", status == 200 and hits
          and hits[0]["chunk"]["filename"] == "aurora_owners_manual.pdf" and hits[0]["chunk"]["page_number"] == 5,
          f"top={hits[0]['chunk']['filename']} p{hits[0]['chunk']['page_number']} score={hits[0]['score']:.2f}" if status == 200 and hits else str(hits)[:120])

    status, chat = post_json("/chat", {"query": "What is the recommended cold tire pressure for the rear tires?"})
    citations = chat.get("citations", []) if status == 200 else []
    check("POST /chat answers a sample question", status == 200 and "2.7 bar" in chat.get("answer", ""),
          f"{chat.get('answer', '')[:90]}..." if status == 200 else str(chat)[:120])
    real = all(c["filename"] in names and 1 <= (c.get("page_number") or 0) <= names[c["filename"]]["page_count"] for c in citations)
    check("Citations point to real documents and pages", bool(citations) and real and chat["citation_check"]["all_verified"],
          ", ".join(f"{c['filename']} p{c['page_number']}" for c in citations))
    agent_steps = len(chat.get("agent_trace") or []) if status == 200 else 0
    check("Answer provider reported", status == 200 and bool(chat.get("provider")),
          f"provider={chat.get('provider')}, agent steps={agent_steps}, LLM configured={info.get('llm_configured')}")

    status, pii = post_json("/chat", {"query": "What is the customer's email and phone number on the service record?"})
    raw = json.dumps(pii)
    check("PII is masked in answers and passages", status == 200 and "jane.example@example.com" not in raw
          and "010-2345" not in raw and pii.get("pii_redactions", 0) > 0, f"masked={pii.get('pii_redactions')}")

    status, neg = post_json("/chat", {"query": "Who won the 2018 FIFA World Cup final?"})
    check("Negative question is refused without citations", status == 200 and neg.get("answer_type") == "insufficient_evidence"
          and not neg.get("citations"), neg.get("answer", "")[:90] if status == 200 else str(neg)[:120])

    if info.get("image_search_enabled"):
        status, imgs = post_form("/search/images", {"query": "fuse box layout diagram", "top_k": "2"}, {})
        check("POST /search/images (text -> image)", status == 200 and imgs and imgs[0]["image"]["filename"] == "aurora_owners_manual.pdf"
              and imgs[0]["image"]["page_number"] == 5,
              f"top={imgs[0]['image']['filename']} p{imgs[0]['image']['page_number']} score={imgs[0]['score']:.2f}" if status == 200 and imgs else str(imgs)[:120])
        if status == 200 and imgs:
            s, h, png = request("GET", imgs[0]["image"]["url_path"])
            check("Image URL serves a PNG", s == 200 and png.startswith(b"\x89PNG"))
            status, sim = post_form("/search/images", {"top_k": "1"}, {"image_file": ("query.png", png, "image/png")})
            check("POST /search/images (image -> image)", status == 200 and sim and sim[0]["score"] > 0.95,
                  f"score={sim[0]['score']:.3f}" if status == 200 and sim else str(sim)[:120])
        status, vis = post_json("/chat", {"query": "Show the front compartment diagram with the 12V battery and coolant"})
        check("Chat returns visual evidence for a diagram request", status == 200 and any(
            i["filename"] == "front_compartment_diagram.png" for i in vis.get("images", [])), vis.get("answer_type", ""))
    else:
        check("Image search disabled on this deployment (skipped)", True)

    for label, files, expected in [
        ("unsupported extension", {"file": ("tool.exe", b"MZ\x90\x00", "application/octet-stream")}, 400),
        ("empty file", {"file": ("empty.txt", b"", "text/plain")}, 422),
        ("fake PDF", {"file": ("fake.pdf", b"not a pdf", "application/pdf")}, 400),
        ("binary disguised as text", {"file": ("blob.txt", b"\x00\x01\x02\x00binary", "text/plain")}, 400),
    ]:
        status, body = post_form("/documents/upload", {}, files)
        code = body.get("detail", {}).get("code") if isinstance(body, dict) else None
        check(f"Invalid upload rejected: {label}", status == expected and code, f"HTTP {status} {code}")

    status, _, raw = request("GET", "/static/images/..%2Fdocuments.json")
    check("Path traversal on image route is blocked", status == 404)
    status, report = get_json("/evaluation")
    check("GET /evaluation serves the committed report", status == 200 and report.get("dataset", {}).get("size"),
          f"{report.get('dataset', {}).get('size')} queries, pass_rate={report.get('summary', {}).get('pass_rate')}")
    status, _, docs_html = request("GET", "/docs")
    check("API docs available at /docs", status == 200 and b"swagger" in docs_html.lower())

    failed = [name for ok, name, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("Failed: " + "; ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
