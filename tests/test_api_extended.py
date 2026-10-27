"""HTTP API behaviour with FastAPI's TestClient (isolated storage, mock embeddings)."""

from __future__ import annotations

import io

from PIL import Image


def _png_bytes(color=(0, 0, 255), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_details_search_delete_flow(isolated_client):
    c = isolated_client
    r = c.post("/documents/upload", files={"file": ("guide.md", b"# Guide\n\n## Ports\n\nThe API listens on port 8443.", "text/markdown")})
    assert r.status_code == 201
    doc = r.json()["document"]
    assert doc["status"] == "ready" and doc["chunk_count"] == 1

    details = c.get(f"/documents/{doc['doc_id']}").json()
    assert details["chunks"][0]["section"] == "Guide > Ports"

    hits = c.post("/search/text", json={"query": "API port", "top_k": 3}).json()
    assert hits[0]["chunk"]["doc_id"] == doc["doc_id"]

    answer = c.post("/chat", json={"query": "Which port does the API listen on?"}).json()
    assert "8443" in answer["answer"]
    assert answer["citations"][0]["page_number"] == 1 and answer["citations"][0]["verified"]

    assert c.delete(f"/documents/{doc['doc_id']}").status_code == 200
    assert c.get(f"/documents/{doc['doc_id']}").status_code == 404
    assert c.get("/documents").json()["total_documents"] == 0


def test_error_format_is_consistent_and_safe(isolated_client):
    c = isolated_client
    for resp, status, code in [
        (c.post("/documents/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}), 400, "unsupported_file_type"),
        (c.post("/documents/upload", files={"file": ("e.txt", b"", "text/plain")}), 422, "malformed_document"),
        (c.post("/chat", json={"query": "   "}), 422, "validation_error"),
        (c.get("/documents/doc_ffffffffffff"), 404, "not_found"),
        (c.post("/search/images", data={"top_k": "2"}), 400, "missing_query"),
    ]:
        assert resp.status_code == status, resp.text
        body = resp.json()["detail"]
        assert body["code"] == code and body["message"]
        assert "Traceback" not in resp.text and "\\Users\\" not in resp.text and "/tmp/" not in resp.text


def test_corrupt_query_image_returns_422(isolated_client):
    r = isolated_client.post("/search/images", files={"image_file": ("q.png", b"\x89PNG\r\n\x1a\nnot really", "image/png")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_image"


def test_image_upload_search_and_safe_serving(isolated_client):
    c = isolated_client
    up = c.post("/documents/upload", files={"file": ("blue.png", _png_bytes(), "image/png")}).json()
    image_id = f"img_{up['document']['doc_id']}"

    res = c.post("/search/images", files={"image_file": ("q.png", _png_bytes((0, 0, 250)), "image/png")}, data={"top_k": "1"})
    assert res.status_code == 200
    hit = res.json()[0]
    assert hit["image"]["image_id"] == image_id and "image_path" not in hit["image"]

    served = c.get(hit["image"]["url_path"])
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    assert served.headers["x-content-type-options"] == "nosniff"

    for bad in ["../documents.json", "..%2Fdocuments.json", "img_doc_000000000000.png", "img_x.png.exe", "documents.json"]:
        assert c.get(f"/static/images/{bad}").status_code == 404


def test_body_size_limit(isolated_client, monkeypatch):
    import drivedocs.api as api

    too_big = b"a" * (api.settings.max_upload_size_bytes + api.MULTIPART_OVERHEAD + 10)
    r = isolated_client.post("/documents/upload", files={"file": ("big.txt", too_big, "text/plain")})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "file_too_large"


def test_body_size_limit_without_content_length(isolated_client):
    import drivedocs.api as api

    limit = api.settings.max_upload_size_bytes + api.MULTIPART_OVERHEAD

    def chunks():
        sent = 0
        while sent <= limit:
            yield b"x" * 1_048_576
            sent += 1_048_576

    r = isolated_client.post("/chat", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "file_too_large"


def test_status_endpoints(isolated_client):
    c = isolated_client
    health = c.get("/health").json()
    assert health["status"] == "healthy" and health["ready"] is True
    ready = c.get("/ready")
    assert ready.status_code == 200 and ready.json()["components"]["registry"]["ready"]
    info = c.get("/system/info").json()
    assert info["storage_mode"] == "ephemeral" and "lost" in info["storage_notice"]
    assert info["authentication"] is False
    assert ".pdf" in info["allowed_extensions"]


def test_evaluation_endpoint_serves_report(isolated_client, tmp_path, monkeypatch):
    from drivedocs.config import settings

    monkeypatch.setattr(settings, "reports_dir", tmp_path / "none")
    assert isolated_client.get("/evaluation").status_code == 404
    (tmp_path / "rep").mkdir()
    (tmp_path / "rep" / "evaluation_report.json").write_text('{"summary": {"text_mrr": 1.0}}')
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "rep")
    assert isolated_client.get("/evaluation").json()["summary"]["text_mrr"] == 1.0


def test_dashboard_security_headers(isolated_client):
    r = isolated_client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    assert r.headers["x-frame-options"] == "DENY"


def test_cors_is_disabled_by_default(isolated_client):
    r = isolated_client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_health_fails_when_startup_fails(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import drivedocs.api as api
    from drivedocs.service import DriveDocsService
    from drivedocs.storage import LocalStorage

    def broken_load(self):
        raise OSError("model weights missing")

    monkeypatch.setattr(DriveDocsService, "_load_state", broken_load)
    monkeypatch.setattr(api, "create_service", lambda: DriveDocsService(storage=LocalStorage(root=tmp_path / "s")))
    with TestClient(api.app) as client:
        client.app.state.service.wait_ready(30)
        health = client.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "unhealthy"
        assert health.json()["phase"] == "failed"
        assert client.get("/ready").status_code == 503
        chat = client.post("/chat", json={"query": "anything"})
        assert chat.status_code == 503 and chat.json()["detail"]["code"] == "service_unavailable"


def test_failed_upload_is_listed(isolated_client):
    c = isolated_client
    r = c.post("/documents/upload", files={"file": ("broken.pdf", b"%PDF-1.7 nothing else", "application/pdf")})
    assert r.status_code == 422
    docs = c.get("/documents").json()
    assert docs["failed_documents"] == 1
    assert docs["documents"][0]["status"] == "failed"
