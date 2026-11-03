"""Abuse protection for the public demo: rate limiting, request IDs and read-only sample documents."""

from __future__ import annotations

import pytest

from drivedocs.ratelimit import RateLimiter, bucket_for, client_id
from drivedocs.service import ProtectedDocumentError


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_sliding_window_allows_limit_then_blocks_until_window_passes():
    clock = FakeClock()
    limiter = RateLimiter({"chat": 3}, window=60, clock=clock)
    assert [limiter.check("chat", "a") for _ in range(3)] == [None, None, None]
    retry = limiter.check("chat", "a")
    assert retry == pytest.approx(60)
    assert limiter.check("chat", "b") is None  # other clients are unaffected
    assert limiter.check("search", "a") is None  # unconfigured buckets are unlimited
    clock.now += 30
    assert limiter.check("chat", "a") == pytest.approx(30)
    clock.now += 31
    assert limiter.check("chat", "a") is None


def test_limiter_memory_is_bounded():
    limiter = RateLimiter({"chat": 1}, max_clients=50)
    for i in range(500):
        limiter.check("chat", f"client-{i}")
    assert len(limiter._hits) == 50


@pytest.mark.parametrize("method,path,bucket", [
    ("POST", "/chat", "chat"),
    ("POST", "/search/text", "search"),
    ("POST", "/search/images", "search"),
    ("POST", "/documents/upload", "upload"),
    ("DELETE", "/documents/doc_abc", "delete"),
    ("GET", "/documents", None),
    ("GET", "/chat", None),
    ("GET", "/health", None),
])
def test_bucket_mapping(method, path, bucket):
    assert bucket_for(method, path) == bucket


def test_client_id_only_trusts_proxy_headers_when_configured():
    headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1", "cf-connecting-ip": "198.51.100.2"}
    assert client_id(headers, "10.0.0.1", trust_proxy_headers=False) == "10.0.0.1"
    assert client_id(headers, "10.0.0.1", trust_proxy_headers=True) == "198.51.100.2"
    assert client_id({"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, "10.0.0.1", True) == "203.0.113.7"
    assert client_id({}, None, True) == "unknown"


@pytest.fixture
def limited_client(isolated_client, monkeypatch):
    import drivedocs.api as api
    from drivedocs.config import settings

    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(api, "rate_limiter", RateLimiter({"chat": 2, "upload": 1}))
    return isolated_client


def test_api_returns_429_with_retry_after_and_request_id(limited_client):
    body = {"query": "anything"}
    assert limited_client.post("/chat", json=body).status_code == 200
    assert limited_client.post("/chat", json=body).status_code == 200
    resp = limited_client.post("/chat", json=body)
    assert resp.status_code == 429
    detail = resp.json()["detail"]
    assert detail["code"] == "rate_limited" and detail["retry_after_seconds"] >= 1
    assert int(resp.headers["Retry-After"]) >= 1
    assert detail["request_id"] == resp.headers["X-Request-ID"]
    # Reads stay available while a client is throttled.
    assert limited_client.get("/documents").status_code == 200


def test_upload_limit_is_separate_from_chat(limited_client):
    files = {"file": ("a.txt", b"Brake fluid must be replaced every 3 years.", "text/plain")}
    assert limited_client.post("/documents/upload", files=files).status_code == 201
    files = {"file": ("b.txt", b"Tire pressure is 35 psi.", "text/plain")}
    assert limited_client.post("/documents/upload", files=files).status_code == 429
    assert limited_client.post("/chat", json={"query": "tire pressure"}).status_code == 200


def test_request_id_is_generated_or_propagated(isolated_client):
    generated = isolated_client.get("/health").headers["X-Request-ID"]
    assert 8 <= len(generated) <= 64
    echoed = isolated_client.get("/health", headers={"X-Request-ID": "trace-12345678"})
    assert echoed.headers["X-Request-ID"] == "trace-12345678"
    unsafe = isolated_client.get("/health", headers={"X-Request-ID": "<script>alert(1)</script>"})
    assert unsafe.headers["X-Request-ID"] != "<script>alert(1)</script>"
    missing = isolated_client.get("/documents/doc_0000000000000000")
    assert missing.status_code == 404 and missing.json()["detail"]["request_id"] == missing.headers["X-Request-ID"]


def test_bundled_samples_are_read_only_by_default(make_service, tmp_path, monkeypatch):
    from drivedocs.config import settings

    service = make_service(root=tmp_path / "ro", seed=True)
    sample = service.list_documents().documents[0]
    with pytest.raises(ProtectedDocumentError):
        service.delete_document(sample.doc_id)
    assert service.list_documents().total_documents == 5  # nothing was removed
    uploaded = service.ingest("notes.txt", b"The horn fuse is fuse 7.").document
    assert service.delete_document(uploaded.doc_id).chunks_removed == 1  # uploads stay deletable
    monkeypatch.setattr(settings, "allow_sample_deletion", True)
    assert service.delete_document(sample.doc_id).doc_id == sample.doc_id


def test_api_rejects_sample_deletion_with_403(isolated_client, monkeypatch):
    service = isolated_client.app.state.service
    service.seed_samples()
    sample = next(d for d in isolated_client.get("/documents").json()["documents"] if d["source"] == "sample")
    resp = isolated_client.delete(f"/documents/{sample['doc_id']}")
    assert resp.status_code == 403 and resp.json()["detail"]["code"] == "sample_document_protected"
    info = isolated_client.get("/system/info").json()
    assert info["sample_deletion_allowed"] is False
