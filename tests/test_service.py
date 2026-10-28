"""Service-level behaviour: persistence, restarts, duplicates, deletion, rollback and readiness."""

from __future__ import annotations

import json
import os

import pytest

from drivedocs.ingestion import DuplicateDocumentError, MalformedDocumentError
from drivedocs.schemas import ChatRequest


def test_registry_and_indexes_survive_restart(make_service, tmp_path):
    root = tmp_path / "persist"
    first = make_service(root=root)
    first.ingest("notes.txt", b"The zebra telemetry system samples at 40 Hz.")
    assert first.health().indexed_documents == 1

    second = make_service(root=root)
    assert second.health().indexed_documents == 1
    assert second.health().indexed_text_chunks == first.health().indexed_text_chunks
    assert second.list_documents().documents[0].filename == "notes.txt"
    with pytest.raises(DuplicateDocumentError):
        second.ingest("renamed.txt", b"The zebra telemetry system samples at 40 Hz.")
    assert second.text_index.count() == first.text_index.count()


def test_index_files_are_written_after_each_upload(make_service, tmp_path):
    service = make_service(root=tmp_path / "s")
    service.ingest("a.txt", b"alpha document content for persistence")
    files = {p.name for p in service.storage.indexes_dir.iterdir()}
    assert {"text_index.faiss", "text_metadata.json", "index_manifest.json"} <= files
    assert json.loads(service.storage.registry_path.read_text())["documents"][0]["filename"] == "a.txt"


def test_delete_removes_everything(make_service, tmp_path, monkeypatch):
    import io
    from PIL import Image

    service = make_service(root=tmp_path / "s")
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), (1, 2, 3)).save(buf, format="PNG")
    img = service.ingest("pic.png", buf.getvalue())
    txt = service.ingest("keep.txt", b"this document must remain after deleting the image")
    image_name = f"{img.document.doc_id}"
    assert service.image_file(f"img_{image_name}.png") is not None

    result = service.delete_document(img.document.doc_id)
    assert result.images_removed == 1
    assert service.image_index.count() == 0
    assert service.image_file(f"img_{image_name}.png") is None
    assert not list(service.storage.uploads_dir.glob(f"{img.document.doc_id}.*"))
    assert service.ingestion.get_document(txt.document.doc_id) is not None
    assert service.text_index.count() == txt.chunks_indexed


def test_failed_ingestion_is_recorded_not_dropped(make_service, tmp_path):
    service = make_service(root=tmp_path / "s")
    with pytest.raises(MalformedDocumentError):
        service.ingest("broken.pdf", b"%PDF-1.4 truncated garbage")
    docs = service.list_documents()
    assert docs.failed_documents == 1 and docs.total_documents == 0
    assert docs.documents[0].status == "failed" and docs.documents[0].error


def test_embedding_failure_rolls_back(make_service, tmp_path, monkeypatch):
    service = make_service(root=tmp_path / "s")
    service.ingest("base.txt", b"baseline document that must stay intact")
    before = (service.text_index.count(), len(service.ingestion.documents_by_id))

    def boom(*args, **kwargs):
        raise RuntimeError("embedding exploded")

    monkeypatch.setattr(service.text_index, "encode_chunks", boom)
    with pytest.raises(RuntimeError):
        service.ingest("new.txt", b"this upload fails during embedding")
    assert (service.text_index.count(), len(service.ingestion.documents_by_id)) == before


def test_legacy_indexes_without_registry_are_discarded(make_service, tmp_path):
    root = tmp_path / "legacy"
    service = make_service(root=root)
    service.ingest("a.txt", b"legacy content")
    service.storage.registry_path.unlink()
    restarted = make_service(root=root)
    assert restarted.text_index.count() == 0
    assert restarted.consistency_problems


def test_inconsistent_index_is_rebuilt_from_originals(make_service, tmp_path):
    root = tmp_path / "rebuild"
    service = make_service(root=root)
    service.ingest("a.txt", b"first rebuild document about lasers")
    service.ingest("b.txt", b"second rebuild document about optics")
    # Simulate a crash between writing the registry and the indexes: drop one doc from the index metadata.
    meta = json.loads((service.storage.indexes_dir / "text_metadata.json").read_text())
    (service.storage.indexes_dir / "text_metadata.json").write_text(json.dumps(meta[:1]))
    restarted = make_service(root=root)
    assert restarted.consistency_problems
    assert restarted.text_index.count() == service.text_index.count()
    assert restarted.health().indexed_documents == 2


def test_model_change_triggers_reembedding(make_service, tmp_path):
    root = tmp_path / "manifest"
    service = make_service(root=root)
    service.ingest("a.txt", b"manifest test document")
    manifest = service.storage.indexes_dir / "index_manifest.json"
    data = json.loads(manifest.read_text())
    data["text_model"] = "some-other-model"
    manifest.write_text(json.dumps(data))
    restarted = make_service(root=root)
    assert json.loads(manifest.read_text())["text_model"] == restarted.text_model.identity
    assert restarted.search_text("manifest test", 1)[0].chunk.filename == "a.txt"


def test_seeding_sample_data(make_service, tmp_path):
    service = make_service(root=tmp_path / "seed", seed=True)
    names = {d.filename for d in service.list_documents().documents}
    assert {"aurora_s5_owner_manual.pdf", "maintenance_schedule.md", "roadside_emergency_guide.txt",
            "service_record_ro58213.txt", "engine_bay_diagram.png"} <= names
    assert all(d.source == "sample" for d in service.list_documents().documents)
    # Bundled samples are not copied into private uploads.
    assert not list(service.storage.uploads_dir.iterdir())
    # The service record's personal data never reaches the index.
    record = next(d for d in service.list_documents().documents if d.filename == "service_record_ro58213.txt")
    assert any("redacted" in w for w in record.warnings)
    text = " ".join(c.text for c in service.document_details(record.doc_id).chunks)
    assert "jordan.avery@example.com" not in text and "1DDAS5E27RA104233" not in text and "[REDACTED_EMAIL]" in text


@pytest.mark.real_models
def test_chat_end_to_end_with_page_citations(make_service, tmp_path, monkeypatch):
    from drivedocs.config import settings

    monkeypatch.setattr(settings, "embedding_backend", os.environ["DRIVEDOCS_TEST_REAL_BACKEND"])
    service = make_service(root=tmp_path / "chat", seed=True)
    resp = service.chat(ChatRequest(query="What is the wheel nut tightening torque?", include_debug=True))
    assert "80 lb-ft" in resp.answer
    assert resp.citations and resp.citations[0].filename == "roadside_emergency_guide.txt"
    assert resp.citations[0].page_number == 1 and resp.citations[0].verified
    assert resp.latency_ms is not None and "fused_ranking" in resp.debug_info
    assert resp.agent is not None and resp.agent.steps[0].tool == "search_manual_text"


def test_health_is_cheap_and_ready_reports_components(make_service, tmp_path):
    service = make_service(root=tmp_path / "h")
    health = service.health()
    assert health.status == "healthy" and health.ready is True
    readiness = service.readiness()
    assert readiness.ready is True
    assert set(readiness.components) >= {"storage", "registry", "text_index", "text_model", "image_model"}
    assert readiness.storage_mode == "ephemeral"


def test_require_ready_raises_while_starting(tmp_path):
    from drivedocs.service import DriveDocsService, ServiceUnavailableError
    from drivedocs.storage import LocalStorage

    service = DriveDocsService(storage=LocalStorage(root=tmp_path / "x"))
    service.config.ready_wait_seconds, original = 0.01, service.config.ready_wait_seconds
    try:
        with pytest.raises(ServiceUnavailableError):
            service.require_ready()
    finally:
        service.config.ready_wait_seconds = original


def test_prebuilt_index_is_reused_without_reseeding(make_service, tmp_path):
    root = tmp_path / "prebuilt"
    built = make_service(root=root, seed=True)
    assert "seed_samples" in built.timings_ms and built.list_documents().total_documents == 5
    # A new process starting from the build output must not re-embed the samples.
    started = make_service(root=root, seed=True)
    assert "seed_samples" not in started.timings_ms
    assert started.list_documents().total_documents == 5
    assert started.text_index.count() == built.text_index.count()
    assert started.image_index.count() == built.image_index.count()


def test_build_demo_index_refuses_mock_backend():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "build_demo_index.py"
    spec = importlib.util.spec_from_file_location("build_demo_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 2  # the test suite runs on mock embeddings


def test_prebuilt_index_is_restored_into_a_different_storage_root(make_service, tmp_path, monkeypatch):
    from drivedocs.config import settings

    prebuilt = tmp_path / "project" / "data" / "prebuilt"
    built = make_service(root=prebuilt, seed=True)  # what scripts/build_demo_index.py produces
    monkeypatch.setattr(settings, "prebuilt_index_dir", prebuilt)
    # Runtime storage elsewhere (e.g. DRIVEDOCS_STORAGE_DIR outside the project, a Docker volume, a new disk).
    runtime = make_service(root=tmp_path / "elsewhere" / "storage", seed=True)
    assert "restore_prebuilt_index" in runtime.timings_ms and "seed_samples" not in runtime.timings_ms
    assert runtime.list_documents().total_documents == built.list_documents().total_documents == 5
    assert runtime.text_index.count() == built.text_index.count()
    hits = runtime.search_text("wheel nut tightening torque", 1)
    assert hits and hits[0].chunk.filename == "roadside_emergency_guide.txt"
    image = runtime.image_index.images[0]
    assert runtime.image_file(f"{image.image_id}.png") is not None  # image files were copied too
    # An existing library is never overwritten, and seeding-disabled deployments get no demo data.
    again = make_service(root=tmp_path / "elsewhere" / "storage", seed=True)
    assert "restore_prebuilt_index" not in again.timings_ms
    empty = make_service(root=tmp_path / "noseed", seed=False)
    assert empty.list_documents().total_documents == 0
