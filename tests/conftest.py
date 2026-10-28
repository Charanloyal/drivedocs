"""Test configuration.

* All storage goes to a temporary directory, never to the repository's data/storage.
* Embeddings use the deterministic mock backend by default, so the suite needs no model downloads.
* Tests marked ``real_models`` use the real MiniLM/CLIP weights. They run only when both models are already in the
  local Hugging Face cache (or DRIVEDOCS_TEST_REAL_MODELS=1 forces them) and never trigger a download.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="drivedocs_test_storage_"))
os.environ["DRIVEDOCS_STORAGE_DIR"] = str(_TEST_STORAGE)
# Never restore a locally built demo index (data/prebuilt) into test storage.
os.environ["DRIVEDOCS_PREBUILT_DIR"] = str(_TEST_STORAGE / "_no_prebuilt_index")
os.environ.setdefault("DRIVEDOCS_EMBEDDING_BACKEND", "mock")
os.environ["DRIVEDOCS_PERSISTENT_STORAGE"] = "false"
os.environ["DRIVEDOCS_READY_WAIT_SECONDS"] = "120"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _key in ("LLM_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY", "DRIVEDOCS_CORS_ORIGINS"):
    os.environ.pop(_key, None)

import pytest  # noqa: E402


def _models_cached() -> bool:
    """Real-model tests use the ONNX backend unless DRIVEDOCS_TEST_REAL_BACKEND=transformers."""
    if os.getenv("DRIVEDOCS_TEST_REAL_MODELS") == "1":
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    from drivedocs.config import settings

    if REAL_BACKEND == "onnx":
        required = [
            (settings.text_onnx_repo, settings.text_onnx_file),
            (settings.image_onnx_repo, settings.image_onnx_text_file),
            (settings.image_onnx_repo, settings.image_onnx_vision_file),
        ]
    else:
        required = [(settings.text_model_name, "config.json"), (settings.image_model_name, "config.json")]
    return all(isinstance(try_to_load_from_cache(repo, f), str) for repo, f in required)


REAL_BACKEND = os.getenv("DRIVEDOCS_TEST_REAL_BACKEND", "onnx")
os.environ.setdefault("DRIVEDOCS_TEST_REAL_BACKEND", REAL_BACKEND)


REAL_MODELS_AVAILABLE = _models_cached()


def pytest_collection_modifyitems(config, items):
    if REAL_MODELS_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="real MiniLM/CLIP weights are not in the local Hugging Face cache")
    for item in items:
        if "real_models" in item.keywords:
            item.add_marker(skip)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_STORAGE, ignore_errors=True)


@pytest.fixture
def make_service(tmp_path):
    """Factory for an isolated, synchronously-initialized DriveDocsService on its own storage root."""
    from drivedocs.service import DriveDocsService
    from drivedocs.storage import LocalStorage

    def factory(root: Path | None = None, seed: bool = False, **overrides):
        from drivedocs.config import settings

        root = root or (tmp_path / "storage")
        service = DriveDocsService(storage=LocalStorage(root=root))
        original_seed = settings.seed_sample_data
        settings.seed_sample_data = seed
        try:
            service.initialize()
        finally:
            settings.seed_sample_data = original_seed
        return service

    return factory


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    """A TestClient whose app uses a fresh storage root (no seeded samples unless requested)."""
    from fastapi.testclient import TestClient

    import drivedocs.api as api
    from drivedocs.config import settings
    from drivedocs.service import DriveDocsService
    from drivedocs.storage import LocalStorage

    root = tmp_path / "api_storage"
    monkeypatch.setattr(settings, "seed_sample_data", False)
    monkeypatch.setattr(api, "create_service", lambda: DriveDocsService(storage=LocalStorage(root=root)))
    with TestClient(api.app) as client:
        assert client.app.state.service.wait_ready(60)
        yield client
