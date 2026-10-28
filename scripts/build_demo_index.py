"""Build the demo document index at build time (run after scripts/download_models.py).

Ingests the bundled sample documents into ``data/prebuilt/`` (DRIVEDOCS_PREBUILT_DIR) and writes the registry, the
FAISS indexes and the extracted images. That directory is part of the project, so it ships with the build output on
Render and in the Docker image. On start-up the service copies it into an empty DRIVEDOCS_STORAGE_DIR (wherever that
points), so instances boot straight into a ready index: no embedding at start-up (faster cold start, lower peak memory)
and the CLIP vision tower is not loaded until a visitor embeds an image. Without a prebuilt index, or if it was built
with other models, the service indexes or re-embeds at start-up as before.

    python scripts/build_demo_index.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drivedocs.config import settings  # noqa: E402
from drivedocs.service import DriveDocsService  # noqa: E402
from drivedocs.storage import LocalStorage  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if settings.embedding_backend == "mock":
        print("[!] Refusing to build the demo index with the mock embedding backend.")
        return 2
    started = time.perf_counter()
    # The index is written inside the project (data/prebuilt by default) so it always ships with the build output;
    # the service copies it into DRIVEDOCS_STORAGE_DIR on start-up when that storage is empty.
    storage = LocalStorage(root=settings.prebuilt_index_dir, config=settings)
    storage.ensure_dirs()
    # Start from an empty library, keeping placeholder files such as .gitkeep.
    storage.registry_path.unlink(missing_ok=True)
    for folder in (storage.indexes_dir, storage.images_dir, storage.uploads_dir):
        for f in folder.iterdir():
            if f.is_file() and not f.name.startswith("."):
                f.unlink()

    settings.seed_sample_data = True
    service = DriveDocsService(storage=storage)
    service.initialize()
    if service.phase != "ready":
        print(f"[!] Demo index build failed: {service.startup_error}")
        return 1
    docs = service.list_documents()
    for d in docs.documents:
        print(f"    {d.filename}: {d.chunk_count} chunks, {d.image_count} images, {d.page_count} page(s), {d.status}")
    print(f"[+] Demo index: {docs.total_documents} documents, {docs.total_chunks} chunks, {docs.total_images} images "
          f"in {storage.root} ({time.perf_counter() - started:.1f}s)")
    expected = len(settings.sample_files)
    if docs.total_documents != expected or docs.failed_documents:
        print(f"[!] Expected {expected} ready documents, got {docs.total_documents} ({docs.failed_documents} failed).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
