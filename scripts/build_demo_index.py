"""Build the demo document index at build time (run after scripts/download_models.py).

Ingests the bundled sample documents into DRIVEDOCS_STORAGE_DIR and writes the registry and FAISS indexes. On
Render the build output is what every new instance starts from, so the server boots straight into a ready index
without re-embedding the samples (faster start-up, and the CLIP vision tower is not needed until a visitor uploads
an image).

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
    started = time.perf_counter()
    storage = LocalStorage(config=settings)
    storage.ensure_dirs()
    # Start from an empty library but keep placeholder files such as .gitkeep.
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
    return 0 if docs.total_documents and not docs.failed_documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
