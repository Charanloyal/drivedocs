"""Storage abstraction for DriveDocs.

Layout under a single root directory (``settings.storage_dir``)::

    <root>/
      documents.json     document registry (metadata + status), written atomically
      indexes/           FAISS indexes + chunk/image metadata, written atomically as a set
      images/            normalized PNGs of uploaded/extracted images (served only via a validated route)
      uploads/           private copies of original uploads (never served)

Only local filesystem storage is implemented. Point the root at a persistent disk to keep data across restarts;
otherwise the data is lost on restart/redeploy (ephemeral demo mode).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from drivedocs.config import Settings, settings as default_settings

_SAFE_IMAGE_NAME = re.compile(r"^img_[A-Za-z0-9_]{1,120}\.png$")
_SAFE_DOC_ID = re.compile(r"^doc_[a-f0-9]{8,64}$")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes via a temp file + os.replace so readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))


def is_safe_image_name(name: str) -> bool:
    return bool(_SAFE_IMAGE_NAME.match(name))


def is_safe_doc_id(doc_id: str) -> bool:
    return bool(_SAFE_DOC_ID.match(doc_id))


class LocalStorage:
    """Local-filesystem storage backend."""

    def __init__(self, root: Optional[Path] = None, config: Optional[Settings] = None) -> None:
        cfg = config or default_settings
        self.root = Path(root) if root is not None else cfg.storage_dir
        self.images_dir = self.root / "images"
        self.indexes_dir = self.root / "indexes"
        self.uploads_dir = self.root / "uploads"
        self.registry_path = self.root / "documents.json"
        self.persistent = cfg.persistent_storage

    @property
    def mode(self) -> str:
        return "persistent" if self.persistent else "ephemeral"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.images_dir, self.indexes_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)

    def is_writable(self) -> bool:
        try:
            self.ensure_dirs()
            probe = self.root / ".write_probe"
            probe.write_bytes(b"ok")
            probe.unlink()
            return True
        except OSError:
            return False

    # Registry --------------------------------------------------------------------------------------------------

    def load_registry(self) -> Optional[dict]:
        if not self.registry_path.exists():
            return None
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_registry(self, payload: dict) -> None:
        atomic_write_json(self.registry_path, payload)

    # Images ----------------------------------------------------------------------------------------------------

    def image_file(self, image_name: str) -> Optional[Path]:
        """Resolve a public image name to a file inside images_dir, or None if invalid/missing."""
        if not is_safe_image_name(image_name):
            return None
        candidate = (self.images_dir / image_name).resolve()
        try:
            candidate.relative_to(self.images_dir.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def resolve_image_path(self, stored_path: str) -> Path:
        """Resolve an ImageItem.image_path (relative name, or legacy absolute path) to a real file path."""
        p = Path(stored_path)
        if not p.is_absolute():
            return self.images_dir / p
        if p.exists():
            return p
        # Legacy absolute path from another machine: fall back to the same file name in our images dir.
        return self.images_dir / p.name

    def delete_image(self, stored_path: str) -> None:
        path = self.resolve_image_path(stored_path)
        try:
            path.resolve().relative_to(self.images_dir.resolve())
        except ValueError:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # Original uploads ------------------------------------------------------------------------------------------

    def save_original(self, doc_id: str, extension: str, content: bytes) -> None:
        if not is_safe_doc_id(doc_id):
            raise ValueError("unsafe document id")
        ext = extension.lower().lstrip(".")
        if not re.fullmatch(r"[a-z0-9]{1,5}", ext):
            raise ValueError("unsafe extension")
        atomic_write_bytes(self.uploads_dir / f"{doc_id}.{ext}", content)

    def delete_original(self, doc_id: str) -> None:
        if not is_safe_doc_id(doc_id):
            return
        for f in self.uploads_dir.glob(f"{doc_id}.*"):
            try:
                f.unlink()
            except OSError:
                pass

    def reset(self) -> None:
        """Remove all stored data (used by tests and the reset script)."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.ensure_dirs()
