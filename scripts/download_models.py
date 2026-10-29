"""Pre-download the embedding models into the Hugging Face cache (run at build time).

With the default ONNX backend only the tokenizer/config files and the selected ONNX graphs are fetched (~210 MB).
With DRIVEDOCS_EMBEDDING_BACKEND=transformers the PyTorch weights are fetched instead (~670 MB; TensorFlow/Flax/ONNX
files skipped). Once the cache is populated the service can run with HF_HUB_OFFLINE=1 and never downloads on boot.

    python scripts/download_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drivedocs.config import settings  # noqa: E402

COMMON = ["*.json", "*.txt", "*.model", "merges.txt", "vocab.*", "1_Pooling/*", "2_Normalize/*", "README.md"]


def _report(label: str, paths: List[str], started: float) -> None:
    size = sum(Path(p).stat().st_size for p in paths) / 1024 / 1024
    print(f"[+] {label}: {len(paths)} files, {size:.0f} MB in {time.perf_counter() - started:.1f}s", flush=True)


def fetch_files(repo_id: str, files: List[str]) -> None:
    from huggingface_hub import hf_hub_download

    if Path(repo_id).is_dir():
        print(f"[=] {repo_id} is a local directory; nothing to download.")
        return
    started = time.perf_counter()
    paths = [hf_hub_download(repo_id, f) for f in files]
    _report(repo_id, paths, started)


def fetch_snapshot(repo_id: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    if Path(repo_id).is_dir():
        print(f"[=] {repo_id} is a local directory; nothing to download.")
        return
    files = set(HfApi().list_repo_files(repo_id))
    weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["pytorch_model.bin"]
    started = time.perf_counter()
    path = snapshot_download(
        repo_id,
        allow_patterns=COMMON + weights,
        ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_model*"],
    )
    _report(repo_id, [str(p) for p in Path(path).rglob("*") if p.is_file()], started)


def main() -> int:
    backend = settings.embedding_backend
    print(f"[*] Embedding backend: {backend}; image search: {settings.enable_image_search}", flush=True)
    if backend == "mock":
        print("[=] Mock backend: no models needed.")
        return 0
    if backend == "onnx":
        fetch_files(settings.text_onnx_repo, ["tokenizer.json", "config.json", settings.text_onnx_file])
        if settings.enable_image_search:
            fetch_files(
                settings.image_onnx_repo,
                ["tokenizer.json", "config.json", "preprocessor_config.json",
                 settings.image_onnx_text_file, settings.image_onnx_vision_file],
            )
        return 0
    fetch_snapshot(settings.text_model_name)
    if settings.enable_image_search:
        fetch_snapshot(settings.image_model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
