"""Unit tests for ImageFAISSIndex and multimodal CLIP retrieval."""

from pathlib import Path
from PIL import Image
import pytest

from drivedocs.image_index import CLIPEmbeddingModel, ImageFAISSIndex
from drivedocs.schemas import ImageItem


@pytest.fixture
def sample_images(tmp_path):
    img1_path = tmp_path / "blue_box.png"
    img2_path = tmp_path / "red_circle.png"

    Image.new("RGB", (100, 100), color=(0, 0, 255)).save(img1_path)
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(img2_path)

    items = [
        ImageItem(
            image_id="img_1",
            doc_id="doc_a",
            filename="diagrams.pdf",
            page_number=1,
            image_path=str(img1_path),
            url_path="/static/images/blue_box.png",
            width=100,
            height=100,
            format="PNG",
        ),
        ImageItem(
            image_id="img_2",
            doc_id="doc_b",
            filename="shapes.png",
            page_number=1,
            image_path=str(img2_path),
            url_path="/static/images/red_circle.png",
            width=100,
            height=100,
            format="PNG",
        ),
    ]
    return items


def test_clip_image_and_text_search(sample_images):
    # Tests real CLIP model (already cached in virtualenv)
    index = ImageFAISSIndex()
    assert index.count() == 0

    added = index.add_images(sample_images)
    assert added == 2
    assert index.count() == 2

    # Text-to-image search
    results_blue = index.search_by_text("a solid blue square box", top_k=2)
    assert len(results_blue) == 2
    assert results_blue[0].image.image_id == "img_1"
    assert 0.0 <= results_blue[0].score <= 1.0

    results_red = index.search_by_text("a bright red circle", top_k=2)
    assert len(results_red) == 2
    assert results_red[0].image.image_id == "img_2"


def test_image_to_image_search(sample_images, tmp_path):
    index = ImageFAISSIndex()
    index.add_images(sample_images)

    # Query image: similar to blue box
    query_blue = Image.new("RGB", (80, 80), color=(10, 10, 240))
    results = index.search_by_image(query_blue, top_k=1)

    assert len(results) == 1
    assert results[0].image.image_id == "img_1"
    assert results[0].score > 0.85  # Very high visual similarity


def test_image_index_persistence(sample_images, tmp_path):
    index = ImageFAISSIndex()
    index.add_images(sample_images)

    save_dir = tmp_path / "faiss_images"
    index.save(save_dir)

    assert (save_dir / "image_index.faiss").exists()
    assert (save_dir / "image_metadata.json").exists()

    new_index = ImageFAISSIndex()
    loaded = new_index.load(save_dir)
    assert loaded is True
    assert new_index.count() == 2
