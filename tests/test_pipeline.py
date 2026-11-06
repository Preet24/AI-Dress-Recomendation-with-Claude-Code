import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image
from qdrant_client import QdrantClient

from outfit.config import Settings
from outfit.db.qdrant_store import VECTOR_SIZE, ItemFilter, WardrobeStore
from outfit.ingestion.background import SegmentationError, SegMode
from outfit.ingestion.embedder import ClipModel
from outfit.ingestion.pipeline import Pipeline
from outfit.ingestion.tagger import TaggingError
from outfit.schemas import Category, stable_item_id
from tests.fakes import FakeClient, reply, tags_reply

H, W = 120, 60


class MaskSession:
    """Fake rembg session returning fixed masks."""

    def __init__(self, *masks: np.ndarray) -> None:
        self.masks = masks

    def predict(self, img: Image.Image) -> list[Image.Image]:
        return [Image.fromarray(m) for m in self.masks]


class Clip512:
    """Fake CLIP with the real vector size: mean RGB, tiled to 512 dims."""

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.mean(dim=(2, 3)).repeat(1, VECTOR_SIZE // 3 + 1)[:, :VECTOR_SIZE]


def fake_clip() -> ClipModel:
    def preprocess(img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)

    return ClipModel(Clip512(), preprocess, tokenizer=lambda texts: torch.zeros(len(texts), 1))


def band(top: int, bottom: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top:bottom, 10:50] = 255
    return mask


@pytest.fixture
def photo(tmp_path):
    """A 'worn' photo: red upper half, navy lower half."""
    img = Image.new("RGB", (W, H), (200, 20, 30))
    img.paste((20, 30, 90), (0, H // 2, W, H))
    path = tmp_path / "raw" / "outfit.jpg"
    path.parent.mkdir()
    img.save(path, quality=95)
    return path


def make_pipeline(tmp_path, client: FakeClient, *masks: np.ndarray) -> Pipeline:
    settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key", clean_dir=tmp_path / "clean")
    store = WardrobeStore(QdrantClient(":memory:"))
    store.ensure_collection()
    asked: list[str] = []

    def session_for(model_name: str) -> MaskSession:
        asked.append(model_name)
        return MaskSession(*masks)

    pipeline = Pipeline(settings, store, client, fake_clip(), session_for=session_for)
    pipeline.asked_models = asked  # type: ignore[attr-defined]
    return pipeline


WORN_MASKS = (band(5, 55), band(65, 115), np.zeros((H, W), dtype=np.uint8))  # upper, lower, full


def test_worn_photo_yields_two_stored_items(tmp_path, photo):
    client = FakeClient(
        tags_reply(),
        tags_reply(category="bottom", subtype="jeans", primary_color_name="navy", length="full"),
    )
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)

    top, bottom = pipeline.ingest_image(photo, SegMode.WORN)

    assert pipeline.asked_models == ["u2net_cloth_seg"]
    assert (top.category, bottom.category) == (Category.TOP, Category.BOTTOM)
    assert top.id == stable_item_id(photo.read_bytes(), "upper")
    assert pipeline.store.count() == 2
    assert pipeline.store.get(bottom.id) == bottom
    assert pipeline.store.count(ItemFilter(categories=[Category.BOTTOM])) == 1


def test_color_is_computed_locally_not_taken_from_the_model(tmp_path, photo):
    # DeepSeek says "green"; the LAB value must still come from the red pixels.
    client = FakeClient(tags_reply(primary_color_name="green"), tags_reply())
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)

    top, _ = pipeline.ingest_image(photo, SegMode.WORN)

    lightness, a, b = top.primary_color_lab
    assert top.primary_color_name == "green"
    assert a > 40 and a > abs(b)


def test_cutouts_are_saved_and_only_cutouts_go_to_deepseek(tmp_path, photo):
    client = FakeClient(tags_reply(), tags_reply())
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)

    items = pipeline.ingest_image(photo, SegMode.WORN)

    hints = []
    for item, request in zip(items, client.requests, strict=True):
        saved = Image.open(item.image_path)
        assert item.image_path.parent == tmp_path / "clean"
        assert item.image_path.name.startswith("outfit_")
        assert saved.mode == "RGBA"
        text_block, image_block = request["messages"][1]["content"]
        data = image_block["image_url"]["url"].split(",", 1)[1]
        sent = Image.open(io.BytesIO(base64.standard_b64decode(data)))
        assert sent.size == saved.size != (W, H)  # the cropped cutout, not the photo
        hints.append(text_block["text"])
    assert "`upper`" in hints[0] and "`lower`" in hints[1]


def test_reingesting_updates_items_and_keeps_created_at(tmp_path, photo):
    client = FakeClient(tags_reply(), tags_reply(), tags_reply(subtype="polo"), tags_reply())
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)

    first, _ = pipeline.ingest_image(photo, SegMode.WORN)
    again, _ = pipeline.ingest_image(photo, SegMode.WORN)

    assert pipeline.store.count() == 2
    assert again.id == first.id
    assert again.created_at == first.created_at
    assert pipeline.store.get(first.id).subtype == "polo"


def test_tagging_failure_stores_nothing(tmp_path, photo):
    client = FakeClient(tags_reply(), reply("bad"), reply("still bad"))
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)

    with pytest.raises(TaggingError):
        pipeline.ingest_image(photo, SegMode.WORN)
    assert pipeline.store.count() == 0


def test_worn_vectors_cut_and_embed_without_tagging_or_saving(tmp_path, photo):
    client = FakeClient()
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)

    found = pipeline.worn_vectors(photo.read_bytes())

    assert [region for region, _ in found] == ["upper", "lower"]
    assert all(v.shape == (512,) and abs(np.linalg.norm(v) - 1) < 1e-5 for _, v in found)
    assert pipeline.asked_models == ["u2net_cloth_seg"]
    assert client.requests == [] and not (tmp_path / "clean").exists()
    assert pipeline.store.count() == 0


def test_photo_without_garment_raises_before_calling_claude(tmp_path, photo):
    client = FakeClient()
    pipeline = make_pipeline(tmp_path, client, np.zeros((H, W), dtype=np.uint8))

    with pytest.raises(SegmentationError):
        pipeline.ingest_image(photo, SegMode.FLAT)
    assert client.requests == []
    assert pipeline.asked_models == ["birefnet-general"]
