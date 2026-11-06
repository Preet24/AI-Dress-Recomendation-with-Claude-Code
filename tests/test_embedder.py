from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from outfit.config import Settings
from outfit.ingestion.embedder import (
    BACKGROUND_RGB,
    ClipModel,
    embed_images,
    embed_texts,
    flatten_on_background,
    get_clip,
    l2_normalize,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClip:
    """Stands in for a CLIP model: the 'embedding' is the mean RGB of the image."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(pixels))
        return pixels.mean(dim=(2, 3))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.float()


def fake_clip() -> ClipModel:
    def preprocess(img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)

    def tokenizer(texts: list[str]) -> torch.Tensor:
        return torch.tensor([[len(t), 1.0] for t in texts])

    return ClipModel(FakeClip(), preprocess, tokenizer)


def test_transparent_pixels_become_neutral_grey():
    img = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    img.putpixel((1, 0), (200, 10, 10, 255))

    flat = flatten_on_background(img)

    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == BACKGROUND_RGB
    assert flat.getpixel((1, 0)) == (200, 10, 10)


def test_rgb_image_is_unchanged():
    img = Image.new("RGB", (3, 3), (1, 2, 3))

    assert flatten_on_background(img).tobytes() == img.tobytes()


def test_l2_normalize_gives_unit_rows_and_tolerates_zero():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))

    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_embed_images_batches_and_normalizes():
    clip = fake_clip()
    images = [Image.new("RGBA", (4, 4), (10 * i, 50, 90, 255)) for i in range(5)]

    vectors = embed_images(images, clip, batch_size=2)

    assert clip.model.batch_sizes == [2, 2, 1]
    assert vectors.shape == (5, 3)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_embed_texts_normalizes():
    vectors = embed_texts(["red shirt", "jeans"], fake_clip())

    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


@pytest.mark.slow
def test_real_clip_matches_red_tshirt_to_its_description():
    settings = Settings()
    clip = get_clip(settings.clip_model, settings.clip_pretrained)
    image = Image.open(FIXTURES / "red_tshirt_flat.png")

    [vector] = embed_images([image], clip)
    texts = embed_texts(["a photo of a red t-shirt", "a photo of blue jeans"], clip)

    assert vector.shape == (512,)
    assert next(clip.model.parameters()).device.type == "cpu"
    shirt, jeans = texts @ vector
    assert shirt > jeans
