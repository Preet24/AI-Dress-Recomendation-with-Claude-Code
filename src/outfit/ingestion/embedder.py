"""CLIP embeddings for garment cutouts (and text queries), with open_clip on CPU.

Image and text vectors share one space, so a cutout can be found by another
cutout ("similar to this") or by a phrase ("a light linen shirt").
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEVICE = "cpu"
# Transparent pixels are composited onto one neutral grey for every item, so the
# background never pulls a vector toward "dark" or "white".
BACKGROUND_RGB = (128, 128, 128)


@dataclass(frozen=True)
class ClipModel:
    model: Any  # open_clip CLIP model: encode_image / encode_text
    preprocess: Callable[[Image.Image], torch.Tensor]
    tokenizer: Callable[[list[str]], torch.Tensor]


@cache
def get_clip(model_name: str, pretrained: str) -> ClipModel:
    """Load an open_clip model once per (name, weights), in eval mode on CPU."""
    import open_clip  # deferred: importing open_clip is slow

    logger.info("Loading CLIP %s (%s) on CPU", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    model.eval()
    return ClipModel(model, preprocess, open_clip.get_tokenizer(model_name))


def flatten_on_background(image: Image.Image) -> Image.Image:
    """RGB version of an image; any transparency is filled with BACKGROUND_RGB."""
    if image.mode not in ("RGBA", "LA", "P"):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGB", rgba.size, BACKGROUND_RGB)
    canvas.paste(rgba, mask=rgba.getchannel("A"))
    return canvas


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so cosine similarity is a dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_images(
    images: Sequence[Image.Image], clip: ClipModel, batch_size: int = 16
) -> np.ndarray:
    """Embed images as unit vectors, shape (n, dim), float32."""
    batches = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        pixels = torch.stack([clip.preprocess(flatten_on_background(img)) for img in chunk])
        with torch.inference_mode():
            batches.append(clip.model.encode_image(pixels).float().numpy())
    return l2_normalize(np.concatenate(batches)).astype(np.float32)


def embed_texts(texts: Sequence[str], clip: ClipModel) -> np.ndarray:
    """Embed text queries as unit vectors in the same space as images, shape (n, dim)."""
    with torch.inference_mode():
        vectors = clip.model.encode_text(clip.tokenizer(list(texts))).float().numpy()
    return l2_normalize(vectors).astype(np.float32)
