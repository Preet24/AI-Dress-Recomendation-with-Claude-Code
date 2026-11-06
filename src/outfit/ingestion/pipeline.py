"""Ingest one wardrobe photo: segment -> tag -> color -> embed -> store.

`describe_photo` does everything but the storing (the shopping assistant checks a
candidate that way); `ingest_image` describes, then stores. All garments in a photo
are tagged and embedded before anything is written, so
a failure (no garment found, unusable tags) leaves the store untouched for that
photo. Item ids come from the photo bytes + region, so re-ingesting a photo
updates its items (keeping their original `created_at`) instead of duplicating.
"""

import hashlib
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient

from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.ingestion.background import (
    Cutout,
    MaskSession,
    SegMode,
    get_session,
    load_image,
    model_for,
    save_cutouts,
    segment,
)
from outfit.ingestion.color import primary_color_lab
from outfit.ingestion.embedder import ClipModel, embed_images, get_clip
from outfit.ingestion.tagger import make_client, tag_garment
from outfit.schemas import ClothingItem, stable_item_id

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    settings: Settings
    store: WardrobeStore
    client: OpenAI  # DeepSeek client (OpenAI-compatible SDK pointed at api.deepseek.com)
    clip: ClipModel
    session_for: Callable[[str], MaskSession] = get_session

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pipeline":
        """Real dependencies: local Qdrant at `qdrant_path`, DeepSeek client, CLIP on CPU."""
        store = WardrobeStore(QdrantClient(path=str(settings.qdrant_path)))
        store.ensure_collection()
        return cls(
            settings=settings,
            store=store,
            client=make_client(settings),
            clip=get_clip(settings.clip_model, settings.clip_pretrained),
        )

    def ingest_image(self, path: Path, mode: SegMode) -> list[ClothingItem]:
        """Ingest every garment in one photo; returns the stored items."""
        items, vectors = self.describe_photo(path, mode)
        self.store.upsert(items, vectors)
        for item in items:
            logger.info("%s: %s %s (%s)", path.name, item.primary_color_name, item.subtype, item.id)
        return items

    def describe_photo(
        self, path: Path, mode: SegMode, clean_dir: Path | None = None
    ) -> tuple[list[ClothingItem], np.ndarray]:
        """Every garment in one photo, tagged and embedded, without storing anything.

        Cutouts are written to `clean_dir` (default: the wardrobe's `clean_dir`)."""
        photo_bytes = path.read_bytes()
        image = load_image(path, self.settings.max_side)
        session = self.session_for(model_for(mode, self.settings))
        cutouts = segment(image, mode, session, self.settings.min_area_ratio)

        # Short content hash keeps same-named photos from different folders apart.
        stem = f"{path.stem}_{hashlib.sha256(photo_bytes).hexdigest()[:8]}"
        image_paths = save_cutouts(cutouts, clean_dir or self.settings.clean_dir, stem)

        items = [
            self._describe(cutout, image_path, photo_bytes)
            for cutout, image_path in zip(cutouts, image_paths, strict=True)
        ]
        vectors = embed_images(
            [c.image for c in cutouts], self.clip, batch_size=self.settings.embed_batch_size
        )
        return items, vectors

    def worn_vectors(self, photo: bytes) -> list[tuple[str, np.ndarray]]:
        """(region, CLIP vector) per garment in a photo of someone wearing them.

        No tagging, nothing saved: the wear log's photo check (`tools/worn_match.py`)."""
        image = load_image(io.BytesIO(photo), self.settings.max_side)
        session = self.session_for(model_for(SegMode.WORN, self.settings))
        cutouts = segment(image, SegMode.WORN, session, self.settings.min_area_ratio)
        vectors = embed_images(
            [c.image for c in cutouts], self.clip, batch_size=self.settings.embed_batch_size
        )
        return [(str(c.region), v) for c, v in zip(cutouts, vectors, strict=True)]

    def _describe(self, cutout: Cutout, image_path: Path, photo_bytes: bytes) -> ClothingItem:
        """Tags from DeepSeek + locally computed color, under a stable id."""
        tags = tag_garment(cutout.image, cutout.region, self.client, self.settings)
        item_id = stable_item_id(photo_bytes, cutout.region)
        fields = {
            **tags.model_dump(),
            "id": item_id,
            "image_path": image_path,
            "primary_color_lab": primary_color_lab(cutout.image),
        }
        if (existing := self.store.get(item_id)) is not None:
            fields["created_at"] = existing.created_at
        return ClothingItem(**fields)
