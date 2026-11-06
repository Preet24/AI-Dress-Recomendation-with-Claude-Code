"""Wardrobe storage in Qdrant: one point per ClothingItem, CLIP vector + tags as payload.

Payload filters do the hard constraints (category, formality, occasion...);
the vector ranks what is left by visual / text similarity.

Local mode (`QdrantClient(path=...)`) scans payloads instead of using the indexes,
which is fine for a personal wardrobe; the indexes are still created so the same
code runs against a Qdrant server. Only one process may open a local path.
"""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient, models

from outfit.schemas import Category, ClothingItem, Occasion

logger = logging.getLogger(__name__)

COLLECTION = "wardrobe"
VECTOR_NAME = "image"
VECTOR_SIZE = 512  # CLIP ViT-B-32

PAYLOAD_INDEXES = {
    "category": models.PayloadSchemaType.KEYWORD,
    "formality": models.PayloadSchemaType.INTEGER,
    "warmth": models.PayloadSchemaType.INTEGER,
    "occasions": models.PayloadSchemaType.KEYWORD,
    "flowy": models.PayloadSchemaType.BOOL,
}


@dataclass(frozen=True)
class ItemFilter:
    """Hard constraints on wardrobe items; unset fields don't constrain."""

    categories: Sequence[Category] = field(default_factory=tuple)
    occasions: Sequence[Occasion] = field(default_factory=tuple)  # item matches any of them
    formality: tuple[int, int] | None = None  # inclusive (min, max)
    warmth: tuple[int, int] | None = None  # inclusive (min, max)
    flowy: bool | None = None


@dataclass(frozen=True)
class SearchHit:
    item: ClothingItem
    score: float  # cosine similarity


def build_filter(f: ItemFilter) -> models.Filter | None:
    """Translate an ItemFilter into a Qdrant filter (None if it has no constraints)."""
    must: list[models.Condition] = []
    if f.categories:
        must.append(_match_any("category", f.categories))
    if f.occasions:
        must.append(_match_any("occasions", f.occasions))
    for key, bounds in (("formality", f.formality), ("warmth", f.warmth)):
        if bounds is not None:
            low, high = bounds
            must.append(models.FieldCondition(key=key, range=models.Range(gte=low, lte=high)))
    if f.flowy is not None:
        must.append(models.FieldCondition(key="flowy", match=models.MatchValue(value=f.flowy)))
    return models.Filter(must=must) if must else None


def _match_any(key: str, values: Sequence[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=[str(v) for v in values]))


def to_payload(item: ClothingItem) -> dict:
    """JSON payload for an item; the id lives in the point id, not the payload."""
    return item.model_dump(mode="json", exclude={"id"})


def from_point(point_id: str | int | UUID, payload: dict | None) -> ClothingItem:
    return ClothingItem.model_validate({**(payload or {}), "id": str(point_id)})


class WardrobeStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION) -> None:
        self.client = client
        self.collection = collection

    def ensure_collection(self) -> None:
        """Create the collection and payload indexes if missing; check the vector config."""
        if self.client.collection_exists(self.collection):
            self._check_vector_config()
            return
        logger.info("Creating Qdrant collection %r", self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                VECTOR_NAME: models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            },
        )
        with warnings.catch_warnings():
            # Expected in local mode (see module docstring); the indexes matter on a server.
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for key, schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(self.collection, key, field_schema=schema)

    def _check_vector_config(self) -> None:
        vectors = self.client.get_collection(self.collection).config.params.vectors
        params = vectors.get(VECTOR_NAME) if isinstance(vectors, dict) else None
        if params is None or params.size != VECTOR_SIZE:
            raise ValueError(
                f"Collection {self.collection!r} has vectors {vectors!r}, expected "
                f"{VECTOR_NAME!r} of size {VECTOR_SIZE}; was it built with another model?"
            )

    def upsert(self, items: Sequence[ClothingItem], vectors: np.ndarray) -> None:
        """Insert or replace items (matched by id) with their image vectors."""
        if vectors.shape != (len(items), VECTOR_SIZE):
            raise ValueError(
                f"Expected vectors of shape ({len(items)}, {VECTOR_SIZE}), got {vectors.shape}"
            )
        points = [
            models.PointStruct(
                id=str(item.id), vector={VECTOR_NAME: vector.tolist()}, payload=to_payload(item)
            )
            for item, vector in zip(items, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)
        logger.info("Upserted %d item(s) into %r", len(points), self.collection)

    def update_tags(self, item: ClothingItem) -> None:
        """Replace an existing item's payload, keeping its vector (the image is unchanged)."""
        if self.get(item.id) is None:
            raise KeyError(f"No item {item.id} in {self.collection!r}")
        self.client.overwrite_payload(
            self.collection, payload=to_payload(item), points=[str(item.id)]
        )
        logger.info("Updated tags of %s", item.id)

    def delete(self, item_ids: Sequence[UUID]) -> None:
        """Remove items by id (ids that don't exist are ignored)."""
        self.client.delete(
            self.collection, points_selector=models.PointIdsList(points=[str(i) for i in item_ids])
        )
        logger.info("Deleted %d item(s) from %r", len(item_ids), self.collection)

    def vector(self, item_id: UUID) -> np.ndarray | None:
        """The stored image vector of an item, or None if there is no such item."""
        records = self.client.retrieve(self.collection, ids=[str(item_id)], with_vectors=True)
        if not records:
            return None
        return np.asarray(records[0].vector[VECTOR_NAME], dtype=np.float32)

    def get(self, item_id: UUID) -> ClothingItem | None:
        records = self.client.retrieve(self.collection, ids=[str(item_id)], with_payload=True)
        return from_point(records[0].id, records[0].payload) if records else None

    def search(
        self, vector: np.ndarray, filters: ItemFilter | None = None, limit: int = 10
    ) -> list[SearchHit]:
        """Most similar items to a CLIP vector (image or text), within the filters."""
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            using=VECTOR_NAME,
            query_filter=build_filter(filters or ItemFilter()),
            limit=limit,
            with_payload=True,
        )
        return [SearchHit(from_point(p.id, p.payload), p.score) for p in response.points]

    def list_items(
        self, filters: ItemFilter | None = None, limit: int = 1000
    ) -> list[ClothingItem]:
        """Items matching the filters, unranked."""
        records, _ = self.client.scroll(
            self.collection,
            scroll_filter=build_filter(filters or ItemFilter()),
            limit=limit,
            with_payload=True,
        )
        return [from_point(r.id, r.payload) for r in records]

    def count(self, filters: ItemFilter | None = None) -> int:
        return self.client.count(
            self.collection, count_filter=build_filter(filters or ItemFilter()), exact=True
        ).count
