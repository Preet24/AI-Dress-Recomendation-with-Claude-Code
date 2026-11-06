"""Wardrobe search: payload filters for hard constraints, CLIP similarity for ranking.

Text queries ("navy linen shirt") and items ("more like this one") are both turned
into vectors in the CLIP space the cutouts were embedded in. Everything runs locally.
"""

from dataclasses import dataclass
from uuid import UUID

from outfit.db.qdrant_store import ItemFilter, SearchHit, WardrobeStore
from outfit.ingestion.embedder import ClipModel, embed_texts

# The standard CLIP caption template. On the real wardrobe it ranked the right item
# first for 8/8 test queries, like bare keywords; a longer suffix (", a piece of
# clothing") shrank the gap to the runner-up by a quarter, so it was dropped.
QUERY_TEMPLATE = "a photo of {query}"


def caption(query: str) -> str:
    """CLIP-friendly caption for a free-text query."""
    query = query.strip()
    if not query:
        raise ValueError("Empty search query")
    return QUERY_TEMPLATE.format(query=query)


@dataclass(frozen=True)
class WardrobeSearch:
    store: WardrobeStore
    clip: ClipModel

    def by_text(
        self, query: str, filters: ItemFilter | None = None, limit: int = 10
    ) -> list[SearchHit]:
        """Items best matching a description, within the filters."""
        [vector] = embed_texts([caption(query)], self.clip)
        return self.store.search(vector, filters, limit)

    def similar_to(
        self, item_id: UUID, filters: ItemFilter | None = None, limit: int = 10
    ) -> list[SearchHit]:
        """Items that look most like the given one (excluding itself), within the filters."""
        vector = self.store.vector(item_id)
        if vector is None:
            raise KeyError(f"No item {item_id} in the wardrobe")
        hits = self.store.search(vector, filters, limit + 1)
        return [hit for hit in hits if hit.item.id != item_id][:limit]
