import numpy as np
import pytest
import torch
from qdrant_client import QdrantClient

from outfit.db.qdrant_store import VECTOR_SIZE, ItemFilter, WardrobeStore
from outfit.ingestion.embedder import ClipModel
from outfit.schemas import Category, Occasion
from outfit.tools.search import WardrobeSearch, caption
from tests.factories import make_item


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=VECTOR_SIZE).astype(np.float32)
    return v / np.linalg.norm(v)


class TextClip:
    """Fake CLIP: each known caption encodes to a fixed vector; records what was encoded."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.captions = list(vectors)
        self.vectors = torch.from_numpy(np.stack(list(vectors.values()) or [unit(0)]))
        self.seen: list[str] = []

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        self.seen.extend(texts)
        return torch.tensor([self.captions.index(t) for t in texts])

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vectors[tokens] * 3.0  # not unit length: search must normalize


@pytest.fixture
def wardrobe():
    """Shirt, jeans and a dress with known vectors."""
    store = WardrobeStore(QdrantClient(":memory:"))
    store.ensure_collection()
    shirt = make_item(subtype="oxford shirt")
    jeans = make_item(category=Category.BOTTOM, subtype="jeans", occasions=[Occasion.CASUAL])
    dress = make_item(category=Category.ONE_PIECE, subtype="midi dress", formality=4)
    # jeans sit close to the shirt; the dress is far from both
    vectors = np.stack([unit(1), unit(1) * 0.8 + unit(2) * 0.6, unit(3)])
    store.upsert([shirt, jeans, dress], vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    return store, shirt, jeans, dress


def searcher(store: WardrobeStore, **captions: np.ndarray) -> tuple[WardrobeSearch, TextClip]:
    fake = TextClip({caption(q.replace("_", " ")): v for q, v in captions.items()})
    clip = ClipModel(fake, preprocess=lambda img: img, tokenizer=fake.tokenize)
    return WardrobeSearch(store, clip), fake


def test_caption_wraps_the_query_for_clip():
    assert caption("  navy linen shirt ") == "a photo of navy linen shirt"


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Empty"):
        caption("   ")


def test_by_text_ranks_by_similarity(wardrobe):
    store, shirt, jeans, dress = wardrobe
    search, fake = searcher(store, a_dress=unit(3))

    hits = search.by_text("a dress", limit=3)

    assert fake.seen == ["a photo of a dress"]
    assert hits[0].item == dress
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score > hits[1].score >= hits[2].score


def test_by_text_applies_hard_filters(wardrobe):
    store, shirt, jeans, dress = wardrobe
    search, _ = searcher(store, a_dress=unit(3))

    hits = search.by_text("a dress", ItemFilter(categories=[Category.TOP, Category.BOTTOM]))

    assert {h.item.id for h in hits} == {shirt.id, jeans.id}


def test_similar_to_excludes_the_item_itself(wardrobe):
    store, shirt, jeans, dress = wardrobe
    search, _ = searcher(store)

    hits = search.similar_to(shirt.id, limit=2)

    assert [h.item.id for h in hits] == [jeans.id, dress.id]


def test_similar_to_respects_filters_and_limit(wardrobe):
    store, shirt, jeans, dress = wardrobe
    search, _ = searcher(store)

    [hit] = search.similar_to(shirt.id, ItemFilter(formality=(4, 5)), limit=1)

    assert hit.item.id == dress.id


def test_similar_to_unknown_item_raises(wardrobe):
    store, *_ = wardrobe
    search, _ = searcher(store)

    with pytest.raises(KeyError):
        search.similar_to(make_item().id)


def test_store_vector_round_trips(wardrobe):
    store, shirt, *_ = wardrobe

    assert np.allclose(store.vector(shirt.id), unit(1), atol=1e-5)
    assert store.vector(make_item().id) is None
