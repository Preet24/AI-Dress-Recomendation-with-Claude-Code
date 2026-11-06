import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from outfit.db.qdrant_store import (
    COLLECTION,
    VECTOR_SIZE,
    ItemFilter,
    WardrobeStore,
    build_filter,
)
from outfit.schemas import Category, Occasion
from tests.factories import make_item


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=VECTOR_SIZE).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store() -> WardrobeStore:
    store = WardrobeStore(QdrantClient(":memory:"))
    store.ensure_collection()
    return store


@pytest.fixture
def wardrobe(store):
    """Three items with known vectors: shirt, jeans, wedding lehenga."""
    items = [
        make_item(subtype="oxford shirt"),
        make_item(
            category=Category.BOTTOM, subtype="jeans", formality=2, occasions=[Occasion.CASUAL]
        ),
        make_item(
            category=Category.ONE_PIECE,
            subtype="lehenga",
            formality=5,
            warmth=3,
            flowy=True,
            occasions=[Occasion.WEDDING, Occasion.INDIAN_FESTIVE],
        ),
    ]
    vectors = np.stack([unit(i) for i in range(len(items))])
    store.upsert(items, vectors)
    return items, vectors


def test_ensure_collection_is_idempotent(store):
    store.ensure_collection()

    assert store.client.collection_exists(COLLECTION)
    assert store.count() == 0


def test_ensure_collection_rejects_wrong_vector_size():
    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION,
        vectors_config={"image": models.VectorParams(size=768, distance=models.Distance.COSINE)},
    )

    with pytest.raises(ValueError, match="another model"):
        WardrobeStore(client).ensure_collection()


def test_upsert_then_get_round_trips_item(store, wardrobe):
    items, _ = wardrobe

    assert store.get(items[2].id) == items[2]
    assert store.count() == 3


def test_get_missing_item_returns_none(store):
    assert store.get(make_item().id) is None


def test_upsert_same_id_replaces_instead_of_duplicating(store, wardrobe):
    items, vectors = wardrobe
    updated = items[0].model_copy(update={"subtype": "linen shirt"})

    store.upsert([updated], vectors[:1])

    assert store.count() == 3
    assert store.get(updated.id).subtype == "linen shirt"


def test_upsert_rejects_mismatched_vectors(store):
    with pytest.raises(ValueError, match="shape"):
        store.upsert([make_item()], np.zeros((1, 10), dtype=np.float32))


def test_search_ranks_by_similarity(store, wardrobe):
    items, vectors = wardrobe

    hits = store.search(vectors[1], limit=3)

    assert hits[0].item.id == items[1].id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 3


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (ItemFilter(categories=[Category.BOTTOM, Category.ONE_PIECE]), {"jeans", "lehenga"}),
        (ItemFilter(occasions=[Occasion.OFFICE, Occasion.WEDDING]), {"oxford shirt", "lehenga"}),
        (ItemFilter(formality=(3, 4)), {"oxford shirt"}),
        (ItemFilter(warmth=(3, 5)), {"lehenga"}),
        (ItemFilter(flowy=False), {"oxford shirt", "jeans"}),
        (ItemFilter(categories=[Category.TOP], flowy=True), set()),
    ],
)
def test_filters_constrain_search_and_listing(store, wardrobe, filters, expected):
    _, vectors = wardrobe

    searched = {h.item.subtype for h in store.search(vectors[0], filters)}
    listed = {i.subtype for i in store.list_items(filters)}

    assert searched == listed == expected
    assert store.count(filters) == len(expected)


def test_empty_filter_means_no_constraint():
    assert build_filter(ItemFilter()) is None


def test_update_tags_keeps_the_vector(store):
    item = make_item()
    store.upsert([item], unit(7)[None, :])

    store.update_tags(item.model_copy(update={"subtype": "linen shirt"}))

    [hit] = store.search(unit(7), limit=1)
    assert hit.item.subtype == "linen shirt"
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_update_tags_of_missing_item_raises(store):
    with pytest.raises(KeyError):
        store.update_tags(make_item())


def test_delete_removes_only_the_given_items(store):
    keep, drop = make_item(), make_item(subtype="jeans")
    store.upsert([keep, drop], np.stack([unit(1), unit(2)]))

    store.delete([drop.id])

    assert store.get(drop.id) is None
    assert store.get(keep.id) == keep
