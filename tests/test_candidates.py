"""Describing a photo without storing it, and candidates that wait to be added."""

import numpy as np

from outfit.ingestion.background import SegMode
from outfit.ingestion.candidates import CandidateStore
from tests.fakes import FakeClient, tags_reply
from tests.test_pipeline import WORN_MASKS, make_pipeline, photo  # noqa: F401 (fixture)


def described(tmp_path, photo):  # noqa: F811
    client = FakeClient(
        tags_reply(),
        tags_reply(category="bottom", subtype="jeans", primary_color_name="navy", length="full"),
    )
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    items, vectors = pipeline.describe_photo(photo, SegMode.WORN, tmp_path / "candidates")
    return pipeline, items, vectors


def test_describing_tags_and_embeds_but_stores_nothing(tmp_path, photo):  # noqa: F811
    pipeline, items, vectors = described(tmp_path, photo)

    assert [i.subtype for i in items] == ["t-shirt", "jeans"]
    assert vectors.shape == (2, 512)
    assert all(i.image_path.parent == tmp_path / "candidates" for i in items)
    assert all(i.image_path.is_file() for i in items)
    assert pipeline.store.count() == 0
    assert not (tmp_path / "clean").exists()


def test_adding_a_candidate_stores_it_without_tagging_again(tmp_path, photo):  # noqa: F811
    pipeline, items, vectors = described(tmp_path, photo)
    candidates = CandidateStore()
    top, jeans = candidates.put(items, vectors, photo)
    requests = len(pipeline.client.requests)

    clean, raw = tmp_path / "clean", tmp_path / "wardrobe_raw"
    added = candidates.add_to_wardrobe(jeans.item.id, pipeline.store, clean, raw)

    assert added.image_path == clean / jeans.item.image_path.name and added.image_path.is_file()
    assert pipeline.store.get(jeans.item.id) == added
    assert np.allclose(pipeline.store.vector(jeans.item.id), jeans.vector, atol=1e-6)
    assert len(pipeline.client.requests) == requests  # no new DeepSeek call
    assert candidates.get(jeans.item.id) is None
    assert photo.is_file()  # the t-shirt from the same photo still needs it

    candidates.add_to_wardrobe(top.item.id, pipeline.store, clean, raw)
    assert (raw / photo.name).is_file() and not photo.exists()
    assert candidates.add_to_wardrobe(top.item.id, pipeline.store, clean, raw) is None


def test_only_the_newest_candidates_are_kept_and_old_files_go(tmp_path, photo):  # noqa: F811
    _, items, vectors = described(tmp_path, photo)
    candidates = CandidateStore(limit=1)
    old, new = candidates.put(items, vectors, photo)

    assert candidates.get(old.item.id) is None and candidates.get(new.item.id) == new
    assert not old.item.image_path.exists()
    assert new.item.image_path.is_file() and photo.is_file()  # still used by the newest
