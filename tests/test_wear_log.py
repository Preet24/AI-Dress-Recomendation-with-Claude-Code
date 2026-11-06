from datetime import date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outfit.wear_log import PHOTO_MAX_SIDE, Issue, Rating, WearEntry, WearLogStore
from tests.test_profile_photo import opened, photo_with_exif

SHIRT, JEANS = uuid4(), uuid4()


def entry(**kw) -> WearEntry:
    return WearEntry(**{"worn_on": date(2026, 10, 20), "pieces": [SHIRT, JEANS],
                        "rating": Rating.LOVED} | kw)  # fmt: skip


def test_an_entry_needs_pieces_and_a_rating():
    with pytest.raises(ValidationError):
        WearEntry(worn_on=date(2026, 10, 20), pieces=[], rating=Rating.FINE)
    with pytest.raises(ValidationError):
        WearEntry(worn_on=date(2026, 10, 20), pieces=[SHIRT])


def test_an_entry_drops_repeats_and_trims_the_note():
    e = entry(pieces=[SHIRT, SHIRT, JEANS], issues=["too_warm", "too_warm"], note="  hot  ")
    assert e.pieces == [SHIRT, JEANS] and e.issues == [Issue.TOO_WARM] and e.note == "hot"


@pytest.mark.parametrize("issues", [["too_warm", "too_cold"], ["too_formal", "too_casual"]])
def test_contradicting_issues_are_refused(issues):
    with pytest.raises(ValidationError, match="can't be both"):
        entry(issues=issues)


def test_the_store_lists_newest_first_and_deletes(tmp_path):
    store = WearLogStore(tmp_path / "wear_log.json")
    old = store.add(entry(worn_on=date(2026, 10, 1)))
    new = store.add(entry(worn_on=date(2026, 10, 20), rating=Rating.NOT_AGAIN))
    same_day_later = store.add(entry(worn_on=date(2026, 10, 20),
                                     created_at=datetime(2030, 1, 1)))  # fmt: skip

    reopened = WearLogStore(tmp_path / "wear_log.json")  # read back from the file
    assert [e.id for e in reopened.entries()] == [same_day_later.id, new.id, old.id]
    assert reopened.get(new.id).rating is Rating.NOT_AGAIN
    assert reopened.delete(old.id) and not reopened.delete(old.id)
    assert [e.id for e in WearLogStore(tmp_path / "wear_log.json").entries()] == [
        same_day_later.id,
        new.id,
    ]
    assert not (tmp_path / "wear_log.tmp").exists()  # swapped in, not left behind


def test_the_in_memory_store_keeps_copies():
    store = WearLogStore()
    added = store.add(entry())
    store.entries()[0].note = "changed outside"
    assert store.get(added.id).note == ""


def test_replacing_an_entry_keeps_its_place():
    store = WearLogStore()
    added = store.add(entry())
    assert store.replace(added.model_copy(update={"pieces": [JEANS]}))
    assert store.get(added.id).pieces == [JEANS]
    assert not store.replace(entry())  # never logged


@pytest.mark.parametrize("on_disk", [True, False])
def test_a_photo_is_kept_without_exif_and_goes_with_its_entry(tmp_path, on_disk):
    store = WearLogStore(tmp_path / "wear_log.json" if on_disk else None)
    added = store.add(entry())
    assert store.photo(added.id) is None and not store.delete_photo(added.id)

    store.save_photo(added.id, photo_with_exif())

    kept = store.photo(added.id)
    assert opened(kept).size == (768, PHOTO_MAX_SIDE) and not opened(kept).getexif()
    if on_disk:
        assert (tmp_path / "worn" / f"{added.id}.jpg").read_bytes() == kept
    store.delete(added.id)
    assert store.photo(added.id) is None
