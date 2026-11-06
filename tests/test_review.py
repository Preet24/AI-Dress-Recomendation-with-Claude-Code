from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from outfit.config import Settings
from outfit.ingestion.background import GarmentRegion, SegmentationError, SegMode
from outfit.ingestion.review import (
    EditError,
    apply_field_edits,
    changes,
    parse_field_edits,
    region_of,
    remove_item,
    replace_with_photo,
    retag_with_note,
)
from outfit.schemas import Category, Length, Occasion
from tests.factories import make_item
from tests.fakes import FakeClient, tags_reply
from tests.test_pipeline import WORN_MASKS, H, MaskSession, W, band, make_pipeline


def test_parse_field_edits_handles_quotes_and_lists():
    edits = parse_field_edits('category=one_piece subtype="slip dress" occasions=party,formal')

    assert edits == {
        "category": "one_piece",
        "subtype": "slip dress",
        "occasions": ["party", "formal"],
    }


@pytest.mark.parametrize(
    "text, message",
    [("colour=red", "Unknown field 'colour'"), ("midi", "key=value"), ("", "No edits")],
)
def test_parse_field_edits_rejects_bad_input(text, message):
    with pytest.raises(EditError, match=message):
        parse_field_edits(text)


def test_pipeline_owned_fields_are_not_editable():
    with pytest.raises(EditError, match="Unknown field 'primary_color_lab'"):
        parse_field_edits("primary_color_lab=1,2,3")


def test_apply_field_edits_validates_and_converts():
    item = make_item()

    edited = apply_field_edits(
        item, {"category": "one_piece", "length": "midi", "formality": "4", "flowy": "true"}
    )

    assert edited.category is Category.ONE_PIECE and edited.length is Length.MIDI
    assert (edited.formality, edited.flowy) == (4, True)
    assert (edited.id, edited.created_at) == (item.id, item.created_at)
    assert set(changes(item, edited)) == {"category", "length", "formality", "flowy"}


def test_apply_field_edits_rejects_invalid_values():
    with pytest.raises(EditError, match="formality"):
        apply_field_edits(make_item(), {"formality": "9"})


@pytest.mark.parametrize(
    "name, region",
    [("photo_ab12cd34_upper.png", "upper"), ("my_photo_ab12_full.png", "full"), ("x.png", "whole")],
)
def test_region_of_reads_the_cutout_name(name, region):
    assert region_of(make_item(image_path=Path(name))) is GarmentRegion(region)


def test_retag_with_note_sends_note_and_keeps_pipeline_fields(tmp_path):
    cutout = tmp_path / "dress_ab12cd34_upper.png"
    Image.new("RGBA", (8, 8), (10, 10, 10, 255)).save(cutout)
    item = make_item(image_path=cutout)
    client = FakeClient(tags_reply(category="one_piece", subtype="slip dress", length="midi"))
    settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")

    new = retag_with_note(item, "black polka-dot midi slip dress", client, settings)

    text_block, _ = client.requests[0]["messages"][1]["content"]
    assert "`upper`" in text_block["text"] and "polka-dot" in text_block["text"]
    assert (new.category, new.subtype) == (Category.ONE_PIECE, "slip dress")
    assert new.id == item.id and new.primary_color_lab == item.primary_color_lab
    assert new.created_at == item.created_at and new.image_path == cutout


@pytest.fixture
def photo(tmp_path):
    img = Image.new("RGB", (W, H), (200, 20, 30))
    img.paste((20, 30, 90), (0, H // 2, W, H))
    path = tmp_path / "outfit.jpg"
    img.save(path, quality=95)
    return path


def test_replace_with_photo_swaps_the_item(tmp_path, photo):
    # Worn mode split a dress into upper + lower; replace the upper with a flat re-ingest.
    client = FakeClient(
        tags_reply(),
        tags_reply(category="bottom"),
        tags_reply(category="one_piece", occasions=["party"]),
    )
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    upper, lower = pipeline.ingest_image(photo, SegMode.WORN)
    pipeline.session_for = lambda name: MaskSession(band(5, 115))

    [dress] = replace_with_photo(pipeline, upper, photo, SegMode.FLAT)

    assert dress.category is Category.ONE_PIECE and dress.occasions == [Occasion.PARTY]
    assert pipeline.store.get(upper.id) is None and not upper.image_path.exists()
    assert pipeline.store.get(lower.id) == lower  # the sibling is reviewed on its own
    assert pipeline.store.count() == 2


def test_replace_with_same_photo_and_mode_updates_in_place(tmp_path, photo):
    client = FakeClient(tags_reply(), tags_reply(), tags_reply(subtype="polo"), tags_reply())
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    upper, _ = pipeline.ingest_image(photo, SegMode.WORN)

    replace_with_photo(pipeline, upper, photo, SegMode.WORN)

    assert pipeline.store.get(upper.id).subtype == "polo"
    assert upper.image_path.exists()
    assert pipeline.store.count() == 2


def test_failed_replacement_keeps_the_old_item(tmp_path, photo):
    client = FakeClient(tags_reply(), tags_reply())
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    upper, _ = pipeline.ingest_image(photo, SegMode.WORN)
    pipeline.session_for = lambda name: MaskSession(np.zeros((H, W), dtype=np.uint8))

    with pytest.raises(SegmentationError):
        replace_with_photo(pipeline, upper, photo, SegMode.FLAT)
    assert pipeline.store.get(upper.id) == upper


def test_remove_item_deletes_point_and_cutout(tmp_path, photo):
    pipeline = make_pipeline(tmp_path, FakeClient(tags_reply(), tags_reply()), *WORN_MASKS)
    upper, _ = pipeline.ingest_image(photo, SegMode.WORN)

    remove_item(pipeline.store, upper)

    assert pipeline.store.get(upper.id) is None and not upper.image_path.exists()
