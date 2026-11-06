"""The wardrobe page, run headless with Streamlit's AppTest against the real app over fakes.

(AppTest can't drive st.file_uploader, so uploads are covered by the API and client tests;
here only the message and the disabled button.)
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from outfit.api.app import create_app
from outfit.schemas import Category, Occasion
from outfit.ui.client import Item, OutfitAPI
from outfit.ui.wardrobe_page import added_message, label, parse_colors, tag_changes
from tests.factories import make_item
from tests.fakes import FakeClient, fake_services, tags_reply


def wardrobe_script():
    import streamlit as st

    from outfit.ui import wardrobe_page

    wardrobe_page.render(st.session_state.api)


@pytest.fixture
def setup(tmp_path):
    """(AppTest after the first run, services, fake DeepSeek client, skirt, shirt)."""
    cutouts = []
    for name in ("skirt", "shirt"):
        path = tmp_path / f"{name}_whole.png"
        Image.new("RGBA", (6, 6), (180, 160, 210, 255)).save(path)
        cutouts.append(path)
    skirt = make_item(
        image_path=cutouts[0],
        category=Category.BOTTOM,
        subtype="skirt",
        primary_color_name="lavender",
        occasions=[Occasion.PARTY],
    )
    shirt = make_item(image_path=cutouts[1], subtype="button-up shirt", primary_color_name="black")
    client = FakeClient()
    services = fake_services([skirt, shirt], client=client)
    with TestClient(create_app(services)) as http:
        at = AppTest.from_function(wardrobe_script, default_timeout=30)
        at.session_state["api"] = OutfitAPI(http=http)
        yield at.run(), services, client, skirt, shirt


def submit(at: AppTest, form: str) -> AppTest:
    [button] = [b for b in at.button if b.proto.form_id == form]
    return button.click().run()


def test_the_grid_shows_each_item_with_its_picture_and_tags(setup):
    at, _, _, skirt, shirt = setup
    assert not at.exception
    assert at.title[0].value == "Your wardrobe"
    captions = [c.value for c in at.caption]
    assert "2 items" in captions
    assert "bottom · formality 3/5 · warmth 2/5 · party" in captions
    pictures = [p.caption for block in at.get("imgs") for p in block.proto.imgs]
    assert pictures == ["black button-up shirt", "lavender skirt"]  # tops first


def test_filters_narrow_the_grid(setup):
    at, *_ = setup
    at.multiselect[0].set_value([Category.BOTTOM]).run()
    assert "1 item" in [c.value for c in at.caption]

    at.multiselect[0].set_value([]).run()
    at.multiselect[1].set_value([Occasion.WEDDING]).run()
    assert at.info[0].value == "Nothing matches these filters."


def test_editing_a_tag_saves_only_that_change(setup):
    at, services, _, skirt, _ = setup
    at.slider(key=f"formality-{skirt.id}").set_value(4)
    at.selectbox(key=f"length-{skirt.id}").set_value("midi")

    at = submit(at, f"edit-{skirt.id}")

    assert at.success[0].value == "Saved lavender skirt: length, formality"
    saved = services.store.get(skirt.id)
    assert (saved.formality, saved.length, saved.subtype) == (4, "midi", "skirt")


def test_saving_an_unchanged_form_changes_nothing(setup):
    at, services, _, skirt, _ = setup
    at = submit(at, f"edit-{skirt.id}")
    assert "Nothing changed" in [i.value for i in at.info]
    assert services.store.get(skirt.id) == skirt


def test_an_edit_the_api_refuses_shows_its_reason(setup):
    at, services, _, skirt, _ = setup
    at.text_input(key=f"subtype-{skirt.id}").set_value("   ")

    at = submit(at, f"edit-{skirt.id}")

    assert "subtype" in at.error[0].value
    assert services.store.get(skirt.id) == skirt


def test_retag_with_a_note(setup):
    at, services, client, skirt, _ = setup
    client.replies.append(tags_reply(category="bottom", subtype="midi skirt", length="midi"))
    at.text_input(key=f"note-{skirt.id}").set_value("it's a midi skirt")

    at = submit(at, f"retag-{skirt.id}")

    assert at.success[0].value == "Re-tagged as red midi skirt"
    assert "it's a midi skirt" in client.requests[0]["messages"][1]["content"][0]["text"]
    assert services.store.get(skirt.id).subtype == "midi skirt"


def test_delete_asks_first_and_cancel_keeps_the_item(setup):
    at, services, _, skirt, _ = setup
    at.button(key=f"delete-{skirt.id}").click().run()
    assert "can't be undone" in at.warning[0].value

    at.button(key=f"cancel-{skirt.id}").click().run()
    assert not at.warning and services.store.get(skirt.id) is not None

    at.button(key=f"delete-{skirt.id}").click().run()
    at.button(key=f"yes-{skirt.id}").click().run()
    assert at.success[0].value == "Deleted lavender skirt"
    assert services.store.get(skirt.id) is None
    assert "1 item" in [c.value for c in at.caption]


def test_an_empty_wardrobe_points_to_the_upload():
    with TestClient(create_app(fake_services())) as http:
        at = AppTest.from_function(wardrobe_script, default_timeout=30)
        at.session_state["api"] = OutfitAPI(http=http)
        at.run()
        assert at.info[0].value == "No items yet: add a photo above."
        assert at.button[0].label == "Add to wardrobe" and at.button[0].disabled


def test_tag_changes_keeps_only_real_changes_as_api_values():
    item = Item.model_validate(
        {
            **make_item().model_dump(mode="json", exclude={"image_path", "created_at"}),
            "name": "light blue oxford shirt",
            "image_url": "/items/x/image",
        }
    )
    values = {
        "category": Category.ONE_PIECE,
        "subtype": item.subtype,
        "secondary_colors": ["white"],
        "occasions": [Occasion.PARTY, Occasion.OFFICE],
    }
    assert tag_changes(item, values) == {"category": "one_piece", "occasions": ["party", "office"]}


def test_small_helpers():
    assert label(Category.ONE_PIECE) == "one piece"
    assert parse_colors(" white, ,navy ") == ["white", "navy"]
    one = Item.model_validate(
        {**make_item().model_dump(mode="json"), "name": "a", "image_url": "u"}
    )
    assert added_message([one]) == "Added 1 garment: a"
    assert added_message([one, one]) == "Added 2 garments: a, a"
