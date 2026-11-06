"""Wardrobe endpoints over an in-memory store (no network, no CLIP)."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from outfit.api.app import create_app
from outfit.schemas import Category, Occasion
from tests.factories import make_item
from tests.fakes import fake_services


@pytest.fixture
def wardrobe(tmp_path):
    cutout = tmp_path / "skirt_lower.png"
    Image.new("RGBA", (4, 4), (180, 160, 210, 255)).save(cutout)
    skirt = make_item(
        image_path=cutout,
        category=Category.BOTTOM,
        subtype="skirt",
        primary_color_name="lavender",
        occasions=[Occasion.PARTY, Occasion.CASUAL],
    )
    shirt = make_item(subtype="button-up shirt", primary_color_name="black")
    services = fake_services([skirt, shirt])
    with TestClient(create_app(services)) as client:
        yield client, services, skirt, shirt


def test_list_shows_every_item_with_name_and_image_url_but_no_local_path(wardrobe):
    client, _, skirt, shirt = wardrobe
    items = client.get("/items").json()
    assert [i["name"] for i in items] == ["black button-up shirt", "lavender skirt"]  # top first
    assert items[1]["image_url"] == f"/items/{skirt.id}/image"
    assert all("image_path" not in i for i in items)


def test_list_filters_by_category_and_occasion(wardrobe):
    client, _, skirt, shirt = wardrobe
    names = lambda params: [i["name"] for i in client.get("/items", params=params).json()]  # noqa: E731
    assert names({"category": "bottom"}) == ["lavender skirt"]
    assert names({"category": ["top", "bottom"]}) == ["black button-up shirt", "lavender skirt"]
    assert names({"occasion": "party"}) == ["lavender skirt"]
    assert names({"occasion": "wedding"}) == []
    assert client.get("/items", params={"category": "hat"}).status_code == 422


def test_get_item_and_unknown_or_malformed_ids(wardrobe):
    client, _, skirt, _ = wardrobe
    response = client.get(f"/items/{skirt.id}")
    assert response.status_code == 200
    assert response.json()["subtype"] == "skirt"
    assert client.get(f"/items/{uuid4()}").status_code == 404
    assert client.get("/items/not-a-uuid").status_code == 422


def test_image_serves_the_cutout_png(wardrobe):
    client, _, skirt, shirt = wardrobe
    response = client.get(f"/items/{skirt.id}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == skirt.image_path.read_bytes()
    missing = client.get(f"/items/{shirt.id}/image")  # the factory's path doesn't exist
    assert missing.status_code == 404
    assert "missing" in missing.json()["detail"]


def test_edit_changes_only_the_given_tags_and_saves_them(wardrobe):
    client, services, skirt, _ = wardrobe
    response = client.patch(f"/items/{skirt.id}", json={"length": "midi", "formality": 4})
    assert response.status_code == 200
    assert response.json()["length"] == "midi"
    saved = services.store.get(skirt.id)
    assert (saved.length, saved.formality, saved.subtype) == ("midi", 4, "skirt")
    assert saved.created_at == skirt.created_at


@pytest.mark.parametrize(
    "body",
    [
        {},  # nothing to change
        {"colour": "lilac"},  # unknown field
        {"id": str(uuid4())},  # pipeline-owned, not a tag
        {"image_path": "elsewhere.png"},
        {"formality": 9},  # out of 1-5
        {"category": "hat"},
        {"subtype": None},
    ],
)
def test_bad_edits_are_rejected_and_nothing_changes(wardrobe, body):
    client, services, skirt, _ = wardrobe
    assert client.patch(f"/items/{skirt.id}", json=body).status_code == 422
    assert services.store.get(skirt.id) == skirt


def test_delete_removes_the_item_and_its_cutout(wardrobe):
    client, services, skirt, _ = wardrobe
    assert client.delete(f"/items/{skirt.id}").status_code == 204
    assert services.store.get(skirt.id) is None
    assert not skirt.image_path.exists()
    assert client.delete(f"/items/{skirt.id}").status_code == 404


def test_openapi_item_schema_has_no_local_path(wardrobe):
    client, *_ = wardrobe
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "image_path" not in schemas["ItemOut"]["properties"]
    assert {"name", "image_url", "occasions"} <= set(schemas["ItemOut"]["properties"])
    assert set(schemas["TagEdits"]["properties"]) == {
        "category", "subtype", "primary_color_name", "secondary_colors", "pattern", "fabric",
        "length", "fit", "formality", "warmth", "flowy", "occasions",
    }  # fmt: skip
