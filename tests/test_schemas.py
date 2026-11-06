import pytest
from pydantic import ValidationError

from outfit.schemas import Category, ClothingItem, Occasion, stable_item_id
from tests.factories import make_item


@pytest.mark.parametrize("field", ["formality", "warmth"])
@pytest.mark.parametrize("value", [0, 6])
def test_scales_are_limited_to_1_to_5(field, value):
    with pytest.raises(ValidationError):
        make_item(**{field: value})


def test_unknown_enum_value_is_rejected():
    with pytest.raises(ValidationError):
        make_item(category="hat")


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_item(brand="acme")


def test_json_round_trip_preserves_item(item):
    payload = item.model_dump(mode="json")

    assert payload["category"] == "top"
    assert payload["occasions"] == ["office", "casual"]
    assert ClothingItem.model_validate(payload) == item


def test_strings_are_coerced_to_enums():
    item = make_item(category="one_piece", occasions=["wedding"])

    assert item.category is Category.ONE_PIECE
    assert item.occasions == [Occasion.WEDDING]


def test_stable_item_id_is_deterministic_per_photo_and_region():
    a = stable_item_id(b"photo-bytes", "upper")

    assert a == stable_item_id(b"photo-bytes", "upper")
    assert a != stable_item_id(b"photo-bytes", "lower")
    assert a != stable_item_id(b"other-photo", "upper")
