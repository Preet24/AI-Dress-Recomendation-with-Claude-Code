import json

import pytest

from outfit.evals.fixtures import WARDROBE_PATH
from outfit.schemas import Category, ClothingItem, Pattern
from outfit.tools.match import best_matches, tag_matches, tokens
from outfit.tools.outfits import label
from tests.factories import make_item


def test_tokens_stem_alias_and_drop_stop_words():
    assert tokens("My blue Jeans") == ["blue", "jean"]
    assert tokens("the white tee") == ["white", "t-shirt"]
    assert tokens("that dress") == ["dress"]  # "dress" keeps its s


def test_the_head_noun_must_be_a_garment_the_item_is():
    jeans = make_item(category=Category.BOTTOM, subtype="jeans", primary_color_name="blue")

    assert tag_matches("blue sweater", [jeans]) == []  # a color alone is not enough
    assert tag_matches("blue jeans", [jeans]) == [(jeans, 2)]
    assert tag_matches("my bottoms", [jeans]) == [(jeans, 1)]  # category words count


def test_hyphenated_tags_match_their_parts():
    tee = make_item(subtype="graphic t-shirt", primary_color_name="off-white")

    assert best_matches("white tee", [tee]) == [tee]
    assert best_matches("white shirt", [tee]) == [tee]


def test_pattern_and_fabric_help_rank():
    plaid = make_item(subtype="shirt", primary_color_name="mauve", pattern=Pattern.CHECKED)
    linen = make_item(subtype="shirt", primary_color_name="white", fabric="linen")

    assert best_matches("linen shirt", [plaid, linen]) == [linen]
    assert best_matches("my shirt", [plaid, linen]) == [plaid, linen]  # a tie for CLIP


def real_items() -> dict[str, ClothingItem]:
    data = json.loads(WARDROBE_PATH.read_text(encoding="utf-8"))
    items = [ClothingItem.model_validate(e["item"]) for e in data["items"]]
    return {label(i): i for i in items}


@pytest.mark.skipif(not WARDROBE_PATH.exists(), reason="no wardrobe snapshot")
@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        # CLIP alone got these wrong on the real wardrobe:
        ("my black sequin top", ["black sequin top"]),  # right item, below 0.24
        ("lavender skirt", ["lavender skirt"]),  # CLIP ranked the beige trousers first
        ("the white tee", ["off-white graphic t-shirt"]),
        ("my heels", ["champagne slingback heel pumps"]),
        ("blue jeans", ["medium blue jeans"]),
        ("pink dress", ["blush pink floral midi dress"]),
        ("plaid shirt", ["mauve plaid shirt"]),
        ("beige trousers", ["beige trousers"]),
        ("black top", ["black button-up shirt", "black sequin top"]),  # tie: CLIP decides
        ("red saree", []),  # not owned: falls back to CLIP and its threshold
        ("leather jacket", []),
        ("sneakers", []),
    ],
)
def test_real_wardrobe_references(reference, expected):
    items = real_items()

    assert sorted(label(i) for i in best_matches(reference, list(items.values()))) == expected
