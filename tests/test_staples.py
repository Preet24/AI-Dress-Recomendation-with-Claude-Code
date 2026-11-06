"""What should I buy? (tools/staples.py) over small hand-made wardrobes."""

import pytest
from pydantic import ValidationError

from outfit.schemas import Category, Occasion
from outfit.tools.climate import WeatherKind
from outfit.tools.outfits import OutfitRequest, label
from outfit.tools.shopping import (
    Situation,
    accessory_gaps,
    baseline,
    check_candidate,
    group_where,
)
from outfit.tools.staples import (
    STAPLES,
    Staple,
    for_gender,
    missing,
    owns_kind,
    pick_color,
    suggest_purchases,
)
from tests.test_outfits import bottom, shoes, top

CASUAL = [Situation("casual", Occasion.CASUAL)]
SPORTS = [Situation("sports", Occasion.SPORTS)]
MILD = [WeatherKind.MILD]
HOT_AND_RAINY = [WeatherKind.HOT, WeatherKind.RAINY]


def staple(category, subtype, colors, formality=2, warmth=2, occasions=(Occasion.CASUAL,)):
    return Staple(
        category=category,
        subtype=subtype,
        colors=colors,
        fabric="cotton",
        formality=formality,
        warmth=warmth,
        occasions=occasions,
    )


SNEAKERS = staple(Category.FOOTWEAR, "sneakers", ("white", "black"))
LOAFERS = staple(Category.FOOTWEAR, "loafers", ("brown", "black"), formality=3)
TEE = staple(Category.TOP, "sports tee", ("black", "grey"), 1, 1, (Occasion.SPORTS,))
TRACKS = staple(Category.BOTTOM, "track pants", ("navy", "grey"), 1, 2, (Occasion.SPORTS,))


def test_the_catalogue_builds_valid_garments():
    for staple in STAPLES:
        for color in staple.colors:
            item = staple.item(color)
            assert item.subtype == staple.subtype and item.primary_color_name == color
    assert len({s.item(s.colors[0]).id for s in STAPLES}) == len(STAPLES)  # stable, distinct


def test_a_staple_with_an_unknown_color_is_refused():
    with pytest.raises(ValidationError, match="colors must be some of"):
        staple(Category.TOP, "t-shirt", ("chartreuse",))


def test_gendered_staples_are_left_out_only_for_the_other_gender():
    names = {s.subtype for s in for_gender("man")}
    assert "polo shirt" in names and "pumps" not in names
    assert {s.subtype for s in for_gender(None)} == {s.subtype for s in STAPLES}
    assert len(for_gender("non_binary")) == len(STAPLES)


def test_owning_a_kind_matches_subtype_words_and_color():
    owned = shoes(subtype="canvas sneakers", primary_color_name="white")
    assert owns_kind(SNEAKERS.item("white"), [owned])
    assert not owns_kind(SNEAKERS.item("black"), [owned])
    assert not owns_kind(LOAFERS.item("brown"), [owned])


def test_the_color_skips_owned_and_avoided_ones():
    wardrobe = [top(), bottom(), shoes()]  # white sneakers already owned
    assert pick_color(SNEAKERS, wardrobe, OutfitRequest()).primary_color_name == "black"
    avoid_black = OutfitRequest(colors_avoided=frozenset({"black"}))
    assert pick_color(SNEAKERS, wardrobe, avoid_black) is None


def test_the_color_goes_with_most_pieces_and_liked_colors_win_ties():
    # white + navy chinos 0.85 (goes with), black + navy 0.715 (flat: too little contrast)
    assert pick_color(SNEAKERS, [bottom()], OutfitRequest()).primary_color_name == "white"
    # with a white tee as well, each color goes with one piece: the liked one wins
    wardrobe = [top(), bottom()]
    for liked in ("white", "black"):
        taste = OutfitRequest(colors_liked=frozenset({liked}))
        assert pick_color(SNEAKERS, wardrobe, taste).primary_color_name == liked


def test_places_and_weathers_are_grouped():
    office = Situation("Office", Occasion.OFFICE)
    found = [(office, WeatherKind.HOT), (office, WeatherKind.RAINY)]
    assert group_where(found, HOT_AND_RAINY) == ["Office in any weather"]
    assert group_where(found, [*HOT_AND_RAINY, WeatherKind.MILD]) == [
        "Office on hot and rainy days"
    ]
    assert group_where(found[:1], [WeatherKind.HOT]) == ["Office on hot days"]


def test_missing_lists_empty_situations_with_their_closest_misses_and_the_gaps():
    found = missing(baseline([top(), bottom()], SPORTS + CASUAL, MILD), MILD)

    [sports] = found.no_outfit
    assert sports.startswith(
        "sports on mild days (no top fits: white t-shirt is not tagged for sports"
    )
    assert found.gaps == {
        "no footwear suits sports": ["sports on mild days"],
        "no footwear suits casual": ["casual on mild days"],
    }


def test_the_nothing_fits_gap_is_an_unlock_not_a_fix():
    report = check_candidate(bottom(), [top(), shoes()], CASUAL, MILD)
    assert report.unlocks == ["casual on mild days"] and report.fixes == []


def test_shoes_are_suggested_for_the_footwear_gap_but_only_one_pair():
    picks = suggest_purchases([bottom(), top()], CASUAL, MILD, staples=(SNEAKERS, LOAFERS))

    [pick] = picks  # the loafers would fix the same gap: once is enough
    assert pick.report.candidate.subtype == "sneakers"
    assert pick.report.fixes == ["no footwear suits casual"]
    assert pick.report.verdict == "buy" and pick.report.color_note in pick.report.reasons


def test_more_shoes_for_a_wardrobe_that_has_them_are_not_suggested():
    wardrobe = [top(), bottom(), shoes()]
    assert suggest_purchases(wardrobe, CASUAL, MILD, staples=(SNEAKERS, LOAFERS)) == []


def test_a_top_and_a_bottom_are_suggested_together_when_neither_unlocks_alone():
    picks = suggest_purchases([top(), bottom()], SPORTS, MILD, staples=(TEE, TRACKS))

    tee, tracks = (label(p.report.candidate) for p in picks)
    assert [p.report.candidate.subtype for p in picks] == ["sports tee", "track pants"]
    assert (picks[0].together_with, picks[1].together_with) == (tracks, tee)
    assert picks[1].after == [tee]
    assert all(p.report.unlocks == ["sports on mild days"] for p in picks)


def test_no_pair_when_the_limit_has_no_room_for_two():
    assert suggest_purchases([top(), bottom()], SPORTS, MILD, limit=1, staples=(TEE, TRACKS)) == []


SUNGLASSES = staple(Category.ACCESSORY, "sunglasses", ("black", "brown"))
COAT = staple(Category.OUTERWEAR, "wool coat", ("navy", "black"), 4, 5)


def test_missing_includes_accessory_gaps():
    wardrobe = [top(), bottom(), shoes()]
    before = baseline(wardrobe, CASUAL, [WeatherKind.HOT])

    found = missing(before, [WeatherKind.HOT], accessory_gaps(wardrobe, before))

    assert found.gaps["no sunglasses for the strong sun"] == ["casual on hot days"]


def test_accessories_are_suggested_for_their_gap_but_after_clothes_that_fill_one():
    wardrobe = [top(), bottom(), shoes()]
    kinds = [WeatherKind.HOT, WeatherKind.COLD]

    picks = suggest_purchases(wardrobe, CASUAL, kinds, staples=(SUNGLASSES, COAT))

    assert [s.staple.subtype for s in picks] == ["wool coat", "sunglasses"]
    assert picks[1].report.fixes == ["no sunglasses for the strong sun"]
