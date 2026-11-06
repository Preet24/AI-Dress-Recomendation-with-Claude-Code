"""The shopping assistant's scoring (tools/shopping.py) and typical weather (tools/climate.py)."""

from datetime import date

import pytest

from outfit.profile import Profile
from outfit.schemas import Category, Occasion
from outfit.tools import shopping
from outfit.tools.climate import (
    ALL_KINDS,
    ARCHIVE_URL,
    WeatherKind,
    day_kind,
    typical_advice,
    typical_kinds,
)
from outfit.tools.outfits import OutfitRequest
from outfit.tools.shopping import (
    CLASHES,
    DUPLICATE,
    GOES_WITH,
    SIMILAR,
    Situation,
    accessory_gaps,
    baseline,
    check_candidate,
    pairings,
    situations,
)
from outfit.tools.weather import GEOCODING_URL, WeatherError
from tests.test_outfits import bottom, dress, shoes, top

CASUAL = [Situation("casual", Occasion.CASUAL)]
MILD = [WeatherKind.MILD]
PUNE = {"name": "Pune", "latitude": 18.5, "longitude": 73.9, "timezone": "Asia/Kolkata"}


# --- Typical weather ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("feels_max", "rain", "kind"),
    [
        (35, 0, "hot"),
        (32, 0, "hot"),
        (31.9, 0, "mild"),
        (16, 0, "mild"),
        (15.9, 0, "cold"),
        (35, 5, "rainy"),
        (10, 12, "rainy"),
    ],
)
def test_each_day_is_one_kind(feels_max, rain, kind):
    assert day_kind(feels_max, rain) == kind


def archive(feels: list[float], rain: list[float]):
    asked = []

    def fetch(url, params):
        asked.append((url, params))
        if url == GEOCODING_URL:
            return {"results": [PUNE]}
        return {"daily": {"apparent_temperature_max": feels, "precipitation_sum": rain}}

    return fetch, asked


def test_kinds_that_happen_on_enough_days_count():
    # 60 hot, 30 rainy, 7 cold (under 8%), 3 missing days of 100
    feels = [35.0] * 60 + [25.0] * 30 + [10.0] * 7 + [None] * 3
    rain = [0.0] * 60 + [9.0] * 30 + [0.0] * 7 + [None] * 3
    fetch, asked = archive(feels, rain)

    kinds, note = typical_kinds("Pune", date(2026, 10, 23), fetch)

    assert kinds == [WeatherKind.HOT, WeatherKind.RAINY]
    assert note == "Pune over the last year: hot 62%, cold 7%, rainy 31% of days"
    url, params = asked[1]
    assert url == ARCHIVE_URL
    assert (params["start_date"], params["end_date"]) == ("2025-10-17", "2026-10-16")


def test_no_city_or_no_archive_means_every_kind():
    assert typical_kinds(None, date(2026, 10, 23))[0] == ALL_KINDS

    def down(url, params):
        raise WeatherError("no network")

    kinds, note = typical_kinds("Pune", date(2026, 10, 23), down)
    assert kinds == ALL_KINDS and "couldn't look up" in note


def test_typical_days_use_the_usual_advice_rules():
    assert typical_advice(WeatherKind.HOT).warmth == (1, 2)
    cold = typical_advice(WeatherKind.COLD)
    assert cold.needs_outerwear and cold.outerwear_min_warmth == 4


def test_only_hot_and_mild_typical_days_have_strong_sun():
    sunny = {kind for kind in WeatherKind if typical_advice(kind).sunny}
    assert sunny == {WeatherKind.HOT, WeatherKind.MILD}


# --- What gets checked ---------------------------------------------------------------


def test_saved_places_come_first_then_ticked_occasions_then_defaults():
    office = ("Office", Occasion.OFFICE, (3, 4))
    assert situations([Occasion.OFFICE, Occasion.CASUAL], [office]) == [
        Situation("Office", Occasion.OFFICE, (3, 4)),
        Situation("casual", Occasion.CASUAL),
    ]
    assert [s.occasion for s in situations([], [])] == [
        Occasion.CASUAL,
        Occasion.OFFICE,
        Occasion.PARTY,
    ]


# --- The report ------------------------------------------------------------------------


def test_a_bottom_for_a_wardrobe_without_one_unlocks_outfits_and_is_a_buy():
    tops = [top(name="white"), top(name="grey", primary_color_lab=(60.0, 0.0, 0.0))]
    chinos = bottom()

    report = check_candidate(chinos, [*tops, shoes()], CASUAL, MILD)

    assert report.verdict == "buy"
    assert report.new_outfits == 2
    assert report.unlocks == ["casual on mild days"]
    assert report.color_note == "navy goes with 2 of your 2 tops"
    assert "it makes outfits possible for casual on mild days" in report.reasons


def test_shoes_fill_the_footwear_gap():
    report = check_candidate(shoes(), [top(), bottom()], CASUAL, MILD)

    assert report.fixes == ["no footwear suits casual"]
    assert report.verdict == "buy" and report.unlocks == []


def test_shoe_variants_of_one_outfit_count_once():
    wardrobe = [top(), bottom(), shoes(), shoes(primary_color_name="black")]
    report = check_candidate(top(name="grey"), wardrobe, CASUAL, MILD)
    assert report.new_outfits == 1


def test_a_near_duplicate_that_unlocks_nothing_is_a_skip():
    owned = top(name="white")
    wardrobe = [owned, bottom(), shoes()]
    again = top(name="white")

    report = check_candidate(again, wardrobe, CASUAL, MILD, look_alikes=[(owned, DUPLICATE + 0.02)])

    assert report.looks_like.item is owned
    assert report.verdict == "skip"
    assert "it is very close to your white t-shirt" in report.reasons


def test_a_similar_piece_is_mentioned_but_does_not_change_the_verdict():
    owned = top(name="white")
    wardrobe = [owned, bottom(), shoes()]
    plain = check_candidate(top(name="grey"), wardrobe, CASUAL, MILD)
    similar = check_candidate(
        top(name="grey"), wardrobe, CASUAL, MILD, look_alikes=[(owned, SIMILAR + 0.01)]
    )
    assert similar.verdict == plain.verdict
    assert "it looks similar to your white t-shirt" in similar.reasons


def test_a_photo_already_in_the_wardrobe_is_recognised():
    owned = top()
    report = check_candidate(owned, [owned, bottom(), shoes()], CASUAL, MILD)
    assert report.owned and report.looks_like.score == 1.0
    assert (report.verdict, report.reasons) == (
        "skip",
        ["your white t-shirt is already in your wardrobe (same photo)"],
    )


def test_a_garment_for_none_of_your_occasions_is_a_skip():
    gown = dress(formality=5, occasions=[Occasion.WEDDING])

    report = check_candidate(gown, [top(), bottom(), shoes()], CASUAL, MILD)

    assert report.new_outfits == 0 and report.verdict == "skip"
    assert "no outfit you'd wear includes the red sundress" in report.reasons


def test_a_couple_of_new_outfits_without_an_unlock_is_a_maybe():
    report = check_candidate(top(name="grey"), [top(), bottom(), shoes()], CASUAL, MILD)
    assert report.new_outfits == 1 and report.verdict == "maybe"


def test_every_situation_and_weather_is_checked():
    two = [Situation("casual", Occasion.CASUAL), Situation("Office", Occasion.OFFICE, (3, 4))]
    report = check_candidate(bottom(), [top()], two, [WeatherKind.HOT, WeatherKind.COLD])
    assert report.situations == [
        "casual on hot days",
        "casual on cold days",
        "Office on hot days",
        "Office on cold days",
    ]


def test_pairings_rank_partners_and_flag_clashes():
    red = bottom(name="red", primary_color_lab=(53.0, 80.0, 67.0))
    tops = [
        top(name="white"),
        top(name="green", primary_color_lab=(46.0, -52.0, 50.0)),
        top(name="yellow", primary_color_lab=(90.0, -5.0, 85.0)),
        top(name="purple", primary_color_lab=(35.0, 60.0, -55.0)),
    ]
    pairs = pairings(red, [*tops, bottom(), shoes()])

    assert {p.item.category for p in pairs} == {Category.TOP}  # only what it is worn with
    assert [p.score for p in pairs] == sorted((p.score for p in pairs), reverse=True)
    report = check_candidate(red, tops, CASUAL, MILD)
    assert [p.item for p in report.clashes] == [
        p.item for p in reversed(pairs) if p.score < CLASHES
    ]
    assert [p.item for p in report.goes_with] == [p.item for p in pairs if p.score >= GOES_WITH]


def test_taste_from_the_profile_is_used(monkeypatch):
    # white tee + navy chinos score 0.925; avoiding navy takes 0.08 off
    monkeypatch.setattr(shopping, "GOOD_OUTFIT", 0.9)
    taste = OutfitRequest(**Profile(colors_avoided=["navy"]).taste_fields())
    tops = [top(name="white")]

    assert check_candidate(bottom(), tops, CASUAL, MILD).new_outfits == 1
    assert check_candidate(bottom(), tops, CASUAL, MILD, taste=taste).new_outfits == 0


# --- Accessories -----------------------------------------------------------------------


def sunglasses(**kw):
    fields = {"category": Category.ACCESSORY, "subtype": "sunglasses", "formality": 2,
              "primary_color_name": "black", "primary_color_lab": (14.0, 0.0, 0.0)}  # fmt: skip
    return top(**fields | kw)


def test_accessories_fill_their_gaps_and_are_a_buy():
    wardrobe = [top(), bottom(), shoes()]

    report = check_candidate(sunglasses(), wardrobe, CASUAL, [WeatherKind.HOT])

    assert report.verdict == "buy"
    assert report.fixes == ["no sunglasses for the strong sun"]
    assert report.new_outfits == 1 and report.unlocks == []
    assert "it goes with 1 of your outfits" in report.reasons


def test_an_accessory_that_fills_no_gap_is_a_maybe_and_one_never_called_for_a_skip():
    wardrobe = [top(), bottom(), shoes()]
    belt = sunglasses(subtype="leather belt")

    assert check_candidate(belt, wardrobe, CASUAL, MILD).verdict == "maybe"
    cold = check_candidate(sunglasses(), wardrobe, CASUAL, [WeatherKind.COLD])
    assert cold.verdict == "skip"
    assert "none of your outfits calls for the black sunglasses" in cold.reasons


def test_accessory_gaps_come_from_the_good_outfits():
    wardrobe = [top(), bottom(), shoes()]
    kinds = [WeatherKind.HOT, WeatherKind.COLD]
    before = baseline(wardrobe, CASUAL, kinds)

    gaps = accessory_gaps(wardrobe, before)

    assert gaps[(CASUAL[0], WeatherKind.HOT)] == ["no sunglasses for the strong sun"]
    assert "no scarf for the cold" in gaps.get((CASUAL[0], WeatherKind.COLD), [])
    assert (
        accessory_gaps([*wardrobe, sunglasses()], before).get((CASUAL[0], WeatherKind.HOT)) is None
    )
