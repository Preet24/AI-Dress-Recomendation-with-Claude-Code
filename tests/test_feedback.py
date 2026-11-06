from datetime import date

import pytest

from outfit.agents.orchestrate import orchestrate_node
from outfit.agents.state import Slots
from outfit.schemas import Category, Occasion
from outfit.tools.feedback import RECENT_DAYS, Feedback, feedback, shifted
from outfit.tools.outfits import OutfitRequest, pairing, suggest_outfits
from outfit.tools.packing import Trip, plan_trip
from outfit.wear_log import Rating, WearEntry
from tests.factories import make_item
from tests.fakes import fake_context
from tests.test_agent_flow import TRIP, WARDROBE, weather_fetch
from tests.test_outfits import advice
from tests.test_packing import GOA, SOON, TripFetch

CASUAL = [Occasion.CASUAL]
TODAY = date(2026, 10, 25)
WHITE = make_item(subtype="t-shirt", primary_color_name="white",
                  primary_color_lab=(95.0, 0.0, 0.0), occasions=CASUAL)  # fmt: skip
GREY = make_item(subtype="t-shirt", primary_color_name="grey",
                 primary_color_lab=(95.0, 0.0, 0.0), occasions=CASUAL)  # fmt: skip
JEANS = make_item(category=Category.BOTTOM, subtype="jeans", primary_color_name="black",
                  primary_color_lab=(10.0, 0.0, 0.0), occasions=CASUAL)  # fmt: skip
DRESS = make_item(category=Category.ONE_PIECE, subtype="dress", occasions=CASUAL)
SHOES = make_item(category=Category.FOOTWEAR, subtype="sneakers", occasions=CASUAL)
BELT = make_item(category=Category.ACCESSORY, subtype="belt", occasions=CASUAL)
ITEMS = [WHITE, GREY, JEANS, DRESS, SHOES, BELT]


EARLIER = date(2026, 10, 1)


def worn(*pieces, on=EARLIER, rating=Rating.LOVED, **kw) -> WearEntry:
    return WearEntry(worn_on=on, pieces=[p.id for p in pieces], rating=rating, **kw)


# --- Pairings ---------------------------------------------------------------


def test_a_pairing_is_top_and_bottom_or_a_one_piece_with_shoes():
    assert pairing([WHITE, JEANS, SHOES, BELT]) == {WHITE.id, JEANS.id}
    assert pairing([DRESS, SHOES, BELT]) == {DRESS.id, SHOES.id}
    assert pairing([DRESS, BELT]) is None and pairing([WHITE]) is None


# --- From the log -----------------------------------------------------------


def test_loved_and_not_again_pairings():
    fb = feedback(
        [worn(WHITE, JEANS, SHOES), worn(DRESS, SHOES, rating=Rating.NOT_AGAIN)], ITEMS, TODAY
    )
    assert fb.liked_pairs == {frozenset({WHITE.id, JEANS.id})}
    assert fb.disliked_pairs == {frozenset({DRESS.id, SHOES.id})}


def test_the_latest_rating_of_a_pairing_wins():
    loved = worn(WHITE, JEANS, on=date(2026, 10, 1))
    then_fine = worn(WHITE, JEANS, on=date(2026, 10, 5), rating=Rating.FINE)
    then_colours = worn(WHITE, JEANS, on=date(2026, 10, 9), issues=["colours"])

    assert feedback([then_fine, loved], ITEMS, TODAY).liked_pairs == frozenset()
    fb = feedback([loved, then_fine, then_colours], ITEMS, TODAY)
    assert fb.disliked_pairs == {frozenset({WHITE.id, JEANS.id})} and not fb.liked_pairs


def test_recent_wear_counts_base_garments_of_the_last_days():
    fb = feedback(
        [
            worn(WHITE, JEANS, SHOES, BELT, on=TODAY),
            worn(GREY, JEANS, on=date(2026, 10, 25 - RECENT_DAYS - 1)),  # too long ago
        ],
        ITEMS,
        TODAY,
    )
    assert fb.recent == {WHITE.id, JEANS.id}


def test_deleted_garments_are_skipped():
    gone = make_item(category=Category.BOTTOM, subtype="shorts", occasions=CASUAL)
    fb = feedback([worn(WHITE, gone, on=TODAY)], ITEMS, TODAY)
    assert fb.liked_pairs == frozenset() and fb.recent == {WHITE.id}


@pytest.mark.parametrize(
    ("cold", "warm", "bias"), [(2, 0, 1), (3, 1, 1), (1, 0, 0), (2, 1, 0), (0, 2, -1)]
)
def test_the_warmth_bias_needs_two_more_reports_one_way(cold, warm, bias):
    entries = [worn(WHITE, JEANS, issues=["too_cold"]) for _ in range(cold)]
    entries += [worn(WHITE, JEANS, issues=["too_warm"]) for _ in range(warm)]
    assert feedback(entries, ITEMS, TODAY).warmth_bias == bias


def test_the_warmth_note_says_why():
    assert Feedback(too_cold=3, too_warm=1).warmth_note() == (
        "you've found outfits too cold 3 times and too warm once, "
        "so this picks one warmth point warmer"
    )
    assert Feedback(too_warm=2).warmth_note().endswith("one warmth point lighter")
    assert Feedback(too_cold=1).warmth_note() is None


def test_shifted_moves_the_warmth_range_within_1_to_5():
    cold = advice(warmth=(2, 4), needs_outerwear=True, outerwear_min_warmth=4)
    warmer = shifted(cold, 1)
    assert warmer.warmth == (3, 5) and warmer.outerwear_min_warmth == 5
    assert shifted(advice(warmth=(1, 2)), -1).warmth == (1, 1)
    assert shifted(None, 1) is None and shifted(cold, 0) is cold


# --- In the ranking ---------------------------------------------------------


def ranked(**history) -> list[set]:
    request = OutfitRequest(occasion=Occasion.CASUAL, **history)
    return [
        {p.id for p in o.pieces} for o in suggest_outfits([WHITE, GREY, JEANS], request).outfits
    ]


def test_a_loved_pairing_rises_with_a_reason():
    first, second = ranked()
    loved = frozenset(second)
    plan = suggest_outfits(
        [WHITE, GREY, JEANS], OutfitRequest(occasion=Occasion.CASUAL, liked_pairs={loved})
    )
    assert {p.id for p in plan.outfits[0].pieces} == second
    assert any(r.startswith("you loved your") for r in plan.outfits[0].reasons)


def test_a_disliked_pairing_sinks_but_stays():
    first, second = ranked()
    after = ranked(disliked_pairs={frozenset(first)})
    assert after == [second, first]


def test_recently_worn_garments_sink_a_little():
    first, second = ranked()
    [top_id] = first - {JEANS.id}
    plan = suggest_outfits(
        [WHITE, GREY, JEANS], OutfitRequest(occasion=Occasion.CASUAL, recent={top_id})
    )
    assert {p.id for p in plan.outfits[0].pieces} == second
    assert any("in the last few days" in r for r in plan.outfits[1].reasons)


# --- Chat and packing -------------------------------------------------------


def test_chat_reads_the_log_and_dresses_a_cold_natured_user_warmer():
    cold = [worn(*WARDROBE, issues=["too_cold"]) for _ in range(2)]
    context = fake_context(WARDROBE, fetch=weather_fetch, wear_log=cold)

    update = orchestrate_node({"slots": Slots(occasion=Occasion.CASUAL, **TRIP)}, context=context)

    assert update["advice"].warmth == (2, 3)  # 29 °C calls for (1, 2)
    assert any("too cold twice" in a for a in update["assumptions"])
    assert any("you loved your" in r for r in update["plan"].outfits[0].reasons)


def test_without_weather_the_bias_is_not_mentioned():
    cold = [worn(*WARDROBE, issues=["too_cold"]) for _ in range(2)]
    context = fake_context(WARDROBE, wear_log=cold)  # the forecast fails

    update = orchestrate_node({"slots": Slots(occasion=Occasion.CASUAL, **TRIP)}, context=context)

    assert update["advice"] is None
    assert not any("warmth point" in a for a in update["assumptions"])


def test_packing_uses_pairings_and_warmth_but_not_recent_wear():
    entries = [worn(WHITE, JEANS, on=SOON, issues=["too_cold"]) for _ in range(2)]
    history = feedback(entries, ITEMS, SOON)
    trip = Trip(city="Goa", place=GOA, first_day=date(2026, 10, 30), days=1, travel_days=False)

    plan = plan_trip([WHITE, GREY, JEANS], trip, SOON, TripFetch(), history=history)

    assert plan.days[0].outfit.ids == {WHITE.id, JEANS.id}  # loved, and not held back
    assert not any("last few days" in r for r in plan.days[0].outfit.reasons)
    assert plan.assumptions[-1].startswith("you've found outfits too cold twice")
