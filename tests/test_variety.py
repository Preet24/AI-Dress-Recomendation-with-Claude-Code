from datetime import datetime

from outfit.schemas import Occasion
from outfit.tools.outfits import Outfit
from outfit.tools.packing import DayOutfit, PackingPlan, Slot, Wears
from outfit.tools.variety import check_variety
from outfit.tools.weather import Outing, Transport
from tests.test_packing import EVERYDAY, plan_for, shoes, top, trip, trousers

WHITE, GREY = top(), top("t-shirt", "grey", (60.0, 0.0, 0.0))
NAVY = trousers()
BEIGE = trousers(primary_color_name="beige", primary_color_lab=(80.0, 3.0, 15.0))
LOAFERS = shoes()


def plan_of(*days) -> PackingPlan:
    """A plan from (day, pieces) pairs: the checker only reads days and pieces."""
    built = []
    for day, pieces in days:
        outing = Outing(start=datetime(2026, 11, 10 + day, 10), hours_away=8,
                        hours_outdoors=1, transport=Transport.WALK)  # fmt: skip
        outfit = Outfit(tuple(pieces), 0.9, 0.9, 1.0, ()) if pieces else None
        built.append(DayOutfit(Slot(day, Occasion.CASUAL, outing), outfit))
    return PackingPlan(pieces=[], days=built, gaps=[], notes=[])


def kinds(repeats) -> list[str]:
    return [r.kind for r in repeats]


def test_different_outfits_every_day_pass():
    plan = plan_of((1, [WHITE, NAVY, LOAFERS]), (2, [GREY, BEIGE, LOAFERS]),
                   (3, [WHITE, BEIGE, LOAFERS]))  # fmt: skip
    assert check_variety(plan, Wears()) == []  # the shoes repeat: fine


def test_the_same_top_and_bottom_on_two_days_is_a_problem():
    plan = plan_of((1, [WHITE, NAVY, LOAFERS]), (2, [GREY, BEIGE]), (3, [WHITE, NAVY]))

    [repeat] = check_variety(plan, Wears())

    assert repeat.problem and repeat.days == (1, 3)
    assert str(repeat) == "same outfit: white oxford shirt + navy trousers on days 1 and 3"


def test_a_piece_past_its_limit_is_a_problem():
    plan = plan_of((1, [WHITE, NAVY]), (2, [GREY, NAVY]), (3, [WHITE, BEIGE]),
                   (4, [GREY, NAVY]), (5, [WHITE, NAVY]))  # fmt: skip

    found = check_variety(plan, Wears(tops=3, bottoms=3))

    over = [r for r in found if r.kind == "over limit"]
    assert [str(r) for r in over] == ["over limit: navy trousers on days 1-2 and 4-5 (limit 3)"]
    assert kinds(found).count("same outfit") == 2  # white+navy on 1, 5; grey+navy on 2, 4


def test_two_slots_on_one_day_count_once_and_a_top_two_days_running_is_a_note():
    plan = plan_of((1, [WHITE, NAVY]), (1, [WHITE, NAVY, LOAFERS]), (2, [WHITE, BEIGE]))

    [repeat] = check_variety(plan, Wears())

    assert repeat.kind == "two days running" and not repeat.problem
    assert repeat.days == (1, 2)


def test_days_with_nothing_to_wear_are_skipped():
    assert check_variety(plan_of((1, []), (2, [])), Wears()) == []


def test_a_real_packed_trip_within_the_limits_has_no_problems():
    items = [WHITE, GREY, NAVY, BEIGE, LOAFERS, top("polo", "red", (45.0, 60.0, 35.0))]
    t = trip(4)

    found = check_variety(plan_for(items, t), t.wears)

    assert not [r for r in found if r.problem]
    assert EVERYDAY  # same everyday occasions as the packing tests
