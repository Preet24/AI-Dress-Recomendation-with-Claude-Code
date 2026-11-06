from datetime import date, datetime, time, timedelta
from itertools import pairwise

import pytest
from pydantic import ValidationError

from outfit.schemas import Category, Occasion
from outfit.tools.climate import ARCHIVE_URL, archive_hours
from outfit.tools.outfits import LOUNGE_HINT, NO_BASE
from outfit.tools.packing import (
    INDOORS,
    NO_CITY,
    Activity,
    Event,
    PlaceNotConfirmed,
    Trip,
    Wears,
    dates_text,
    days_text,
    pack,
    plan_trip,
    preview_trip,
    trip_slots,
    trip_weather,
    weather_line,
)
from outfit.tools.weather import (
    FORECAST_URL,
    GEOCODING_URL,
    Place,
    Transport,
    WeatherError,
    find_places,
)
from tests.factories import make_item
from tests.test_outfits import advice
from tests.test_weather import PUNE, hourly_payload

EVERYDAY = [Occasion.CASUAL, Occasion.TRAVEL, Occasion.PARTY]
FRI = date(2026, 10, 30)


def top(subtype="oxford shirt", color="white", lab=(95.0, 0.0, 0.0), **kw):
    fields = {"subtype": subtype, "primary_color_name": color, "primary_color_lab": lab,
              "occasions": EVERYDAY}  # fmt: skip
    return make_item(**fields | kw)


def trousers(**kw):
    fields = {"category": Category.BOTTOM, "subtype": "trousers", "primary_color_name": "navy",
              "primary_color_lab": (20.0, 5.0, -30.0), "occasions": EVERYDAY}  # fmt: skip
    return make_item(**fields | kw)


def shoes(**kw):
    fields = {"category": Category.FOOTWEAR, "subtype": "loafers", "primary_color_name": "black",
              "primary_color_lab": (10.0, 0.0, 0.0), "occasions": EVERYDAY}  # fmt: skip
    return make_item(**fields | kw)


def trip(days, **kw):
    return Trip(first_day=FRI, days=days, **{"travel_days": False} | kw)


def plan_for(items, t, weather=None, **kw):
    slots = trip_slots(t)
    return pack(items, slots, [weather] * len(slots), t.wears, **kw)


def packed_ids(plan) -> set:
    return {p.item.id for p in plan.pieces}


def tops_by_day(plan) -> list:
    return [next(p.id for p in d.outfit.pieces if p.category is Category.TOP) for d in plan.days]


# --- Trip and slots ---------------------------------------------------------


def test_slots_are_each_day_then_its_events_with_travel_days():
    party = Event(day=2, occasion=Occasion.PARTY, start=time(20), hours_away=4,
                  hours_outdoors=0.5, transport=Transport.CAR)  # fmt: skip

    slots = trip_slots(Trip(first_day=FRI, days=3, events=[party]))

    assert [s.name for s in slots] == [
        "day 1 travel",
        "day 2 casual",
        "day 2 party",
        "day 3 travel",
    ]
    assert slots[2].outing.start == datetime(2026, 10, 31, 20, 0)
    assert slots[2].outing.transport is Transport.CAR
    assert slots[1].outing.start == datetime(2026, 10, 31, 10, 0)  # the everyday default


def test_travel_days_can_be_off():
    assert {s.occasion for s in trip_slots(trip(3))} == {Occasion.CASUAL}


def test_an_event_after_the_trip_is_refused():
    late = Event(day=4, occasion=Occasion.PARTY, start=time(20), hours_away=4,
                 hours_outdoors=0, transport=Transport.CAR)  # fmt: skip
    with pytest.raises(ValidationError, match="day 4 of a 3-day trip"):
        Trip(first_day=FRI, days=3, events=[late])


def test_wear_limits_default_and_validate():
    assert Wears() == Wears(tops=2, bottoms=3, one_pieces=1)
    with pytest.raises(ValidationError):
        Wears(tops=0)
    assert Wears(one_pieces=0).one_pieces == 0


def test_zero_one_pieces_packs_none():
    dress = make_item(category=Category.ONE_PIECE, subtype="dress", primary_color_name="red",
                      primary_color_lab=(50.0, 60.0, 40.0), occasions=EVERYDAY)  # fmt: skip
    shirt, pants = top(), trousers()

    plan = plan_for([dress, shirt, pants, shoes()], trip(3, wears=Wears(one_pieces=0, tops=3)))

    assert dress.id not in packed_ids(plan)
    assert all(d.outfit and dress.id not in d.outfit.ids for d in plan.days)
    # Not even "worn again" when nothing else fits
    alone = plan_for([dress, shoes()], trip(2, wears=Wears(one_pieces=0)))
    assert not packed_ids(alone) and alone.days[0].outfit is None


# --- Packing ----------------------------------------------------------------


def test_one_outfit_covers_the_days_its_limits_allow():
    shirt, pants, loafers = top(), trousers(), shoes()

    plan = plan_for([shirt, pants, loafers], trip(2))

    assert packed_ids(plan) == {shirt.id, pants.id, loafers.id}
    assert all(d.outfit for d in plan.days)
    assert {p.item.id: p.days for p in plan.pieces}[shirt.id] == (1, 2)
    # Nothing else to wear, so the same outfit comes back, and the plan says so
    assert plan.days[0].notes == ()
    assert plan.days[1].notes == (
        "your white oxford shirt with navy trousers again, as on day 1: nothing else fits",
    )


def test_the_same_top_and_bottom_is_not_worn_twice_when_a_new_piece_helps():
    white, grey = top(), top("t-shirt", "grey", (60.0, 0.0, 0.0))

    plan = plan_for([white, grey, trousers(), shoes()], trip(2))

    assert {white.id, grey.id} <= packed_ids(plan)
    assert not any(d.notes for d in plan.days)


def test_two_tops_and_two_bottoms_are_mixed_rather_than_repeated():
    white, grey = top(), top("t-shirt", "grey", (60.0, 0.0, 0.0))
    navy, beige = trousers(), trousers(primary_color_name="beige",
                                       primary_color_lab=(80.0, 3.0, 15.0))  # fmt: skip

    plan = plan_for([white, grey, navy, beige, shoes()], trip(4))

    bases = [frozenset(p.id for p in d.outfit.pieces if p.category is not Category.FOOTWEAR)
             for d in plan.days]  # fmt: skip
    assert len(set(bases)) == 4


def test_two_slots_on_one_day_may_share_an_outfit():
    dinner = Event(day=1, occasion=Occasion.CASUAL, start=time(19), hours_away=3,
                   hours_outdoors=0, transport=Transport.CAR)  # fmt: skip

    plan = plan_for([top(), trousers(), shoes()], trip(1, events=[dinner]))

    assert plan.days[0].outfit.ids == plan.days[1].outfit.ids
    assert not any(d.notes for d in plan.days)


def test_when_bottoms_run_out_a_fresh_top_goes_with_a_bottom_worn_again():
    colors = [("white", 95.0), ("grey", 60.0), ("black", 10.0), ("ivory", 90.0)]
    tops = [top(primary_color_name=c, primary_color_lab=(light, 0.0, 0.0)) for c, light in colors]

    plan = plan_for([*tops, trousers(), shoes()], trip(4, wears=Wears(tops=1, bottoms=3)))

    worn = {p.item.id: p.days for p in plan.pieces}
    assert all(len(worn[t.id]) == 1 for t in tops)  # no top past its limit
    assert [n for d in plan.days for n in d.notes] == [
        "your navy trousers again: worn on more than 3 days from here, nothing else fits"
    ]


def test_each_day_can_choose_from_many_tops_not_one_in_many_shoes():
    pairs = [shoes(subtype=f"loafers {n}") for n in range(40)]
    tops = [top(primary_color_name=c, primary_color_lab=(light, 0.0, 0.0))
            for c, light in [("white", 95.0), ("grey", 60.0), ("ivory", 90.0)]]  # fmt: skip

    plan = plan_for([*tops, trousers(), *pairs], trip(3, wears=Wears(tops=1)))

    assert not any(d.notes for d in plan.days)


def test_the_top_limit_packs_a_second_top_and_no_top_two_days_running():
    white, grey = top(), top("t-shirt", "grey", (60.0, 0.0, 0.0))
    items = [white, grey, trousers(), shoes()]

    plan = plan_for(items, trip(4))

    assert {white.id, grey.id} <= packed_ids(plan)
    worn = tops_by_day(plan)
    assert all(a != b for a, b in pairwise(worn))  # alternates
    assert worn.count(white.id) == worn.count(grey.id) == 2


def test_spare_pieces_stay_home():
    loafers, sneakers = shoes(), shoes(subtype="sneakers")
    items = [top(), trousers(), loafers, sneakers]

    plan = plan_for(items, trip(2))

    assert len([p for p in plan.pieces if p.item.category is Category.FOOTWEAR]) == 1


def test_a_party_gets_its_own_outfit():
    tee = top("t-shirt", "grey", (60.0, 0.0, 0.0), formality=2, occasions=[Occasion.CASUAL])
    shirt = top(formality=4, occasions=[Occasion.PARTY])
    party = Event(day=1, occasion=Occasion.PARTY, start=time(20), hours_away=4,
                  hours_outdoors=0, transport=Transport.CAR)  # fmt: skip

    plan = plan_for([tee, shirt, trousers(), shoes()], trip(1, events=[party]))

    day, evening = plan.days
    assert tee.id in day.outfit.ids and shirt.id in evening.outfit.ids


def test_past_the_limit_a_piece_is_worn_again_with_a_note():
    shirt = top()

    plan = plan_for([shirt, trousers(), shoes()], trip(4, wears=Wears(bottoms=4)))

    assert all(d.outfit for d in plan.days)
    assert plan.days[2].notes == (
        "your white oxford shirt again: worn on more than 2 days from here, nothing else fits",
    )
    assert plan.days[3].notes == ()  # said once
    assert {p.item.id: p.days for p in plan.pieces}[shirt.id] == (1, 2, 3, 4)


@pytest.mark.parametrize(
    ("days", "text"),
    [([2], "day 2"), ([4, 2], "days 2 and 4"), ([1, 3, 4, 5, 6, 9], "days 1, 3-6 and 9"),
     ([2, 3], "days 2-3")],
)  # fmt: skip
def test_days_text(days, text):
    assert days_text(days) == text


def test_a_slot_nothing_fits_is_a_gap_with_its_day():
    wedding = Event(day=2, occasion=Occasion.WEDDING, start=time(18), hours_away=5,
                    hours_outdoors=0, transport=Transport.CAR)  # fmt: skip

    plan = plan_for([top(), trousers(), shoes()], trip(2, events=[wedding]))

    assert plan.days[2].outfit is None
    assert any(g.startswith(NO_BASE) and g.endswith("(day 2)") for g in plan.gaps)


def test_each_slot_has_its_own_weather():
    coat = make_item(category=Category.OUTERWEAR, subtype="wool coat", primary_color_name="grey",
                     primary_color_lab=(50.0, 0.0, 0.0), warmth=4, occasions=EVERYDAY)  # fmt: skip
    items = [top(), trousers(), shoes(), coat]
    t = trip(2)
    cold = advice(effective_temp=8, needs_outerwear=True, outerwear_min_warmth=3)

    plan = pack(items, trip_slots(t), [advice(), cold], t.wears)

    assert {p.item.id: p.days for p in plan.pieces}[coat.id] == (2,)


def test_accessories_finish_the_days_and_are_packed():
    belt = make_item(category=Category.ACCESSORY, subtype="leather belt",
                     primary_color_name="black", primary_color_lab=(10.0, 0.0, 0.0),
                     fabric="leather", warmth=1, occasions=EVERYDAY)  # fmt: skip

    plan = plan_for([top(), trousers(), shoes(), belt], trip(2))

    assert all(belt.id in d.outfit.ids for d in plan.days)
    assert plan.pieces[-1].item.id == belt.id  # accessories last


def lounge_tee():
    return top("t-shirt", "grey", (60.0, 0.0, 0.0), formality=1,
               occasions=[Occasion.LOUNGE, Occasion.CASUAL])  # fmt: skip


def joggers():
    return trousers(subtype="joggers", primary_color_name="black",
                    primary_color_lab=(10.0, 0.0, 0.0), formality=1,
                    occasions=[Occasion.LOUNGE])  # fmt: skip


def test_evenings_in_add_a_lounge_slot_every_night_but_the_last_day():
    slots = trip_slots(trip(3, evenings_in=True))

    assert [s.name for s in slots] == [
        "day 1 casual",
        "day 1 evening in",
        "day 2 casual",
        "day 2 evening in",
        "day 3 casual",
    ]
    evening = slots[1]
    assert evening.indoors and evening.occasion is Occasion.LOUNGE
    assert evening.outing.start == datetime(2026, 10, 30, 21, 0)
    assert not any(s.indoors for s in trip_slots(trip(3)))  # off by default


def test_evenings_in_get_lounge_clothes_without_shoes_or_accessories():
    belt = make_item(category=Category.ACCESSORY, subtype="leather belt",
                     primary_color_name="black", primary_color_lab=(10.0, 0.0, 0.0),
                     fabric="leather", warmth=1, occasions=EVERYDAY)  # fmt: skip
    tee, lounge_pants = lounge_tee(), joggers()
    items = [top(), trousers(), shoes(), belt, tee, lounge_pants]

    plan = plan_for(items, trip(2, evenings_in=True, wears=Wears(tops=3)))

    day1, evening, _ = plan.days
    assert evening.slot.indoors
    assert evening.outfit.ids == {tee.id, lounge_pants.id}  # no shoes, no belt
    assert lounge_pants.id not in day1.outfit.ids
    assert not any("footwear" in g or "lounge" in g for g in plan.gaps)


def test_casual_bottoms_are_not_lounge_wear_so_nothing_to_sleep_in_is_a_gap():
    plan = plan_for([lounge_tee(), trousers(), shoes()], trip(2, evenings_in=True))

    assert plan.days[1].outfit is None
    [gap] = [g for g in plan.gaps if g.startswith(NO_BASE)]
    assert "no bottom fits: navy trousers is not tagged for lounge" in gap
    assert gap.endswith(f"{LOUNGE_HINT}) (day 1)")


def test_evenings_in_need_no_weather():
    # Days end at 06:00, so the evening (21:00 to 07:00) would stretch the forecast
    early = Activity(occasion=Occasion.CASUAL, start=time(5), hours_away=1, hours_outdoors=1,
                     transport=Transport.WALK)  # fmt: skip
    fetch = TripFetch()
    slots = trip_slots(trip(2, everyday=early, evenings_in=True))

    weather = trip_weather("Pune", slots, SOON, fetch)

    assert weather.sources == ["forecast", "indoors", "forecast"]
    assert weather.summaries[1] is None and weather.advice[1] is None
    without = TripFetch()
    trip_weather("Pune", trip_slots(trip(2, everyday=early)), SOON, without)
    assert fetch.calls == without.calls  # the evenings fetch nothing more
    assert weather_line(None, indoors=True) == INDOORS


def test_notes_count_socks_for_every_day():
    plan = plan_for([top(), trousers(), shoes()], trip(3))
    assert plan.notes == ["plus socks and underwear for 3 days"]


# --- Weather ----------------------------------------------------------------


class TripFetch:
    """Answers geocoding, forecast and archive requests; `feels(time)` sets the weather."""

    def __init__(self, feels=lambda t: 22.0, rain=lambda t: 0.0, missing=lambda t: False, fail=()):
        self.feels, self.rain, self.missing, self.fail = feels, rain, missing, set(fail)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, params: dict) -> dict:
        self.calls.append((url, params))
        if url in self.fail:
            raise WeatherError("service down")
        if url == GEOCODING_URL:
            return {"results": [PUNE.model_dump()]}
        if url == FORECAST_URL:
            first = datetime.fromisoformat(params["start_hour"])
            last = datetime.fromisoformat(params["end_hour"])
        else:
            first = datetime.fromisoformat(params["start_date"])
            last = datetime.fromisoformat(params["end_date"]) + timedelta(hours=23)
        times = [
            first + timedelta(hours=h)
            for h in range(int((last - first).total_seconds() // 3600) + 1)
        ]
        return hourly_payload(
            [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            apparent_temperature=[
                None if url == FORECAST_URL and self.missing(t) else self.feels(t) for t in times
            ],
            temperature_2m=[self.feels(t) for t in times],
            precipitation=[self.rain(t) for t in times],
        )

    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


SOON = date(2026, 10, 25)  # the trip starts 5 days later: all within the forecast
EARLY = date(2026, 10, 1)  # 29 days before: past the forecast's reach


def test_no_city_means_no_weather_and_no_requests():
    fetch = TripFetch()

    weather = trip_weather(None, trip_slots(trip(2)), SOON, fetch)

    assert weather.advice == [None, None] and weather.assumptions == [NO_CITY]
    assert fetch.calls == []


def test_one_forecast_covers_the_trip_and_each_slot_gets_its_hours():
    party = Event(day=2, occasion=Occasion.PARTY, start=time(20), hours_away=4,
                  hours_outdoors=0.5, transport=Transport.CAR)  # fmt: skip
    slots = trip_slots(trip(2, events=[party]))
    fetch = TripFetch(feels=lambda t: 5.0 if t.day == 31 else 22.0)

    weather = trip_weather("Pune", slots, SOON, fetch)

    assert fetch.urls() == [GEOCODING_URL, FORECAST_URL]
    params = fetch.calls[1][1]
    assert (params["start_hour"], params["end_hour"]) == ("2026-10-30T10:00", "2026-10-31T23:00")
    day1, day2, evening = weather.summaries
    assert (day1.start, day1.end) == (datetime(2026, 10, 30, 10), datetime(2026, 10, 30, 18))
    assert evening.start == datetime(2026, 10, 31, 20)
    assert not weather.advice[0].needs_outerwear and weather.advice[1].needs_outerwear
    assert weather.assumptions == []


def test_days_past_the_forecast_use_last_year():
    fetch = TripFetch()

    weather = trip_weather("Pune", trip_slots(trip(2)), EARLY, fetch)

    assert fetch.urls() == [GEOCODING_URL, ARCHIVE_URL]
    params = fetch.calls[1][1]
    assert (params["start_date"], params["end_date"]) == ("2025-10-30", "2025-10-31")
    assert weather.summaries[0].start == datetime(2026, 10, 30, 10)  # moved to the trip
    assert weather.assumptions == [
        "no forecast for days 1-2: planned on last year's weather on the same dates"
    ]


def test_a_trip_across_the_forecast_edge_uses_both():
    fetch = TripFetch()
    edge = date(2026, 10, 30) - timedelta(days=15)  # day 1 is the forecast's last day

    weather = trip_weather("Pune", trip_slots(trip(3)), edge, fetch)

    assert fetch.urls() == [GEOCODING_URL, FORECAST_URL, ARCHIVE_URL]
    assert all(weather.advice)
    assert "days 2-3" in weather.assumptions[0]


def test_an_unknown_city_plans_without_weather():
    weather = trip_weather("Pune", trip_slots(trip(1)), SOON, TripFetch(fail={GEOCODING_URL}))
    assert weather.advice == [None]
    assert weather.sources == ["not checked"]
    assert weather.assumptions == ["couldn't find Pune, so the weather was not checked"]


def test_a_failed_forecast_falls_back_to_last_year():
    fetch = TripFetch(fail={FORECAST_URL})

    weather = trip_weather("Pune", trip_slots(trip(2)), SOON, fetch)

    assert fetch.urls() == [GEOCODING_URL, FORECAST_URL, ARCHIVE_URL]
    assert all(weather.advice)
    assert "last year's weather" in weather.assumptions[0]


def test_slots_the_forecast_has_no_hours_for_use_last_year():
    fetch = TripFetch(missing=lambda t: t.day == 31)  # the forecast has no feels-like on day 2

    weather = trip_weather("Pune", trip_slots(trip(2)), SOON, fetch)

    assert fetch.urls() == [GEOCODING_URL, FORECAST_URL, ARCHIVE_URL]
    assert (fetch.calls[2][1]["start_date"], fetch.calls[2][1]["end_date"]) == (
        "2025-10-31",
        "2025-10-31",
    )
    assert all(weather.advice)
    assert weather.assumptions == [
        "no forecast for day 2: planned on last year's weather on the same dates"
    ]


def test_no_forecast_and_no_archive_is_an_assumption():
    fetch = TripFetch(fail={FORECAST_URL, ARCHIVE_URL})

    weather = trip_weather("Pune", trip_slots(trip(2)), SOON, fetch)

    assert weather.advice == [None, None]
    assert weather.assumptions == [
        "no forecast or past weather for days 1-2, so it was not checked"
    ]


def test_a_city_found_under_another_name_is_said():
    weather = trip_weather("goa", trip_slots(trip(1)), SOON, TripFetch())  # PUNE stands in

    assert weather.place == PUNE
    assert weather.assumptions == ["took goa to be Pune, Maharashtra, India"]


def test_archive_hours_mark_rain_and_take_uv_from_the_kind_of_day():
    fetch = TripFetch(feels=lambda t: 34.0, rain=lambda t: 0.5 if t.hour == 15 else 0.0)

    hours = archive_hours(PUNE, date(2025, 10, 30), date(2025, 10, 30), fetch)

    assert len(hours) == 24
    assert {h.rain_chance for h in hours if h.time.hour == 15} == {100.0}
    assert {h.rain_chance for h in hours if h.time.hour != 15} == {0.0}
    assert {h.uv_index for h in hours} == {9.0}  # a hot day


def test_plan_trip_puts_the_weather_on_each_day():
    coat = make_item(category=Category.OUTERWEAR, subtype="wool coat", primary_color_name="grey",
                     primary_color_lab=(50.0, 0.0, 0.0), warmth=4, occasions=EVERYDAY)  # fmt: skip
    fetch = TripFetch(feels=lambda t: 5.0 if t.day == 31 else 22.0)

    plan = plan_trip([top(), trousers(), shoes(), coat], confirmed(2), SOON, fetch)

    assert fetch.urls() == [FORECAST_URL]  # the confirmed place: no geocoding

    assert [d.weather.feels_like_min for d in plan.days] == [22.0, 5.0]
    assert plan.days[1].advice.needs_outerwear
    assert {p.item.id: p.days for p in plan.pieces}[coat.id] == (2,)
    assert plan.assumptions == () and plan.place == "Pune, Maharashtra, India"


# --- Confirming the trip ----------------------------------------------------

GOA = Place(name="Panjim", admin1="Goa", country="India", latitude=15.49, longitude=73.82,
            timezone="Asia/Kolkata")  # fmt: skip
GENOA = Place(name="Genoa", admin1="Liguria", country="Italy", latitude=44.4, longitude=8.94,
              timezone="Europe/Rome")  # fmt: skip


def confirmed(days, **kw):
    return trip(days, city="Pune", place=PUNE, **kw)


class PlacesFetch(TripFetch):
    """TripFetch whose geocoding returns these places."""

    def __init__(self, *places: Place, **kw):
        super().__init__(**kw)
        self.places = places

    def __call__(self, url: str, params: dict) -> dict:
        if url == GEOCODING_URL and url not in self.fail:
            self.calls.append((url, params))
            return {"results": [p.model_dump() for p in self.places]}
        return super().__call__(url, params)


def test_find_places_offers_one_of_each_name_state_and_country():
    fetch = PlacesFetch(GENOA, GOA, GOA.model_copy(update={"latitude": 15.5}))

    places = find_places(" Goa ", fetch, count=5)

    assert [p.label for p in places] == ["Genoa, Liguria, Italy", "Panjim, Goa, India"]
    assert fetch.calls[0][1] == {"name": "Goa", "count": 5, "language": "en"}


def test_find_places_raises_when_nothing_matches():
    with pytest.raises(WeatherError, match="Zzz"):
        find_places("Zzz", PlacesFetch())


def test_preview_offers_the_matches_and_shows_the_first_ones_weather():
    party = Event(day=2, occasion=Occasion.PARTY, start=time(20), hours_away=4,
                  hours_outdoors=0.5, transport=Transport.CAR)  # fmt: skip
    fetch = PlacesFetch(GENOA, GOA)

    preview = preview_trip(trip(2, city="Goa", events=[party]), SOON, fetch)

    assert [p.label for p in preview.matches] == ["Genoa, Liguria, Italy", "Panjim, Goa, India"]
    assert preview.place == GENOA
    assert preview.dates == "Fri 30 Oct - Sat 31 Oct 2026, 2 days"
    assert [r.slot.name for r in preview.rows] == ["day 1 casual", "day 2 casual", "day 2 party"]
    assert [r.source for r in preview.rows] == ["forecast"] * 3
    assert preview.rows[2].weather.start == datetime(2026, 10, 31, 20)
    assert preview.wears == Wears()
    assert preview.assumptions == []  # the place is shown, not assumed


def test_preview_for_a_chosen_place_does_not_search_again():
    fetch = PlacesFetch(GENOA, GOA)

    preview = preview_trip(trip(1, city="Goa", place=GOA), EARLY, fetch)

    assert preview.matches == [GOA] and preview.place == GOA
    assert fetch.urls() == [ARCHIVE_URL]
    assert preview.rows[0].source == "last year"


def test_preview_of_an_unknown_city_says_so():
    preview = preview_trip(trip(1, city="Zzz"), SOON, PlacesFetch())

    assert preview.matches == [] and preview.place is None
    assert preview.rows[0].source == "not checked"
    assert preview.assumptions == ["couldn't find Zzz: try a town or city name"]


def test_preview_without_a_city_checks_no_weather():
    fetch = PlacesFetch(GOA)

    preview = preview_trip(trip(1), SOON, fetch)

    assert preview.place is None and fetch.calls == []
    assert preview.assumptions == [NO_CITY]


def test_packing_needs_the_confirmed_place():
    with pytest.raises(PlaceNotConfirmed, match="Goa"):
        plan_trip([top(), trousers()], trip(1, city="Goa"), SOON, PlacesFetch(GOA))


def test_packing_without_a_city_plans_without_weather():
    plan = plan_trip([top(), trousers(), shoes()], trip(1), SOON, TripFetch())
    assert plan.days[0].outfit and plan.assumptions == (NO_CITY,)


@pytest.mark.parametrize(
    ("first", "days", "text"),
    [
        (FRI, 1, "Fri 30 Oct 2026, 1 day"),
        (FRI, 4, "Fri 30 Oct - Mon 2 Nov 2026, 4 days"),
        (date(2026, 12, 30), 3, "Wed 30 Dec 2026 - Fri 1 Jan 2027, 3 days"),
    ],
)
def test_dates_text(first, days, text):
    assert dates_text(Trip(first_day=first, days=days)) == text


def test_weather_line():
    summary = trip_weather("Pune", trip_slots(trip(1)), SOON, TripFetch()).summaries[0]
    assert weather_line(summary) == (
        "feels 22 °C, rain up to 10%, wind up to 10 km/h, UV 4, mainly clear"
    )
    assert weather_line(None) == "weather not checked"
