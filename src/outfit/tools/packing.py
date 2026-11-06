"""Trip packing: the fewest pieces that cover every day of a trip, and what to wear when.

A trip is cut into slots: each day's everyday plan (a travel outfit on the first and
last day, if asked) and the extra events, each with its own occasion and weather.

- Options: per slot, the planner's good outfits (`all_outfits`, score >= `GOOD_OUTFIT`;
  the best it has when none is good).
- Pack: greedily add the outfit that lets the most still-undressed slots be dressed per
  new piece. Shoes and outerwear count as pieces, so one pair that goes with everything
  wins over three.
- Re-wear limits: the user's cap on how many days a top, bottom or one-piece is worn;
  shoes, outerwear and accessories repeat freely. The same base (top + bottom, or a
  one-piece) isn't worn on two days either, so pieces get mixed differently or one more
  is packed. When the packed pieces can't dress a slot within these rules and no new
  piece helps, a piece or a base is worn again, with a note.
- Days: each slot gets an outfit made only of packed pieces (no top two days running
  when there's a choice, then the least-worn pieces, then the best score), finished
  with accessories (tools/accessories.py). Only pieces some day uses are packed.
- Weather (`trip_weather`): one geocode, one forecast for the slots it reaches (16
  days), one archive fetch of the same days last year for the rest and for slots the
  forecast left without hours; every slot gets the usual `advise()`. Where the weather
  came from, or that it wasn't checked, is an assumption, never a gap.

- Confirm first: `preview_trip` shows the places the city name could be (full name,
  state, country), the dates and every day and event with its weather, using no
  wardrobe; `plan_trip` packs only for a trip carrying the place the user confirmed.

`pack` is pure (the caller gives each slot's advice); `trip_weather`, `preview_trip`
and `plan_trip` take the HTTP `fetch`, so tests need no store or network.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from outfit.schemas import Category, ClothingItem, Occasion
from outfit.tools.accessories import finish_outfit
from outfit.tools.climate import archive_hours
from outfit.tools.feedback import Feedback, shifted
from outfit.tools.outfits import (
    ACCESSORY_SLOTS,
    BOTTOMS,
    TOPS,
    Outfit,
    OutfitRequest,
    all_outfits,
    label,
)
from outfit.tools.shopping import GOOD_OUTFIT, and_join
from outfit.tools.weather import (
    FORECAST_DAYS_MAX,
    FetchJson,
    HourlyWeather,
    Outing,
    Place,
    Transport,
    WeatherAdvice,
    WeatherError,
    WeatherSummary,
    advise,
    fetch_hours,
    find_places,
    geocode,
    http_get_json,
    outing_hours,
    summarize,
)

logger = logging.getLogger(__name__)

MAX_DAYS = 14
LAST_YEAR = timedelta(days=365)  # days past the forecast are planned on these days back
TOP_WEARS = 2  # default re-wear limits, in days; the user sets their own
BOTTOM_WEARS = 3
ONE_PIECE_WEARS = 1
SLOT_OPTIONS = 30  # best different bases kept per slot, each with its shoes and outerwear
# Order of the packing list
PACK_ORDER = [
    Category.TOP,
    Category.ETHNIC_TOP,
    Category.BOTTOM,
    Category.ETHNIC_BOTTOM,
    Category.ONE_PIECE,
    Category.OUTERWEAR,
    Category.FOOTWEAR,
    Category.ACCESSORY,
]


class Activity(BaseModel):
    """What the user does in one part of a day."""

    model_config = ConfigDict(frozen=True)

    occasion: Occasion
    start: time  # local time at the destination
    hours_away: float = Field(gt=0, le=24)
    hours_outdoors: float = Field(ge=0, le=24)
    transport: Transport

    @model_validator(mode="after")
    def _outdoors_within_away(self) -> "Activity":
        if self.hours_outdoors > self.hours_away:
            raise ValueError("hours_outdoors cannot exceed hours_away")
        return self


EVERYDAY = Activity(
    occasion=Occasion.CASUAL,
    start=time(10, 0),
    hours_away=8,
    hours_outdoors=3,
    transport=Transport.PUBLIC_TRANSIT,
)
# "Evenings in": relaxing and sleeping at the stay, every night but the last day's
# (indoors, so no weather; the outing only places it in the day).
EVENING_IN = Activity(
    occasion=Occasion.LOUNGE,
    start=time(21, 0),
    hours_away=10,
    hours_outdoors=0,
    transport=Transport.CAR,
)


class Event(Activity):
    """An extra activity on one day of the trip ("party on day 2 at 20:00")."""

    day: int = Field(ge=1, le=MAX_DAYS)  # 1 = the first day


class Wears(BaseModel):
    """How many days the user will wear one piece. One-pieces may be 0: don't pack any
    (tops and bottoms can't, every outfit without a one-piece needs them)."""

    model_config = ConfigDict(frozen=True)

    tops: int = Field(TOP_WEARS, ge=1, le=MAX_DAYS)
    bottoms: int = Field(BOTTOM_WEARS, ge=1, le=MAX_DAYS)
    one_pieces: int = Field(ONE_PIECE_WEARS, ge=0, le=MAX_DAYS)

    def limit(self, item: ClothingItem) -> int | None:
        """Days `item` may be worn; None = no limit (shoes, outerwear, accessories)."""
        if item.category in TOPS:
            return self.tops
        if item.category in BOTTOMS:
            return self.bottoms
        if item.category is Category.ONE_PIECE:
            return self.one_pieces
        return None


class Trip(BaseModel):
    city: str | None = None  # as typed; `preview_trip` offers the places it could be
    place: Place | None = None  # the place the user confirmed; packing needs it
    first_day: date
    days: int = Field(ge=1, le=MAX_DAYS)
    everyday: Activity = EVERYDAY
    events: list[Event] = Field(default_factory=list)
    travel_days: bool = True  # the first and last day's everyday plan is travel
    evenings_in: bool = False  # a lounge slot (relax / sleep) every night of the stay
    wears: Wears = Wears()

    @model_validator(mode="after")
    def _events_within_trip(self) -> "Trip":
        if late := [e.day for e in self.events if e.day > self.days]:
            raise ValueError(f"event on day {late[0]} of a {self.days}-day trip")
        return self


@dataclass(frozen=True)
class Slot:
    day: int
    occasion: Occasion
    outing: Outing
    indoors: bool = False  # an evening in: no weather

    @property
    def name(self) -> str:
        return f"day {self.day} {'evening in' if self.indoors else self.occasion}"


@dataclass(frozen=True)
class DayOutfit:
    slot: Slot
    outfit: Outfit | None  # None: nothing fits (the plan's gaps say why)
    notes: tuple[str, ...] = ()
    weather: WeatherSummary | None = None  # None: not checked (see the assumptions)
    advice: WeatherAdvice | None = None


@dataclass(frozen=True)
class Packed:
    item: ClothingItem
    days: tuple[int, ...]


@dataclass(frozen=True)
class PackingPlan:
    pieces: list[Packed]
    days: list[DayOutfit]
    gaps: list[str]  # "no footwear suits party (days 2 and 4)"
    notes: list[str]
    assumptions: tuple[str, ...] = ()  # how the weather was (or wasn't) checked
    place: str | None = None  # where the weather is for: "Genoa, Liguria, Italy"


def trip_slots(trip: Trip) -> list[Slot]:
    """Every day's everyday plan, then that day's events by start time, then (if asked)
    the evening in, on every day but the last (a 3-day trip has 2 nights)."""
    slots = []
    for day in range(1, trip.days + 1):
        on = trip.first_day.toordinal() + day - 1
        everyday = trip.everyday
        if trip.travel_days and day in {1, trip.days}:
            everyday = everyday.model_copy(update={"occasion": Occasion.TRAVEL})
        events = sorted((e for e in trip.events if e.day == day), key=lambda e: e.start)
        activities = [(everyday, False)] + [(e, False) for e in events]
        if trip.evenings_in and day < trip.days:
            activities.append((EVENING_IN, True))
        for activity, indoors in activities:
            outing = Outing(
                start=datetime.combine(date.fromordinal(on), activity.start),
                hours_away=activity.hours_away,
                hours_outdoors=activity.hours_outdoors,
                transport=activity.transport,
            )
            slots.append(Slot(day, activity.occasion, outing, indoors))
    return slots


# --- Assigning outfits to slots ---------------------------------------------

Base = frozenset[UUID]  # an outfit without its shoes, outerwear and accessories


def base_of(outfit: Outfit) -> Base:
    return frozenset(p.id for p in outfit.pieces if p.category not in ACCESSORY_SLOTS)


@dataclass
class Worn:
    """The days each piece and each base is worn on."""

    pieces: dict[UUID, set[int]] = field(default_factory=lambda: defaultdict(set))
    bases: dict[Base, set[int]] = field(default_factory=lambda: defaultdict(set))

    def repeats(self, outfit: Outfit, day: int) -> bool:
        """This base was worn on another day (the same day is one outfit, twice)."""
        return bool(self.bases.get(base_of(outfit), set()) - {day})


def within(outfit: Outfit, day: int, worn: Worn, wears: Wears) -> bool:
    """Wearing it on `day` repeats no base and keeps every piece within its limit."""
    return not worn.repeats(outfit, day) and all(
        (limit := wears.limit(p)) is None
        or day in worn.pieces[p.id]
        or len(worn.pieces[p.id]) < limit
        for p in outfit.pieces
    )


def preference(outfit: Outfit, day: int, worn: Worn, wears: Wears) -> tuple:
    """Lower is better: no repeated base, no top two days running, then least-worn
    pieces, then score."""
    limited = [p for p in outfit.pieces if wears.limit(p) is not None]
    again = any(
        day - 1 in worn.pieces[p.id] for p in limited if p.category in TOPS | {Category.ONE_PIECE}
    )
    used = sum(len(worn.pieces[p.id] - {day}) for p in limited)
    return (worn.repeats(outfit, day), again, used, -outfit.score)


def wear(outfit: Outfit, day: int, worn: Worn) -> None:
    for p in outfit.pieces:
        worn.pieces[p.id].add(day)
    worn.bases[base_of(outfit)].add(day)


def assign(
    slots: Sequence[Slot], options: Sequence[list[Outfit]], packed: set[UUID], wears: Wears
) -> list[Outfit | None]:
    """An outfit of packed pieces for each slot, in day order, within the limits."""
    worn = Worn()
    chosen: list[Outfit | None] = []
    for slot, slot_options in zip(slots, options, strict=True):
        fits = [o for o in slot_options if o.ids <= packed and within(o, slot.day, worn, wears)]
        pick = min(fits, key=lambda o: preference(o, slot.day, worn, wears), default=None)
        if pick is not None:
            wear(pick, slot.day, worn)
        chosen.append(pick)
    return chosen


def dressed(chosen: list[Outfit | None]) -> int:
    return sum(o is not None for o in chosen)


def greedy_pack(
    slots: Sequence[Slot], options: Sequence[list[Outfit]], wears: Wears
) -> tuple[set[UUID], list[Outfit | None]]:
    """Add the outfit that dresses the most new slots per new piece until none helps."""
    packed: set[UUID] = set()
    chosen = assign(slots, options, packed, wears)
    while True:
        best, best_key = None, None
        tried: set[frozenset[UUID]] = set()
        for i, slot_options in enumerate(options):
            if chosen[i] is not None:
                continue
            for outfit in slot_options:
                new = outfit.ids - packed
                if not new or outfit.ids in tried:
                    continue
                tried.add(outfit.ids)
                gain = dressed(assign(slots, options, packed | outfit.ids, wears)) - dressed(chosen)
                key = (gain / len(new), outfit.score)
                if gain > 0 and (best_key is None or key > best_key):
                    best, best_key = outfit, key
        if best is None:
            return packed, chosen
        packed |= best.ids
        chosen = assign(slots, options, packed, wears)


def worn_by(slots: Sequence[Slot], chosen: Sequence[Outfit | None]) -> Worn:
    worn = Worn()
    for slot, outfit in zip(slots, chosen, strict=True):
        if outfit is not None:
            wear(outfit, slot.day, worn)
    return worn


def wear_again(
    slots: Sequence[Slot],
    options: Sequence[list[Outfit]],
    packed: set[UUID],
    chosen: list[Outfit | None],
    wears: Wears,
) -> list[Outfit | None]:
    """Dress the slots the limits left bare: no repeated base if possible, then the fewest
    pieces past their limit, then the fewest new pieces (so when the bottoms run out, a
    fresh top still goes with a bottom worn once more)."""
    chosen, packed = list(chosen), set(packed)
    for i, slot in enumerate(slots):
        if chosen[i] is not None or not options[i]:
            continue
        worn = worn_by(slots, chosen)
        chosen[i] = min(options[i], key=lambda o: again_cost(o, slot.day, worn, packed, wears))  # noqa: B023  (called at once)
        packed |= chosen[i].ids
    return chosen


def again_cost(outfit: Outfit, day: int, worn: Worn, packed: set[UUID], wears: Wears) -> tuple:
    """Lower is better, for a slot the limits left bare (`wear_again`)."""
    over = sum(
        (limit := wears.limit(p)) is not None
        and day not in worn.pieces[p.id]
        and len(worn.pieces[p.id]) >= limit
        for p in outfit.pieces
    )
    new = len(outfit.ids - packed)
    return (worn.repeats(outfit, day), over, new, *preference(outfit, day, worn, wears))


def repeat_notes(
    slots: Sequence[Slot], chosen: Sequence[Outfit | None], wears: Wears
) -> dict[int, list[str]]:
    """Per slot index, a note for each piece first worn past its limit there, and for
    each base first worn again there."""
    worn = Worn()
    notes: dict[int, list[str]] = defaultdict(list)
    said: set[Base] = set()
    for i, (slot, outfit) in enumerate(zip(slots, chosen, strict=True)):
        if outfit is None:
            continue
        base = base_of(outfit)
        if worn.repeats(outfit, slot.day) and base not in said:
            said.add(base)
            names = " with ".join(label(p) for p in outfit.pieces if p.id in base)
            first = min(worn.bases[base])
            notes[i].append(f"your {names} again, as on day {first}: nothing else fits")
        for p in outfit.pieces:
            if slot.day in worn.pieces[p.id]:
                continue
            limit = wears.limit(p)
            if limit is not None and len(worn.pieces[p.id]) == limit:
                notes[i].append(
                    f"your {label(p)} again: worn on more than {limit} days from here, "
                    "nothing else fits"
                )
        wear(outfit, slot.day, worn)
    return notes


# --- The plan ---------------------------------------------------------------


def varied_options(outfits: Sequence[Outfit]) -> list[Outfit]:
    """Best first, the best `SLOT_OPTIONS` different bases with all their shoes and
    outerwear: a plain top-N would be a few bases in every pair of shoes, and the re-wear
    limits would run out while the wardrobe still had other tops."""
    kept: set[Base] = set()
    options = []
    for outfit in outfits:
        base = base_of(outfit)
        if base not in kept and len(kept) == SLOT_OPTIONS:
            continue
        kept.add(base)
        options.append(outfit)
    return options


def days_text(days: Sequence[int]) -> str:
    """ "day 2", "days 2 and 4", "days 1, 3-6 and 9"."""
    runs: list[list[int]] = []
    for day in sorted(set(days)):
        if runs and day == runs[-1][-1] + 1:
            runs[-1].append(day)
        else:
            runs.append([day])
    parts = [str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs]
    plural = len(runs) > 1 or len(runs[0]) > 1
    return f"day{'s' * plural} {and_join(parts)}"


def pack(
    items: list[ClothingItem],
    slots: Sequence[Slot],
    advice: Sequence[WeatherAdvice | None],
    wears: Wears | None = None,
    taste: OutfitRequest | None = None,
) -> PackingPlan:
    """What to pack and what to wear in each slot. `advice` is each slot's weather
    (None: not checked); `taste` carries the profile's likes, as in chat."""
    wears = wears or Wears()
    taste = taste or OutfitRequest()
    items = [i for i in items if wears.limit(i) != 0]  # 0 = leave that kind at home
    requests = [
        replace(taste, occasion=s.occasion, advice=a) for s, a in zip(slots, advice, strict=True)
    ]
    options: list[list[Outfit]] = []
    slot_notes: dict[int, list[str]] = defaultdict(list)
    gap_days: dict[str, list[int]] = {}
    for i, (slot, request) in enumerate(zip(slots, requests, strict=True)):
        plan = all_outfits(items, request)
        good = [o for o in plan.outfits if o.score >= GOOD_OUTFIT]
        if plan.outfits and not good:
            slot_notes[i].append("nothing scores well here, so it's the best of what fits")
        options.append(varied_options(good or plan.outfits))
        for gap in plan.gaps:
            gap_days.setdefault(gap, []).append(slot.day)

    packed, chosen = greedy_pack(slots, options, wears)
    chosen = wear_again(slots, options, packed, chosen, wears)
    for i, notes in repeat_notes(slots, chosen, wears).items():
        slot_notes[i] += notes

    accessories = [i for i in items if i.category is Category.ACCESSORY]
    days = []
    for i, (slot, outfit, request) in enumerate(zip(slots, chosen, requests, strict=True)):
        if outfit is not None:
            outfit, missed = finish_outfit(outfit, accessories, request)
            for gap in missed:
                gap_days.setdefault(gap, []).append(slot.day)
        days.append(DayOutfit(slot, outfit, tuple(slot_notes[i]), advice=request.advice))

    on_days: dict[UUID, set[int]] = defaultdict(set)
    by_id: dict[UUID, ClothingItem] = {}
    for day in days:
        for p in day.outfit.pieces if day.outfit else ():
            on_days[p.id].add(day.slot.day)
            by_id[p.id] = p
    pieces = [Packed(by_id[i], tuple(sorted(d))) for i, d in on_days.items()]
    pieces.sort(key=lambda p: (PACK_ORDER.index(p.item.category), label(p.item)))
    trip_days = len({s.day for s in slots})
    return PackingPlan(
        pieces=pieces,
        days=days,
        gaps=[f"{gap} ({days_text(d)})" for gap, d in gap_days.items()],
        notes=[f"plus socks and underwear for {trip_days} day{'s' * (trip_days != 1)}"],
    )


# --- Weather ----------------------------------------------------------------

NO_CITY = "no city given, so the weather was not checked"
PLACE_MATCHES = 5  # places offered for the user to confirm

Source = Literal["forecast", "last year", "not checked", "indoors"]


@dataclass(frozen=True)
class TripWeather:
    summaries: list[WeatherSummary | None]  # per slot; None = not checked
    advice: list[WeatherAdvice | None]
    sources: list[Source]
    assumptions: list[str]
    place: Place | None = None  # where the weather is for


def within_span(hours: list[HourlyWeather], span: tuple[datetime, datetime]) -> list[HourlyWeather]:
    return [h for h in hours if span[0] <= h.time <= span[1]]


def unchecked(slots: Sequence[Slot], why: str) -> TripWeather:
    n = len(slots)
    return TripWeather([None] * n, [None] * n, ["not checked"] * n, [why])


def trip_weather(
    where: str | Place | None, slots: Sequence[Slot], today: date, fetch: FetchJson = http_get_json
) -> TripWeather:
    """Each slot's weather, for a confirmed place (or a city name, taking its best match).
    Evenings in get none (source "indoors") and fetch nothing."""
    outdoor = [i for i, s in enumerate(slots) if not s.indoors]
    found = outdoor_weather(where, [slots[i] for i in outdoor], today, fetch)
    n = len(slots)
    summaries: list[WeatherSummary | None] = [None] * n
    advice: list[WeatherAdvice | None] = [None] * n
    sources: list[Source] = ["indoors"] * n
    for j, i in enumerate(outdoor):
        summaries[i], advice[i], sources[i] = found.summaries[j], found.advice[j], found.sources[j]
    return TripWeather(summaries, advice, sources, found.assumptions, found.place)


def outdoor_weather(
    where: str | Place | None, slots: Sequence[Slot], today: date, fetch: FetchJson
) -> TripWeather:
    """`trip_weather` for slots that go out: the forecast where it has the hours, else
    last year's archive (slots past the forecast's reach, slots it has no complete
    hours for, or a failed forecast)."""
    if not where:
        return unchecked(slots, NO_CITY)
    assumptions = []
    if isinstance(where, Place):
        place = where
    else:
        try:
            place = geocode(where, fetch)
        except WeatherError as exc:
            logger.warning("No weather for the trip to %s: %s", where, exc)
            return unchecked(
                slots, f"couldn't find {where.strip()}, so the weather was not checked"
            )
        if place.name.casefold() != where.strip().casefold():
            assumptions.append(f"took {where.strip()} to be {place.label}")
    reach = today + timedelta(days=FORECAST_DAYS_MAX - 1)
    spans = [outing_hours(s.outing) for s in slots]
    near = [i for i, (_, end) in enumerate(spans) if end.date() <= reach]
    forecast: list[HourlyWeather] = []
    if near:
        try:
            first, last = min(spans[i][0] for i in near), max(spans[i][1] for i in near)
            forecast = fetch_hours(place, first, last, fetch)
        except WeatherError as exc:
            logger.warning("No forecast for the trip to %s: %s", place.label, exc)
    hours = [within_span(forecast, span) for span in spans]
    sources: list[Source] = ["forecast" if h else "not checked" for h in hours]
    far = [i for i, found in enumerate(hours) if not found]
    if far:
        days = days_text([slots[i].day for i in far])
        try:
            first = min(spans[i][0] for i in far).date() - LAST_YEAR
            last = max(spans[i][1] for i in far).date() - LAST_YEAR
            past = [
                h.model_copy(update={"time": h.time + LAST_YEAR})
                for h in archive_hours(place, first, last, fetch)
            ]
            for i in far:
                hours[i] = within_span(past, spans[i])
                sources[i] = "last year" if hours[i] else "not checked"
            assumptions.append(
                f"no forecast for {days}: planned on last year's weather on the same dates"
            )
        except WeatherError as exc:
            logger.warning("No weather archive for %s: %s", place.label, exc)
            assumptions.append(f"no forecast or past weather for {days}, so it was not checked")
    summaries: list[WeatherSummary | None] = []
    advice: list[WeatherAdvice | None] = []
    for slot, picked in zip(slots, hours, strict=True):
        summary = summarize(place, picked) if picked else None
        summaries.append(summary)
        advice.append(advise(summary, slot.outing) if summary else None)
    return TripWeather(summaries, advice, sources, assumptions, place)


INDOORS = "indoors, no weather needed"


def weather_line(summary: WeatherSummary | None, indoors: bool = False) -> str:
    """ "feels 22-26 °C, rain up to 10%, wind up to 15 km/h, UV 6, partly cloudy"."""
    if indoors:
        return INDOORS
    if summary is None:
        return "weather not checked"
    low, high = f"{summary.feels_like_min:.0f}", f"{summary.feels_like_max:.0f}"
    return (
        f"feels {low if low == high else f'{low}-{high}'} °C, "
        f"rain up to {summary.rain_chance_max:.0f}%, wind up to {summary.wind_speed_max:.0f} km/h, "
        f"UV {summary.uv_index_max:.0f}, {', '.join(summary.conditions)}"
    )


# --- Confirming the trip ----------------------------------------------------


def day_text(day: date, year: bool = False) -> str:
    return f"{day:%a} {day.day} {day:%b}" + (f" {day.year}" if year else "")


def dates_text(trip: Trip) -> str:
    """ "Fri 30 Oct - Mon 2 Nov 2026, 4 days"."""
    last = trip.first_day + timedelta(days=trip.days - 1)
    span = (
        day_text(last, year=True)
        if trip.days == 1
        else f"{day_text(trip.first_day, year=trip.first_day.year != last.year)} - "
        f"{day_text(last, year=True)}"
    )
    return f"{span}, {trip.days} day{'s' * (trip.days != 1)}"


@dataclass(frozen=True)
class PreviewRow:
    slot: Slot
    weather: WeatherSummary | None
    source: Source


@dataclass(frozen=True)
class TripPreview:
    """What the user confirms before anything is packed."""

    matches: list[Place]  # places the city name could be, best first
    place: Place | None  # the one this preview is for (None: weather not checked)
    dates: str
    rows: list[PreviewRow]
    wears: Wears
    assumptions: list[str]


def preview_trip(trip: Trip, today: date, fetch: FetchJson = http_get_json) -> TripPreview:
    """The trip as the user should confirm it: which place (full name, state, country),
    the dates, and every day and event with its weather. Uses no wardrobe."""
    slots = trip_slots(trip)
    matches: list[Place] = []
    if trip.place is not None:
        matches = [trip.place]
    elif trip.city:
        try:
            matches = find_places(trip.city, fetch, PLACE_MATCHES)
        except WeatherError as exc:
            logger.warning("No place called %s: %s", trip.city, exc)
    place = matches[0] if matches else None
    if trip.city and place is None:
        weather = unchecked(slots, f"couldn't find {trip.city.strip()}: try a town or city name")
    else:
        weather = trip_weather(place, slots, today, fetch)
    rows = [
        PreviewRow(slot, summary, source)
        for slot, summary, source in zip(slots, weather.summaries, weather.sources, strict=True)
    ]
    return TripPreview(matches, place, dates_text(trip), rows, trip.wears, weather.assumptions)


class PlaceNotConfirmed(ValueError):
    """A trip with a city must carry the place the user confirmed (`preview_trip`)."""


def plan_trip(
    items: list[ClothingItem],
    trip: Trip,
    today: date,
    fetch: FetchJson = http_get_json,
    taste: OutfitRequest | None = None,
    history: Feedback | None = None,
) -> PackingPlan:
    """The whole tool, after the user confirmed the trip: weather per slot -> what to
    pack and wear. A city without a confirmed place is refused. `history` (the wear log)
    adds liked / disliked pairings and the user's warmth bias; recent wear doesn't count
    for a trip that hasn't started."""
    if trip.city and trip.place is None:
        raise PlaceNotConfirmed(f"confirm which {trip.city.strip()} first")
    history = history or Feedback()
    taste = replace(taste or OutfitRequest(), **history.request_fields(recent=False))
    slots = trip_slots(trip)
    weather = trip_weather(trip.place, slots, today, fetch)
    advice = [shifted(a, history.warmth_bias) for a in weather.advice]
    assumptions = list(weather.assumptions)
    if any(advice) and (note := history.warmth_note()):
        assumptions.append(note)
    plan = pack(items, slots, advice, trip.wears, taste)
    days = [replace(d, weather=w) for d, w in zip(plan.days, weather.summaries, strict=True)]
    return replace(
        plan,
        days=days,
        assumptions=tuple(assumptions),
        place=weather.place.label if weather.place else None,
    )
