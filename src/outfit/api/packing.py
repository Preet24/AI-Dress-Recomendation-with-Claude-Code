"""Packing endpoints: check a trip, then, once the user confirms it, pack for it.

`POST /packing/preview` shows the places the city could be (full name, state and
country), the dates, and every day and event with its weather; it uses no wardrobe.
`POST /packing` takes the same trip with the place the user confirmed and returns
what to pack and what to wear each day (`tools/packing.py`). Neither calls DeepSeek.
"""

from datetime import date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from outfit.api.chat import OutfitOut
from outfit.api.items import ItemOut
from outfit.api.services import Services, get_services
from outfit.schemas import Occasion
from outfit.tools.feedback import feedback
from outfit.tools.outfits import OutfitRequest
from outfit.tools.packing import (
    DayOutfit,
    PackingPlan,
    Slot,
    Source,
    Trip,
    TripPreview,
    Wears,
    dates_text,
    plan_trip,
    preview_trip,
    weather_line,
)
from outfit.tools.weather import Place, Transport, WeatherSummary

router = APIRouter(prefix="/packing", tags=["packing"])
ServicesDep = Annotated[Services, Depends(get_services)]


class PlaceOut(BaseModel):
    """A place with its full label; send it back unchanged as the trip's `place`."""

    label: str  # "Manali, Himachal Pradesh, India"
    name: str
    admin1: str  # state / province
    country: str
    latitude: float
    longitude: float
    timezone: str

    @classmethod
    def of(cls, place: Place) -> "PlaceOut":
        return cls(**place.model_dump(), label=place.label)


class SlotOut(BaseModel):
    day: int
    name: str  # "day 2 party"
    occasion: Occasion
    start: datetime  # local time at the destination
    hours_away: float
    hours_outdoors: float
    transport: Transport
    indoors: bool  # an evening in (relax / sleep): no weather, no shoes

    @classmethod
    def of(cls, slot: Slot) -> "SlotOut":
        outing = slot.outing
        return cls(
            day=slot.day,
            name=slot.name,
            occasion=slot.occasion,
            start=outing.start,
            hours_away=outing.hours_away,
            hours_outdoors=outing.hours_outdoors,
            transport=outing.transport,
            indoors=slot.indoors,
        )


class RowOut(BaseModel):
    slot: SlotOut
    weather: str  # "feels 11-18 °C, rain up to 22%, ..."
    source: Source  # forecast / last year / not checked / indoors
    summary: WeatherSummary | None


class PreviewOut(BaseModel):
    city: str | None  # the name searched (the home city if the trip gave none)
    matches: list[PlaceOut]  # best first; send the chosen one back as `place`
    place: PlaceOut | None  # the place this preview's weather is for
    dates: str  # "Fri 30 Oct - Mon 2 Nov 2026, 4 days"
    rows: list[RowOut]
    wears: Wears
    assumptions: list[str]

    @classmethod
    def of(cls, city: str | None, preview: TripPreview, assumed: list[str]) -> "PreviewOut":
        return cls(
            city=city,
            matches=[PlaceOut.of(p) for p in preview.matches],
            place=PlaceOut.of(preview.place) if preview.place else None,
            dates=preview.dates,
            rows=[
                RowOut(
                    slot=SlotOut.of(r.slot),
                    weather=weather_line(r.weather, r.slot.indoors),
                    source=r.source,
                    summary=r.weather,
                )
                for r in preview.rows
            ],
            wears=preview.wears,
            assumptions=assumed + preview.assumptions,
        )


class PackedOut(BaseModel):
    item: ItemOut
    days: list[int]  # the days it is worn on


class DayOut(BaseModel):
    slot: SlotOut
    outfit: OutfitOut | None  # None: nothing fits (see gaps)
    notes: list[str]  # "your navy shirt again: ..."
    weather: str

    @classmethod
    def of(cls, day: DayOutfit) -> "DayOut":
        outfit = day.outfit
        return cls(
            slot=SlotOut.of(day.slot),
            outfit=OutfitOut(
                pieces=[ItemOut.of(p) for p in outfit.pieces],
                reasons=list(outfit.reasons),
                warnings=list(outfit.warnings),
            )
            if outfit
            else None,
            notes=list(day.notes),
            weather=weather_line(day.weather, day.slot.indoors),
        )


class PlanOut(BaseModel):
    place: str | None  # "Manali, Himachal Pradesh, India"
    dates: str
    pieces: list[PackedOut]  # tops, bottoms, one-pieces, outerwear, shoes, accessories
    days: list[DayOut]
    gaps: list[str]  # "no footwear suits party (days 2 and 4)"
    notes: list[str]  # "plus socks and underwear for 4 days"
    assumptions: list[str]

    @classmethod
    def of(cls, trip: Trip, plan: PackingPlan, assumed: list[str]) -> "PlanOut":
        return cls(
            place=plan.place,
            dates=dates_text(trip),
            pieces=[PackedOut(item=ItemOut.of(p.item), days=list(p.days)) for p in plan.pieces],
            days=[DayOut.of(d) for d in plan.days],
            gaps=plan.gaps,
            notes=plan.notes,
            assumptions=assumed + list(plan.assumptions),
        )


def today(s: Services) -> date:
    return (s.context.now or datetime.now()).date()


def with_home_city(trip: Trip, s: Services) -> tuple[Trip, list[str]]:
    """The trip, with the profile's home city when it names neither a city nor a place."""
    if trip.city or trip.place:
        return trip, []
    home = s.profile.load().home_city
    if not home:
        return trip, []
    return trip.model_copy(update={"city": home}), [f"no city given: your home city, {home}"]


@router.post("/preview")
def preview(trip: Trip, s: ServicesDep) -> PreviewOut:
    """Check the trip before packing: where (pick one of `matches` and send it back as
    `place`), when, and each day's and event's weather. Free: no wardrobe, no DeepSeek."""
    trip, assumed = with_home_city(trip, s)
    return PreviewOut.of(trip.city, preview_trip(trip, today(s), s.context.fetch), assumed)


@router.post("")
def pack(trip: Trip, s: ServicesDep) -> PlanOut:
    """Pack for a confirmed trip: it must carry the `place` the user picked from the
    preview (422 otherwise). What to pack, and what to wear on each day."""
    trip, assumed = with_home_city(trip, s)
    items = s.store.list_items()
    taste = OutfitRequest(**s.profile.load().taste_fields())
    history = feedback(s.wear_log.entries(), items, today(s))
    plan = plan_trip(items, trip, today(s), s.context.fetch, taste, history)
    return PlanOut.of(trip, plan, assumed)
