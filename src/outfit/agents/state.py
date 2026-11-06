"""Shared graph state: every node reads `ChatState` and returns the fields it changes."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from outfit.schemas import ClothingItem, Occasion
from outfit.tools.outfits import OutfitPlan
from outfit.tools.weather import Transport, WeatherAdvice, WeatherSummary


class Slots(BaseModel):
    """What the user wants, filled in over one or more turns."""

    model_config = ConfigDict(extra="forbid")

    occasion: Occasion | None = None
    formality: tuple[int, int] | None = None
    city: str | None = None
    skip_weather: bool = False  # user does not know or care: plan without weather
    start: str | None = None  # parse_start format: "now", "19:00", "tomorrow 9:00"
    hours_away: float | None = Field(default=None, gt=0, le=24)
    hours_outdoors: float | None = Field(default=None, ge=0, le=24)
    transport: Transport | None = None
    must_include: list[str] = Field(default_factory=list)  # "my white tee" or item ids
    exclude: list[str] = Field(default_factory=list)
    place: str | None = None  # a saved place from the profile ("Office")
    use_saved_trip: bool | None = None  # the answer to "same as usual?"; None = not asked yet


class ChatState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # chat history: {"role", "content"}
    intent: Literal["new", "follow_up", "chat"] | None
    slots: Slots
    previous_slots: Slots | None  # the request before a "new" one, for "same city as before"
    missing: list[str]  # slot names intake still needs, in asking order
    asks: int  # questions asked for the current request
    assumptions: list[str]  # defaults applied instead of asking, for the stylist to mention
    weather: WeatherSummary | None
    advice: WeatherAdvice | None
    plan: OutfitPlan | None
    found: list[ClothingItem]  # refine's search results this turn, for the stylist to show
    react: list[dict]  # refine's tool-calling transcript this turn (OpenAI message format)
    tool_calls: int  # tools run this turn (max refine.MAX_TOOL_CALLS)
    draft: str | None
    critique: list[str]
    revisions: int
    reply: str | None
