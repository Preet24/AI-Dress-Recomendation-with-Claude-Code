"""Intake agent: turn chat messages into a complete request, asking for what is missing.

Two graph nodes (docs/agentic-architecture.md, section 5):

- `intake`: one DeepSeek JSON call extracts a *patch* (only what the latest message
  says); code merges it into the known slots, validates each value, and decides what
  is still missing. No interrupt here, so resuming the graph never repeats this call.
- `ask`: builds one combined question from a template and pauses the graph with
  `interrupt()`; the user's answer comes back as the next user message.

Code, not the model, owns defaults and validation: `start` defaults to "now", times
must parse with `parse_start`, and after `MAX_ASKS` questions intake stops asking and
falls back to safe assumptions, which are recorded for the stylist to mention.

The user's profile fills in more: a named saved place ("office tomorrow") brings its
occasion and dress code at once, and its usual trip after the user confirms it
("same as usual?"); a request without a city uses the home city. Each of these is an
assumption too.
"""

import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from langgraph.types import interrupt
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from outfit.agents.llm import chat_json, load_prompt
from outfit.agents.state import ChatState, Slots
from outfit.config import Settings
from outfit.profile import DRESS_CODE_FORMALITY, Profile, SavedPlace
from outfit.schemas import Occasion
from outfit.tools.weather import Transport, parse_start

logger = logging.getLogger(__name__)

MAX_ASKS = 3
WEATHER_SLOTS = ("city", "hours_away", "hours_outdoors", "transport")
QUESTIONS = {
    "occasion": "What's the occasion?",
    "city": "Which city will you be in?",
    "hours_away": "About how many hours will you be out?",
    "hours_outdoors": "How much of that time will you spend outdoors?",
    "transport": (
        "How will you get there: walking, bicycle, scooter or motorbike, "
        "public transport, or car / cab?"
    ),
}
FALLBACK_OCCASION = Occasion.CASUAL
TRIP_SLOTS = ("start", "hours_away", "hours_outdoors", "transport")  # a saved place's trip
SAVED_TRIP = "saved_trip"  # asked like a slot: "same as usual for Office?"
TRANSPORT_WORDS = {
    Transport.WALK: "on foot",
    Transport.BIKE: "by bicycle",
    Transport.TWO_WHEELER: "by scooter or motorbike",
    Transport.PUBLIC_TRANSIT: "by public transport",
    Transport.CAR: "by car or cab",
}


class IntakePatch(BaseModel):
    """What one message says; every field is optional and `None` means "not mentioned"."""

    model_config = ConfigDict(extra="forbid")

    occasion: Occasion | None = None
    formality: tuple[int, int] | None = None
    city: str | None = None
    skip_weather: bool | None = None
    start: str | None = None
    hours_away: float | None = None
    hours_outdoors: float | None = None
    transport: Transport | None = None
    must_include: list[str] | None = None
    exclude: list[str] | None = None
    place: str | None = None
    use_saved_trip: bool | None = None


# --- Pure logic -------------------------------------------------------------


def merge_slots(slots: Slots, patch: IntakePatch, now: datetime) -> tuple[Slots, list[str]]:
    """Apply the patch field by field; values that don't validate are dropped with a note."""
    values = slots.model_dump()
    rejected = []
    for key, value in patch.model_dump(exclude_none=True).items():
        if key == "start":
            try:
                parse_start(value, now)
            except ValueError:
                rejected.append(f"start={value!r}")
                continue
        try:
            Slots.model_validate({**values, key: value})
        except ValidationError:
            rejected.append(f"{key}={value!r}")
            continue
        values[key] = value
        if key == "city":
            values["skip_weather"] = False
    if rejected:
        logger.info("Intake dropped invalid values: %s", ", ".join(rejected))
    return Slots.model_validate(values), rejected


def missing_slots(slots: Slots) -> list[str]:
    """Slots still needed before planning, in the order they are asked."""
    missing = [] if slots.occasion else ["occasion"]
    if slots.skip_weather:
        return missing
    missing += [name for name in WEATHER_SLOTS if getattr(slots, name) is None]
    if (
        slots.hours_outdoors is not None
        and slots.hours_away is not None
        and slots.hours_outdoors > slots.hours_away
        and "hours_outdoors" not in missing
    ):
        missing.append("hours_outdoors")
    return missing


def adopt_place(slots: Slots, place: SavedPlace, patch: IntakePatch) -> tuple[Slots, list[str]]:
    """A newly named saved place: its occasion (unless the message gave one) and its dress
    code (unless a formality is already known, e.g. from "an interview")."""
    update: dict[str, Any] = {}
    assumptions = []
    if patch.occasion is None:
        update["occasion"] = place.occasion
    if slots.formality is None and place.dress_code:
        update["formality"] = DRESS_CODE_FORMALITY[place.dress_code]
        low, high = update["formality"]
        assumptions.append(
            f"{place.name}'s dress code is {place.dress_code.replace('_', ' ')} "
            f"(formality {low}-{high}), from your profile"
        )
    return slots.model_copy(update=update), assumptions


def trip_offer(slots: Slots, place: SavedPlace) -> dict[str, Any]:
    """The saved trip's values for the trip slots still unknown."""
    saved = {
        "start": place.start.strftime("%H:%M") if place.start else None,
        "hours_away": place.hours_away,
        "hours_outdoors": place.hours_outdoors,
        "transport": place.transport,
    }
    return {k: v for k, v in saved.items() if v is not None and getattr(slots, k) is None}


def describe_trip(offer: dict[str, Any]) -> str:
    """'leaving at 09:00, 9 h away, 0.25 h outdoors, by car or cab'."""
    parts = [
        f"leaving at {offer['start']}" if "start" in offer else None,
        f"{offer['hours_away']:g} h away" if "hours_away" in offer else None,
        f"{offer['hours_outdoors']:g} h outdoors" if "hours_outdoors" in offer else None,
        TRANSPORT_WORDS[offer["transport"]] if "transport" in offer else None,
    ]
    return ", ".join(p for p in parts if p)


def use_saved_trip(slots: Slots, place: SavedPlace) -> tuple[Slots, list[str]]:
    """Fill the trip slots still unknown from the saved place."""
    offer = trip_offer(slots, place)
    if not offer:
        return slots, []
    return slots.model_copy(update=offer), [f"your usual {place.name} trip: {describe_trip(offer)}"]


def with_trip_question(missing: list[str], slots: Slots, place: SavedPlace | None) -> list[str]:
    """Put "same as usual?" first while the saved trip is unconfirmed; it covers the
    slots it would fill."""
    if place is None or slots.use_saved_trip is not None:
        return missing
    offer = trip_offer(slots, place)
    if not offer:
        return missing
    return [SAVED_TRIP] + [name for name in missing if name not in offer]


def home_city(slots: Slots, profile: Profile) -> tuple[Slots, list[str]]:
    """No city said: the profile's home city, without asking."""
    if slots.city is not None or slots.skip_weather or profile.home_city is None:
        return slots, []
    return slots.model_copy(update={"city": profile.home_city}), [
        f"the city is {profile.home_city}, the home city in your profile (no city was given)"
    ]


def question_for(missing: list[str], slots: Slots, place: SavedPlace | None = None) -> str:
    """One combined, friendly question for everything still missing."""
    parts = [QUESTIONS[name] for name in missing if name != SAVED_TRIP]
    if "hours_outdoors" in missing and slots.hours_away is not None and len(missing) == 1:
        parts = [f"Of the {slots.hours_away:g} hours, how much will you spend outdoors?"]
    if SAVED_TRIP in missing and place is not None:
        usual = describe_trip(trip_offer(slots, place))
        confirm = f"Same as usual for {place.name}: {usual}?"
        tail = "Say yes, or tell me what's different."
        return " ".join([confirm, *parts, tail])
    intro = "Got it. " if len(missing) < len(QUESTIONS) else ""
    return intro + " ".join(parts)


def apply_defaults(slots: Slots) -> tuple[Slots, list[str]]:
    """Defaults that are safe to assume without asking."""
    assumptions = []
    if slots.start is None and not slots.skip_weather:
        slots = slots.model_copy(update={"start": "now"})
        assumptions.append("you're leaving now")
    return slots, assumptions


def give_up_asking(
    slots: Slots, missing: list[str], place: SavedPlace | None = None
) -> tuple[Slots, list[str]]:
    """After MAX_ASKS questions: fill what is still missing with safe fallbacks (an
    unconfirmed saved trip is the safest guess there is)."""
    assumptions = []
    if SAVED_TRIP in missing and place is not None:
        slots, assumptions = use_saved_trip(slots, place)
        missing = missing_slots(slots)
    update: dict[str, Any] = {}
    if "occasion" in missing:
        update["occasion"] = FALLBACK_OCCASION
        assumptions.append(f"a {FALLBACK_OCCASION} occasion")
    if any(name in missing for name in WEATHER_SLOTS):
        update["skip_weather"] = True
        assumptions.append("no weather check, since the trip details were unclear")
    return slots.model_copy(update=update), assumptions


def last_message(messages: list[dict], role: str) -> str | None:
    return next((m["content"] for m in reversed(messages) if m["role"] == role), None)


def extract(
    message: str,
    slots: Slots,
    last_question: str | None,
    client: OpenAI,
    settings: Settings,
    now: datetime,
    previous: Slots | None = None,
    places: Sequence[SavedPlace] = (),
) -> IntakePatch:
    """The DeepSeek call: what does this message say about the request?"""
    payload: dict[str, Any] = {
        "current_request": slots.model_dump(mode="json"),
        "previous_request": previous.model_dump(mode="json", exclude_defaults=True)
        if previous
        else None,
        "last_question": last_question,
        "message": message,
        "now": now.strftime("%A %Y-%m-%d %H:%M"),
    }
    if places:  # left out when there are none, so requests without a profile don't change
        payload["saved_places"] = [
            {
                "name": p.name,
                "occasion": p.occasion,
                "usual_start": p.start.strftime("%H:%M") if p.start else None,
            }
            for p in places
        ]
    messages = [
        {"role": "system", "content": load_prompt("intake", IntakePatch)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return chat_json(client, settings, messages, IntakePatch)


# --- Graph nodes ------------------------------------------------------------


def known_place(patch: IntakePatch, profile: Profile) -> IntakePatch:
    """The patch's place as saved (exact name), or dropped if there is no such place."""
    if patch.place is None:
        return patch
    place = profile.place(patch.place)
    if place is None:
        logger.info("Intake dropped unknown place %r", patch.place)
    return patch.model_copy(update={"place": place.name if place else None})


def intake_node(
    state: ChatState,
    *,
    client: OpenAI,
    settings: Settings,
    now: datetime | None = None,
    profile: Callable[[], Profile] = Profile,
) -> dict:
    now = now or datetime.now().replace(second=0, microsecond=0)
    user = profile()
    messages = state.get("messages", [])
    slots = state.get("slots") or Slots()
    message = last_message(messages, "user") or ""

    # The last assistant message is a question only while asking; otherwise it is the
    # reply to an earlier request.
    question = last_message(messages, "assistant") if state.get("asks") else None
    previous = state.get("previous_slots")
    patch = extract(message, slots, question, client, settings, now, previous, user.places)
    patch = known_place(patch, user)
    place_before = slots.place
    slots, _ = merge_slots(slots, patch, now)
    assumptions = list(state.get("assumptions", []))

    place = user.place(slots.place) if slots.place else None
    if place and slots.place != place_before:
        slots, notes = adopt_place(slots, place, patch)
        assumptions += notes
    if place and slots.use_saved_trip:
        slots, notes = use_saved_trip(slots, place)
        assumptions += notes
    slots, notes = home_city(slots, user)
    assumptions += notes
    missing = with_trip_question(missing_slots(slots), slots, place)

    if missing and state.get("asks", 0) >= MAX_ASKS:
        slots, given_up = give_up_asking(slots, missing, place)
        assumptions += given_up
        missing = missing_slots(slots)
    if not missing:
        slots, defaults = apply_defaults(slots)
        assumptions += defaults
    return {"slots": slots, "missing": missing, "assumptions": assumptions}


def ask_node(state: ChatState, profile: Callable[[], Profile] = Profile) -> dict:
    """Pause the graph with one question; the answer becomes the next user message."""
    slots = state.get("slots") or Slots()
    place = profile().place(slots.place) if slots.place else None
    question = question_for(state["missing"], slots, place)
    answer = interrupt(question)
    return {
        "messages": [
            {"role": "assistant", "content": question},
            {"role": "user", "content": str(answer)},
        ],
        "asks": state.get("asks", 0) + 1,
    }


def route_after_intake(state: ChatState) -> str:
    return "ask" if state.get("missing") else "ready"
