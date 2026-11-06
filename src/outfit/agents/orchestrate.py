"""Orchestrate node: the fixed pipeline, plain Python, no LLM.

slots -> (resolve "my white tee" to item ids) -> weather for the outing -> outfits.

Everything it needs is injected (store, weather fetcher, item resolver), so tests run
without the network, Qdrant or CLIP. Failures that should not end the conversation
become gaps or assumptions the stylist passes on: weather down, an item reference
that matches nothing.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from outfit.agents.state import ChatState, Slots
from outfit.db.qdrant_store import WardrobeStore
from outfit.profile import Profile
from outfit.schemas import ClothingItem
from outfit.tools.accessories import finish_plan
from outfit.tools.feedback import feedback, shifted
from outfit.tools.match import best_matches
from outfit.tools.outfits import OutfitPlan, OutfitRequest, label, suggest_outfits
from outfit.tools.search import WardrobeSearch
from outfit.tools.weather import (
    FetchJson,
    Outing,
    WeatherAdvice,
    WeatherError,
    WeatherSummary,
    advise,
    fetch_forecast,
    geocode,
    http_get_json,
    local_now,
    parse_start,
    summarize,
)
from outfit.wear_log import WearEntry

logger = logging.getLogger(__name__)

PLAN_LIMIT = 3  # outfits the stylist gets: one to recommend, two alternatives

# CLIP text->cutout score needed to accept a reference no tag matches ("something
# sparkly"). On the real wardrobe, items that exist matched at 0.258-0.340 ("white
# tee", "blue jeans", "heels"...); items that don't ("red saree", "leather jacket",
# "sneakers") topped out at 0.222.
MIN_MATCH_SCORE = 0.24
FILLER_WORDS = ("my ", "the ", "a ", "that ", "those ")
NO_WEATHER = "no weather check, since there's no place and time to check it for"
# Prefix of the assumption when the forecast fails; critic.ASSUMPTION_TOPICS keys on it.
WEATHER_FAILED = "couldn't check the weather"

# Turns a description ("my white tee") into the best-matching item, or None.
Resolver = Callable[[str, Sequence[ClothingItem]], ClothingItem | None]


def clip_query(ref: str) -> str:
    query = ref.strip().lower()
    for filler in FILLER_WORDS:
        query = query.removeprefix(filler)
    return query


def wardrobe_resolver(search_for: Callable[[], WardrobeSearch]) -> Resolver:
    """Tags first (tools/match.py), CLIP for ties and for references no tag matches.

    `search_for` builds the search lazily, so CLIP only loads when it is needed.
    """

    def resolve(ref: str, items: Sequence[ClothingItem]) -> ClothingItem | None:
        tied = best_matches(ref, list(items))
        if len(tied) == 1:
            return tied[0]
        allowed = {i.id for i in (tied or items)}
        hits = [
            h
            for h in search_for().by_text(clip_query(ref), limit=len(items))
            if h.item.id in allowed
        ]
        if tied:  # several items fit the words equally: the one that looks most like them
            return hits[0].item if hits else tied[0]
        return hits[0].item if hits and hits[0].score >= MIN_MATCH_SCORE else None

    return resolve


@dataclass(frozen=True)
class Context:
    """What orchestrate needs from the outside world."""

    store: WardrobeStore
    resolve: Resolver
    search: Callable[[], WardrobeSearch]  # built lazily: CLIP loads on the first search
    fetch: FetchJson = http_get_json
    now: datetime | None = None  # a fixed clock for evals; None = the real time
    profile: Callable[[], Profile] = Profile  # read every turn; default: an empty profile
    wear_log: Callable[[], list[WearEntry]] = list  # read every turn; default: nothing worn


def resolve_references(
    refs: list[str], items: Sequence[ClothingItem], resolve: Resolver
) -> tuple[frozenset[UUID], list[str]]:
    """Item ids for user references; an id prefix matches directly, text goes to `resolve`."""
    ids, notes = set(), []
    for ref in refs:
        by_id = [i for i in items if str(i.id).startswith(ref.strip())] if len(ref) >= 6 else []
        item = by_id[0] if len(by_id) == 1 else resolve(ref, items)
        if item is None:
            notes.append(f"couldn't find anything matching '{ref}' in your wardrobe")
            continue
        ids.add(item.id)
        if not by_id:
            notes.append(f"took '{ref}' to mean your {label(item)}")
    return frozenset(ids), notes


def weather_for(
    slots: Slots, fetch: FetchJson, now: datetime | None = None
) -> tuple[WeatherSummary | None, WeatherAdvice | None, list[str]]:
    """Forecast + advice for the outing; on failure, no advice and an assumption saying
    so (not a gap: the wardrobe is not missing anything)."""
    if slots.skip_weather or not slots.city:
        return None, None, []
    try:
        place = geocode(slots.city, fetch)
        outing = Outing(
            start=parse_start(slots.start or "now", now or local_now(place)),
            hours_away=slots.hours_away,
            hours_outdoors=slots.hours_outdoors,
            transport=slots.transport,
        )
        summary = summarize(place, fetch_forecast(place, outing, fetch, now))
    except (WeatherError, ValueError) as exc:
        logger.warning("Weather unavailable: %s", exc)
        return None, None, [f"{WEATHER_FAILED} ({exc}), so this ignores it"]
    return summary, advise(summary, outing), []


def orchestrate_node(state: ChatState, *, context: Context) -> dict:
    slots = state.get("slots") or Slots()
    items = context.store.list_items()
    include, include_notes = resolve_references(slots.must_include, items, context.resolve)
    exclude, exclude_notes = resolve_references(slots.exclude, items, context.resolve)
    weather, advice, weather_notes = weather_for(slots, context.fetch, context.now)
    profile = context.profile()
    worn = feedback(context.wear_log(), items, (context.now or datetime.now()).date())
    if advice is not None and (note := worn.warmth_note()):
        advice = shifted(advice, worn.warmth_bias)
        weather_notes.append(note)

    request = OutfitRequest(
        occasion=slots.occasion,
        formality=slots.formality,
        advice=advice,
        must_include=include,
        exclude=exclude,
        limit=PLAN_LIMIT,
        **profile.taste_fields(),
        **worn.request_fields(),
    )
    plan: OutfitPlan = finish_plan(suggest_outfits(items, request), items, request)
    logger.info("Planned %d outfit(s), gaps: %s", len(plan.outfits), plan.gaps)
    assumptions = [*state.get("assumptions", []), *include_notes, *exclude_notes, *weather_notes]
    if slots.skip_weather and not any("weather" in a for a in assumptions):
        # The user said where/when is unknown: the reply should say weather was left out.
        assumptions.append(NO_WEATHER)
    return {"weather": weather, "advice": advice, "plan": plan, "assumptions": assumptions}
