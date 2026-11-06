"""Stylist agent: write the reply from the plan's facts, and nothing but those facts.

The model gets a compact JSON of what was decided (outfits with their reasons,
warnings and gaps, the weather and its clothing notes, any assumptions) and returns
the chat message as JSON. It never sees photos or the rest of the wardrobe, so it
can only talk about items that are in the plan; the critic checks that it did.
"""

import json
import logging
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from outfit.agents.intake import last_message
from outfit.agents.llm import LLMError, chat_json, load_prompt
from outfit.agents.state import ChatState, Slots
from outfit.config import Settings
from outfit.tools.outfits import Outfit, label

logger = logging.getLogger(__name__)


class StylistDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)  # the chat reply, plain text with simple bullets


def outfit_facts(n: int, outfit: Outfit) -> dict[str, Any]:
    return {
        "option": n,
        "score": outfit.score,
        "pieces": [{"name": label(p), "category": p.category} for p in outfit.pieces],
        "why": list(outfit.reasons),
        "warnings": list(outfit.warnings),
    }


def facts(state: ChatState) -> dict[str, Any]:
    """Everything the stylist may talk about, as plain JSON."""
    slots = state.get("slots") or Slots()
    plan = state.get("plan")
    weather = state.get("weather")
    advice = state.get("advice")
    result: dict[str, Any] = {
        "user_message": last_message(state.get("messages", []), "user"),
        "request": slots.model_dump(mode="json", exclude_defaults=True),
        "outfits": [outfit_facts(n, o) for n, o in enumerate(plan.outfits, 1)] if plan else [],
        "gaps": plan.gaps if plan else [],
        "assumptions": state.get("assumptions", []),
    }
    if found := state.get("found"):
        result["items_to_show"] = [{"name": label(i), "category": i.category} for i in found]
    if weather and advice:
        result["weather"] = {
            "place": weather.place,
            "time": f"{weather.start:%a %H:%M}-{weather.end:%H:%M}",
            "temperature_c": [round(weather.temp_min), round(weather.temp_max)],
            "feels_like_min_c": round(weather.feels_like_min),
            "rain_chance_max_pct": round(weather.rain_chance_max),
            "uv_index_max": round(weather.uv_index_max),
            "wind_gusts_max_kmh": round(weather.wind_gusts_max),
            "sky": weather.conditions,
            "clothing_notes": advice.notes,
        }
    if critique := state.get("critique"):
        result["previous_draft"] = state.get("draft")
        result["fix_these_issues"] = critique
    return result


def write_reply(state: ChatState, client: OpenAI, settings: Settings) -> str:
    messages = [
        {"role": "system", "content": load_prompt("stylist", StylistDraft)},
        {"role": "user", "content": json.dumps(facts(state), ensure_ascii=False)},
    ]
    return chat_json(client, settings, messages, StylistDraft).message.strip()


def stylist_node(state: ChatState, *, client: OpenAI, settings: Settings) -> dict:
    """Draft the reply; if the model fails, leave no draft so `respond` uses the template."""
    try:
        return {"draft": write_reply(state, client, settings)}
    except LLMError as exc:
        logger.warning("Stylist failed, falling back to the template reply: %s", exc)
        return {"draft": None}
