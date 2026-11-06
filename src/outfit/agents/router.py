"""Router agent: is this message a new request, a follow-up on the last plan, or chat?

One small DeepSeek JSON call per new message (resumed answers to a question skip it;
they go straight back to `ask`). Code owns the rules around it:

- with no plan yet, a follow-up is impossible, so `follow_up` becomes `new`;
- `new` starts the request over: slots, plan and weather from the previous request
  are cleared, so intake does not mix two outings (the old slots are kept aside as
  `previous_slots`, for "same city as before");
- `chat` ("thanks!", "what can you do?") is answered by the router in the same call
  and goes straight to `respond`; that reply may not name garments or recommend
  anything, so the critic has nothing to check;
- if the call fails, the message is treated as `new` (intake will ask if unclear).
"""

import json
import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from outfit.agents.intake import last_message
from outfit.agents.llm import LLMError, chat_json, load_prompt
from outfit.agents.state import ChatState, Slots
from outfit.config import Settings
from outfit.tools.outfits import label

logger = logging.getLogger(__name__)

Intent = Literal["new", "follow_up", "chat"]


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    reply: str | None = Field(default=None, max_length=600)  # only for "chat"

    @model_validator(mode="after")
    def chat_needs_a_reply(self) -> "RouterDecision":
        if self.intent == "chat" and not (self.reply and self.reply.strip()):
            raise ValueError("intent is chat, so give the reply")
        return self


def previous(state: ChatState) -> dict | None:
    """The last request and its top outfit, so the model can tell a follow-up apart."""
    plan = state.get("plan")
    if plan is None:
        return None
    slots = state.get("slots") or Slots()
    return {
        "request": slots.model_dump(mode="json", exclude_defaults=True),
        "top_outfit": [label(p) for p in plan.outfits[0].pieces] if plan.outfits else [],
        "last_reply": last_message(state.get("messages", []), "assistant"),
    }


def decide(state: ChatState, client: OpenAI, settings: Settings) -> RouterDecision:
    payload = {
        "message": last_message(state.get("messages", []), "user") or "",
        "previous": previous(state),
    }
    messages = [
        {"role": "system", "content": load_prompt("router", RouterDecision)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        decision = chat_json(client, settings, messages, RouterDecision)
    except LLMError as exc:
        logger.warning("Router failed, treating the message as a new request: %s", exc)
        return RouterDecision(intent="new")
    if decision.intent == "follow_up" and state.get("plan") is None:
        return RouterDecision(intent="new")
    return decision


def router_node(state: ChatState, *, client: OpenAI, settings: Settings) -> dict:
    decision = decide(state, client, settings)
    logger.info("Router: %s", decision.intent)
    if decision.intent == "chat":
        return {"intent": "chat", "draft": decision.reply.strip()}
    if decision.intent == "new":
        return {
            "intent": "new",
            "previous_slots": state.get("slots") if state.get("plan") else None,
            "slots": Slots(),
            "plan": None,
            "weather": None,
            "advice": None,
            "assumptions": [],
        }
    return {"intent": "follow_up"}


def route_after_router(state: ChatState) -> str:
    return state["intent"]
