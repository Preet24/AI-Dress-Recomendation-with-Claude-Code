"""Refine agent: the one ReAct loop, for open-ended follow-ups on the last plan.

"Something with my blue jeans instead", "swap the shoes", "what if I go by scooter?",
"show me my other tops": the path is not known in advance, so DeepSeek picks tools
(function calling) and sees their results. Two graph nodes keep the loop explicit:

    refine (LLM call) -> tools (plain Python) -> refine -> ... -> stylist

- `refine` sends the transcript (`react` in state) with the tool schemas; a reply
  with tool calls goes to `tools`, a reply without them ends the loop.
- `tools` validates each call's arguments with Pydantic (DeepSeek documents that
  they may be invalid JSON; errors go back to the model as the tool result), runs it,
  and counts it. After `MAX_TOOL_CALLS` the loop ends without asking the model again.

Tools only change state (`slots`, `plan`, `found`); the model's own closing words are
dropped. The reply is still written by the stylist from those facts and checked by
the critic, like every other reply. Weather questions go through `update_request`,
which re-plans with the new forecast, so weather facts reach the stylist via state.
"""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from openai import APIError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from outfit.agents.intake import IntakePatch, last_message, merge_slots
from outfit.agents.llm import load_prompt
from outfit.agents.orchestrate import MIN_MATCH_SCORE, Context, orchestrate_node
from outfit.agents.state import ChatState, Slots
from outfit.config import Settings
from outfit.db.qdrant_store import ItemFilter, SearchHit
from outfit.schemas import Category, ClothingItem, Occasion
from outfit.tools.match import tag_matches
from outfit.tools.outfits import OutfitPlan, label

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 4
HISTORY = 6  # recent chat messages the model sees
ID_CHARS = 8  # item ids are shown to the model as short prefixes
MAX_SHOWN = 6


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description='What to look for, e.g. "white tee"')
    category: Category | None = None
    occasion: Occasion | None = None
    limit: int = Field(default=5, ge=1, le=8)


class SimilarArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=6, description="Id (prefix) of an item from a result")
    category: Category | None = None
    limit: int = Field(default=5, ge=1, le=8)


class ShowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_ids: list[str] = Field(min_length=1, max_length=MAX_SHOWN)


TOOL_ARGS: dict[str, tuple[type[BaseModel], str]] = {
    "search_wardrobe": (
        SearchArgs,
        "Find items in the user's wardrobe by description: items whose tags match the "
        "words come first, then the closest by image search. Returns ids, names and a "
        "match score; `weak` matches are probably not it.",
    ),
    "similar_items": (
        SimilarArgs,
        "Items that look like a given item (same look, not 'goes with').",
    ),
    "update_request": (
        IntakePatch,
        "Change the request and re-plan the outfits (with the weather for the outing). "
        "Send only the fields that change. `must_include` / `exclude` replace the "
        "current lists: send the full list, with item ids or descriptions.",
    ),
    "show_items": (
        ShowArgs,
        "Put these items in front of the user, for questions like 'show me my tops'.",
    ),
}


def inline_refs(schema: dict) -> dict:
    """Pydantic puts enums in `$defs`; tool schemas are simpler without references."""
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": inline_refs(model.model_json_schema()),
        },
    }
    for name, (model, description) in TOOL_ARGS.items()
]


# --- What the model sees -------------------------------------------------------


def brief(item: ClothingItem) -> dict[str, Any]:
    return {
        "id": str(item.id)[:ID_CHARS],
        "name": label(item),
        "category": item.category.value,
        "formality": item.formality,
        "occasions": [o.value for o in item.occasions],
    }


def plan_summary(plan: OutfitPlan | None, slots: Slots) -> dict[str, Any]:
    return {
        "request": slots.model_dump(mode="json", exclude_defaults=True),
        "outfits": [
            {"option": n, "pieces": [brief(p) for p in o.pieces], "warnings": list(o.warnings)}
            for n, o in enumerate(plan.outfits, 1)
        ]
        if plan
        else [],
        "gaps": plan.gaps if plan else [],
    }


def first_messages(state: ChatState) -> list[dict]:
    messages = state.get("messages", [])
    payload = {
        "conversation": [{"role": m["role"], "content": m["content"]} for m in messages[-HISTORY:]],
        "current": plan_summary(state.get("plan"), state.get("slots") or Slots()),
        "message": last_message(messages, "user") or "",
    }
    return [
        {"role": "system", "content": load_prompt("refine")},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def as_dict(message: Any) -> dict:
    """The assistant message in the plain-dict form the API takes back (and state stores)."""
    result: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if calls := getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in calls
        ]
    if reasoning := getattr(message, "reasoning_content", None):
        result["reasoning_content"] = reasoning  # DeepSeek wants it back within a turn
    return result


# --- Tools -----------------------------------------------------------------------


def by_prefix(ref: str, items: list[ClothingItem]) -> ClothingItem | None:
    ref = ref.strip().lower()
    matches = [i for i in items if str(i.id).startswith(ref)] if len(ref) >= 4 else []
    return matches[0] if len(matches) == 1 else None


def hits_result(hits: list[SearchHit], by_tags: set[UUID] | None = None) -> dict:
    """Hits for the model; items whose tags match the query are never `weak`."""
    by_tags = by_tags or set()
    return {
        "items": [
            {
                **brief(h.item),
                "match": round(h.score, 3),
                "weak": h.score < MIN_MATCH_SCORE and h.item.id not in by_tags,
            }
            for h in hits
        ]
    }


def search_hits(args: SearchArgs, context: Context) -> tuple[list[SearchHit], set[UUID]]:
    """Tag matches first (tools/match.py), then the rest by CLIP similarity."""
    f = item_filter(args.category, args.occasion)
    pool = context.store.list_items(f)
    order = {item.id: n for n, (item, _) in enumerate(tag_matches(args.query, pool))}
    hits = context.search().by_text(args.query, f, max(len(pool), 1))
    hits.sort(key=lambda h: order.get(h.item.id, len(order)))  # stable: CLIP order otherwise
    return hits[: args.limit], set(order)


def item_filter(category: Category | None, occasion: Occasion | None = None) -> ItemFilter:
    return ItemFilter(
        categories=(category,) if category else (), occasions=(occasion,) if occasion else ()
    )


def run_tool(
    name: str, arguments: str, state: ChatState, context: Context, now: datetime
) -> tuple[dict, dict]:
    """(result for the model, state update). Bad calls return {"error": ...} and no update."""
    if name not in TOOL_ARGS:
        return {"error": f"unknown tool {name!r}"}, {}
    model = TOOL_ARGS[name][0]
    try:
        args = model.model_validate(json.loads(arguments or "{}"))
    except (json.JSONDecodeError, ValidationError) as exc:
        return {"error": f"invalid arguments: {exc}"}, {}

    if isinstance(args, SearchArgs):
        hits, by_tags = search_hits(args, context)
        if args.category or by_tags or any(h.score >= MIN_MATCH_SCORE for h in hits):
            return hits_result(hits, by_tags), {}
        # Looking for one item and it is not there: tell the user, like orchestrate does.
        note = f"couldn't find anything matching '{args.query}' in your wardrobe"
        return hits_result(hits), {"assumptions": [*state.get("assumptions", []), note]}

    items = context.store.list_items()
    if isinstance(args, SimilarArgs):
        item = by_prefix(args.item_id, items)
        if item is None:
            return {"error": f"no item with id {args.item_id!r}"}, {}
        return hits_result(
            context.search().similar_to(item.id, item_filter(args.category), args.limit)
        ), {}

    if isinstance(args, ShowArgs):
        shown = [i for ref in args.item_ids if (i := by_prefix(ref, items))]
        unknown = [ref for ref in args.item_ids if by_prefix(ref, items) is None]
        return {"shown": [label(i) for i in shown], "unknown_ids": unknown}, {"found": shown}

    # update_request: merge like intake does, then re-plan with the fixed pipeline.
    slots, rejected = merge_slots(state.get("slots") or Slots(), args, now)
    planned = orchestrate_node({**state, "slots": slots}, context=context)
    planned["assumptions"] = list(dict.fromkeys(planned["assumptions"]))  # re-plans repeat notes
    result = {"rejected": rejected, **plan_summary(planned["plan"], slots)}
    return result, {"slots": slots, **planned}


# --- Graph nodes -------------------------------------------------------------------


def refine_node(state: ChatState, *, client: OpenAI, settings: Settings) -> dict:
    """One model step: either tool calls (-> tools) or a final answer (-> stylist)."""
    react = state.get("react") or first_messages(state)
    try:
        response = client.chat.completions.create(
            model=settings.agent_model,
            messages=react,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=settings.agent_max_tokens,
        )
    except APIError as exc:
        logger.warning("Refine call failed, replying from the current plan: %s", exc)
        return {"react": react}
    message = as_dict(response.choices[0].message)
    if not message.get("tool_calls"):
        logger.info("Refine done: %s", message["content"][:200])
    return {"react": [*react, message]}


def tools_node(state: ChatState, *, context: Context, now: datetime | None = None) -> dict:
    """Run the tool calls of the last refine step, in order, within the budget."""
    now = now or context.now or datetime.now().replace(second=0, microsecond=0)
    current: dict = dict(state)
    update: dict = {}
    results = []
    used = state.get("tool_calls", 0)
    for call in state["react"][-1]["tool_calls"]:
        name = call["function"]["name"]
        if used >= MAX_TOOL_CALLS:
            result: dict = {"error": "tool budget used up for this message"}
        else:
            used += 1
            result, changes = run_tool(name, call["function"]["arguments"], current, context, now)
            current.update(changes)
            update.update(changes)
            logger.info("Tool %s -> %s", name, "error" if "error" in result else "ok")
        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    return {**update, "react": [*state["react"], *results], "tool_calls": used}


def route_after_refine(state: ChatState) -> str:
    react = state.get("react") or [{}]
    return "tools" if react[-1].get("tool_calls") else "done"


def route_after_tools(state: ChatState) -> str:
    return "done" if state.get("tool_calls", 0) >= MAX_TOOL_CALLS else "refine"
