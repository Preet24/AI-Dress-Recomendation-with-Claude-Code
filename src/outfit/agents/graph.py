"""The LangGraph workflow (docs/agentic-architecture.md, section 3).

    START -> router -> new:       intake <-> ask (interrupt); intake -> orchestrate -> stylist
                       follow_up: refine <-> tools (ReAct, max 4 calls); refine -> stylist
                       chat:      respond (the router wrote the reply)
    stylist <-> critic (max 1 revision); critic -> respond -> END

A resumed run (the answer to `ask`'s question) continues in `ask`, not at the router.
"""

from collections.abc import Callable
from datetime import datetime
from functools import cache, partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI
from qdrant_client import QdrantClient

from outfit.agents.critic import critic_node, route_after_critic
from outfit.agents.intake import ask_node, intake_node, route_after_intake
from outfit.agents.orchestrate import Context, orchestrate_node, wardrobe_resolver
from outfit.agents.refine import refine_node, route_after_refine, route_after_tools, tools_node
from outfit.agents.respond import respond_node
from outfit.agents.router import route_after_router, router_node
from outfit.agents.state import ChatState, Slots
from outfit.agents.stylist import stylist_node
from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.ingestion.embedder import get_clip
from outfit.profile import Profile
from outfit.schemas import Category, ClothingItem, Fit, Length, Occasion, Pattern
from outfit.tools.outfits import Outfit, OutfitPlan
from outfit.tools.search import WardrobeSearch
from outfit.tools.weather import (
    FetchJson,
    Transport,
    WeatherAdvice,
    WeatherSummary,
    http_get_json,
)
from outfit.wear_log import WearEntry

# Our types stored in ChatState. The checkpointer only restores allow-listed classes
# (LangGraph is phasing out unregistered ones), so every state type is listed here.
STATE_TYPES = (
    Slots,
    Occasion,
    Transport,
    WeatherSummary,
    WeatherAdvice,
    OutfitPlan,
    Outfit,
    ClothingItem,
    Category,
    Pattern,
    Length,
    Fit,
)


def checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in STATE_TYPES]
    )


def clip_context(
    store: WardrobeStore,
    settings: Settings,
    fetch: FetchJson = http_get_json,
    now: datetime | None = None,
    profile: Callable[[], Profile] = Profile,
    wear_log: Callable[[], list[WearEntry]] = list,
) -> Context:
    """Context over a store with real CLIP search (loaded on first use)."""

    @cache
    def search() -> WardrobeSearch:
        return WardrobeSearch(store, get_clip(settings.clip_model, settings.clip_pretrained))

    return Context(
        store=store,
        resolve=wardrobe_resolver(search),
        search=search,
        fetch=fetch,
        now=now,
        profile=profile,
        wear_log=wear_log,
    )


def live_context(settings: Settings) -> Context:
    """Real dependencies: the local wardrobe, Open-Meteo, CLIP and the real clock."""
    return clip_context(WardrobeStore(QdrantClient(path=str(settings.qdrant_path))), settings)


def build_graph(
    client: OpenAI,
    settings: Settings,
    context: Context,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the graph; state is saved per `thread_id` by the checkpointer."""
    llm = {"client": client, "settings": settings}
    graph = StateGraph(ChatState)
    graph.add_node("router", partial(router_node, **llm))
    graph.add_node("intake", partial(intake_node, **llm, now=context.now, profile=context.profile))
    graph.add_node("ask", partial(ask_node, profile=context.profile))
    graph.add_node("orchestrate", partial(orchestrate_node, context=context))
    graph.add_node("refine", partial(refine_node, **llm))
    graph.add_node("tools", partial(tools_node, context=context))
    graph.add_node("stylist", partial(stylist_node, **llm))
    graph.add_node("critic", partial(critic_node, **llm))
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"new": "intake", "follow_up": "refine", "chat": "respond"},
    )
    graph.add_conditional_edges("refine", route_after_refine, {"tools": "tools", "done": "stylist"})
    graph.add_conditional_edges("tools", route_after_tools, {"refine": "refine", "done": "stylist"})
    graph.add_conditional_edges(
        "intake", route_after_intake, {"ask": "ask", "ready": "orchestrate"}
    )
    graph.add_edge("ask", "intake")
    graph.add_edge("orchestrate", "stylist")
    graph.add_edge("stylist", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"revise": "stylist", "done": "respond"}
    )
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serde()))
