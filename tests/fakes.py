"""Shared test double for the DeepSeek client (chat completions via the OpenAI-compatible SDK)."""

import json
from types import SimpleNamespace

VALID_TAGS = {
    "category": "top",
    "subtype": "t-shirt",
    "primary_color_name": "red",
    "secondary_colors": [],
    "pattern": "solid",
    "fabric": "cotton",
    "length": "short",
    "fit": "regular",
    "formality": 2,
    "warmth": 1,
    "flowy": False,
    "occasions": ["casual", "travel"],
}


def reply(text: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    """A chat completion with one choice."""
    message = SimpleNamespace(role="assistant", content=text)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def tags_reply(**overrides) -> SimpleNamespace:
    return reply(json.dumps({**VALID_TAGS, **overrides}))


class FakeClient:
    """Stands in for the DeepSeek client: returns queued replies, records requests."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self.replies.pop(0)


# Passes the critic's code checks for any plan: names no piece, mentions shoes and weather.
NEUTRAL_MESSAGE = "I couldn't check the weather, and found no outfit or shoes for this."


def stylist_reply(message: str = NEUTRAL_MESSAGE) -> SimpleNamespace:
    return reply(json.dumps({"message": message}))


def critic_reply(*issues: str) -> SimpleNamespace:
    """The critic rubric's verdict: ok without issues, otherwise these issues."""
    return reply(json.dumps({"ok": not issues, "issues": list(issues)}))


def router_reply(intent: str = "new", reply_text: str | None = None) -> SimpleNamespace:
    return reply(json.dumps({"intent": intent, "reply": reply_text}))


def tool_call_reply(*calls: tuple[str, dict]) -> SimpleNamespace:
    """A refine step that calls these tools: (name, arguments) pairs."""
    tool_calls = [
        SimpleNamespace(
            id=f"call_{n}",
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for n, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(role="assistant", content="", tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=message)])


class FakeSearch:
    """WardrobeSearch stand-in: by_text scores by shared words with the item's name, so
    tests need no CLIP; similar_to is the real one (it only needs the store's vectors)."""

    def __init__(self, store) -> None:
        from outfit.tools.search import WardrobeSearch

        self.store = store
        self.similar_to = WardrobeSearch(store, clip=None).similar_to
        self.queries: list[str] = []

    def by_text(self, query, filters=None, limit=10):
        from outfit.db.qdrant_store import SearchHit
        from outfit.tools.outfits import label

        self.queries.append(query)
        wanted = set(query.lower().split())
        hits = []
        for item in self.store.list_items():
            if filters and filters.categories and item.category not in filters.categories:
                continue
            shared = len(wanted & set(label(item).lower().split()))
            hits.append(SearchHit(item, 0.2 + 0.05 * shared))  # 1 shared word clears 0.24
        return sorted(hits, key=lambda h: -h.score)[:limit]


def fake_context(items=(), resolve=None, fetch=None, profile=None, wear_log=None):
    """orchestrate Context over an in-memory wardrobe; weather fails unless `fetch` is given;
    `profile` is a Profile or a loader (default: empty); `wear_log` a list of WearEntry or
    a loader (default: nothing worn)."""
    import numpy as np
    from qdrant_client import QdrantClient

    from outfit.agents.orchestrate import Context
    from outfit.db.qdrant_store import VECTOR_SIZE, WardrobeStore
    from outfit.profile import Profile
    from outfit.tools.weather import WeatherError

    store = WardrobeStore(QdrantClient(":memory:"))
    store.ensure_collection()
    if items:
        vectors = np.random.default_rng(0).normal(size=(len(items), VECTOR_SIZE))
        store.upsert(list(items), vectors / np.linalg.norm(vectors, axis=1, keepdims=True))

    def no_network(url, params):
        raise WeatherError("no network in tests")

    search = FakeSearch(store)
    return Context(
        store=store,
        resolve=resolve or (lambda ref, items: None),
        search=lambda: search,
        fetch=fetch or no_network,
        profile=profile if callable(profile) else (lambda: profile or Profile()),
        wear_log=wear_log if callable(wear_log) else (lambda: list(wear_log or [])),
    )


def fake_services(items=(), client=None, pipeline=None, settings=None, fetch=None):
    """API services over fake_context: in-memory wardrobe, FakeClient, no CLIP or network."""
    from outfit.api.services import Services
    from outfit.config import Settings
    from outfit.profile import ProfileStore

    settings = settings or Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")
    from outfit.wear_log import WearLogStore

    profile = ProfileStore()  # in memory; the chat reads what PUT /profile saved
    wear_log = WearLogStore()  # in memory; the chat and packing read what POST /worn logged
    context = fake_context(items, fetch=fetch, profile=profile.load, wear_log=wear_log.entries)

    def no_pipeline():
        raise AssertionError("this test does not ingest photos")

    return Services(
        settings,
        context.store,
        client or FakeClient(),
        context,
        pipeline or no_pipeline,
        profile=profile,
        wear_log=wear_log,
    )
