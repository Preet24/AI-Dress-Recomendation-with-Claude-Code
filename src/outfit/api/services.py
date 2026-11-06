"""Everything the API needs, built once per process.

Local Qdrant lets only one process open `qdrant_path`, so the app opens it once and
shares that store between ingestion and chat (both already take the store as an
argument). Tests build `Services` from fakes instead of `live_services`.

Conversations live in one checkpointer, keyed by `thread_id`. Each chat turn compiles
the graph around its own `RecordingClient` (milliseconds), so a turn's record holds
exactly its own LLM calls even while other conversations run in parallel.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from threading import Lock

from fastapi import Request
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI
from qdrant_client import QdrantClient

from outfit.agents.graph import build_graph, checkpoint_serde, clip_context
from outfit.agents.orchestrate import Context
from outfit.agents.recording import Recorder, RecordingClient
from outfit.api.errors import Busy
from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.ingestion.candidates import CandidateStore
from outfit.ingestion.embedder import get_clip
from outfit.ingestion.pipeline import Pipeline
from outfit.ingestion.tagger import make_client
from outfit.profile import ProfileStore
from outfit.tools.climate import WeatherKind, typical_kinds
from outfit.wear_log import WearLogStore


@dataclass
class Services:
    settings: Settings
    store: WardrobeStore
    client: OpenAI  # DeepSeek client (OpenAI-compatible SDK), shared by tagger and agents
    context: Context  # the agents' store, CLIP search, weather and clock
    pipeline: Callable[[], Pipeline]  # ingestion; loads CLIP and rembg on first use
    profile: ProfileStore = field(default_factory=ProfileStore)  # in memory unless given a path
    candidates: CandidateStore = field(default_factory=CandidateStore)  # shopping, in memory
    wear_log: WearLogStore = field(default_factory=WearLogStore)  # in memory unless given a path
    _climates: dict[tuple[str | None, date], tuple[list[WeatherKind], str]] = field(
        default_factory=dict
    )
    _climates_lock: Lock = field(default_factory=Lock)
    # In memory: conversations are lost on restart (a SQLite saver would need new packages).
    checkpointer: BaseCheckpointSaver = field(
        default_factory=lambda: InMemorySaver(serde=checkpoint_serde())
    )
    # One photo at a time: ingestion is CPU-heavy and re-tagging rewrites an item.
    ingest_lock: Lock = field(default_factory=Lock)
    _thread_locks: dict[str, Lock] = field(default_factory=dict)
    _locks_lock: Lock = field(default_factory=Lock)
    recorders: dict[str, Recorder] = field(default_factory=dict)  # when record_chats is on

    def graph(self, client: RecordingClient | OpenAI | None = None) -> CompiledStateGraph:
        """The agent graph over the shared conversations, calling DeepSeek via `client`."""
        return build_graph(client or self.client, self.settings, self.context, self.checkpointer)

    def thread_lock(self, thread_id: str) -> Lock:
        """Held for a whole turn; a second message to a busy conversation is refused."""
        with self._locks_lock:
            lock = self._thread_locks.setdefault(thread_id, Lock())
        if not lock.acquire(blocking=False):
            raise Busy("This conversation is still answering the previous message")
        return lock

    def recorder(self, thread_id: str) -> Recorder:
        if thread_id not in self.recorders:
            self.recorders[thread_id] = Recorder(
                self.settings.conversations_dir, thread_id, self.settings.agent_model
            )
        return self.recorders[thread_id]

    def climate(self, city: str | None, today: date) -> tuple[list[WeatherKind], str]:
        """The city's typical weather kinds, looked up once per city and day."""
        with self._climates_lock:
            if (city, today) not in self._climates:
                self._climates[(city, today)] = typical_kinds(city, today, self.context.fetch)
            return self._climates[(city, today)]

    def close(self) -> None:
        self.store.client.close()


def live_services(settings: Settings) -> Services:
    """Real dependencies: the local wardrobe, DeepSeek, CLIP on CPU and Open-Meteo."""
    store = WardrobeStore(QdrantClient(path=str(settings.qdrant_path)))
    store.ensure_collection()
    client = make_client(settings)

    @cache
    def pipeline() -> Pipeline:
        clip = get_clip(settings.clip_model, settings.clip_pretrained)
        return Pipeline(settings=settings, store=store, client=client, clip=clip)

    profile = ProfileStore(settings.profile_path)
    wear_log = WearLogStore(settings.wear_log_path)
    context = clip_context(store, settings, profile=profile.load, wear_log=wear_log.entries)
    return Services(settings, store, client, context, pipeline, profile=profile, wear_log=wear_log)


def get_services(request: Request) -> Services:
    """FastAPI dependency: the services the app was started with."""
    return request.app.state.services
