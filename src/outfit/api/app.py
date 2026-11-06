"""The FastAPI app. Run it with `python scripts/serve.py`; interactive docs at /docs.

Endpoints are plain `def` functions: the store, CLIP and the agent graph are blocking
calls, so FastAPI runs them in its thread pool.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from outfit.api import chat, items, packing, profile, shopping, worn
from outfit.api.errors import add_error_handlers
from outfit.api.services import Services, get_services, live_services
from outfit.config import Settings


class Health(BaseModel):
    status: str
    items: int  # garments in the wardrobe
    model: str  # DeepSeek model the agents use


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    """The app; tests pass their own `services`, otherwise the live ones are built at
    startup from `settings` (default: `.env` and `OUTFIT_*` variables)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or live_services(settings or Settings())
        try:
            yield
        finally:
            if services is None:  # close only what the app opened itself
                app.state.services.close()

    app = FastAPI(
        title="Outfit recommender",
        description="Your wardrobe and the outfit-recommending agents, over HTTP.",
        version="0.1.0",
        lifespan=lifespan,
    )
    add_error_handlers(app)
    app.include_router(items.router)
    app.include_router(chat.router)
    app.include_router(profile.router)
    app.include_router(shopping.router)
    app.include_router(packing.router)
    app.include_router(worn.router)

    @app.get("/health")
    def health(s: Annotated[Services, Depends(get_services)]) -> Health:
        return Health(status="ok", items=s.store.count(), model=s.settings.agent_model)

    return app
