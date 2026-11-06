"""Fixed inputs for evals, so the agents are the only thing that varies between runs.

- Weather: a fake Open-Meteo per named fixture. Any city geocodes (to Pune's
  coordinates and time zone) and every hour asked for gets the same values.
- Wardrobe: a snapshot of the real store (items + CLIP vectors) in
  `evals/wardrobe.json`, loaded into an in-memory Qdrant. CLIP text search stays
  real, so "my blue jeans" resolves exactly as it does live.
- Clock: `eval_context(..., now=...)` fixes "now" for every node.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient

from outfit.agents.graph import clip_context
from outfit.agents.orchestrate import Context
from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.evals.scenarios import Weather
from outfit.schemas import ClothingItem
from outfit.tools.weather import GEOCODING_URL, FetchJson

WARDROBE_PATH = Path(__file__).resolve().parents[3] / "evals" / "wardrobe.json"

# Hourly values per fixture, in Open-Meteo's variable names. Chosen to sit clearly
# inside one advice band each (tools/weather.py), not on a threshold.
WEATHER: dict[str, dict[str, float]] = {
    "warm_dry": {  # feels 29 °C: warmth 1-2, no outerwear, no notes
        "temperature_2m": 30.0,
        "apparent_temperature": 29.0,
        "relative_humidity_2m": 50.0,
        "precipitation_probability": 5.0,
        "precipitation": 0.0,
        "uv_index": 3.0,
        "wind_speed_10m": 10.0,
        "wind_gusts_10m": 20.0,
        "weather_code": 1,
    },
    "rainy": {  # 80% rain, humid; riding: 24 - 5 = 19 °C and a rain layer
        "temperature_2m": 25.0,
        "apparent_temperature": 24.0,
        "relative_humidity_2m": 85.0,
        "precipitation_probability": 80.0,
        "precipitation": 1.5,
        "uv_index": 1.0,
        "wind_speed_10m": 15.0,
        "wind_gusts_10m": 30.0,
        "weather_code": 63,
    },
    "cold": {  # feels 8 °C: warmth 2-5 and outerwear 4+ when exposed
        "temperature_2m": 9.0,
        "apparent_temperature": 8.0,
        "relative_humidity_2m": 60.0,
        "precipitation_probability": 0.0,
        "precipitation": 0.0,
        "uv_index": 0.0,
        "wind_speed_10m": 8.0,
        "wind_gusts_10m": 15.0,
        "weather_code": 0,
    },
}
PLACE = {"country": "India", "latitude": 18.52, "longitude": 73.86, "timezone": "Asia/Kolkata"}


def weather_fetch(name: Weather) -> FetchJson:
    """A fake Open-Meteo that answers with fixture `name` for any place and hours."""
    values = WEATHER[name]

    def fetch(url: str, params: dict[str, Any]) -> dict[str, Any]:
        if url == GEOCODING_URL:
            return {"results": [{"name": params["name"].split(",")[0].title(), **PLACE}]}
        hour = datetime.fromisoformat(params["start_hour"])
        end = datetime.fromisoformat(params["end_hour"])
        times = []
        while hour <= end:
            times.append(f"{hour:%Y-%m-%dT%H:%M}")
            hour += timedelta(hours=1)
        return {"hourly": {"time": times, **{k: [v] * len(times) for k, v in values.items()}}}

    return fetch


# --- Wardrobe snapshot -------------------------------------------------------------


def snapshot(store: WardrobeStore) -> dict[str, Any]:
    """Every item with its image vector, as JSON data."""
    items = sorted(store.list_items(), key=lambda i: str(i.id))
    return {
        "items": [
            {"item": i.model_dump(mode="json"), "vector": store.vector(i.id).tolist()}
            for i in items
        ]
    }


def restore(data: dict[str, Any]) -> WardrobeStore:
    """An in-memory store holding a snapshot."""
    store = WardrobeStore(QdrantClient(":memory:"))
    store.ensure_collection()
    items = [ClothingItem.model_validate(entry["item"]) for entry in data["items"]]
    if items:
        vectors = np.asarray([entry["vector"] for entry in data["items"]], dtype=np.float32)
        store.upsert(items, vectors)
    return store


def load_snapshot(path: Path = WARDROBE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No wardrobe snapshot at {path}; run scripts/snapshot_wardrobe.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_wardrobe(path: Path = WARDROBE_PATH) -> WardrobeStore:
    return restore(load_snapshot(path))


def eval_context(
    store: WardrobeStore, settings: Settings, weather: Weather, now: datetime
) -> Context:
    """Real CLIP search over the snapshot, fixture weather, a fixed clock."""
    return clip_context(store, settings, fetch=weather_fetch(weather), now=now)
