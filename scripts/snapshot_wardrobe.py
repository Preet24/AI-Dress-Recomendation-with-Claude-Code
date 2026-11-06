"""Snapshot the wardrobe (items + CLIP vectors) to evals/wardrobe.json for the agent evals.

Run again after ingesting or fixing items, then re-check evals/scenarios.toml, which
is written against the snapshot's items:
    python scripts/snapshot_wardrobe.py
"""

import argparse
import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.evals.fixtures import WARDROBE_PATH, snapshot

logger = logging.getLogger("snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=WARDROBE_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = Settings()
    client = QdrantClient(path=str(settings.qdrant_path))
    try:
        data = snapshot(WardrobeStore(client))
    finally:
        client.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    for entry in data["items"]:
        item = entry["item"]
        logger.info("  %s: %s %s", item["category"], item["primary_color_name"], item["subtype"])
    logger.info("Wrote %d item(s) to %s", len(data["items"]), args.out)


if __name__ == "__main__":
    main()
