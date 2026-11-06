"""Search the wardrobe by description or by similarity, with optional hard filters.

Runs fully locally (CLIP on CPU + local Qdrant), e.g.:
    python scripts/search_wardrobe.py "black dress"
    python scripts/search_wardrobe.py "something light for summer" --warmth 1-2 --category top
    python scripts/search_wardrobe.py --like 0fdff64b --category bottom   # bottoms like an item
"""

import argparse
import logging

from qdrant_client import QdrantClient

from outfit.cli import scale_range
from outfit.config import Settings
from outfit.db.qdrant_store import ItemFilter, WardrobeStore
from outfit.ingestion.embedder import get_clip
from outfit.schemas import Category, Occasion
from outfit.tools.search import WardrobeSearch

logger = logging.getLogger("search_wardrobe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", help="what you are looking for")
    parser.add_argument("--like", metavar="ID", help="find items similar to this item (id prefix)")
    parser.add_argument("--category", type=Category, choices=list(Category), action="append")
    parser.add_argument("--occasion", type=Occasion, choices=list(Occasion), action="append")
    parser.add_argument("--formality", type=scale_range, metavar="N[-M]")
    parser.add_argument("--warmth", type=scale_range, metavar="N[-M]")
    parser.add_argument("--flowy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if bool(args.query) == bool(args.like):
        parser.error("give either a query or --like ID")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    settings = Settings()
    store = WardrobeStore(QdrantClient(path=str(settings.qdrant_path)))
    search = WardrobeSearch(store, get_clip(settings.clip_model, settings.clip_pretrained))
    filters = ItemFilter(
        categories=args.category or (),
        occasions=args.occasion or (),
        formality=args.formality,
        warmth=args.warmth,
        flowy=args.flowy,
    )

    if args.like:
        matches = [i for i in store.list_items() if str(i.id).startswith(args.like)]
        if len(matches) != 1:
            parser.error(f"--like {args.like!r} matches {len(matches)} items; give more of the id")
        logger.info("Items like: %s %s", matches[0].primary_color_name, matches[0].subtype)
        hits = search.similar_to(matches[0].id, filters, args.limit)
    else:
        hits = search.by_text(args.query, filters, args.limit)

    if not hits:
        logger.info("Nothing matches those filters.")
    for hit in hits:
        item = hit.item
        logger.info(
            "%.3f  %s %s [%s] F%d W%d  %s  (%s)",
            hit.score,
            item.primary_color_name,
            item.subtype,
            item.category,
            item.formality,
            item.warmth,
            item.image_path.name,
            str(item.id)[:8],
        )


if __name__ == "__main__":
    main()
