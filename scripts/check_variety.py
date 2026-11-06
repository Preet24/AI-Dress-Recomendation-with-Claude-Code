"""Check that multi-day packing plans don't repeat the same clothes or outfits.

Packs a trip of each length given on the wardrobe (rules only: no weather, no
DeepSeek; the profile's taste and the wear log as on the Packing page), then reports
per trip:
  - problem: the same top + bottom (or the same one-piece) on 2+ days
  - problem: a top, bottom or one-piece worn on more days than its re-wear limit
  - note:    the same top or one-piece two days running
Shoes, outerwear and accessories may repeat and are not checked. Exits 1 on a problem.

    python scripts/check_variety.py                       # 2, 3, 5, 7 and 14 days
    python scripts/check_variety.py --days 3 7 --evenings-in --occasion office
    python scripts/check_variety.py --snapshot evals/wardrobe.json --show
"""

import argparse
import logging
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from qdrant_client import QdrantClient

from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.evals.fixtures import load_wardrobe
from outfit.profile import ProfileStore
from outfit.schemas import ClothingItem, Occasion
from outfit.tools.feedback import feedback
from outfit.tools.outfits import OutfitRequest, label
from outfit.tools.packing import EVERYDAY, MAX_DAYS, Trip, Wears, pack, trip_slots
from outfit.tools.variety import check_variety
from outfit.wear_log import WearLogStore

logger = logging.getLogger("check_variety")
DEFAULT_DAYS = [2, 3, 5, 7, 14]


def load_items(snapshot: Path | None, settings: Settings) -> list[ClothingItem]:
    store = load_wardrobe(snapshot) if snapshot else None
    if store is None:
        store = WardrobeStore(QdrantClient(path=str(settings.qdrant_path)))
    try:
        return store.list_items()
    finally:
        store.client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, nargs="+", default=DEFAULT_DAYS,
                        choices=range(2, MAX_DAYS + 1), metavar="N",
                        help=f"trip lengths to check (default {DEFAULT_DAYS})")  # fmt: skip
    parser.add_argument("--occasion", type=Occasion, choices=list(Occasion),
                        default=EVERYDAY.occasion, help="the everyday occasion")  # fmt: skip
    parser.add_argument("--no-travel-days", dest="travel_days", action="store_false")
    parser.add_argument("--evenings-in", action="store_true")
    parser.add_argument("--tops", type=int, default=Wears().tops)
    parser.add_argument("--bottoms", type=int, default=Wears().bottoms)
    parser.add_argument("--one-pieces", type=int, default=Wears().one_pieces)
    parser.add_argument("--snapshot", type=Path, help="a wardrobe snapshot instead of the store")
    parser.add_argument("--no-taste", action="store_true",
                        help="ignore the profile and the wear log")  # fmt: skip
    parser.add_argument("--show", action="store_true", help="print every day's outfit too")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    settings = Settings()
    try:
        wears = Wears(tops=args.tops, bottoms=args.bottoms, one_pieces=args.one_pieces)
    except ValueError as exc:
        parser.error(str(exc))
    items = load_items(args.snapshot, settings)
    today = date.today()
    taste = OutfitRequest()
    if not args.no_taste:
        taste = OutfitRequest(**ProfileStore(settings.profile_path).load().taste_fields())
        history = feedback(WearLogStore(settings.wear_log_path).entries(), items, today)
        taste = replace(taste, **history.request_fields(recent=False))
    everyday = EVERYDAY.model_copy(update={"occasion": args.occasion})
    logger.info(
        "%d items; everyday %s; limits: top %d, bottom %d, one-piece %d days\n",
        len(items), args.occasion, wears.tops, wears.bottoms, wears.one_pieces,
    )  # fmt: skip

    problems = 0
    for days in sorted(set(args.days)):
        trip = Trip(first_day=today + timedelta(days=1), days=days, everyday=everyday,
                    travel_days=args.travel_days, evenings_in=args.evenings_in,
                    wears=wears)  # fmt: skip
        slots = trip_slots(trip)
        plan = pack(items, slots, [None] * len(slots), wears, taste)
        repeats = check_variety(plan, wears)
        bad = [r for r in repeats if r.problem]
        problems += len(bad)
        empty = sum(d.outfit is None for d in plan.days)
        status = "FAIL" if bad else "ok"
        logger.info(
            "%-4s %2d days: %d pieces packed, %d problem(s), %d note(s)%s",
            status, days, len(plan.pieces), len(bad), len(repeats) - len(bad),
            f", {empty} slot(s) with nothing to wear" if empty else "",
        )  # fmt: skip
        for repeat in repeats:
            logger.info("       %s %s", "x" if repeat.problem else "-", repeat)
        if args.show:
            for day in plan.days:
                pieces = ", ".join(label(p) for p in day.outfit.pieces) if day.outfit else "-"
                logger.info("         %-18s %s", day.slot.name, pieces)
    logger.info("\n%s", "No repeats beyond the limits." if not problems else
                f"{problems} problem(s) found.")  # fmt: skip
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
