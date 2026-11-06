"""Pack for a trip: check the place, dates and weather, then, once you confirm, pack.

Local rules on the stored tags plus Open-Meteo (no CLIP, no DeepSeek), e.g.:
    python scripts/pack_trip.py --city Manali --first-day 2026-11-06 --days 4 \
        --event 2,party,20:00,4,0.5,car --event 4,party,20:00,4,0.5,car
    python scripts/pack_trip.py --city Manali --first-day 2026-11-06 --days 4 --place 2
    python scripts/pack_trip.py ... --tops 3 --bottoms 4 --yes   # own limits, no question
It prints the check first and packs only after "y" (or with --yes); --place N picks
the Nth match when the city name could be several places.
"""

import argparse
import logging
from dataclasses import replace
from datetime import date, datetime

from qdrant_client import QdrantClient

from outfit.cli import trip_event
from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.profile import ProfileStore
from outfit.schemas import Occasion
from outfit.tools.outfits import OutfitRequest, label
from outfit.tools.packing import (
    EVERYDAY,
    MAX_DAYS,
    Activity,
    PlaceNotConfirmed,
    Trip,
    TripPreview,
    Wears,
    plan_trip,
    preview_trip,
    weather_line,
)
from outfit.tools.weather import Transport

logger = logging.getLogger("pack_trip")


def show_preview(preview: TripPreview) -> None:
    logger.info("Where: %s", preview.place.label if preview.place else "(weather not checked)")
    if len(preview.matches) > 1:
        for n, place in enumerate(preview.matches, 1):
            logger.info("  %d. %s", n, place.label)
    logger.info("When:  %s", preview.dates)
    for row in preview.rows:
        outing = row.slot.outing
        when = f"{outing.start:%a %d %b %H:%M}"
        if row.slot.indoors:
            logger.info("  %-18s %s, at the stay", row.slot.name, when)
            continue
        logger.info(
            "  %-18s %s, %g h away (%g h outdoors, %s)",
            row.slot.name,
            when,
            outing.hours_away,
            outing.hours_outdoors,
            outing.transport,
        )
        logger.info("  %18s %s [%s]", "", weather_line(row.weather), row.source)
    wears = preview.wears
    logger.info(
        "Re-wear: a top up to %d days, a bottom %d, a one-piece %d",
        wears.tops,
        wears.bottoms,
        wears.one_pieces,
    )
    for assumption in preview.assumptions:
        logger.info("Note: %s", assumption)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", help="town or city (default: the profile's home city)")
    parser.add_argument("--first-day", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--days", type=int, required=True, choices=range(1, MAX_DAYS + 1))
    parser.add_argument("--place", type=int, default=1, metavar="N", help="pick the Nth match")
    parser.add_argument("--yes", action="store_true", help="pack without asking")
    plan = parser.add_argument_group("each day (default: casual, 10:00, 8 h, 3 h out, transit)")
    plan.add_argument(
        "--occasion", type=Occasion, choices=list(Occasion), default=EVERYDAY.occasion
    )
    plan.add_argument("--start", type=lambda t: datetime.strptime(t, "%H:%M").time(),
                      default=EVERYDAY.start, metavar="HH:MM")  # fmt: skip
    plan.add_argument("--hours-away", type=float, default=EVERYDAY.hours_away)
    plan.add_argument("--hours-outdoors", type=float, default=EVERYDAY.hours_outdoors)
    plan.add_argument("--transport", type=Transport, choices=list(Transport),
                      default=EVERYDAY.transport)  # fmt: skip
    plan.add_argument("--no-travel-days", dest="travel_days", action="store_false",
                      help="don't plan the first and last day as travel")  # fmt: skip
    plan.add_argument("--evenings-in", action="store_true",
                      help="something to lounge and sleep in every night")  # fmt: skip
    plan.add_argument("--event", type=trip_event, action="append", default=[],
                      metavar="DAY,OCCASION,HH:MM,HOURS,OUTDOORS,TRANSPORT")  # fmt: skip
    limits = parser.add_argument_group("re-wear limits, in days")
    limits.add_argument("--tops", type=int, default=Wears().tops)
    limits.add_argument("--bottoms", type=int, default=Wears().bottoms)
    limits.add_argument("--one-pieces", type=int, default=Wears().one_pieces,
                        help="0: don't pack any")  # fmt: skip
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    settings = Settings()
    profile = ProfileStore(settings.profile_path).load()
    try:
        trip = Trip(
            city=args.city or profile.home_city,
            first_day=args.first_day,
            days=args.days,
            everyday=Activity(
                occasion=args.occasion,
                start=args.start,
                hours_away=args.hours_away,
                hours_outdoors=args.hours_outdoors,
                transport=args.transport,
            ),
            events=args.event,
            travel_days=args.travel_days,
            evenings_in=args.evenings_in,
            wears=Wears(tops=args.tops, bottoms=args.bottoms, one_pieces=args.one_pieces),
        )
    except ValueError as exc:
        parser.error(str(exc))
    today = date.today()
    preview = preview_trip(trip, today)
    if preview.matches:
        if not 1 <= args.place <= len(preview.matches):
            parser.error(f"--place must be 1-{len(preview.matches)}")
        trip = trip.model_copy(update={"place": preview.matches[args.place - 1]})
        if args.place != 1:  # the chosen place's weather, still listing every match
            preview = replace(preview_trip(trip, today), matches=preview.matches)
    show_preview(preview)
    if not args.yes and input("\nPack for this? [y/N] ").strip().lower() not in {"y", "yes"}:
        logger.info("Not packed. Change the trip, or --place N for another match.")
        return

    store = WardrobeStore(QdrantClient(path=str(settings.qdrant_path)))
    try:
        result = plan_trip(
            store.list_items(), trip, today, taste=OutfitRequest(**profile.taste_fields())
        )
    except PlaceNotConfirmed as exc:
        parser.error(str(exc))
    logger.info("\nPack (%d pieces):", len(result.pieces))
    for packed in result.pieces:
        days = ", ".join(map(str, packed.days))
        logger.info("  - %-35s [%s] days %s", label(packed.item), packed.item.category, days)
    for note in result.notes:
        logger.info("  + %s", note)
    for day in result.days:
        logger.info("\n%s  %s", day.slot.name, f"{day.slot.outing.start:%a %d %b %H:%M}")
        logger.info("   %s", weather_line(day.weather, day.slot.indoors))
        if day.outfit is None:
            logger.info("   nothing fits (see the gaps)")
            continue
        logger.info("   %s", ", ".join(label(p) for p in day.outfit.pieces))
        for note in day.notes:
            logger.info("   ! %s", note)
    for gap in result.gaps:
        logger.info("\nGap: %s", gap)
    for assumption in result.assumptions:
        logger.info("Note: %s", assumption)


if __name__ == "__main__":
    main()
