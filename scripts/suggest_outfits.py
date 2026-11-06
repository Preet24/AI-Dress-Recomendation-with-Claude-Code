"""Suggest outfits from the wardrobe for an occasion, optionally for the weather.

Runs locally on the stored tags (no CLIP, no DeepSeek); --city adds an Open-Meteo
forecast for the outing, e.g.:
    python scripts/suggest_outfits.py --occasion casual
    python scripts/suggest_outfits.py --occasion party --city Pune --start 19:00 \
        --hours-away 4 --hours-outdoors 0.5 --transport car
    python scripts/suggest_outfits.py --with 0fdff64b          # outfits with this item
    python scripts/suggest_outfits.py --occasion casual --avoid black --like navy
"""

import argparse
import logging

from qdrant_client import QdrantClient

from outfit.cli import scale_range
from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.schemas import ClothingItem, Fit, Occasion
from outfit.tools.accessories import finish_plan
from outfit.tools.outfits import OutfitRequest, label, suggest_outfits
from outfit.tools.weather import (
    Outing,
    Transport,
    WeatherError,
    advise,
    fetch_forecast,
    geocode,
    local_now,
    parse_start,
    summarize,
)

logger = logging.getLogger("suggest_outfits")


def resolve(prefixes: list[str] | None, items: list[ClothingItem], parser) -> frozenset:
    ids = set()
    for prefix in prefixes or []:
        matches = [i for i in items if str(i.id).startswith(prefix)]
        if len(matches) != 1:
            parser.error(f"id {prefix!r} matches {len(matches)} items; give more of the id")
        ids.add(matches[0].id)
    return frozenset(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--occasion", type=Occasion, choices=list(Occasion))
    parser.add_argument("--formality", type=scale_range, metavar="N[-M]")
    parser.add_argument("--with", dest="include", action="append", metavar="ID")
    parser.add_argument("--without", dest="exclude", action="append", metavar="ID")
    parser.add_argument("--limit", type=int, default=3)
    taste = parser.add_argument_group("taste (optional, like the profile)")
    taste.add_argument("--like", action="append", default=[], metavar="COLOR")
    taste.add_argument("--avoid", action="append", default=[], metavar="COLOR")
    taste.add_argument("--fit-tops", type=Fit, choices=list(Fit))
    taste.add_argument("--fit-bottoms", type=Fit, choices=list(Fit))
    weather = parser.add_argument_group("weather (optional)")
    weather.add_argument("--city")
    weather.add_argument("--start", default="now")
    weather.add_argument("--hours-away", type=float, default=3)
    weather.add_argument("--hours-outdoors", type=float, default=1)
    weather.add_argument("--transport", type=Transport, choices=list(Transport), default="walk")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    store = WardrobeStore(QdrantClient(path=str(Settings().qdrant_path)))
    items = store.list_items()

    advice = None
    if args.city:
        try:
            place = geocode(args.city)
            outing = Outing(
                start=parse_start(args.start, local_now(place)),
                hours_away=args.hours_away,
                hours_outdoors=args.hours_outdoors,
                transport=args.transport,
            )
            summary = summarize(place, fetch_forecast(place, outing))
        except (WeatherError, ValueError) as exc:
            parser.error(str(exc))
        advice = advise(summary, outing)
        logger.info(
            "%s: %.0f-%.0f°C (feels %.0f), rain %.0f%%, gusts %.0f km/h -> warmth %d-%d%s",
            summary.place,
            summary.temp_min,
            summary.temp_max,
            summary.feels_like_min,
            summary.rain_chance_max,
            summary.wind_gusts_max,
            *advice.warmth,
            ", no flowy" if advice.avoid_flowy else "",
        )
        for note in advice.notes:
            logger.info("  • %s", note)

    request = OutfitRequest(
        occasion=args.occasion,
        formality=args.formality,
        advice=advice,
        must_include=resolve(args.include, items, parser),
        exclude=resolve(args.exclude, items, parser),
        limit=args.limit,
        fit_tops=args.fit_tops,
        fit_bottoms=args.fit_bottoms,
        colors_liked=frozenset(c.strip().lower() for c in args.like),
        colors_avoided=frozenset(c.strip().lower() for c in args.avoid),
    )
    plan = finish_plan(suggest_outfits(items, request), items, request)

    low, high = request.formality_range
    logger.info(
        "\n%d outfit(s) for %s (formality %d-%d) from %d items",
        len(plan.outfits),
        request.occasion or "any occasion",
        low,
        high,
        len(items),
    )
    for n, outfit in enumerate(plan.outfits, 1):
        logger.info(
            "\n%d. %.2f  (colors %.2f, fit %.2f)", n, outfit.score, outfit.harmony, outfit.fit
        )
        for piece in outfit.pieces:
            logger.info("   - %s [%s]  (%s)", label(piece), piece.category, str(piece.id)[:8])
        for reason in outfit.reasons:
            logger.info("   · %s", reason)
        for warning in outfit.warnings:
            logger.info("   ! %s", warning)
    for gap in plan.gaps:
        logger.info("\nGap: %s", gap)


if __name__ == "__main__":
    main()
