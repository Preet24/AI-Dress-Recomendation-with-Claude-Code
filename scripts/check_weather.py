"""Check the weather for an outing and what it means for clothes.

Asks where you are going, when you leave, how long you will be out, how much of
that time is outdoors, and how you will travel. Any answer can be given as an
option instead, e.g.:
    python scripts/check_weather.py
    python scripts/check_weather.py --city Pune --start 18:30 --hours-away 4 --hours-outdoors 1 \
        --transport two_wheeler
"""

import argparse
import logging

from pydantic import ValidationError

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

logger = logging.getLogger("check_weather")

TRANSPORT_HELP = {
    Transport.WALK: "walking",
    Transport.BIKE: "bicycle",
    Transport.TWO_WHEELER: "scooter / motorbike",
    Transport.PUBLIC_TRANSIT: "bus, metro or train",
    Transport.CAR: "car or cab",
}


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix} > ").strip() or default


def ask_hours(prompt: str, default: str) -> float:
    while True:
        try:
            return float(ask(prompt, default))
        except ValueError:
            logger.warning("  Please enter a number of hours, e.g. 2 or 0.5")


def ask_transport() -> Transport:
    choices = list(Transport)
    for n, transport in enumerate(choices, 1):
        logger.info("  %d. %s (%s)", n, transport, TRANSPORT_HELP[transport])
    while True:
        answer = ask("How will you travel? (number or name)", "1")
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in set(Transport):
            return Transport(answer)
        logger.warning("  Pick one of 1-%d", len(choices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city")
    parser.add_argument("--start", help='"now", "18:30", "tomorrow 9:00" or "2026-10-18 09:00"')
    parser.add_argument("--hours-away", type=float)
    parser.add_argument("--hours-outdoors", type=float)
    parser.add_argument("--transport", type=Transport, choices=list(Transport))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    try:
        place = geocode(args.city or ask("Where are you going? (city)"))
        now = local_now(place)
        logger.info("  %s (local time %s)", place.label, now.strftime("%a %H:%M"))
        start = parse_start(args.start or ask("When do you leave?", "now"), now)
        away = args.hours_away or ask_hours("How many hours will you be out in total?", "3")
        outdoors = (
            args.hours_outdoors
            if args.hours_outdoors is not None
            else ask_hours("Of that, how many hours outdoors?", str(min(away, 1.0)))
        )
        outing = Outing(
            start=start,
            hours_away=away,
            hours_outdoors=outdoors,
            transport=args.transport or ask_transport(),
        )
        summary = summarize(place, fetch_forecast(place, outing))
    except (WeatherError, ValidationError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(1) from None

    advice = advise(summary, outing)
    s = summary
    logger.info(
        "\n%s, %s-%s\n"
        "  Temperature   %.0f-%.0f°C (feels like %.0f-%.0f°C)\n"
        "  Humidity      %.0f%%\n"
        "  Rain          up to %.0f%% chance, %.1f mm total\n"
        "  UV index      up to %.0f\n"
        "  Wind          up to %.0f km/h, gusts %.0f km/h\n"
        "  Sky           %s",
        s.place,
        s.start.strftime("%a %d %b %H:%M"),
        s.end.strftime("%H:%M"),
        s.temp_min,
        s.temp_max,
        s.feels_like_min,
        s.feels_like_max,
        s.humidity_mean,
        s.rain_chance_max,
        s.precipitation_total,
        s.uv_index_max,
        s.wind_speed_max,
        s.wind_gusts_max,
        ", ".join(s.conditions),
    )
    outer = (
        f"yes, warmth {advice.outerwear_min_warmth}+" if advice.needs_outerwear else "not needed"
    )
    logger.info(
        "\nWhat to wear (%s, %.1f h outdoors)\n"
        "  Dress for     %.0f°C\n"
        "  Warmth        %d-%d (of 5)\n"
        "  Outerwear     %s\n"
        "  Flowy pieces  %s\n"
        "  Avoid fabric  %s\n"
        "  Prefer fabric %s\n"
        "  Light colors  %s",
        TRANSPORT_HELP[outing.transport],
        outing.hours_outdoors,
        advice.effective_temp,
        *advice.warmth,
        outer,
        "avoid" if advice.avoid_flowy else "fine",
        ", ".join(advice.avoid_fabrics) or "-",
        ", ".join(advice.prefer_fabrics) or "-",
        "yes" if advice.prefer_light_colors else "no preference",
    )
    for note in advice.notes:
        logger.info("  • %s", note)


if __name__ == "__main__":
    main()
