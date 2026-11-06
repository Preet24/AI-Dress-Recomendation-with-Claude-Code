"""Small argument parsers shared by the scripts in scripts/."""

import argparse

from pydantic import ValidationError

from outfit.tools.packing import Event


def scale_range(text: str) -> tuple[int, int]:
    """'3' -> (3, 3); '2-4' -> (2, 4); both ends within 1..5."""
    low, _, high = text.partition("-")
    try:
        bounds = (int(low), int(high or low))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"expected N or N-M, got {text!r}") from err
    if not 1 <= bounds[0] <= bounds[1] <= 5:
        raise argparse.ArgumentTypeError(f"expected a range within 1-5, got {text!r}")
    return bounds


def trip_event(text: str) -> Event:
    """'2,party,20:00,4,0.5,car' -> day 2: a party at 20:00, 4 h away, 0.5 h outdoors, by car."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"expected DAY,OCCASION,HH:MM,HOURS,OUTDOORS,TRANSPORT, got {text!r}"
        )
    day, occasion, start, away, outdoors, transport = parts
    try:
        return Event(
            day=day,
            occasion=occasion,
            start=start,
            hours_away=away,
            hours_outdoors=outdoors,
            transport=transport,
        )
    except ValidationError as err:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors())
        raise argparse.ArgumentTypeError(f"{text!r}: {problems}") from err
