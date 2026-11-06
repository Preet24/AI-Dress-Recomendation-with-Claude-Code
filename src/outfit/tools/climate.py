"""Typical weather for a city: which of hot / mild / cold / rainy days it actually gets.

The shopping assistant plans for the user's home city over a whole year, not for one
forecast. Last year's daily weather comes from Open-Meteo's archive (free, no key, the
same `fetch` as the forecast); each day is put in one kind, and kinds with at least
`MIN_SHARE` of the days count. Each kind has a representative outing so the usual
`advise()` rules turn it into clothing advice.

The packing list uses the archive too (`archive_hours`): trip days past the forecast's
reach are planned on the same days last year, hour by hour.
"""

import logging
from collections import Counter
from datetime import date, datetime, timedelta
from enum import StrEnum

from outfit.tools.weather import (
    WET_HOUR_MM,
    FetchJson,
    HourlyWeather,
    Outing,
    Place,
    Transport,
    WeatherAdvice,
    WeatherError,
    WeatherSummary,
    advise,
    geocode,
    http_get_json,
)

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_LAG_DAYS = 7  # the archive trails today by a few days
DAYS = 365
RAINY_MM = 5.0  # a day with this much rain is a rainy day, whatever the temperature
HOT_FEELS_MAX = 32.0  # the warmest hour feels at least this hot
COLD_FEELS_MAX = 16.0  # even the warmest hour feels colder than this
MIN_SHARE = 0.08  # about a month of the year


class WeatherKind(StrEnum):
    HOT = "hot"
    MILD = "mild"
    COLD = "cold"
    RAINY = "rainy"


ALL_KINDS = list(WeatherKind)

# (feels-like min, max, rain chance %, rain mm, UV) of a typical day of each kind; UV
# decides "strong sun" (weather.UV_HIGH: sunglasses), so overcast kinds stay below it
TYPICAL_DAY = {
    WeatherKind.HOT: (32.0, 36.0, 10.0, 0.0, 9.0),
    WeatherKind.MILD: (21.0, 26.0, 10.0, 0.0, 6.0),
    WeatherKind.COLD: (8.0, 13.0, 10.0, 0.0, 3.0),
    WeatherKind.RAINY: (23.0, 26.0, 85.0, 8.0, 3.0),
}
# A daytime outing: 4 hours out, 1 outdoors, by public transport (no riding rules).
TYPICAL_OUTING = Outing(
    start=datetime(2026, 1, 1, 11, 0),
    hours_away=4,
    hours_outdoors=1,
    transport=Transport.PUBLIC_TRANSIT,
)


def day_kind(feels_max: float, rain_mm: float) -> WeatherKind:
    if rain_mm >= RAINY_MM:
        return WeatherKind.RAINY
    if feels_max >= HOT_FEELS_MAX:
        return WeatherKind.HOT
    if feels_max < COLD_FEELS_MAX:
        return WeatherKind.COLD
    return WeatherKind.MILD


def typical_kinds(
    city: str | None, today: date, fetch: FetchJson = http_get_json
) -> tuple[list[WeatherKind], str]:
    """The kinds of weather `city` gets over a year, and a note saying how that was decided.

    No city, or the archive failing, gives all four kinds."""
    if not city:
        return ALL_KINDS, "no home city in your profile, so every kind of weather"
    try:
        place = geocode(city, fetch)
        end = today - timedelta(days=ARCHIVE_LAG_DAYS)
        data = fetch(
            ARCHIVE_URL,
            {
                "latitude": place.latitude,
                "longitude": place.longitude,
                "timezone": place.timezone,
                "start_date": (end - timedelta(days=DAYS - 1)).isoformat(),
                "end_date": end.isoformat(),
                "daily": "apparent_temperature_max,precipitation_sum",
            },
        )
        daily = data["daily"]
        days = [
            day_kind(feels, rain or 0.0)
            for feels, rain in zip(
                daily["apparent_temperature_max"], daily["precipitation_sum"], strict=True
            )
            if feels is not None
        ]
    except (WeatherError, KeyError, TypeError, ValueError) as exc:
        logger.warning("No climate for %s: %s", city, exc)
        return ALL_KINDS, f"couldn't look up the weather in {city}, so every kind of weather"
    if not days:
        return ALL_KINDS, f"no weather history for {city}, so every kind of weather"
    counts = Counter(days)
    kinds = [k for k in ALL_KINDS if counts[k] / len(days) >= MIN_SHARE]
    shares = ", ".join(f"{k} {counts[k] / len(days):.0%}" for k in ALL_KINDS if counts[k])
    return kinds, f"{place.name} over the last year: {shares} of days"


ARCHIVE_HOURLY = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
)


def archive_hours(
    place: Place, first: date, last: date, fetch: FetchJson = http_get_json
) -> list[HourlyWeather]:
    """Every hour of the days `first`..`last` as they really were (the archive).

    The archive has no rain chance and no UV: an hour with rain gets a 100% chance, and
    every hour gets the UV of a typical day of its day's kind (hot and mild are sunny)."""
    data = fetch(
        ARCHIVE_URL,
        {
            "latitude": place.latitude,
            "longitude": place.longitude,
            "timezone": place.timezone,
            "start_date": first.isoformat(),
            "end_date": last.isoformat(),
            "hourly": ",".join(ARCHIVE_HOURLY),
        },
    )
    try:
        hourly = data["hourly"]
        rows = [
            dict(zip(("time", *ARCHIVE_HOURLY), values, strict=True))
            for values in zip(*(hourly[k] for k in ("time", *ARCHIVE_HOURLY)), strict=True)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError(f"The weather archive had no hourly data: {exc}") from exc
    rows = [r for r in rows if all(v is not None for v in r.values())]
    by_day: dict[str, list[dict]] = {}
    for row in rows:
        by_day.setdefault(row["time"][:10], []).append(row)
    uv = {
        day: TYPICAL_DAY[
            day_kind(
                max(r["apparent_temperature"] for r in day_rows),
                sum(r["precipitation"] for r in day_rows),
            )
        ][4]
        for day, day_rows in by_day.items()
    }
    hours = [
        HourlyWeather(
            time=datetime.fromisoformat(r["time"]),
            temperature=r["temperature_2m"],
            feels_like=r["apparent_temperature"],
            humidity=r["relative_humidity_2m"],
            rain_chance=100.0 if r["precipitation"] >= WET_HOUR_MM else 0.0,
            precipitation=r["precipitation"],
            uv_index=uv[r["time"][:10]],
            wind_speed=r["wind_speed_10m"],
            wind_gusts=r["wind_gusts_10m"],
            weather_code=r["weather_code"],
        )
        for r in rows
    ]
    if not hours:
        raise WeatherError("The weather archive has nothing for those days")
    return hours


def typical_advice(kind: WeatherKind) -> WeatherAdvice:
    """Clothing advice for a typical day of `kind`, from the same rules as a forecast."""
    feels_min, feels_max, rain_chance, rain_mm, uv = TYPICAL_DAY[kind]
    summary = WeatherSummary(
        place=f"a typical {kind} day",
        start=TYPICAL_OUTING.start,
        end=TYPICAL_OUTING.start + timedelta(hours=TYPICAL_OUTING.hours_away),
        temp_min=feels_min,
        temp_max=feels_max,
        feels_like_min=feels_min,
        feels_like_max=feels_max,
        humidity_mean=60.0,
        rain_chance_max=rain_chance,
        precipitation_total=rain_mm,
        uv_index_max=uv,
        wind_speed_max=12.0,
        wind_gusts_max=25.0,
        conditions=["rain"] if rain_mm else ["clear"],
    )
    return advise(summary, TYPICAL_OUTING)
