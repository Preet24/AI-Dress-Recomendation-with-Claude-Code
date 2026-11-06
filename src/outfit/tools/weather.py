"""Weather for an outing, turned into clothing constraints.

Open-Meteo (no API key) gives the hourly forecast for the hours the user is out:
temperature, feels-like, humidity, chance of rain, rainfall, UV index, wind and
gusts. `advise()` then turns that forecast, plus how long the user is outdoors and
how they travel, into constraints the outfit step can apply: a warmth range,
outerwear, flowy or not, fabrics to avoid or prefer, light colors, and notes.

Every rule is a plain threshold below, so the advice is predictable and testable.
Uses only the standard library for HTTP (two GET requests, no new dependency).
"""

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta
from enum import StrEnum
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from outfit.db.qdrant_store import ItemFilter

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS_MAX = 16
WET_HOUR_MM = 0.1  # an hour this wet counts as "rain chance 100%" when no chance is given
HOURLY_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "uv_index",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
)

# WMO weather codes used by Open-Meteo, grouped into plain words.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}

FetchJson = Callable[[str, dict[str, Any]], dict[str, Any]]


class WeatherError(RuntimeError):
    """The weather could not be fetched (unknown place, service down, out of range)."""


# --- HTTP -------------------------------------------------------------------


def http_get_json(
    url: str,
    params: dict[str, Any],
    *,
    retries: int = 2,
    timeout: float = 20,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """GET a JSON API; retries server errors and network failures with backoff."""
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries + 1):
        try:
            with urlopen(full_url, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            if err.code < 500 or attempt == retries:
                raise WeatherError(f"Weather service error {err.code} for {url}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == retries:
                raise WeatherError(f"Could not reach the weather service: {err}") from err
        logger.warning("Weather request failed (attempt %d), retrying", attempt + 1)
        sleep(2**attempt)
    raise AssertionError("unreachable")


# --- Data -------------------------------------------------------------------


class Transport(StrEnum):
    WALK = "walk"
    BIKE = "bike"  # bicycle
    TWO_WHEELER = "two_wheeler"  # scooter / motorbike
    PUBLIC_TRANSIT = "public_transit"  # bus, metro, train (includes waiting outside)
    CAR = "car"  # own car or cab, door to door


class Place(BaseModel):
    name: str
    country: str = ""
    admin1: str = ""  # state / province
    latitude: float
    longitude: float
    timezone: str

    @property
    def label(self) -> str:
        return ", ".join(part for part in (self.name, self.admin1, self.country) if part)


class Outing(BaseModel):
    """What the user tells us about their plans."""

    start: datetime  # local time at the place, naive
    hours_away: float = Field(gt=0, le=24)  # from leaving until back indoors for good
    hours_outdoors: float = Field(ge=0, le=24)  # of that, time actually outside
    transport: Transport

    @model_validator(mode="after")
    def _outdoors_within_away(self) -> "Outing":
        if self.hours_outdoors > self.hours_away:
            raise ValueError("hours_outdoors cannot exceed hours_away")
        return self


class HourlyWeather(BaseModel):
    time: datetime
    temperature: float  # °C
    feels_like: float  # °C, apparent temperature (wind, humidity, sun)
    humidity: float  # %
    rain_chance: float  # %
    precipitation: float  # mm in that hour
    uv_index: float
    wind_speed: float  # km/h
    wind_gusts: float  # km/h
    weather_code: int


class WeatherSummary(BaseModel):
    """The forecast over the outing's hours."""

    place: str
    start: datetime
    end: datetime
    temp_min: float
    temp_max: float
    feels_like_min: float
    feels_like_max: float
    humidity_mean: float
    rain_chance_max: float
    precipitation_total: float
    uv_index_max: float
    wind_speed_max: float
    wind_gusts_max: float
    conditions: list[str]  # distinct sky conditions, in order of appearance


class WeatherAdvice(BaseModel):
    """Constraints for the outfit step; `warmth` and `avoid_flowy` are hard filters."""

    effective_temp: float  # °C the user will feel, after travel adjustments
    warmth: tuple[int, int]  # allowed warmth for base garments (tops, bottoms, one-pieces)
    needs_outerwear: bool
    outerwear_min_warmth: int | None
    avoid_flowy: bool
    avoid_fabrics: list[str]
    prefer_fabrics: list[str]
    prefer_light_colors: bool
    notes: list[str]
    sunny: bool = False  # strong UV with an hour or more outside (sunglasses, a hat)
    riding: bool = False  # bicycle or two-wheeler

    def item_filter(self) -> ItemFilter:
        """Hard constraints for base garments as a store filter."""
        return ItemFilter(warmth=self.warmth, flowy=False if self.avoid_flowy else None)


# --- Fetching ---------------------------------------------------------------


def local_now(place: Place) -> datetime:
    """Current wall-clock time at the place (naive, like the forecast's times)."""
    return datetime.now(ZoneInfo(place.timezone)).replace(tzinfo=None, second=0, microsecond=0)


def parse_start(text: str, now: datetime) -> datetime:
    """When the user leaves: "now", "18:30", "tomorrow 9:00" or "2026-10-18 09:00".

    A bare time that has already passed today means tomorrow.
    """
    text = text.strip().lower()
    if text in ("", "now"):
        return now
    day = now.date()
    if text.startswith("tomorrow"):
        day += timedelta(days=1)
        text = text.removeprefix("tomorrow").strip() or "09:00"
    try:
        return datetime.fromisoformat(text) if "-" in text else _at(day, text, now)
    except ValueError as err:
        raise ValueError(
            f"Could not read the time {text!r}; try 18:30 or 2026-10-18 09:00"
        ) from err


def _at(day: date, clock: str, now: datetime) -> datetime:
    start = datetime.combine(day, datetime.strptime(clock, "%H:%M").time())
    return start + timedelta(days=1) if start < now - timedelta(minutes=30) else start


def geocode(name: str, fetch: FetchJson = http_get_json) -> Place:
    """Best match for a place name ("Pune", "Paris, France")."""
    return find_places(name, fetch, count=1)[0]


def find_places(name: str, fetch: FetchJson = http_get_json, count: int = 5) -> list[Place]:
    """Up to `count` matches for a place name, best first, one per name + state + country
    (the search often returns several villages of one name in one state)."""
    data = fetch(GEOCODING_URL, {"name": name.strip(), "count": count, "language": "en"})
    results = data.get("results") or []  # no "results" key at all when nothing matches
    places: list[Place] = []
    for place in map(Place.model_validate, results):
        if place.label not in {p.label for p in places}:
            places.append(place)
    if not places:
        raise WeatherError(f"Could not find a place called {name!r}")
    return places


def fetch_forecast(
    place: Place, outing: Outing, fetch: FetchJson = http_get_json, now: datetime | None = None
) -> list[HourlyWeather]:
    """Hourly forecast covering the outing, in the place's local time."""
    start, end = outing_hours(outing)
    today = (now or local_now(place)).date()
    if end.date() > today + timedelta(days=FORECAST_DAYS_MAX - 1):
        raise WeatherError(f"The forecast only reaches {FORECAST_DAYS_MAX} days ahead")
    return fetch_hours(place, start, end, fetch)


def outing_hours(outing: Outing) -> tuple[datetime, datetime]:
    """The first and last forecast hour the outing covers."""
    start = outing.start.replace(minute=0, second=0, microsecond=0)
    return start, start + timedelta(hours=max(math.ceil(outing.hours_away) - 1, 0))


def fetch_hours(
    place: Place, start: datetime, end: datetime, fetch: FetchJson = http_get_json
) -> list[HourlyWeather]:
    """Hourly forecast from `start` to `end` (whole hours, local time, both included)."""
    data = fetch(
        FORECAST_URL,
        {
            "latitude": place.latitude,
            "longitude": place.longitude,
            "timezone": place.timezone,
            "hourly": ",".join(HOURLY_VARIABLES),
            "start_hour": start.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": end.strftime("%Y-%m-%dT%H:%M"),
        },
    )
    return parse_hourly(data)


def parse_hourly(data: dict[str, Any]) -> list[HourlyWeather]:
    hourly = data.get("hourly")
    if not hourly or not hourly.get("time"):
        raise WeatherError("The forecast had no hourly data for that time")
    columns = {
        "temperature": "temperature_2m",
        "feels_like": "apparent_temperature",
        "humidity": "relative_humidity_2m",
        "rain_chance": "precipitation_probability",
        "precipitation": "precipitation",
        "uv_index": "uv_index",
        "wind_speed": "wind_speed_10m",
        "wind_gusts": "wind_gusts_10m",
        "weather_code": "weather_code",
    }
    hours = []
    for i, stamp in enumerate(hourly["time"]):
        values = {field: hourly[key][i] for field, key in columns.items()}
        if values["rain_chance"] is None and values["precipitation"] is not None:
            # far ahead some models give rainfall but no chance: read it off the rainfall
            values["rain_chance"] = 100.0 if values["precipitation"] >= WET_HOUR_MM else 0.0
        if any(v is None for v in values.values()):
            continue  # Open-Meteo leaves other gaps as null
        hours.append(HourlyWeather(time=datetime.fromisoformat(stamp), **values))
    if not hours:
        raise WeatherError("The forecast for that time is incomplete")
    return hours


def summarize(place: Place, hours: list[HourlyWeather]) -> WeatherSummary:
    conditions = [WEATHER_CODES.get(h.weather_code, f"code {h.weather_code}") for h in hours]
    return WeatherSummary(
        place=place.label,
        start=hours[0].time,
        end=hours[-1].time + timedelta(hours=1),
        temp_min=min(h.temperature for h in hours),
        temp_max=max(h.temperature for h in hours),
        feels_like_min=min(h.feels_like for h in hours),
        feels_like_max=max(h.feels_like for h in hours),
        humidity_mean=round(mean(h.humidity for h in hours), 1),
        rain_chance_max=max(h.rain_chance for h in hours),
        precipitation_total=round(sum(h.precipitation for h in hours), 1),
        uv_index_max=max(h.uv_index for h in hours),
        wind_speed_max=max(h.wind_speed for h in hours),
        wind_gusts_max=max(h.wind_gusts for h in hours),
        conditions=list(dict.fromkeys(conditions)),
    )


# --- Rules ------------------------------------------------------------------

OPEN_TRANSPORT = {Transport.WALK, Transport.BIKE, Transport.TWO_WHEELER, Transport.PUBLIC_TRANSIT}
RIDING = {Transport.BIKE, Transport.TWO_WHEELER}
# Riding makes its own headwind; only matters when it is not hot.
RIDE_CHILL_C = {Transport.BIKE: 2.0, Transport.TWO_WHEELER: 5.0}
RIDE_CHILL_BELOW_C = 25.0
EXPOSED_HOURS = 0.5  # outdoors at least this long counts as exposed, whatever the transport

# (feels like at least °C, base warmth range, outerwear min warmth or None)
WARMTH_BANDS: list[tuple[float, tuple[int, int], int | None]] = [
    (26, (1, 2), None),
    (18, (1, 3), None),
    (10, (2, 4), 3),
    (0, (2, 5), 4),
    (-math.inf, (3, 5), 5),
]
RAIN_LIKELY_PCT = 50
RAIN_POSSIBLE_PCT = 30
RAIN_TOTAL_MM = 1.0
RAIN_SENSITIVE_FABRICS = ["suede", "silk", "satin", "velvet", "leather"]
WIND_FLOWY_KMH = 25  # sustained wind
GUST_FLOWY_KMH = 40
HUMID_PCT = 70
HUMID_HOT_C = 25
BREATHABLE_FABRICS = ["cotton", "linen"]
HEAVY_IN_HUMIDITY = ["synthetic", "wool", "leather", "velvet"]
UV_HIGH = 6
UV_VERY_HIGH = 8
HOT_C = 30


def advise(summary: WeatherSummary, outing: Outing) -> WeatherAdvice:
    """Clothing constraints for this weather, outdoor time and way of travel."""
    exposed = outing.hours_outdoors >= EXPOSED_HOURS or outing.transport in OPEN_TRANSPORT
    riding = outing.transport in RIDING
    notes: list[str] = []
    avoid_fabrics: list[str] = []
    prefer_fabrics: list[str] = []

    # Temperature: the coldest the user will feel, plus the headwind when riding.
    effective = summary.feels_like_min
    if riding and effective < RIDE_CHILL_BELOW_C:
        effective -= RIDE_CHILL_C[outing.transport]
        notes.append(f"Riding adds wind chill: dress for about {effective:.0f}°C.")
    warmth, outerwear_min = next(
        (band, outer) for low, band, outer in WARMTH_BANDS if effective >= low
    )
    needs_outerwear = outerwear_min is not None and exposed
    if outerwear_min is not None and not exposed:
        notes.append("Mostly indoors or in a car: a light layer is enough to carry.")
    if summary.temp_max - summary.temp_min >= 8:
        notes.append(
            f"Temperature swings from {summary.temp_min:.0f} to {summary.temp_max:.0f}°C: "
            "layers you can take off help."
        )

    # Rain.
    rain_likely = (
        summary.rain_chance_max >= RAIN_LIKELY_PCT or summary.precipitation_total >= RAIN_TOTAL_MM
    )
    if rain_likely and exposed:
        avoid_fabrics += RAIN_SENSITIVE_FABRICS
        if riding:
            notes.append("Rain likely while riding: wear a rain jacket; avoid long hems.")
            needs_outerwear = True
            outerwear_min = outerwear_min or 1
        else:
            notes.append("Rain likely: take an umbrella or a water-resistant layer.")
        notes.append("Choose closed shoes that can get wet; skip suede and satin.")
    elif rain_likely or summary.rain_chance_max >= RAIN_POSSIBLE_PCT:
        notes.append(f"Chance of rain up to {summary.rain_chance_max:.0f}%: an umbrella is wise.")

    # Wind: flowy pieces flap in gusts and can catch in wheels and chains when riding.
    windy = summary.wind_speed_max >= WIND_FLOWY_KMH or summary.wind_gusts_max >= GUST_FLOWY_KMH
    avoid_flowy = riding or (windy and exposed)
    if riding:
        notes.append("Riding: avoid flowy or very long pieces that can catch in the wheels.")
    elif avoid_flowy:
        notes.append(f"Windy (gusts up to {summary.wind_gusts_max:.0f} km/h): skip flowy pieces.")

    # Humidity and heat.
    if summary.temp_max >= HUMID_HOT_C and summary.humidity_mean >= HUMID_PCT:
        prefer_fabrics += BREATHABLE_FABRICS
        avoid_fabrics += HEAVY_IN_HUMIDITY
        notes.append(f"Humid ({summary.humidity_mean:.0f}%): breathable cotton or linen.")
    prefer_light = summary.temp_max >= HOT_C and exposed

    # Sun.
    sunny = summary.uv_index_max >= UV_HIGH and outing.hours_outdoors >= 1
    if sunny:
        prefer_light = True
        strength = "very high" if summary.uv_index_max >= UV_VERY_HIGH else "high"
        notes.append(
            f"UV {strength} ({summary.uv_index_max:.0f}): sunglasses, sunscreen, and sleeves "
            "or a hat if you are out long."
        )

    return WeatherAdvice(
        effective_temp=round(effective, 1),
        warmth=warmth,
        needs_outerwear=needs_outerwear,
        outerwear_min_warmth=outerwear_min if needs_outerwear else None,
        avoid_flowy=avoid_flowy,
        avoid_fabrics=list(dict.fromkeys(avoid_fabrics)),
        prefer_fabrics=prefer_fabrics,
        prefer_light_colors=prefer_light,
        notes=notes,
        sunny=sunny,
        riding=riding,
    )


def weather_for_outing(
    place_name: str, outing: Outing, fetch: FetchJson = http_get_json
) -> tuple[Place, WeatherSummary, WeatherAdvice]:
    """The whole tool: place name + outing -> where, what the weather is, what to wear."""
    place = geocode(place_name, fetch)
    summary = summarize(place, fetch_forecast(place, outing, fetch))
    return place, summary, advise(summary, outing)
