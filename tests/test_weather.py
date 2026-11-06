import io
import json
import urllib.error
from datetime import datetime

import pytest
from pydantic import ValidationError

from outfit.tools.weather import (
    FORECAST_URL,
    GEOCODING_URL,
    HOURLY_VARIABLES,
    Outing,
    Place,
    Transport,
    WeatherError,
    WeatherSummary,
    advise,
    fetch_forecast,
    geocode,
    http_get_json,
    parse_hourly,
    parse_start,
    summarize,
    weather_for_outing,
)

PUNE = Place(
    name="Pune",
    country="India",
    admin1="Maharashtra",
    latitude=18.52,
    longitude=73.86,
    timezone="Asia/Kolkata",
)
NOW = datetime(2026, 10, 17, 12, 0)


def outing(**overrides) -> Outing:
    fields = {"start": NOW, "hours_away": 3, "hours_outdoors": 2, "transport": Transport.WALK}
    return Outing(**(fields | overrides))


def summary(**overrides) -> WeatherSummary:
    """Mild, dry, calm weather; override what a test is about."""
    fields = {
        "place": "Pune",
        "start": NOW,
        "end": NOW,
        "temp_min": 21,
        "temp_max": 24,
        "feels_like_min": 21,
        "feels_like_max": 24,
        "humidity_mean": 50,
        "rain_chance_max": 5,
        "precipitation_total": 0,
        "uv_index_max": 3,
        "wind_speed_max": 8,
        "wind_gusts_max": 15,
        "conditions": ["clear sky"],
    }
    return WeatherSummary(**(fields | overrides))


def hourly_payload(times: list[str], **columns) -> dict:
    base = {
        "temperature_2m": 25.0,
        "apparent_temperature": 26.0,
        "relative_humidity_2m": 60,
        "precipitation_probability": 10,
        "precipitation": 0.0,
        "uv_index": 4.0,
        "wind_speed_10m": 10.0,
        "wind_gusts_10m": 20.0,
        "weather_code": 1,
    }
    hourly = {"time": times}
    for key, value in base.items():
        hourly[key] = columns.get(key, [value] * len(times))
    return {"hourly": hourly}


class FakeFetch:
    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, params: dict) -> dict:
        self.calls.append((url, params))
        return self.responses.pop(0)


# --- HTTP -------------------------------------------------------------------


class FakeUrlopen:
    """Raises the queued errors, then returns the JSON body."""

    def __init__(self, *errors: Exception, body: dict | None = None) -> None:
        self.errors = list(errors)
        self.body = body or {}
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.urls.append(url)
        if self.errors:
            raise self.errors.pop(0)
        return io.BytesIO(json.dumps(self.body).encode())


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, None)


def test_http_get_json_retries_server_errors():
    urlopen = FakeUrlopen(http_error(503), body={"ok": True})

    data = http_get_json("https://x/api", {"a": 1}, urlopen=urlopen, sleep=lambda s: None)

    assert data == {"ok": True}
    assert urlopen.urls == ["https://x/api?a=1"] * 2


def test_http_get_json_does_not_retry_client_errors():
    urlopen = FakeUrlopen(http_error(400))

    with pytest.raises(WeatherError, match="400"):
        http_get_json("https://x", {}, urlopen=urlopen, sleep=lambda s: None)
    assert len(urlopen.urls) == 1


def test_http_get_json_gives_up_after_retries():
    urlopen = FakeUrlopen(*[urllib.error.URLError("offline")] * 3)

    with pytest.raises(WeatherError, match="Could not reach"):
        http_get_json("https://x", {}, retries=2, urlopen=urlopen, sleep=lambda s: None)
    assert len(urlopen.urls) == 3


# --- Fetching ---------------------------------------------------------------


def test_geocode_returns_best_match():
    fetch = FakeFetch({"results": [PUNE.model_dump() | {"id": 1, "population": 3e6}]})

    place = geocode(" Pune ", fetch)

    assert place == PUNE and place.label == "Pune, Maharashtra, India"
    assert fetch.calls[0] == (GEOCODING_URL, {"name": "Pune", "count": 1, "language": "en"})


def test_geocode_unknown_place_raises():
    with pytest.raises(WeatherError, match="Zzz"):
        geocode("Zzz", FakeFetch({"generationtime_ms": 0.1}))  # no "results" key


def test_forecast_requests_the_outing_hours_in_local_time():
    fetch = FakeFetch(hourly_payload(["2026-10-17T18:00"]))

    fetch_forecast(PUNE, outing(start=datetime(2026, 10, 17, 18, 40), hours_away=2.5), fetch, NOW)

    url, params = fetch.calls[0]
    assert url == FORECAST_URL
    assert params["timezone"] == "Asia/Kolkata"
    assert (params["start_hour"], params["end_hour"]) == ("2026-10-17T18:00", "2026-10-17T20:00")
    assert params["hourly"].split(",") == list(HOURLY_VARIABLES)


def test_forecast_beyond_16_days_raises():
    with pytest.raises(WeatherError, match="16 days"):
        fetch_forecast(PUNE, outing(start=datetime(2026, 11, 5, 9, 0)), FakeFetch(), NOW)


def test_parse_hourly_skips_hours_with_gaps():
    data = hourly_payload(["2026-10-17T12:00", "2026-10-17T13:00"], uv_index=[None, 4.0])

    [hour] = parse_hourly(data)

    assert hour.time == datetime(2026, 10, 17, 13, 0)


def test_parse_hourly_reads_a_missing_rain_chance_off_the_rainfall():
    data = hourly_payload(
        ["2026-10-17T12:00", "2026-10-17T13:00", "2026-10-17T14:00"],
        precipitation_probability=[None, None, 40],
        precipitation=[0.0, 0.4, 0.0],
    )

    assert [h.rain_chance for h in parse_hourly(data)] == [0.0, 100.0, 40]


def test_parse_hourly_without_data_raises():
    with pytest.raises(WeatherError):
        parse_hourly({"hourly": {"time": []}})


def test_summarize_takes_extremes_over_the_window():
    hours = parse_hourly(
        hourly_payload(
            ["2026-10-17T12:00", "2026-10-17T13:00"],
            temperature_2m=[24.0, 28.0],
            apparent_temperature=[25.0, 31.0],
            relative_humidity_2m=[60, 80],
            precipitation_probability=[20, 70],
            precipitation=[0.0, 1.2],
            uv_index=[7.0, 5.0],
            wind_gusts_10m=[20.0, 45.0],
            weather_code=[1, 61],
        )
    )

    s = summarize(PUNE, hours)

    assert (s.temp_min, s.temp_max, s.feels_like_max) == (24, 28, 31)
    assert (s.humidity_mean, s.rain_chance_max, s.precipitation_total) == (70, 70, 1.2)
    assert (s.uv_index_max, s.wind_gusts_max) == (7, 45)
    assert s.conditions == ["mainly clear", "light rain"]
    assert s.end == datetime(2026, 10, 17, 14, 0)


def test_weather_for_outing_end_to_end():
    fetch = FakeFetch({"results": [PUNE.model_dump()]}, hourly_payload(["2026-10-17T12:00"]))

    place, s, advice = weather_for_outing("Pune", outing(start=datetime.now()), fetch)

    assert place == PUNE and s.temp_max == 25
    assert advice.warmth == (1, 2)  # feels like 26°C


# --- Inputs -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("now", NOW),
        ("", NOW),
        ("18:30", datetime(2026, 10, 17, 18, 30)),
        ("9:00", datetime(2026, 10, 18, 9, 0)),  # already past today -> tomorrow
        ("tomorrow 7:15", datetime(2026, 10, 18, 7, 15)),
        ("2026-10-20 09:00", datetime(2026, 10, 20, 9, 0)),
    ],
)
def test_parse_start(text, expected):
    assert parse_start(text, NOW) == expected


def test_parse_start_rejects_nonsense():
    with pytest.raises(ValueError, match="Could not read"):
        parse_start("after lunch", NOW)


def test_outing_cannot_be_outdoors_longer_than_away():
    with pytest.raises(ValidationError, match="hours_outdoors"):
        outing(hours_away=1, hours_outdoors=2)


# --- Rules ------------------------------------------------------------------


def test_mild_dry_calm_weather_needs_nothing_special():
    advice = advise(summary(), outing())

    assert advice.warmth == (1, 3)
    assert not advice.needs_outerwear and not advice.avoid_flowy
    assert advice.avoid_fabrics == [] and advice.notes == []
    assert advice.item_filter().warmth == (1, 3) and advice.item_filter().flowy is None


@pytest.mark.parametrize(
    "feels_like, warmth, outer",
    [(30, (1, 2), None), (20, (1, 3), None), (12, (2, 4), 3), (3, (2, 5), 4), (-5, (3, 5), 5)],
)
def test_warmth_bands_follow_feels_like_temperature(feels_like, warmth, outer):
    advice = advise(summary(feels_like_min=feels_like), outing())

    assert advice.warmth == warmth
    assert advice.outerwear_min_warmth == outer
    assert advice.needs_outerwear is (outer is not None)


def test_car_with_little_time_outdoors_makes_outerwear_optional():
    advice = advise(summary(feels_like_min=8), outing(transport=Transport.CAR, hours_outdoors=0.2))

    assert not advice.needs_outerwear
    assert any("light layer" in n for n in advice.notes)


def test_riding_adds_wind_chill_and_forbids_flowy():
    advice = advise(summary(feels_like_min=19), outing(transport=Transport.TWO_WHEELER))

    assert advice.effective_temp == 14
    assert advice.warmth == (2, 4) and advice.needs_outerwear
    assert advice.avoid_flowy and advice.item_filter().flowy is False


def test_no_wind_chill_when_hot():
    advice = advise(summary(feels_like_min=30), outing(transport=Transport.BIKE))

    assert advice.effective_temp == 30


def test_gusts_rule_out_flowy_when_outdoors_but_not_in_a_car():
    windy = summary(wind_gusts_max=50)

    assert advise(windy, outing()).avoid_flowy
    assert not advise(windy, outing(transport=Transport.CAR, hours_outdoors=0)).avoid_flowy


def test_rain_outdoors_avoids_sensitive_fabrics():
    advice = advise(summary(rain_chance_max=80), outing())

    assert {"suede", "satin", "silk"} <= set(advice.avoid_fabrics)
    assert any("umbrella" in n for n in advice.notes)


def test_rain_when_riding_requires_a_rain_layer_even_if_warm():
    advice = advise(
        summary(precipitation_total=3, feels_like_min=27), outing(transport=Transport.TWO_WHEELER)
    )

    assert advice.needs_outerwear and advice.outerwear_min_warmth == 1
    assert any("rain jacket" in n for n in advice.notes)


def test_rain_by_car_is_only_a_note():
    advice = advise(
        summary(rain_chance_max=80), outing(transport=Transport.CAR, hours_outdoors=0.1)
    )

    assert advice.avoid_fabrics == []
    assert any("umbrella" in n for n in advice.notes)


def test_hot_and_humid_prefers_breathable_fabrics():
    advice = advise(
        summary(temp_max=32, feels_like_min=30, humidity_mean=80), outing(hours_outdoors=2)
    )

    assert advice.prefer_fabrics == ["cotton", "linen"]
    assert "synthetic" in advice.avoid_fabrics
    assert advice.prefer_light_colors


def test_high_uv_matters_only_with_real_time_outdoors():
    sunny = summary(uv_index_max=9)

    long_out = advise(sunny, outing(hours_outdoors=2))
    short_out = advise(sunny, outing(hours_outdoors=0.5))

    assert long_out.prefer_light_colors and any("UV very high" in n for n in long_out.notes)
    assert not short_out.prefer_light_colors
    assert long_out.sunny and not short_out.sunny  # sunglasses / hat (tools/accessories)


def test_advice_says_when_riding():
    assert advise(summary(), outing(transport=Transport.TWO_WHEELER)).riding
    assert not advise(summary(), outing(transport=Transport.WALK)).riding
