"""The packing endpoints over fake services: check the trip, confirm the place, pack."""

from dataclasses import replace
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from outfit.api.app import create_app
from outfit.profile import Profile
from outfit.tools.weather import FORECAST_URL, GEOCODING_URL
from tests.fakes import fake_services
from tests.test_packing import GENOA, GOA, PlacesFetch, shoes, top, trousers

NOW = datetime(2026, 10, 25, 12, 0)
PARTY = {"day": 2, "occasion": "party", "start": "20:00", "hours_away": 4,
         "hours_outdoors": 0.5, "transport": "car"}  # fmt: skip


def trip(**kw) -> dict:
    return {"city": "Goa", "first_day": "2026-10-30", "days": 2, "travel_days": False,
            "events": [PARTY]} | kw  # fmt: skip


@pytest.fixture
def setup():
    """(TestClient, services, weather fetch) with a shirt, trousers and shoes owned."""
    fetch = PlacesFetch(GENOA, GOA)
    services = fake_services([top(), trousers(), shoes()], fetch=fetch)
    services.context = replace(services.context, now=NOW)
    with TestClient(create_app(services)) as http:
        yield http, services, fetch


def test_preview_shows_the_places_dates_and_each_rows_weather(setup):
    http, services, fetch = setup

    response = http.post("/packing/preview", json=trip())

    assert response.status_code == 200
    body = response.json()
    assert [m["label"] for m in body["matches"]] == ["Genoa, Liguria, Italy", "Panjim, Goa, India"]
    assert body["place"]["label"] == "Genoa, Liguria, Italy"
    assert body["dates"] == "Fri 30 Oct - Sat 31 Oct 2026, 2 days"
    rows = body["rows"]
    assert [r["slot"]["name"] for r in rows] == ["day 1 casual", "day 2 casual", "day 2 party"]
    party = rows[2]["slot"]
    assert party["start"] == "2026-10-31T20:00:00" and party["transport"] == "car"
    assert rows[0]["weather"].startswith("feels 22 °C") and rows[0]["source"] == "forecast"
    assert body["wears"] == {"tops": 2, "bottoms": 3, "one_pieces": 1}
    assert services.client.requests == []  # no DeepSeek


def test_preview_for_the_picked_place_and_home_city_default(setup):
    http, services, fetch = setup
    services.profile.save(Profile(home_city="Goa"))
    picked = http.post("/packing/preview", json=trip(city=None)).json()["matches"][1]

    body = http.post("/packing/preview", json=trip(city=None, place=picked)).json()

    assert body["place"]["label"] == "Panjim, Goa, India"
    assert body["assumptions"] == []  # a picked place is not the home city default
    first = http.post("/packing/preview", json=trip(city=None)).json()
    assert first["city"] == "Goa"
    assert first["assumptions"] == ["no city given: your home city, Goa"]


def test_packing_without_a_confirmed_place_is_refused(setup):
    http, _, _ = setup

    response = http.post("/packing", json=trip())

    assert response.status_code == 422
    assert response.json()["detail"] == "confirm which Goa first"


def test_packing_for_the_confirmed_place(setup):
    http, services, fetch = setup
    picked = http.post("/packing/preview", json=trip()).json()["matches"][1]
    fetch.calls.clear()

    response = http.post("/packing", json=trip(place=picked))

    assert response.status_code == 200
    body = response.json()
    assert body["place"] == "Panjim, Goa, India"
    assert [url for url, _ in fetch.calls] == [FORECAST_URL]  # no new place search
    assert {p["item"]["name"] for p in body["pieces"]} == {
        "white oxford shirt",
        "navy trousers",
        "black loafers",
    }
    assert body["pieces"][0]["item"]["image_url"].startswith("/items/")
    assert all(d["outfit"] for d in body["days"])
    assert body["days"][2]["slot"]["name"] == "day 2 party"
    assert body["days"][0]["weather"].startswith("feels 22 °C")
    assert body["notes"] == ["plus socks and underwear for 2 days"]
    assert services.client.requests == []


def test_an_event_after_the_trip_is_a_422(setup):
    http, _, fetch = setup

    response = http.post("/packing/preview", json=trip(events=[PARTY | {"day": 5}]))

    assert response.status_code == 422
    assert GEOCODING_URL not in [url for url, _ in fetch.calls]
