"""The packing page headless, against the real API over fakes: check, confirm, pack."""

from dataclasses import replace
from datetime import date, time

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from outfit.api.app import create_app
from outfit.ui.client import OutfitAPI, PackingPlan, Trip, TripEvent, TripPreview
from tests.fakes import fake_services
from tests.test_api_packing import NOW
from tests.test_packing import GENOA, GOA, PlacesFetch, shoes, top, trousers

FIRST = date(2026, 10, 30)


def packing_script():
    import streamlit as st

    from outfit.ui import packing_page

    packing_page.render(st.session_state.api)


@pytest.fixture
def setup():
    """(AppTest, services, weather fetch, api) with a shirt, trousers and shoes owned."""
    fetch = PlacesFetch(GENOA, GOA)
    services = fake_services([top(), trousers(), shoes()], fetch=fetch)
    services.context = replace(services.context, now=NOW)
    with TestClient(create_app(services)) as http:
        api = OutfitAPI(http=http)
        at = AppTest.from_function(packing_script, default_timeout=30)
        at.session_state["api"] = api
        yield at.run(), services, fetch, api


def submit(at, form_id):
    return next(b for b in at.button if b.proto.form_id == form_id).click().run()


def check_trip(at, city="Goa", days=2):
    at.text_input(key="trip-city").set_value(city)
    at.date_input(key="trip-first").set_value(FIRST)
    at.number_input(key="trip-days").set_value(days)
    return submit(at, "trip")


def test_the_client_mirrors_the_packing_responses(setup):
    _, _, _, api = setup
    trip = Trip(city="Goa", first_day=FIRST, days=2,
                events=[TripEvent(day=2, occasion="party", start=time(20), hours_away=4,
                                  hours_outdoors=0.5, transport="car")])  # fmt: skip

    preview = api.preview_trip(trip)
    plan = api.pack_trip(trip.model_copy(update={"place": preview.matches[1]}))

    assert isinstance(preview, TripPreview) and isinstance(plan, PackingPlan)
    assert [m.label for m in preview.matches] == ["Genoa, Liguria, Italy", "Panjim, Goa, India"]
    assert preview.rows[-1].slot.name == "day 2 party"
    assert plan.place == "Panjim, Goa, India" and plan.days[-1].outfit.pieces


def test_nothing_is_checked_or_packed_until_asked(setup):
    at, services, fetch, _ = setup
    assert not at.exception
    assert at.title[0].value == "Packing"
    assert not at.table and not [b for b in at.button if b.key == "pack"]
    assert fetch.calls == []


def test_check_trip_shows_the_places_dates_and_weather_before_packing(setup):
    at, services, _, _ = setup

    at = check_trip(at)

    assert not at.exception
    assert at.radio(key="packing-place").options == ["Genoa, Liguria, Italy", "Panjim, Goa, India"]
    shown = " ".join(m.value for m in at.markdown)
    assert "**Where:** Genoa, Liguria, Italy" in shown
    assert "Fri 30 Oct - Sat 31 Oct 2026, 2 days" in shown
    table = at.table[0].value
    assert list(table["Occasion"]) == ["travel", "travel"]
    assert all(w.startswith("feels 22 °C") for w in table["Weather"])
    assert at.button(key="pack")  # waiting for the user
    assert "packing_plan" not in at.session_state
    assert services.client.requests == []


def test_picking_another_place_rechecks_its_weather(setup):
    at, _, fetch, _ = setup
    at = check_trip(at)
    fetch.calls.clear()

    at = at.radio(key="packing-place").set_value("Panjim, Goa, India").run()

    assert not at.exception
    assert "**Where:** Panjim, Goa, India" in " ".join(m.value for m in at.markdown)
    assert at.radio(key="packing-place").options == ["Genoa, Liguria, Italy", "Panjim, Goa, India"]
    assert [url for url, _ in fetch.calls] == ["https://api.open-meteo.com/v1/forecast"]


def test_pack_for_this_packs_for_the_place_shown(setup):
    at, services, _, _ = setup
    at = check_trip(at)
    at = at.radio(key="packing-place").set_value("Panjim, Goa, India").run()

    at = at.button(key="pack").click().run()

    assert not at.exception
    assert at.subheader[-1].value == "Pack 3 pieces for Panjim, Goa, India"
    captions = [c.value for c in at.caption]
    assert "days 1, 2" in captions
    assert any(m.value == "- plus socks and underwear for 2 days" for m in at.markdown)
    assert sum("**Day " in m.value for m in at.markdown) == 2  # a card per day
    assert services.client.requests == []
    # "I wore this" on each day, with the day's occasion filled in
    worn_forms = {b.proto.form_id for b in at.button if b.proto.form_id.startswith("worn-")}
    assert len(worn_forms) == 2
    occasions = [s.value for s in at.selectbox if s.key.startswith("worn-occasion")]
    assert occasions == ["travel", "travel"]  # 2 days: arrive and leave


def test_events_are_added_and_a_change_throws_the_check_away(setup):
    at, _, _, _ = setup
    at = check_trip(at)
    assert at.table

    at.number_input(key="event-day").set_value(2)
    at = submit(at, "event-0")

    assert not at.exception
    assert any("day 2: party at 20:00" in m.value for m in at.markdown)
    assert not at.table  # checked again only on request
    at = check_trip(at)
    assert list(at.table[0].value["Occasion"]) == ["travel", "travel", "party"]


def test_evenings_in_and_no_dresses_show_in_the_check_and_the_plan(setup):
    at, _, _, _ = setup
    at.checkbox(key="trip-evenings").check()
    at.number_input(key="wears-one").set_value(0)

    at = check_trip(at)

    assert not at.exception
    table = at.table[0].value
    assert list(table["Occasion"]) == ["travel", "Evening in (lounge / sleep)", "travel"]
    assert table["Weather"].iloc[1] == "indoors, no weather needed"
    assert (table["Out"].iloc[1], table["By"].iloc[1]) == ("in", "-")
    assert any("a bottom 3, no dresses." in c.value for c in at.caption)

    at = at.button(key="pack").click().run()

    assert not at.exception
    cards = [m.value for m in at.markdown if m.value.startswith("**Day ")]
    assert cards[1].startswith("**Day 1 · Evening in (lounge / sleep)**")
    # Nothing in this wardrobe is for lounging: the evening says so
    assert any("Nothing in your wardrobe fits" in w.value for w in at.warning)


def test_an_unknown_town_cannot_be_packed(setup):
    at, _, fetch, _ = setup
    fetch.places = ()

    at = check_trip(at, city="Zzz")

    assert any("No place found" in w.value for w in at.warning)
    assert not [b for b in at.button if b.key == "pack"]
