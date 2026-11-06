"""The profile page, run headless with AppTest against the real API over fakes."""

from datetime import time
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from outfit import profile as server
from outfit.api.app import create_app
from outfit.schemas import Occasion
from outfit.tools.weather import Transport
from outfit.ui import client
from outfit.ui.client import OutfitAPI, Profile, SavedPlace
from outfit.ui.profile_page import DRESS_CODES, place_summary
from tests.fakes import fake_services

OFFICE = server.SavedPlace(
    name="Office",
    occasion=Occasion.OFFICE,
    dress_code=server.DressCode.BUSINESS_CASUAL,
    start=time(9, 0),
    hours_away=9,
    hours_outdoors=0.25,
    transport=Transport.CAR,
)


def profile_script():
    import streamlit as st

    from outfit.ui import profile_page

    profile_page.render(st.session_state.api)


@pytest.fixture
def setup():
    """(AppTest after the first run, services) with a saved home city and office."""
    services = fake_services()
    services.profile.save(server.Profile(home_city="Pune", places=[OFFICE]))
    with TestClient(create_app(services)) as http:
        at = AppTest.from_function(profile_script, default_timeout=30)
        at.session_state["api"] = OutfitAPI(http=http)
        yield at.run(), services


def submit(at: AppTest, form: str) -> AppTest:
    [button] = [b for b in at.button if b.proto.form_id == form]
    return button.click().run()


def test_the_client_enums_mirror_the_servers():
    pairs = [
        (client.Gender, server.Gender),
        (client.Build, server.Build),
        (client.DressCode, server.DressCode),
        (client.Transport, Transport),
    ]
    for mirror, enum in pairs:
        assert list(get_args(mirror)) == [e.value for e in enum]
    assert list(DRESS_CODES) == [d.value for d in server.DressCode]
    for mirror, model in [
        (client.Profile, server.Profile),
        (client.SavedPlace, server.SavedPlace),
        (client.Colouring, server.Colouring),
    ]:
        assert list(mirror.model_fields) == list(model.model_fields)


def test_the_client_parses_and_sends_the_profile():
    services = fake_services()
    with TestClient(create_app(services)) as http:
        api = OutfitAPI(http=http)
        assert api.profile() == Profile()
        saved = api.save_profile(
            Profile(colors_liked=["Navy"], places=[SavedPlace(**dict(OFFICE))])
        )
    assert saved.colors_liked == ["navy"]  # cleaned by the server
    assert services.profile.load().places == [OFFICE]


def test_place_summary_lists_what_is_set():
    assert place_summary(SavedPlace(**dict(OFFICE))) == (
        "Office · office · business casual (3-4) · 09:00, 9 h, 0.25 h outside, car"
    )
    assert place_summary(SavedPlace(name="Gym", occasion="sports")) == "Gym · sports"


def test_the_page_shows_the_saved_profile(setup):
    at, _ = setup
    assert not at.exception
    assert at.title[0].value == "Your profile"
    assert at.text_input(key="home_city").value == "Pune"
    assert at.selectbox(key="gender").value is None
    assert at.expander[0].label.startswith("Office · office")


def test_saving_about_you_keeps_the_places(setup):
    at, services = setup
    at.text_input(key="home_city").set_value("Bangalore")
    at.selectbox(key="gender").set_value("woman")
    at.number_input(key="height_cm").set_value(165)
    at.text_input(key="colors_avoided").set_value("Yellow, mustard")
    at = submit(at, "about")

    saved = services.profile.load()
    assert saved.home_city == "Bangalore"
    assert saved.gender == server.Gender.WOMAN
    assert saved.height_cm == 165
    assert saved.colors_avoided == ["yellow", "mustard"]
    assert saved.places == [OFFICE]
    assert at.success[0].value == "Saved your details"


def test_adding_editing_and_removing_a_place(setup):
    at, services = setup
    at.text_input(key="new-place-0-name").set_value("Gym")
    at.selectbox(key="new-place-0-occasion").set_value(Occasion.SPORTS)
    at.selectbox(key="new-place-0-transport").set_value("two_wheeler")
    at = submit(at, "new-place-0")
    assert [p.name for p in services.profile.load().places] == ["Office", "Gym"]
    assert at.success[0].value == "Added Gym"
    assert at.text_input(key="new-place-1-name").value == ""  # a fresh form

    at.selectbox(key="place-office-dress").set_value("formal")
    at = submit(at, "place-office")
    assert services.profile.load().place("office").dress_code == server.DressCode.FORMAL

    at = at.button(key="place-gym-remove").click().run()
    assert [p.name for p in services.profile.load().places] == ["Office"]
    assert at.success[0].value == "Removed Gym"


def test_the_apis_refusal_is_shown_and_nothing_changes(setup):
    at, services = setup
    at.text_input(key="new-place-0-name").set_value("office")  # same name, other case
    at = submit(at, "new-place-0")
    assert "unique" in at.error[0].value
    assert services.profile.load().places == [OFFICE]


def test_a_place_needs_a_name(setup):
    at, services = setup
    at = submit(at, "new-place-0")
    assert at.error[0].value == "Give the place a name"
    assert services.profile.load().places == [OFFICE]
