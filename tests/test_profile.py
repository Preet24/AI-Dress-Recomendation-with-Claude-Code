"""The profile schema, its JSON file and the /profile endpoints."""

from datetime import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from outfit.api.app import create_app
from outfit.profile import DressCode, Profile, ProfileStore, SavedPlace
from outfit.schemas import Occasion
from outfit.tools.weather import Transport
from tests.fakes import fake_services

OFFICE = SavedPlace(
    name="Office",
    occasion=Occasion.OFFICE,
    dress_code=DressCode.BUSINESS_CASUAL,
    start=time(9, 0),
    hours_away=9,
    hours_outdoors=0.25,
    transport=Transport.CAR,
)
GYM = SavedPlace(name="Gym", occasion=Occasion.SPORTS, transport=Transport.TWO_WHEELER)


def test_an_empty_profile_is_valid():
    profile = Profile()
    assert profile.home_city is None
    assert profile.places == []


def test_dress_codes_set_the_formality_and_none_keeps_the_occasions_default():
    assert OFFICE.formality() == (3, 4)
    assert GYM.formality() is None
    assert SavedPlace(name="Bank", occasion="office", dress_code="business").formality() == (4, 4)


def test_saved_places_are_found_by_name_in_any_case():
    profile = Profile(places=[OFFICE, GYM])
    assert profile.place(" office ") is OFFICE
    assert profile.place("college") is None


def test_colors_are_trimmed_lower_cased_and_not_repeated():
    profile = Profile(colors_liked=[" Navy", "olive", "navy", ""], colors_avoided=["Yellow"])
    assert profile.colors_liked == ["navy", "olive"]
    assert profile.colors_avoided == ["yellow"]


@pytest.mark.parametrize(
    "data",
    [
        {"places": [OFFICE.model_dump(), {**OFFICE.model_dump(), "name": "OFFICE"}]},
        {"places": [{"name": "  ", "occasion": "casual"}]},
        {"places": [{"name": "Gym", "occasion": "sports", "hours_away": 0}]},
        {"places": [{"name": "Gym", "occasion": "gym"}]},  # not an occasion
        {"height_cm": 20},
        {"age": 30},  # dropped on purpose
        {"home_city": ""},
    ],
)
def test_invalid_profiles_are_refused(data):
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


def test_the_profile_round_trips_through_its_file(tmp_path):
    path = tmp_path / "data" / "profile.json"
    store = ProfileStore(path)
    assert store.load() == Profile()  # no file yet

    profile = Profile(home_city="Bangalore", colors_liked=["navy"], places=[OFFICE, GYM])
    store.save(profile)

    assert ProfileStore(path).load() == profile
    assert not path.with_suffix(".tmp").exists()


def test_the_in_memory_store_hands_out_copies():
    store = ProfileStore()
    store.save(Profile(colors_liked=["navy"]))
    store.load().colors_liked.append("red")
    assert store.load().colors_liked == ["navy"]


def test_get_and_put_profile():
    with TestClient(create_app(fake_services())) as client:
        assert client.get("/profile").json()["places"] == []

        body = Profile(home_city="Pune", places=[OFFICE]).model_dump(mode="json")
        saved = client.put("/profile", json=body)
        assert saved.status_code == 200
        assert saved.json() == body

        again = client.get("/profile").json()
        assert again["places"][0]["start"] == "09:00:00"
        assert again["places"][0]["dress_code"] == "business_casual"


def test_put_refuses_an_invalid_profile_and_keeps_the_old_one():
    with TestClient(create_app(fake_services())) as client:
        client.put("/profile", json={"home_city": "Pune"})
        response = client.put("/profile", json={"home_city": "Pune", "shoe": "9"})
        assert response.status_code == 422
        assert client.get("/profile").json()["home_city"] == "Pune"
