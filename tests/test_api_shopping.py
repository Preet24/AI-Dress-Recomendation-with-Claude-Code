"""The shopping endpoints over fake services: describe without storing, report, add."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from outfit.api.app import create_app
from outfit.config import Settings
from outfit.profile import DressCode, Profile, SavedPlace
from outfit.schemas import Category, Occasion
from outfit.tools.weather import GEOCODING_URL, WeatherError
from tests.factories import make_item
from tests.fakes import FakeClient, fake_services, tags_reply
from tests.test_api_upload import jpeg_bytes
from tests.test_pipeline import WORN_MASKS, make_pipeline

SHIRT = make_item(
    category=Category.TOP,
    subtype="button-up shirt",
    primary_color_name="white",
    primary_color_lab=(95.0, 0.0, 0.0),
    formality=3,
    occasions=[Occasion.CASUAL, Occasion.OFFICE],
)


@pytest.fixture
def setup(tmp_path):
    """(TestClient, services, fake DeepSeek client, climate lookups) with one shirt owned."""
    client = FakeClient()
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    settings = Settings(
        _env_file=None,
        DEEPSEEK_API_KEY="test-key",
        raw_dir=tmp_path / "raw_wardrobe",
        clean_dir=tmp_path / "clean",
        candidates_dir=tmp_path / "candidates",
    )
    lookups = []

    def no_archive(url, params):
        lookups.append(url)
        raise WeatherError("no network in tests")

    services = fake_services([SHIRT], client=client, pipeline=lambda: pipeline, settings=settings)
    services.context = replace(services.context, fetch=no_archive)
    pipeline.store = services.store
    with TestClient(create_app(services)) as http:
        yield http, services, client, lookups


def check(http, mode="worn"):
    files = {"photo": ("shop.jpg", jpeg_bytes(), "image/jpeg")}
    return http.post("/shopping/check", files=files, data={"mode": mode})


def queue_worn_tags(client):
    client.replies += [
        tags_reply(),  # the fake tagger's white t-shirt
        tags_reply(
            category="bottom",
            subtype="chinos",
            primary_color_name="navy",
            length="full",
            formality=3,
            occasions=["casual", "office"],
        ),
    ]


def test_a_check_reports_each_garment_and_stores_nothing(setup, tmp_path):
    http, services, client, _ = setup
    queue_worn_tags(client)

    response = check(http)

    assert response.status_code == 200
    body = response.json()
    assert [r["candidate"]["subtype"] for r in body["reports"]] == ["t-shirt", "chinos"]
    assert services.store.count() == 1  # still only the shirt
    chinos = body["reports"][1]
    assert chinos["verdict"] in ("buy", "maybe")
    assert chinos["new_outfits"] >= 1
    assert chinos["goes_with"][0]["item"]["name"] == "white button-up shirt"
    assert body["climate"] == "no home city in your profile, so every kind of weather"
    assert body["checked"][0] == "casual on hot days" and len(body["checked"]) == 12
    image = http.get(chinos["candidate"]["image_url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert (tmp_path / "candidates" / "shop.jpg").is_file()
    assert not (tmp_path / "clean").exists()


def test_adding_a_candidate_stores_it_without_a_new_tagging_call(setup, tmp_path):
    http, services, client, _ = setup
    queue_worn_tags(client)
    chinos = check(http).json()["reports"][1]["candidate"]
    calls = len(client.requests)

    added = http.post(f"/shopping/candidates/{chinos['id']}/add")

    assert added.status_code == 201
    assert added.json()["image_url"] == f"/items/{chinos['id']}/image"
    assert services.store.count() == 2 and len(client.requests) == calls
    assert http.get(added.json()["image_url"]).status_code == 200
    assert http.get(chinos["image_url"]).status_code == 404  # no longer a candidate
    assert http.post(f"/shopping/candidates/{chinos['id']}/add").status_code == 404


def test_the_profile_decides_what_is_checked_and_the_climate_is_looked_up_once(setup):
    http, services, client, lookups = setup
    office = SavedPlace(name="Office", occasion=Occasion.OFFICE, dress_code=DressCode.BUSINESS)
    services.profile.save(Profile(home_city="Pune", places=[office], occasions=["casual"]))
    queue_worn_tags(client)
    queue_worn_tags(client)

    first = check(http).json()
    check(http)

    assert first["checked"][:2] == ["Office on hot days", "Office on mild days"]
    assert first["checked"][4] == "casual on hot days"
    assert first["climate"].startswith("couldn't look up the weather in Pune")
    assert lookups == [GEOCODING_URL]  # once for both checks (cached for the day)


def test_a_failed_check_keeps_no_photo(setup, tmp_path):
    http, _, client, _ = setup
    client.replies += [tags_reply(category="spaceship")] * 2  # invalid twice: TaggingError

    response = check(http)

    assert response.status_code == 502
    assert list((tmp_path / "candidates").glob("*.jpg")) == []


def test_a_non_photo_is_refused(setup):
    http, _, _, _ = setup
    files = {"photo": ("x.jpg", b"not a photo", "image/jpeg")}
    assert http.post("/shopping/check", files=files).status_code == 422


def test_unknown_candidates_are_404(setup):
    http, _, _, _ = setup
    unknown = "00000000-0000-0000-0000-000000000000"
    assert http.get(f"/shopping/candidates/{unknown}/image").status_code == 404
    assert http.post(f"/shopping/candidates/{unknown}/add").status_code == 404


# --- What should I buy? ----------------------------------------------------------------


def test_suggest_lists_what_is_missing_and_what_to_buy_without_deepseek(setup):
    http, services, client, _ = setup
    services.profile.save(Profile(gender="man", occasions=[Occasion.CASUAL]))

    response = http.post("/shopping/suggest", json={"occasions": []})

    assert response.status_code == 200 and client.requests == []
    body = response.json()
    assert body["checked"][0] == "casual on hot days"  # no home city: every weather
    assert body["no_outfit"][0].startswith("casual in any weather")  # a shirt, no bottom
    names = [s["name"] for s in body["suggestions"]]
    assert names and all("skirt" not in n and "dress" not in n for n in names)  # gender
    first = body["suggestions"][0]
    assert first["category"] in ("bottom", "ethnic_bottom") and first["unlocks"]
    assert all(p["item"]["image_url"].startswith("/items/") for p in first["goes_with"])


def test_suggest_for_chosen_occasions_uses_the_saved_places_for_them(setup):
    http, services, _, _ = setup
    office = SavedPlace(name="Office", occasion=Occasion.OFFICE, dress_code=DressCode.BUSINESS)
    gym = SavedPlace(name="Gym", occasion=Occasion.SPORTS)
    services.profile.save(Profile(places=[office, gym], occasions=[Occasion.PARTY]))

    body = http.post("/shopping/suggest", json={"occasions": ["office"]}).json()

    assert {c.split(" on ")[0] for c in body["checked"]} == {"Office"}
