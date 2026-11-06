"""The wear-log endpoints over fake services."""

import json
from dataclasses import replace
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from outfit.api.app import create_app
from outfit.ingestion.background import SegmentationError
from outfit.schemas import Category
from tests.factories import make_item
from tests.fakes import FakeClient, fake_services, reply
from tests.test_look import LOOK
from tests.test_packing import shoes, top, trousers
from tests.test_profile_photo import opened, photo_with_exif

NOW = datetime(2026, 10, 25, 12, 0)
SHIRT, PANTS, SHOES = top(), trousers(), shoes()
CHINOS = make_item(category=Category.BOTTOM, subtype="chinos", primary_color_name="grey")


@pytest.fixture
def setup():
    services = fake_services([SHIRT, PANTS, SHOES])
    services.context = replace(services.context, now=NOW)
    with TestClient(create_app(services)) as http:
        yield http, services


def log(http, **kw):
    body = {"pieces": [str(SHIRT.id), str(PANTS.id)], "rating": "loved"} | kw
    return http.post("/worn", json=body)


def test_logging_an_outfit_keeps_it_with_its_pieces(setup):
    http, services = setup

    response = log(http, issues=["too_warm"], note="great at dinner", occasion="party",
                   felt=[24, 27], source="chat")  # fmt: skip

    assert response.status_code == 201
    body = response.json()
    assert body["worn_on"] == "2026-10-25"  # today, by the app's clock
    assert [p["name"] for p in body["pieces"]] == ["white oxford shirt", "navy trousers"]
    assert body["pieces"][0]["image_url"] == f"/items/{SHIRT.id}/image"
    assert (body["rating"], body["issues"], body["felt"]) == ("loved", ["too_warm"], [24, 27])
    assert body["removed"] == 0
    assert len(services.wear_log.entries()) == 1
    assert services.client.requests == []  # free


def test_the_log_lists_newest_first(setup):
    http, _ = setup
    log(http, worn_on="2026-10-01")
    log(http, worn_on="2026-10-20", rating="not_again")

    listed = http.get("/worn").json()

    assert [(e["worn_on"], e["rating"]) for e in listed] == [
        ("2026-10-20", "not_again"),
        ("2026-10-01", "loved"),
    ]
    assert len(http.get("/worn", params={"limit": 1}).json()) == 1


def test_a_deleted_garment_shows_as_removed(setup):
    http, services = setup
    log(http)
    services.store.delete([PANTS.id])

    [entry] = http.get("/worn").json()

    assert [p["name"] for p in entry["pieces"]] == ["white oxford shirt"]
    assert entry["removed"] == 1


@pytest.mark.parametrize(
    ("body", "says"),
    [
        ({"pieces": [str(uuid4())]}, "not in your wardrobe"),
        ({"worn_on": "2026-10-26"}, "in the future"),
        ({"issues": ["too_warm", "too_cold"]}, "both too warm and too cold"),
    ],
)
def test_bad_entries_are_refused_and_nothing_is_kept(setup, body, says):
    http, services = setup

    response = log(http, **body)

    assert response.status_code == 422 and says in response.json()["detail"]
    assert services.wear_log.entries() == []


def test_a_rating_is_required(setup):
    http, _ = setup
    response = http.post("/worn", json={"pieces": [str(SHIRT.id)]})
    assert response.status_code == 422


def test_deleting_an_entry(setup):
    http, services = setup
    entry_id = log(http).json()["id"]

    assert http.delete(f"/worn/{entry_id}").status_code == 204
    assert http.delete(f"/worn/{entry_id}").status_code == 404
    assert services.wear_log.entries() == []


# --- The photo ----------------------------------------------------------------


class WornPipeline:
    """Fake segmentation + CLIP: each photo shows the given (region, item) garments."""

    def __init__(self, services, *shown):
        self.services, self.shown, self.photos = services, list(shown), []

    def worn_vectors(self, photo: bytes):
        self.photos.append(photo)
        if not self.shown:
            raise SegmentationError("No garment covering >= 2% of the image")
        return [(region, self.services.store.vector(item.id)) for region, item in self.shown]


@pytest.fixture
def with_photo(setup):
    """`setup` plus grey chinos, and a photo showing the shirt with the chinos."""
    http, services = setup
    services.store.upsert([CHINOS], services.store.vector(SHIRT.id)[None, ::-1].copy())
    pipeline = WornPipeline(services, ("upper", SHIRT), ("lower", CHINOS))
    services.pipeline = lambda: pipeline
    entry = log(http, pieces=[str(SHIRT.id), str(PANTS.id), str(SHOES.id)]).json()
    return http, services, pipeline, entry["id"]


def put_photo(http, entry_id, data=None):
    return http.put(
        f"/worn/{entry_id}/photo", files={"photo": ("me.jpg", data or photo_with_exif())}
    )


def test_a_photo_is_matched_locally_and_kept_without_exif(with_photo):
    http, services, pipeline, entry_id = with_photo

    response = put_photo(http, entry_id)

    assert response.status_code == 200
    body = response.json()
    upper, lower = body["match"]["garments"]
    assert (upper["verdict"], upper["message"]) == ("yours", "your white oxford shirt: yes")
    assert lower["verdict"] == "unsure" and lower["best"] == str(CHINOS.id)
    assert "more like your grey chinos than your navy trousers" in lower["message"]
    assert body["match"]["unchecked"] == [str(SHOES.id)]
    assert body["photo_url"] == f"/worn/{entry_id}/photo"
    kept = http.get(body["photo_url"])
    assert kept.headers["content-type"] == "image/jpeg" and not opened(kept.content).getexif()
    assert not opened(pipeline.photos[0]).getexif()  # checked on the re-encoded copy
    assert services.client.requests == []  # never sent to DeepSeek
    assert http.get("/worn").json()[0]["match"] == body["match"]


def test_correcting_the_pieces_checks_the_photo_again(with_photo):
    http, services, pipeline, entry_id = with_photo
    put_photo(http, entry_id)

    response = http.patch(f"/worn/{entry_id}", json={"pieces": [str(SHIRT.id), str(CHINOS.id)]})

    assert response.status_code == 200
    body = response.json()
    assert [p["name"] for p in body["pieces"]] == ["white oxford shirt", "grey chinos"]
    assert [g["verdict"] for g in body["match"]["garments"]] == ["yours", "yours"]
    assert len(pipeline.photos) == 2
    assert services.wear_log.get(UUID(entry_id)).pieces == [SHIRT.id, CHINOS.id]


def test_a_photo_without_garments_is_not_kept(with_photo):
    http, services, pipeline, entry_id = with_photo
    pipeline.shown = []

    response = put_photo(http, entry_id)

    assert response.status_code == 422
    assert services.wear_log.photo(UUID(entry_id)) is None
    assert http.get("/worn").json()[0]["photo_url"] is None


def test_deleting_the_photo_keeps_the_entry(with_photo):
    http, services, _, entry_id = with_photo
    put_photo(http, entry_id)

    assert http.delete(f"/worn/{entry_id}/photo").status_code == 204

    [entry] = http.get("/worn").json()
    assert entry["photo_url"] is None and entry["match"] is None
    assert http.get(f"/worn/{entry_id}/photo").status_code == 404
    assert http.delete(f"/worn/{entry_id}/photo").status_code == 404


def test_look_scores_are_asked_for_and_kept_until_the_photo_changes(with_photo):
    http, services, _, entry_id = with_photo
    assert http.post(f"/worn/{entry_id}/look").status_code == 404  # no photo yet
    put_photo(http, entry_id)
    services.client = FakeClient(reply(json.dumps(LOOK)))

    response = http.post(f"/worn/{entry_id}/look")

    assert response.status_code == 200 and response.json()["fit"]["score"] == 4
    assert len(services.client.requests) == 1
    assert http.get("/worn").json()[0]["look"] == response.json()
    put_photo(http, entry_id)  # a new photo: the old scores no longer apply
    assert http.get("/worn").json()[0]["look"] is None


def test_no_person_in_the_photo_keeps_no_scores(with_photo):
    http, services, _, entry_id = with_photo
    put_photo(http, entry_id)
    services.client = FakeClient(reply(json.dumps({"person_visible": False})))

    response = http.post(f"/worn/{entry_id}/look")

    assert response.status_code == 422 and "No one" in response.json()["detail"]
    assert http.get("/worn").json()[0]["look"] is None


def test_photo_and_correction_errors(with_photo):
    http, _, _, entry_id = with_photo
    unknown = uuid4()

    assert put_photo(http, unknown).status_code == 404
    assert put_photo(http, entry_id, b"not a photo").status_code == 422
    assert http.patch(f"/worn/{unknown}", json={"pieces": [str(SHIRT.id)]}).status_code == 404
    bad = http.patch(f"/worn/{entry_id}", json={"pieces": [str(unknown)]})
    assert bad.status_code == 422 and "not in your wardrobe" in bad.json()["detail"]
