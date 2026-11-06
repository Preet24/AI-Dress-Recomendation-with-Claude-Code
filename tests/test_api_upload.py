"""Photo upload and re-tagging over HTTP, with the pipeline tests' fake rembg and CLIP."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from outfit.api.app import create_app
from outfit.api.items import photo_path
from outfit.config import Settings
from tests.factories import make_item
from tests.fakes import FakeClient, reply, tags_reply
from tests.fakes import fake_services as base_services
from tests.test_pipeline import WORN_MASKS, H, MaskSession, W, make_pipeline


def jpeg_bytes(size=(W, H)) -> bytes:
    """A 'worn' photo like the pipeline tests': red upper half, navy lower half."""
    img = Image.new("RGB", size, (200, 20, 30))
    img.paste((20, 30, 90), (0, size[1] // 2, size[0], size[1]))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def setup(tmp_path):
    """(TestClient, services, fake DeepSeek client): one store shared by API and pipeline."""
    client = FakeClient()
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    settings = Settings(
        _env_file=None, DEEPSEEK_API_KEY="test-key", raw_dir=tmp_path / "raw", max_upload_mb=1
    )
    services = base_services(client=client, pipeline=lambda: pipeline, settings=settings)
    pipeline.store = services.store
    with TestClient(create_app(services)) as http:
        yield http, services, client


def upload(http, data: bytes, name="outfit.jpg", mode="worn"):
    return http.post("/items", files={"photo": (name, data, "image/jpeg")}, data={"mode": mode})


def test_worn_photo_upload_stores_both_garments_and_keeps_the_photo(setup, tmp_path):
    http, services, client = setup
    client.replies += [tags_reply(), tags_reply(category="bottom", subtype="jeans")]

    response = upload(http, jpeg_bytes())

    assert response.status_code == 201
    names = [i["name"] for i in response.json()]
    assert names == ["red t-shirt", "red jeans"]
    assert services.store.count() == 2
    assert (tmp_path / "raw" / "outfit.jpg").read_bytes() == jpeg_bytes()
    image = http.get(response.json()[0]["image_url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"


def test_reuploading_the_same_photo_updates_its_items(setup, tmp_path):
    http, services, client = setup
    client.replies += [tags_reply(), tags_reply(), tags_reply(subtype="polo"), tags_reply()]

    first = upload(http, jpeg_bytes()).json()
    again = upload(http, jpeg_bytes()).json()

    assert [i["id"] for i in again] == [i["id"] for i in first]
    assert services.store.count() == 2
    assert services.store.get(first[0]["id"]).subtype == "polo"
    assert list((tmp_path / "raw").iterdir()) == [tmp_path / "raw" / "outfit.jpg"]


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        (b"not an image at all", "Not a readable image"),
        (jpeg_bytes(size=(1200, 1200))[:2000], "Not a readable image"),
        (b"\0" * (1024 * 1024 + 1), "larger than 1 MB"),
    ],
    ids=["not-an-image", "truncated-jpeg", "too-large"],
)
def test_bad_files_are_rejected_before_anything_runs(setup, tmp_path, data, detail):
    http, services, client = setup
    response = upload(http, data)
    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert client.requests == [] and services.store.count() == 0
    assert not (tmp_path / "raw").exists()


def test_unsupported_image_type_is_rejected(setup):
    http, *_ = setup
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    response = upload(http, buffer.getvalue(), name="x.gif")
    assert response.status_code == 422
    assert "GIF" in response.json()["detail"]


def test_bad_mode_is_rejected(setup):
    http, *_ = setup
    assert upload(http, jpeg_bytes(), mode="hanging").status_code == 422


def test_failed_tagging_stores_nothing_and_drops_the_new_photo(setup, tmp_path):
    http, services, client = setup
    client.replies += [tags_reply(), reply("bad"), reply("still bad")]

    response = upload(http, jpeg_bytes())

    assert response.status_code == 502
    assert services.store.count() == 0
    assert list((tmp_path / "raw").iterdir()) == []


def test_photo_without_a_garment_is_a_client_error(setup, tmp_path):
    http, services, client = setup
    empty = np.zeros((H, W), dtype=np.uint8)
    services.pipeline().session_for = lambda name: MaskSession(empty)

    response = upload(http, jpeg_bytes(), mode="flat")

    assert response.status_code == 422
    assert client.requests == []


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("outfit.jpg", "outfit.jpg"),
        ("..\\..\\secrets/evil name!.png", "evil-name.jpg"),
        ("../../etc/passwd", "passwd.jpg"),
        (None, "photo.jpg"),
        ("....jpg", "photo.jpg"),
    ],
)
def test_upload_names_stay_inside_the_raw_folder(tmp_path, filename, expected):
    path = photo_path(tmp_path, filename, b"data", ".jpg")
    assert path == tmp_path / expected


def test_a_different_photo_with_the_same_name_gets_its_own_file(tmp_path):
    (tmp_path / "outfit.jpg").write_bytes(b"old photo")
    assert photo_path(tmp_path, "outfit.jpg", b"old photo", ".jpg") == tmp_path / "outfit.jpg"
    other = photo_path(tmp_path, "outfit.jpg", b"new photo", ".jpg")
    assert other.parent == tmp_path and other.name.startswith("outfit-") and other.exists() is False


def test_retag_sends_the_note_and_saves_the_new_tags(setup, tmp_path):
    http, services, client = setup
    cutout = tmp_path / "dress_ab12cd34_upper.png"
    Image.new("RGBA", (8, 8), (10, 10, 10, 255)).save(cutout)
    item = make_item(image_path=cutout)
    services.store.upsert([item], np.ones((1, 512)) / np.sqrt(512))
    client.replies.append(tags_reply(category="one_piece", subtype="slip dress", length="midi"))

    response = http.post(f"/items/{item.id}/retag", json={"note": "it's a midi slip dress"})

    assert response.status_code == 200
    assert response.json()["subtype"] == "slip dress"
    assert "midi slip dress" in client.requests[0]["messages"][1]["content"][0]["text"]
    saved = services.store.get(item.id)
    assert (saved.category, saved.created_at) == ("one_piece", item.created_at)


def test_retag_needs_a_note_and_an_existing_cutout(setup):
    http, services, _ = setup
    item = make_item()  # its cutout file doesn't exist
    services.store.upsert([item], np.ones((1, 512)) / np.sqrt(512))
    assert http.post(f"/items/{item.id}/retag", json={"note": ""}).status_code == 422
    assert http.post(f"/items/{item.id}/retag", json={"note": "a skirt"}).status_code == 404
