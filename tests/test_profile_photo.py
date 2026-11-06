"""The profile photo (kept locally, no EXIF), its colour analysis, and the taste rules
the analysis feeds."""

import base64
import io
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from outfit.agents.llm import LLMRefused
from outfit.agents.orchestrate import orchestrate_node
from outfit.agents.state import Slots
from outfit.api.app import create_app
from outfit.colouring import SEND_MAX_SIDE, NoFaceVisible, analyse_colouring
from outfit.config import Settings
from outfit.profile import PHOTO_MAX_SIDE, Colouring, Profile, ProfileStore, reencode
from outfit.schemas import Occasion
from outfit.tools.outfits import GO_EASY_COLOR, SUITS_COLORING, OutfitRequest, taste
from outfit.ui.client import OutfitAPI
from tests.fakes import FakeClient, fake_context, fake_services, reply
from tests.test_outfits import bottom, top

SETTINGS = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")
ANALYSIS = {
    "face_visible": True,
    "summary": "warm undertone, medium depth, dark brown hair and eyes, medium contrast",
    "suits": ["Olive", "rust", "cream"],
    "go_easy": ["neon", "icy pastels"],
}
GPS = 0x8825  # EXIF GPS IFD tag


def photo_with_exif(size=(2000, 1500)) -> bytes:
    """A JPEG carrying camera and GPS data, stored sideways (orientation 6)."""
    image = Image.new("RGB", size, (190, 140, 110))
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # camera make
    exif[0x0112] = 6  # rotate 90° to show upright
    exif[GPS] = {1: "N", 2: (18.0, 31.0, 0.0)}
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def opened(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# --- Keeping the photo -----------------------------------------------------------


def test_the_kept_copy_is_upright_smaller_and_has_no_exif():
    kept = opened(reencode(photo_with_exif(), PHOTO_MAX_SIDE))

    assert kept.format == "JPEG"
    assert kept.size == (768, PHOTO_MAX_SIDE)  # turned upright, then shrunk
    assert not kept.getexif()
    assert b"PhoneMaker" not in reencode(photo_with_exif(), PHOTO_MAX_SIDE)


@pytest.mark.parametrize("on_disk", [True, False])
def test_the_store_keeps_and_deletes_the_photo(tmp_path, on_disk):
    store = ProfileStore(tmp_path / "profile.json" if on_disk else None)
    assert store.photo() is None and store.delete_photo() is False

    store.save_photo(photo_with_exif())

    assert opened(store.photo()).format == "JPEG"
    if on_disk:
        assert (tmp_path / "profile_photo.jpg").is_file()
    assert store.delete_photo() is True and store.photo() is None


# --- The analysis ------------------------------------------------------------------


def test_the_analysis_sends_a_small_copy_without_exif_to_the_vision_model():
    client = FakeClient(reply(json.dumps(ANALYSIS)))

    colouring = analyse_colouring(photo_with_exif(), client, SETTINGS)

    assert colouring == Colouring(
        summary=ANALYSIS["summary"], suits=["olive", "rust", "cream"], go_easy=ANALYSIS["go_easy"]
    )
    [request] = client.requests
    assert request["model"] == SETTINGS.tagger_model
    system, user = request["messages"]
    assert "Never comment on or guess ethnicity" in system["content"]
    url = user["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    sent = base64.standard_b64decode(url.split(",", 1)[1])
    assert max(opened(sent).size) == SEND_MAX_SIDE
    assert not opened(sent).getexif() and b"PhoneMaker" not in sent


def test_no_visible_face_is_an_error_not_a_guess():
    client = FakeClient(reply(json.dumps({"face_visible": False})))
    with pytest.raises(NoFaceVisible):
        analyse_colouring(photo_with_exif(), client, SETTINGS)


def test_a_refusal_is_not_retried():
    client = FakeClient(reply(None, finish_reason="content_filter"))
    with pytest.raises(LLMRefused):
        analyse_colouring(photo_with_exif(), client, SETTINGS)
    assert len(client.requests) == 1


# --- The API -----------------------------------------------------------------------


def test_upload_analyse_and_delete_over_the_api():
    client = FakeClient(reply(json.dumps(ANALYSIS)))
    services = fake_services(client=client)
    with TestClient(create_app(services)) as http:
        assert http.get("/profile/photo").status_code == 404
        assert http.post("/profile/analyse").status_code == 404  # nothing to analyse
        assert client.requests == []

        files = {"photo": ("me.jpg", photo_with_exif())}
        assert http.put("/profile/photo", files=files).status_code == 204
        shown = http.get("/profile/photo")
        assert shown.headers["content-type"] == "image/jpeg" and not opened(shown.content).getexif()

        analysed = http.post("/profile/analyse")
        assert analysed.status_code == 200
        assert analysed.json()["suits"] == ["olive", "rust", "cream"]

        assert http.delete("/profile/photo").status_code == 204
        assert http.get("/profile").json()["colouring"]["summary"] == ANALYSIS["summary"]
        assert http.delete("/profile/photo").status_code == 404
    assert len(client.requests) == 1


def test_a_failed_analysis_saves_nothing():
    services = fake_services(client=FakeClient(reply(json.dumps({"face_visible": False}))))
    with TestClient(create_app(services)) as http:
        http.put("/profile/photo", files={"photo": ("me.jpg", photo_with_exif())})
        response = http.post("/profile/analyse")
    assert response.status_code == 422 and "No face" in response.json()["detail"]
    assert services.profile.load().colouring is None


def test_a_non_image_is_refused():
    with TestClient(create_app(fake_services())) as http:
        response = http.put("/profile/photo", files={"photo": ("me.jpg", b"not a photo")})
    assert response.status_code == 422


def test_the_ui_client_round_trip():
    services = fake_services(client=FakeClient(reply(json.dumps(ANALYSIS))))
    with TestClient(create_app(services)) as http:
        api = OutfitAPI(http=http)
        assert api.profile_photo() is None
        api.upload_profile_photo("me.jpg", photo_with_exif())
        assert api.profile_photo()
        assert api.analyse_colouring().summary == ANALYSIS["summary"]
        assert api.profile().colouring.go_easy == ["neon", "icy pastels"]
        api.delete_profile_photo()
        assert api.profile_photo() is None


# --- Taste from the analysis ---------------------------------------------------------

CASUAL = OutfitRequest(
    occasion=Occasion.CASUAL,
    colors_suit=frozenset({"olive"}),
    colors_go_easy=frozenset({"neon"}),
)


def test_colours_that_suit_and_to_go_easy_on_move_the_score():
    assert taste(top(name="olive"), CASUAL) == (
        SUITS_COLORING,
        ["olive t-shirt is in olive, which suits your colouring"],
    )
    assert taste(top(name="neon green"), CASUAL) == (
        -GO_EASY_COLOR,
        ["neon green t-shirt is in neon, which your colouring suggests going easy on"],
    )


def test_the_users_own_choice_beats_the_analysis():
    own = replace(CASUAL, colors_liked=frozenset({"neon"}), colors_avoided=frozenset({"olive"}))
    assert taste(top(name="neon"), own)[1] == ["neon t-shirt is in neon, a color you like"]
    assert taste(top(name="olive"), own)[1] == [
        "olive t-shirt is in olive, a color you usually avoid"
    ]


def test_orchestrate_takes_the_colouring_from_the_profile():
    olive, grey, navy = top(name="olive"), top(name="grey"), bottom()
    profile = Profile(colouring=Colouring(summary="warm", suits=["olive"]))
    context = fake_context([grey, olive, navy], profile=profile)

    update = orchestrate_node(
        {"slots": Slots(occasion=Occasion.CASUAL, skip_weather=True)}, context=context
    )

    assert update["plan"].outfits[0].pieces[0].id == olive.id


# --- The page ------------------------------------------------------------------------


def profile_script():
    import streamlit as st

    from outfit.ui import profile_page

    profile_page.render(st.session_state.api)


@pytest.fixture
def page():
    services = fake_services(client=FakeClient(reply(json.dumps(ANALYSIS))))
    services.profile.save_photo(photo_with_exif())
    with TestClient(create_app(services)) as http:
        at = AppTest.from_function(profile_script, default_timeout=30)
        at.session_state["api"] = OutfitAPI(http=http)
        yield at.run(), services


def test_analysing_from_the_page_fills_the_colouring_form(page):
    at, services = page
    assert not at.exception
    assert any("sends a smaller copy" in c.value for c in at.caption)

    at = at.button(key="analyse").click().run()

    assert services.profile.load().colouring.summary == ANALYSIS["summary"]
    assert at.text_input(key="colouring_suits").value == "olive, rust, cream"
    assert at.success[0].value.startswith("Your colouring: warm undertone")


def test_the_colouring_can_be_edited_and_cleared(page):
    at, services = page
    at.text_input(key="colouring_summary").set_value("cool, light")
    at.text_input(key="colouring_suits").set_value("Navy, berry")
    at = [b for b in at.button if b.proto.form_id == "colouring"][0].click().run()
    assert services.profile.load().colouring == Colouring(
        summary="cool, light", suits=["navy", "berry"]
    )

    # AppTest drops a second submit of one form right after st.rerun(); a plain run
    # in between stands in for the user's next separate click.
    at = at.run()
    at.text_input(key="colouring_summary").set_value("")
    at = [b for b in at.button if b.proto.form_id == "colouring"][0].click().run()
    assert services.profile.load().colouring is None
    assert at.success[0].value == "Cleared your colouring"


def test_deleting_the_photo_from_the_page(page):
    at, services = page
    at = at.button(key="delete-photo").click().run()
    assert services.profile.photo() is None
    assert at.success[0].value == "Deleted your photo"
    assert not [b for b in at.button if b.key == "analyse"]  # nothing left to analyse
