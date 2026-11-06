"""The shopping page headless, against the real API over fakes.

AppTest can't drive st.file_uploader, so the check runs through the UI client (which
also proves the client parses the real response) and its result is put where the
page keeps it, `session_state.shopping_check`.
"""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from outfit.api.app import create_app
from outfit.config import Settings
from outfit.profile import Profile, SavedPlace
from outfit.schemas import Occasion
from outfit.tools.weather import WeatherError
from outfit.ui.client import OutfitAPI, ShoppingCheck, ShoppingSuggestions
from tests.fakes import FakeClient, fake_services
from tests.test_api_shopping import SHIRT, queue_worn_tags
from tests.test_api_upload import jpeg_bytes
from tests.test_pipeline import WORN_MASKS, make_pipeline


def shopping_script():
    import streamlit as st

    from outfit.ui import shopping_page

    shopping_page.render(st.session_state.api)


def no_archive(url, params):
    raise WeatherError("no network in tests")


@pytest.fixture
def setup(tmp_path):
    """(AppTest after a check was run, services, the check)."""
    client = FakeClient()
    pipeline = make_pipeline(tmp_path, client, *WORN_MASKS)
    settings = Settings(
        _env_file=None,
        DEEPSEEK_API_KEY="test-key",
        raw_dir=tmp_path / "raw_wardrobe",
        clean_dir=tmp_path / "clean",
        candidates_dir=tmp_path / "candidates",
    )
    services = fake_services([SHIRT], client=client, pipeline=lambda: pipeline, settings=settings)
    services.context = replace(services.context, fetch=no_archive)
    services.profile.save(
        Profile(places=[SavedPlace(name="Office", occasion=Occasion.OFFICE)], occasions=["casual"])
    )
    pipeline.store = services.store
    queue_worn_tags(client)
    with TestClient(create_app(services)) as http:
        api = OutfitAPI(http=http)
        result = api.check_item("shop.jpg", jpeg_bytes())
        at = AppTest.from_function(shopping_script, default_timeout=30)
        at.session_state["api"] = api
        at.session_state["shopping_check"] = result
        yield at.run(), services, result


def test_the_client_parses_a_check(setup):
    _, _, result = setup
    assert isinstance(result, ShoppingCheck)
    assert [r.candidate.subtype for r in result.reports] == ["t-shirt", "chinos"]
    assert result.reports[1].goes_with[0].item.name == "white button-up shirt"


def test_each_garment_gets_a_card_with_its_verdict_and_reasons(setup):
    at, _, result = setup
    assert not at.exception
    assert at.title[0].value == "Shopping"
    verdicts = [e.value for e in (*at.success, *at.info, *at.warning)]
    assert any("chinos" in v for v in verdicts) and any("t-shirt" in v for v in verdicts)
    chinos = result.reports[1]
    shown = [m.value for m in at.markdown]
    assert f"- {chinos.color_note}" in shown
    assert "**Goes with**" in shown
    assert any("Checked 8 situations" in c.value for c in at.caption)
    assert any("Office (office)" in c.value for c in at.caption)  # saved places count too


def test_ticking_occasions_saves_them_to_the_profile(setup):
    at, services, _ = setup
    at.multiselect(key="occasions").set_value([Occasion.CASUAL, Occasion.PARTY])
    at = [b for b in at.button if b.proto.form_id == "occasions"][0].click().run()

    assert services.profile.load().occasions == [Occasion.CASUAL, Occasion.PARTY]
    assert at.success[0].value == "Saved the occasions you dress for"


def test_add_to_wardrobe_stores_the_candidate_once(setup):
    at, services, result = setup
    chinos = result.reports[1].candidate

    at = at.button(key=f"add-{chinos.id}").click().run()

    assert services.store.get(chinos.id) is not None and services.store.count() == 2
    assert at.success[0].value == "Added navy chinos to your wardrobe"
    at = at.run()  # AppTest keeps elements from the run that called st.rerun(); refresh
    assert not [b for b in at.button if b.key == f"add-{chinos.id}"]
    assert any(c.value == "Added to your wardrobe." for c in at.caption)


def test_no_check_yet_shows_only_the_upload(tmp_path):
    services = fake_services()
    with TestClient(create_app(services)) as http:
        at = AppTest.from_function(shopping_script, default_timeout=30)
        at.session_state["api"] = OutfitAPI(http=http)
        at.run()
    assert not at.exception
    assert at.button(key="check").disabled


def test_suggest_shows_what_is_missing_and_a_card_per_suggestion(setup):
    at, services, _ = setup
    assert at.multiselect(key="shop_for").value == [Occasion.CASUAL, Occasion.OFFICE]

    at = at.button(key="suggest").click().run()

    assert not at.exception
    result = at.session_state["shopping_suggest"]
    assert isinstance(result, ShoppingSuggestions) and result.suggestions
    shown = [m.value for m in at.markdown]
    assert "**What your wardrobe is missing**" in shown
    assert any(m.startswith("- no outfit for") for m in shown)  # one shirt, no bottoms
    cards = [s.value for s in at.success]
    assert all(any(f"**{s.name}**" in c for c in cards) for s in result.suggestions)
