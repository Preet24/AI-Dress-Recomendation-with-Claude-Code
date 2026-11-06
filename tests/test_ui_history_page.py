"""'I wore this' and the History page headless, against the real API over fakes."""

import json
from dataclasses import replace
from datetime import date
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from outfit import wear_log as server
from outfit.api.app import create_app
from outfit.api.worn import WornOut
from outfit.ui import client
from outfit.ui.client import OutfitAPI, WornEntry
from tests.fakes import FakeClient, fake_services, reply
from tests.test_api_worn import CHINOS, NOW, PANTS, SHIRT, SHOES, WornPipeline
from tests.test_look import LOOK
from tests.test_profile_photo import photo_with_exif


def history_script():
    import streamlit as st

    from outfit.ui import history_page

    history_page.render(st.session_state.api)


def form_script():
    import streamlit as st

    from outfit.ui.widgets import show_flash
    from outfit.ui.worn_form import wore_this

    show_flash()
    wore_this(st.session_state.api, st.session_state.pieces, "test", "chat")


@pytest.fixture
def setup():
    """(services, api): shirt, trousers, shoes and chinos; photos show shirt + chinos."""
    services = fake_services([SHIRT, PANTS, SHOES, CHINOS])
    services.context = replace(services.context, now=NOW)
    pipeline = WornPipeline(services, ("upper", SHIRT), ("lower", CHINOS))
    services.pipeline = lambda: pipeline
    with TestClient(create_app(services)) as http:
        yield services, OutfitAPI(http=http)


def page(api, script=history_script, **state) -> AppTest:
    at = AppTest.from_function(script, default_timeout=30)
    at.session_state["api"] = api
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def logged_with_photo(api) -> WornEntry:
    entry = api.log_worn([SHIRT.id, PANTS.id, SHOES.id], "fine", worn_on=date(2026, 10, 24))
    return api.upload_worn_photo(entry.id, "me.jpg", photo_with_exif())


def submit(at, form_id):
    return next(b for b in at.button if b.proto.form_id == form_id).click().run()


# --- The client ---------------------------------------------------------------


def test_the_client_mirrors_the_wear_log():
    for mirror, enum in [
        (client.Rating, server.Rating),
        (client.Issue, server.Issue),
        (client.Verdict, server.Verdict),
    ]:
        assert list(get_args(mirror)) == [e.value for e in enum]
    assert list(get_args(client.WearSource)) == list(get_args(server.Source))
    for mirror, model in [
        (client.WornEntry, WornOut),
        (client.GarmentMatch, server.GarmentMatch),
        (client.LookScores, server.LookScores),
    ]:
        assert list(mirror.model_fields) == list(model.model_fields)


def test_the_client_logs_checks_corrects_and_scores(setup):
    services, api = setup

    entry = logged_with_photo(api)
    assert [g.verdict for g in entry.match.garments] == ["yours", "unsure"]
    fixed = api.correct_worn(entry.id, [SHIRT.id, CHINOS.id])
    assert [g.verdict for g in fixed.match.garments] == ["yours", "yours"]
    services.client = FakeClient(reply(json.dumps(LOOK)))
    assert api.ask_look(entry.id).fit.score == 4
    assert api.worn()[0].look.tips == ["Tuck the shirt in", "Try brown shoes"]
    api.delete_worn_photo(entry.id)
    assert api.worn()[0].photo_url is None
    api.delete_worn(entry.id)
    assert api.worn() == []


# --- I wore this ----------------------------------------------------------------


def test_i_wore_this_needs_a_rating_then_logs(setup):
    services, api = setup
    at = page(api, form_script, pieces=api.items()[:2])

    at = submit(at, "worn-test")
    assert at.error[0].value.startswith("Pick a rating") and services.wear_log.entries() == []

    at.radio(key="worn-rating-test").set_value("loved")
    at.multiselect(key="worn-issues-test").set_value(["too_warm"])
    at = submit(at, "worn-test")

    assert not at.exception
    [entry] = services.wear_log.entries()
    assert entry.rating == "loved" and entry.issues == ["too_warm"] and entry.source == "chat"
    assert len(entry.pieces) == 2
    assert "Logged: loved it." in at.success[0].value


# --- History ---------------------------------------------------------------------


def test_an_empty_history_says_how_to_start(setup):
    _, api = setup
    at = page(api)
    assert not at.exception and "I wore this" in at.info[0].value


def test_history_shows_the_photo_check_and_fixes_a_piece(setup):
    services, api = setup
    entry = logged_with_photo(api)

    at = page(api)

    assert not at.exception
    shown = [m.value for m in at.markdown]
    assert "✅ your white oxford shirt: yes" in shown
    assert any(m.startswith("❓ this bottom looks more like your grey chinos") for m in shown)
    assert any("Not checked" in c.value and "black loafers" in c.value for c in at.caption)

    at = at.button(key=f"fix-{entry.id}-1").click().run()

    assert not at.exception
    assert set(services.wear_log.get(entry.id).pieces) == {SHIRT.id, SHOES.id, CHINOS.id}
    assert "Updated: you wore your grey chinos" in at.success[0].value


def test_asking_deepseek_shows_the_scores(setup):
    services, api = setup
    entry = logged_with_photo(api)
    services.client = FakeClient(reply(json.dumps(LOOK)))
    at = page(api)
    assert services.client.requests == []  # nothing sent until asked

    at = at.button(key=f"look-{entry.id}").click().run().run()

    assert not at.exception and len(services.client.requests) == 1
    assert at.table and "- Tip: Tuck the shirt in" in [m.value for m in at.markdown]
    assert not [b for b in at.button if b.key == f"look-{entry.id}"]


def test_deleting_an_entry_asks_first(setup):
    services, api = setup
    entry = logged_with_photo(api)
    at = page(api)

    at = at.button(key=f"forget-{entry.id}").click().run()
    assert at.warning and services.wear_log.entries()
    at = at.button(key=f"forget-yes-{entry.id}").click().run()

    assert services.wear_log.entries() == []
    assert services.wear_log.photo(entry.id) is None
