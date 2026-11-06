"""The UI's API client against the real FastAPI app over fake services (no network)."""

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from outfit.api.app import create_app
from outfit.schemas import Category
from outfit.ui.client import APIError, OutfitAPI
from tests.fakes import FakeClient, fake_services
from tests.test_agent_flow import HOT_DAY, JEANS, TEE, WARDROBE, weather_fetch
from tests.test_api_chat import FIRST_TURN
from tests.test_api_upload import jpeg_bytes


@pytest.fixture
def api():
    services = fake_services(WARDROBE, client=FakeClient(*FIRST_TURN), fetch=weather_fetch)
    with TestClient(create_app(services)) as http:
        yield OutfitAPI(http=http)


def test_wardrobe_calls_parse_into_typed_items(api):
    assert api.health().items == 2
    items = api.items()
    assert [i.name for i in items] == ["white t-shirt", "medium blue jeans"]
    assert items[1].category is Category.BOTTOM and items[1].id == JEANS.id
    assert [i.name for i in api.items(categories=[Category.BOTTOM])] == ["medium blue jeans"]

    edited = api.edit(TEE.id, {"primary_color_name": "ivory"})
    assert edited.name == "ivory t-shirt"
    api.delete(TEE.id)
    assert [i.name for i in api.items()] == ["medium blue jeans"]


def test_a_question_then_the_answer(api):
    asked = api.chat("casual day out")
    assert asked.kind == "question"
    assert api.conversation(asked.thread_id).question == asked.text

    answer = api.chat("Pune, 3 hours, 1 hour outside, walking", asked.thread_id)
    assert (answer.kind, answer.text) == ("reply", HOT_DAY)
    assert {p.name for p in answer.outfits[0].pieces} == {"white t-shirt", "medium blue jeans"}


def test_api_errors_carry_the_servers_message(api):
    with pytest.raises(APIError) as missing:
        api.chat("hello", "no-such-thread")
    assert missing.value.status == 404 and "server may have restarted" in missing.value.detail

    with pytest.raises(APIError) as invalid:
        api.edit(JEANS.id, {"formality": 9})
    assert invalid.value.status == 422 and "formality" in invalid.value.detail

    with pytest.raises(APIError) as bad_field:  # FastAPI's own validation error list
        api.edit(JEANS.id, {"colour": "red"})
    assert bad_field.value.status == 422 and "colour" in bad_field.value.detail


def test_upload_errors_reach_the_user_in_plain_words(api):
    with pytest.raises(APIError) as bad:
        api.upload("notes.txt", b"not a photo", "flat")
    assert bad.value.status == 422 and bad.value.detail == "Not a readable image"
    assert jpeg_bytes()  # a valid photo would go on to the pipeline (tested in test_api_upload)


def test_an_unreachable_api_says_how_to_start_it():
    with pytest.raises(APIError) as down:
        OutfitAPI("http://127.0.0.1:9").health()  # nothing listens on port 9
    assert down.value.status == 0 and "serve.py" in down.value.detail


def test_the_ui_client_loads_no_server_code():
    probe = (
        "import sys, outfit.ui.client; "
        "print(sorted({m.split('.')[0] if not m.startswith('outfit') else m for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'qdrant_client', 'fastapi', 'openai', 'langgraph') "
        "or m.startswith(('outfit.api', 'outfit.agents', 'outfit.ingestion'))}))"
    )
    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "[]"
