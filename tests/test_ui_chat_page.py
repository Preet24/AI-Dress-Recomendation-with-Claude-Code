"""The chat page, run headless with Streamlit's AppTest against the real app over fakes."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from outfit.api.app import create_app
from outfit.ui.chat_page import WELCOME, Entry, history_from, turn_summary
from outfit.ui.client import ChatTurn, Conversation, OutfitAPI
from tests.fakes import FakeClient, fake_services
from tests.test_agent_flow import HOT_DAY, WARDROBE, weather_fetch
from tests.test_api_chat import FIRST_TURN

ANSWER = "Pune, 3 hours, 1 hour outside, walking"


def chat_script():
    import streamlit as st

    from outfit.ui import chat_page

    chat_page.render(st.session_state.api)


@pytest.fixture
def api(tmp_path):
    """The tee-and-jeans wardrobe, with real cutout files so pictures render."""
    wardrobe = []
    for item in WARDROBE:
        cutout = tmp_path / f"{item.subtype}.png"
        Image.new("RGBA", (6, 6), (120, 140, 200, 255)).save(cutout)
        wardrobe.append(item.model_copy(update={"image_path": cutout}))
    services = fake_services(wardrobe, client=FakeClient(*FIRST_TURN), fetch=weather_fetch)
    with TestClient(create_app(services)) as http:
        yield OutfitAPI(http=http)


def page(api, thread=None) -> AppTest:
    at = AppTest.from_function(chat_script, default_timeout=30)
    at.session_state["api"] = api
    if thread:
        at.query_params["thread"] = thread
    return at.run()


def bubbles(at: AppTest) -> list[tuple[str, str]]:
    return [(m.name, m.markdown[0].value) for m in at.chat_message]


def test_a_question_then_a_recommendation_with_outfit_pictures(api):
    at = page(api)
    assert bubbles(at) == [("assistant", WELCOME)]

    at.chat_input[0].set_value("casual day out").run()
    question = bubbles(at)[-1]
    assert question[0] == "assistant" and "hours" in question[1]
    thread = at.query_params["thread"][0]

    at.chat_input[0].set_value(ANSWER).run()

    assert not at.exception
    roles = [role for role, _ in bubbles(at)]
    assert roles == ["assistant", "user", "assistant", "user", "assistant"]
    assert bubbles(at)[-1][1] == HOT_DAY
    assert at.query_params["thread"][0] == thread  # same conversation
    assert "**Top pick**" in [m.value for m in at.markdown]
    assert any("colors" in c.value or "neutral" in c.value for c in at.caption)  # reasons
    assert any("footwear" in i.value for i in at.info)  # the gap
    assert "no picture" not in " ".join(c.value for c in at.caption)
    pictures = [p for p in at.get("imgs") for p in p.proto.imgs]
    assert sorted(p.caption for p in pictures) == ["medium blue jeans", "white t-shirt"]


def test_a_reload_reopens_the_conversation_and_its_open_question(api):
    asked = api.chat("casual day out")

    at = page(api, thread=asked.thread_id)

    assert bubbles(at)[1:] == [("user", "casual day out"), ("assistant", asked.text)]
    at.chat_input[0].set_value(ANSWER).run()
    assert bubbles(at)[-1] == ("assistant", HOT_DAY)


def test_an_unknown_conversation_starts_fresh_with_a_warning(api):
    at = page(api, thread="no-such-thread")

    assert "server may have restarted" in at.warning[0].value
    assert bubbles(at) == [("assistant", WELCOME)]
    assert "thread" not in at.query_params


def test_new_conversation_clears_the_chat(api):
    at = page(api)
    at.chat_input[0].set_value("casual day out").run()

    at.sidebar.button[0].click().run()

    assert bubbles(at) == [("assistant", WELCOME)]
    assert "thread" not in at.query_params and "thread_id" not in at.session_state


def test_a_failed_turn_says_so_in_the_chat():
    at = page(OutfitAPI("http://127.0.0.1:9"))  # nothing listens there

    at.chat_input[0].set_value("hello").run()

    assert not at.exception
    assert bubbles(at)[-1][1].startswith("Sorry, that failed: Can't reach the API")


APP = "src/outfit/ui/app.py"


def test_the_app_shows_the_wardrobe_size_and_opens_on_the_chat(api):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = api
    at.run()

    assert not at.exception
    assert at.sidebar.caption[0].value.startswith("2 items in your wardrobe")
    assert at.title[0].value == "Outfit chat"


def test_the_app_stops_with_instructions_when_the_api_is_down():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api"] = OutfitAPI("http://127.0.0.1:9")
    at.run()

    assert "serve.py" in at.error[0].value
    assert not at.title  # no page was drawn


def test_history_from_a_waiting_conversation_ends_with_the_question():
    waiting = Conversation(
        thread_id="t", messages=[{"role": "user", "content": "hi"}], question="Which city?"
    )
    assert history_from(waiting) == [Entry("user", "hi"), Entry("assistant", "Which city?")]


def test_turn_summary_reads_like_a_trace():
    turn = ChatTurn(
        thread_id="t", kind="reply", text="", outfits=[], gaps=[], shown=[],
        path=["router", "respond"], llm_calls=1, seconds=0.84,
    )  # fmt: skip
    assert turn_summary(turn) == "router → respond · 1 DeepSeek call · 0.8s"
