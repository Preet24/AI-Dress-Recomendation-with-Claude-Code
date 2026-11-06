"""Chat over HTTP: questions and answers, outfits in the response, busy and unknown threads."""

import json

import pytest
from fastapi.testclient import TestClient

from outfit.api.app import create_app
from tests.fakes import (
    FakeClient,
    critic_reply,
    fake_services,
    reply,
    router_reply,
    stylist_reply,
    tool_call_reply,
)
from tests.test_agent_flow import HOT_DAY, JEANS, WARDROBE, weather_fetch

TRIP = {"city": "Pune", "hours_away": 3, "hours_outdoors": 1, "transport": "walk"}
FIRST_TURN = (  # "casual day out" -> a question; the answer -> a recommendation
    router_reply(),
    reply(json.dumps({"occasion": "casual", "start": "now"})),
    reply(json.dumps(TRIP)),
    stylist_reply(HOT_DAY),
    critic_reply(),
)


@pytest.fixture
def chat():
    """(TestClient, services, fake DeepSeek client) over the two-item wardrobe."""
    client = FakeClient(*FIRST_TURN)
    services = fake_services(WARDROBE, client=client, fetch=weather_fetch)
    with TestClient(create_app(services)) as http:
        yield http, services, client


def send(http, message, thread_id=None):
    body = {"message": message} | ({"thread_id": thread_id} if thread_id else {})
    return http.post("/chat", json=body)


def test_a_question_then_the_answer_gives_a_recommendation_with_its_outfits(chat):
    http, _, client = chat

    first = send(http, "casual day out")
    assert first.status_code == 200
    asked = first.json()
    assert asked["kind"] == "question"
    assert asked["path"] == ["router", "intake"]  # paused inside ask: no update yet
    assert asked["outfits"] == [] and asked["llm_calls"] == 2

    answer = send(http, "Pune, 3 hours, 1 hour outside, walking", asked["thread_id"]).json()

    assert answer["kind"] == "reply" and answer["text"] == HOT_DAY
    assert answer["thread_id"] == asked["thread_id"]
    assert answer["path"][:2] == ["ask", "intake"]  # resumed at the question, not routed again
    assert answer["llm_calls"] == 3  # intake, stylist, critic: nothing repeated
    assert len(client.requests) == 5
    [outfit] = answer["outfits"]
    assert {p["name"] for p in outfit["pieces"]} == {"white t-shirt", "medium blue jeans"}
    assert outfit["pieces"][0]["image_url"].startswith("/items/")
    assert any("footwear" in gap for gap in answer["gaps"])


def test_the_conversation_can_be_read_back(chat):
    http, *_ = chat
    asked = send(http, "casual day out").json()
    thread_id = asked["thread_id"]

    waiting = http.get(f"/chat/{thread_id}").json()
    assert waiting["question"] == asked["text"]  # a reloaded page can show the open question
    assert waiting["messages"] == [{"role": "user", "content": "casual day out"}]

    send(http, "Pune, 3 hours, 1 hour outside, walking", thread_id)
    done = http.get(f"/chat/{thread_id}").json()
    assert done["question"] is None
    assert [m["role"] for m in done["messages"]] == ["user", "assistant", "user", "assistant"]
    assert done["messages"][1]["content"] == asked["text"]
    assert done["messages"][-1] == {"role": "assistant", "content": HOT_DAY}


def test_a_follow_up_returns_the_items_shown_and_no_new_outfits(chat):
    http, _, client = chat
    thread_id = send(http, "casual day out").json()["thread_id"]
    send(http, "Pune, 3 hours, 1 hour outside, walking", thread_id)
    jeans = f"Here are your medium blue jeans. {HOT_DAY}"
    client.replies += [
        router_reply("follow_up"),
        tool_call_reply(("show_items", {"item_ids": [str(JEANS.id)[:8]]})),
        reply("Showed the jeans."),
        stylist_reply(jeans),
        critic_reply(),
    ]

    out = send(http, "show me my jeans", thread_id).json()

    assert out["text"] == jeans
    assert [i["name"] for i in out["shown"]] == ["medium blue jeans"]
    assert out["outfits"] == []  # nothing was re-planned this turn


def test_unknown_conversations_are_404(chat):
    http, *_ = chat
    assert send(http, "hello", "no-such-thread").status_code == 404
    assert http.get("/chat/no-such-thread").status_code == 404


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "   "}, {"text": "hi"}])
def test_empty_or_malformed_messages_are_422(chat, body):
    http, _, client = chat
    assert http.post("/chat", json=body).status_code == 422
    assert client.requests == []


def test_a_busy_conversation_refuses_a_second_message(chat):
    http, services, client = chat
    thread_id = send(http, "casual day out").json()["thread_id"]
    lock = services.thread_lock(thread_id)  # as if a turn were still running
    try:
        busy = send(http, "Pune, 3 hours, 1 hour outside, walking", thread_id)
    finally:
        lock.release()
    assert busy.status_code == 409
    assert len(client.requests) == 2  # the refused message never reached the model
    assert send(http, "Pune, 3 hours, 1 hour outside, walking", thread_id).json()["kind"] == "reply"


def test_turns_are_recorded_when_record_chats_is_on(chat, tmp_path):
    http, services, _ = chat
    services.settings = services.settings.model_copy(
        update={"record_chats": True, "conversations_dir": tmp_path}
    )
    thread_id = send(http, "casual day out").json()["thread_id"]
    send(http, "Pune, 3 hours, 1 hour outside, walking", thread_id)

    [path] = tmp_path.iterdir()
    turns = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [t["turn"] for t in turns] == [1, 2]
    assert turns[0]["thread_id"] == thread_id and turns[1]["reply"] == HOT_DAY
    assert [len(step["llm"]) for step in turns[1]["steps"] if step["node"] == "intake"] == [1]
