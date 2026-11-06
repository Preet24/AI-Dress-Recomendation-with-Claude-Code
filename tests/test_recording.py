import json
from types import SimpleNamespace

import pytest

from outfit.agents.graph import build_graph
from outfit.agents.recording import Recorder, RecordingClient, jsonable, prompts_version, run_turn
from tests.fakes import FakeClient, critic_reply, reply, router_reply, stylist_reply
from tests.test_agent_flow import SETTINGS, TRIP, planned_state
from tests.test_router_refine import NO_SHOES, context

ANSWER = "Your white t-shirt and beige chinos." + NO_SHOES


def test_jsonable_shortens_garments_and_flattens_state():
    state = planned_state()

    data = json.loads(json.dumps(jsonable(state)))  # survives a JSON round trip

    piece = data["plan"]["outfits"][0]["pieces"][0]
    assert set(piece) == {"id", "name", "category"} and len(piece["id"]) == 8
    assert data["slots"]["transport"] == "walk"
    assert data["advice"]["warmth"] == [1, 2]


def test_a_turn_through_a_question_is_recorded_node_by_node():
    fake = FakeClient(
        router_reply(),
        reply(json.dumps({"occasion": "casual"})),
        reply(json.dumps(TRIP)),
        stylist_reply(ANSWER),
        critic_reply(),
    )
    client = RecordingClient(fake)
    graph = build_graph(client, SETTINGS, context())
    config = {"configurable": {"thread_id": "t"}}

    asked = run_turn(graph, config, "casual day out", resume=False, client=client)
    answered = run_turn(graph, config, "Pune, 3 hours, 1 outside, walking", True, client)

    assert asked.reply is None and "Which city" in asked.question
    assert asked.record["path"] == ["router", "intake"]
    assert [len(s["llm"]) for s in asked.record["steps"]] == [1, 1]  # router, then intake
    assert answered.record["path"] == [
        "ask",
        "intake",
        "orchestrate",
        "stylist",
        "critic",
        "respond",
    ]
    assert answered.reply == answered.record["reply"] == ANSWER
    assert answered.record["llm_calls"] == 3  # intake, stylist, critic: the resume repeats none
    stylist = answered.record["steps"][3]
    assert stylist["update"]["draft"] == ANSWER  # respond clears it, the recording keeps it
    facts = json.loads(stylist["llm"][0]["request"][0]["content"])
    assert facts["outfits"]  # what the stylist was shown
    assert all(
        m["role"] != "system"
        for s in answered.record["steps"]
        for c in s["llm"]
        for m in c["request"]
    )


def test_recording_client_logs_usage_and_errors():
    with_usage = reply("{}")
    with_usage.usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    client = RecordingClient(FakeClient(with_usage))  # second call: queue is empty

    client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(IndexError):
        client.chat.completions.create(messages=[], tools=[])

    ok, failed = client.take()
    assert ok.usage["total_tokens"] == 10 and ok.finish_reason == "stop"
    assert ok.request == [{"role": "user", "content": "hi"}] and not ok.tools
    assert failed.tools and "IndexError" in failed.error
    assert client.take() == []


def test_recorder_appends_numbered_turns(tmp_path):
    recorder = Recorder(tmp_path, "abcdef123456", "deepseek-flash")

    recorder.write({"user": "hi", "reply": "Hello!"})
    recorder.write({"user": "thanks", "reply": "You're welcome!"})

    lines = [json.loads(line) for line in recorder.path.read_text("utf-8").splitlines()]
    assert recorder.path.name.endswith("_abcdef12.jsonl")
    assert [line["turn"] for line in lines] == [1, 2]
    assert lines[0]["prompts"] == prompts_version() and lines[0]["model"] == "deepseek-flash"
