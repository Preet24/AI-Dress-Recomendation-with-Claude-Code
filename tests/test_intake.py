import json
from datetime import datetime

import pytest
from langgraph.types import Command

from outfit.agents.graph import build_graph
from outfit.agents.intake import (
    MAX_ASKS,
    IntakePatch,
    apply_defaults,
    give_up_asking,
    intake_node,
    merge_slots,
    missing_slots,
    question_for,
)
from outfit.agents.state import Slots
from outfit.config import Settings
from outfit.schemas import Occasion
from outfit.tools.weather import Transport
from tests.fakes import (
    FakeClient,
    critic_reply,
    fake_context,
    reply,
    router_reply,
    stylist_reply,
)

SETTINGS = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")
NOW = datetime(2026, 10, 17, 16, 0)
COMPLETE = {"city": "Pune", "hours_away": 4, "hours_outdoors": 0.5, "transport": "car"}


def patch_reply(**fields):
    return reply(json.dumps(fields))


# --- Pure logic -------------------------------------------------------------


def test_merge_applies_only_what_the_message_says():
    slots = Slots(occasion=Occasion.PARTY, city="Pune")

    merged, rejected = merge_slots(slots, IntakePatch(hours_away=4), NOW)

    assert (merged.occasion, merged.city, merged.hours_away) == (Occasion.PARTY, "Pune", 4)
    assert rejected == []


def test_merge_drops_invalid_values_and_keeps_the_rest():
    patch = IntakePatch(start="after dinner", hours_away=30, transport=Transport.CAR)

    merged, rejected = merge_slots(Slots(), patch, NOW)

    assert merged.start is None and merged.hours_away is None
    assert merged.transport is Transport.CAR
    assert rejected == ["start='after dinner'", "hours_away=30.0"]


def test_naming_a_city_turns_weather_back_on():
    merged, _ = merge_slots(Slots(skip_weather=True), IntakePatch(city="Pune"), NOW)

    assert merged.city == "Pune" and not merged.skip_weather


def test_missing_slots_in_asking_order():
    assert missing_slots(Slots()) == [
        "occasion",
        "city",
        "hours_away",
        "hours_outdoors",
        "transport",
    ]
    assert missing_slots(Slots(occasion=Occasion.PARTY, **COMPLETE)) == []


def test_skipping_weather_only_needs_the_occasion():
    assert missing_slots(Slots(skip_weather=True)) == ["occasion"]
    assert missing_slots(Slots(skip_weather=True, occasion=Occasion.OFFICE)) == []


def test_outdoors_longer_than_away_is_asked_again():
    slots = Slots(occasion=Occasion.PARTY, **COMPLETE | {"hours_away": 1, "hours_outdoors": 3})

    assert missing_slots(slots) == ["hours_outdoors"]


def test_question_combines_everything_missing():
    question = question_for(["city", "transport"], Slots(occasion=Occasion.PARTY))

    assert question.startswith("Got it.")
    assert "Which city" in question and "car / cab" in question


def test_question_for_outdoor_time_mentions_the_hours_away():
    question = question_for(["hours_outdoors"], Slots(hours_away=4))

    assert question == "Got it. Of the 4 hours, how much will you spend outdoors?"


def test_start_defaults_to_now_as_a_recorded_assumption():
    slots, assumptions = apply_defaults(Slots(occasion=Occasion.PARTY, **COMPLETE))

    assert slots.start == "now"
    assert assumptions == ["you're leaving now"]


def test_giving_up_falls_back_to_casual_without_weather():
    slots, assumptions = give_up_asking(Slots(city="Pune"), ["occasion", "hours_away"])

    assert slots.occasion is Occasion.CASUAL and slots.skip_weather
    assert len(assumptions) == 2


# --- The intake node ----------------------------------------------------------


def test_intake_sends_known_slots_last_question_and_message():
    client = FakeClient(patch_reply(hours_away=3))
    state = {
        "messages": [
            {"role": "user", "content": "party in Pune"},
            {"role": "assistant", "content": "How long will you be out?"},
            {"role": "user", "content": "about 3 hours"},
        ],
        "slots": Slots(occasion=Occasion.PARTY, city="Pune"),
        "asks": 1,
    }

    update = intake_node(state, client=client, settings=SETTINGS, now=NOW)

    system, user = client.requests[0]["messages"]
    payload = json.loads(user["content"])
    assert payload["message"] == "about 3 hours"
    assert payload["last_question"] == "How long will you be out?"
    assert payload["current_request"]["city"] == "Pune"
    assert payload["now"] == "Saturday 2026-10-17 16:00"
    assert "two_wheeler" in system["content"]  # schema and guidance are in the prompt
    assert update["slots"].hours_away == 3
    assert update["missing"] == ["hours_outdoors", "transport"]


def test_intake_stops_asking_after_max_asks():
    client = FakeClient(patch_reply())  # the user still gave nothing usable
    state = {"messages": [{"role": "user", "content": "dunno"}], "asks": MAX_ASKS}

    update = intake_node(state, client=client, settings=SETTINGS, now=NOW)

    assert update["missing"] == []
    assert update["slots"].occasion is Occasion.CASUAL and update["slots"].skip_weather
    assert any("casual" in a for a in update["assumptions"])


# --- The graph: multi-turn via interrupt / resume ------------------------------


def run(graph, thread: str, message: str | None = None, resume: str | None = None):
    config = {"configurable": {"thread_id": thread}}
    payload = (
        Command(resume=resume)
        if resume is not None
        else {"messages": [{"role": "user", "content": message}]}
    )
    return graph.invoke(payload, config)


def test_complete_first_message_needs_no_question():
    client = FakeClient(
        router_reply(),
        patch_reply(occasion="party", start="19:00", **COMPLETE),
        stylist_reply(),
        critic_reply(),
    )
    graph = build_graph(client, SETTINGS, fake_context())

    out = run(graph, "t", "party in Pune at 7, 4 hours, 30 min outside, cab")

    assert "__interrupt__" not in out
    assert out["missing"] == [] and out["slots"].start == "19:00"


def test_missing_details_are_asked_and_the_answer_resumes_the_run():
    client = FakeClient(
        router_reply(),
        patch_reply(occasion="party", start="19:00"),
        patch_reply(**COMPLETE),
        stylist_reply(),
        critic_reply(),
    )
    graph = build_graph(client, SETTINGS, fake_context())

    first = run(graph, "t", "what should I wear to a party tonight?")
    [question] = first["__interrupt__"]
    second = run(graph, "t", resume="Pune, 4 hours, half an hour outside, by cab")

    assert "Which city" in question.value and "outdoors" in question.value
    assert second["missing"] == [] and second["slots"].transport is Transport.CAR
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]  # ... question, answer, reply
    assert second["messages"][1]["content"] == question.value
    assert len(client.requests) == 5  # router, 2 extractions, stylist, critic: no repeats


def test_threads_keep_separate_state():
    client = FakeClient(
        router_reply(),
        patch_reply(occasion="party"),
        router_reply(),
        patch_reply(occasion="office"),
    )
    graph = build_graph(client, SETTINGS, fake_context())

    run(graph, "a", "party")
    run(graph, "b", "work")

    config = {"configurable": {"thread_id": "a"}}
    assert graph.get_state(config).values["slots"].occasion is Occasion.PARTY


@pytest.mark.parametrize("answers", [MAX_ASKS])
def test_conversation_ends_with_assumptions_after_max_asks(answers):
    client = FakeClient(
        router_reply(),
        *[patch_reply() for _ in range(answers + 1)],
        stylist_reply(),
        critic_reply(),
    )
    graph = build_graph(client, SETTINGS, fake_context())

    out = run(graph, "t", "hmm")
    for _ in range(answers):
        out = run(graph, "t", resume="not sure")

    assert "__interrupt__" not in out
    assert out["slots"].occasion is Occasion.CASUAL and out["slots"].skip_weather
    assert len(client.requests) == MAX_ASKS + 4  # router, each extraction, stylist, critic
    stylist_facts = json.loads(client.requests[-2]["messages"][1]["content"])
    assert any("casual" in a for a in stylist_facts["assumptions"])
