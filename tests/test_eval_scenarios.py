import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from outfit.agents.graph import build_graph
from outfit.agents.recording import RecordingClient, run_turn
from outfit.evals.checks import check_turn, mentions, slot_matches, tools_called
from outfit.evals.scenarios import Expect, Scenario, load_scenarios
from tests.fakes import (
    FakeClient,
    critic_reply,
    reply,
    router_reply,
    stylist_reply,
    tool_call_reply,
)
from tests.test_agent_flow import SETTINGS, TRIP
from tests.test_router_refine import CHINOS, JEANS_ID, NO_SHOES, context

NOW = datetime(2026, 10, 17, 16, 0)
ANSWER = "Your white t-shirt and medium blue jeans." + NO_SHOES  # option 1 of the plan


def test_the_scenario_file_is_valid():
    scenarios = load_scenarios()

    assert scenarios.now == NOW
    assert len(scenarios.scenarios) >= 15
    assert {"intake", "router", "refine", "weather"} <= {
        tag for s in scenarios.scenarios for tag in s.tags
    }


@pytest.mark.parametrize(
    "expect",
    [
        {"slots": {"town": "Pune"}},  # not a slot
        {"tools": ["get_weather"]},  # not a tool
        {"missing": ["venue"]},
        {"reply_contains": ["x"]},  # misspelled expectation
    ],
)
def test_typos_in_expectations_are_errors(expect):
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {"id": "x", "about": "x", "turns": [{"say": "hi", "expect": expect}]}
        )


@pytest.mark.parametrize(
    ("name", "actual", "wanted", "ok"),
    [
        ("start", "tomorrow 9:00", "tomorrow 09:00", True),
        ("start", "19:00", "18:00", False),
        ("start", "soon", "18:00", False),
        ("hours_outdoors", 0.33, [0.3, 0.35], True),
        ("hours_outdoors", 0.5, 0.52, True),
        ("hours_away", 4.0, 5, False),
        ("formality", (4, 5), [4, 5], True),
        ("city", "pune", "Pune", True),
        ("occasion", "travel", ["casual", "travel"], True),
        ("occasion", "party", ["casual", "travel"], False),
        ("skip_weather", True, True, True),
        ("city", None, "Pune", False),
    ],
)
def test_slot_matching(name, actual, wanted, ok):
    assert slot_matches(name, actual, wanted, NOW) is ok


def test_mentions_whole_words_and_alternatives():
    assert mentions("Bring a light Jacket.", "coat|jacket")
    assert not mentions("A rainbow print", "rain")


def asked_then_answered():
    """A recorded two-turn conversation: question, then a full recommendation."""
    client = RecordingClient(
        FakeClient(
            router_reply(),
            reply(json.dumps({"occasion": "casual"})),
            reply(json.dumps(TRIP)),
            stylist_reply(ANSWER),
            critic_reply(),
        )
    )
    graph = build_graph(client, SETTINGS, context())
    config = {"configurable": {"thread_id": "t"}}
    first = run_turn(graph, config, "casual day out", resume=False, client=client)
    first_state = graph.get_state(config).values
    second = run_turn(graph, config, "Pune, 3 hours, 1 outside, walking", True, client)
    return (first, first_state), (second, graph.get_state(config).values)


def test_expectations_that_hold_pass():
    (first, first_state), (second, state) = asked_then_answered()

    asked = Expect(intent="new", asks=True, missing=["city", "transport"])
    answered = Expect(
        asks=False,
        slots={"occasion": "casual", "city": "Pune", "hours_away": 3, "transport": "walk"},
        reply_mentions=["t-shirt", "shoes|footwear"],
        reply_avoids=["chinos"],
        plan_includes=["t-shirt"],
        names_top_outfit=True,
    )

    assert check_turn(asked, first.record, first_state, NOW) == []
    assert check_turn(answered, second.record, state, NOW) == []


def test_expectations_that_fail_say_why():
    (first, first_state), (second, state) = asked_then_answered()

    failures = check_turn(
        Expect(intent="chat", asks=False, missing=["occasion"]), first.record, first_state, NOW
    )
    assert failures[0] == "intent: wanted chat, got ['new']"
    assert failures[1].startswith("asks: wanted a reply, got question 'Got it. Which city")
    assert failures[2] == (
        "missing: should ask for ['occasion'], asked for "
        "['city', 'hours_away', 'hours_outdoors', 'transport']"
    )

    failures = check_turn(
        Expect(
            intent="new",  # a resumed turn has no router step
            slots={"transport": "car"},
            tools=["update_request"],
            reply_mentions=["linen"],
            plan_excludes=["chinos"],
        ),
        second.record,
        state,
        NOW,
    )
    assert failures[0] == "intent: wanted new, got no router step"
    assert failures[1] == "slot transport: wanted 'car', got <Transport.WALK: 'walk'>"
    assert failures[2] == "tools: ['update_request'] not called (called [])"
    assert failures[3] == "reply does not mention 'linen'"
    assert failures[4].startswith("an outfit has 'chinos'")


def test_follow_up_tools_and_shown_items_are_checked():
    client = RecordingClient(
        FakeClient(
            router_reply(),
            reply(json.dumps({"occasion": "casual", **TRIP})),
            stylist_reply(ANSWER),
            critic_reply(),
            router_reply("follow_up"),
            tool_call_reply(("show_items", {"item_ids": [JEANS_ID, str(CHINOS.id)[:8]]})),
            reply("Shown."),
            stylist_reply("Your bottoms: your medium blue jeans and beige chinos." + NO_SHOES),
            critic_reply(),
        )
    )
    graph = build_graph(client, SETTINGS, context())
    config = {"configurable": {"thread_id": "t"}}
    run_turn(graph, config, "casual day out", resume=False, client=client)

    turn = run_turn(graph, config, "show me my bottoms", resume=False, client=client)
    state = graph.get_state(config).values

    assert tools_called(turn.record) == ["show_items"]
    assert (
        check_turn(
            Expect(intent="follow_up", tools=["show_items"], found_min=2), turn.record, state, NOW
        )
        == []
    )
    assert check_turn(Expect(found_min=3), turn.record, state, NOW) == [
        "found: wanted at least 3 items shown, got 2"
    ]
