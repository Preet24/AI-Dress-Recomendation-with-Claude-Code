import json

from outfit.agents.graph import build_graph
from outfit.agents.refine import (
    MAX_TOOL_CALLS,
    TOOLS,
    route_after_refine,
    route_after_tools,
    run_tool,
    tools_node,
)
from outfit.agents.router import router_node
from outfit.agents.state import Slots
from outfit.agents.stylist import facts
from outfit.schemas import Category, Occasion
from tests.factories import make_item
from tests.fakes import (
    FakeClient,
    critic_reply,
    fake_context,
    reply,
    router_reply,
    stylist_reply,
    tool_call_reply,
)
from tests.test_agent_flow import CASUAL, JEANS, SETTINGS, TEE, TRIP, planned_state, weather_fetch

CHINOS = make_item(
    category=Category.BOTTOM,
    subtype="chinos",
    primary_color_name="beige",
    primary_color_lab=(75.0, 2.0, 12.0),
    formality=3,
    occasions=CASUAL,
)
WARDROBE = [TEE, JEANS, CHINOS]
JEANS_ID = str(JEANS.id)[:8]
NO_SHOES = " You have no shoes for this yet."


def context():
    return fake_context(WARDROBE, fetch=weather_fetch)


def call(name, **args):
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# --- router -----------------------------------------------------------------------


def test_a_follow_up_without_a_plan_is_a_new_request():
    client = FakeClient(router_reply("follow_up"))

    update = router_node(
        {"messages": [{"role": "user", "content": "with jeans"}]}, client=client, settings=SETTINGS
    )

    assert update["intent"] == "new"
    payload = json.loads(client.requests[0]["messages"][1]["content"])
    assert payload == {"message": "with jeans", "previous": None}


def test_a_new_request_clears_the_previous_one():
    state = {**planned_state(), "messages": [{"role": "user", "content": "wedding on Sunday?"}]}

    update = router_node(state, client=FakeClient(router_reply("new")), settings=SETTINGS)

    assert update["slots"] == Slots() and update["plan"] is None and update["advice"] is None
    assert update["previous_slots"].city == "Pune"  # kept aside for "same city as before"


def test_intake_sees_the_previous_request_for_back_references():
    graph, client, config = first_turn(
        router_reply("new"), reply(json.dumps({"occasion": "office", "start": "tomorrow 9:00"}))
    )

    out = say(graph, config, "what about office tomorrow at 9")

    payload = json.loads(client.requests[-1]["messages"][1]["content"])
    assert payload["previous_request"]["city"] == "Pune"
    assert payload["current_request"]["city"] is None  # nothing carried over by code
    assert payload["last_question"] is None  # the last reply was not a question
    assert "Which city" in out["__interrupt__"][0].value


def test_follow_up_keeps_the_plan_and_shows_it_to_the_model():
    client = FakeClient(router_reply("follow_up"))
    state = {**planned_state(), "messages": [{"role": "user", "content": "less formal?"}]}

    assert router_node(state, client=client, settings=SETTINGS) == {"intent": "follow_up"}
    previous = json.loads(client.requests[0]["messages"][1]["content"])["previous"]
    assert previous["top_outfit"] == ["white t-shirt", "medium blue jeans"]


def test_chat_is_answered_by_the_router():
    client = FakeClient(router_reply("chat"), router_reply("chat", "Hi! Tell me the occasion."))

    update = router_node(
        {"messages": [{"role": "user", "content": "hi"}]}, client=client, settings=SETTINGS
    )

    assert update == {"intent": "chat", "draft": "Hi! Tell me the occasion."}
    assert len(client.requests) == 2  # chat without a reply was retried


def test_a_failed_router_treats_the_message_as_new():
    client = FakeClient(reply("?"), reply("??"))

    update = router_node(
        {"messages": [{"role": "user", "content": "x"}]}, client=client, settings=SETTINGS
    )

    assert update["intent"] == "new"


# --- refine tools --------------------------------------------------------------------


def test_tool_schemas_are_self_contained():
    names = [t["function"]["name"] for t in TOOLS]

    assert names == ["search_wardrobe", "similar_items", "update_request", "show_items"]
    assert "$ref" not in json.dumps(TOOLS) and "$defs" not in json.dumps(TOOLS)


def test_search_returns_short_ids_and_a_note_when_nothing_matches():
    state = planned_state()

    found, no_change = run_tool(
        "search_wardrobe", '{"query": "blue jeans"}', state, context(), None
    )
    missing, update = run_tool("search_wardrobe", '{"query": "red saree"}', state, context(), None)

    assert found["items"][0] == {**found["items"][0], "id": JEANS_ID, "weak": False}
    assert no_change == {}
    assert all(i["weak"] for i in missing["items"])
    assert "couldn't find anything matching 'red saree' in your wardrobe" in update["assumptions"]


def test_invalid_arguments_go_back_to_the_model():
    state = planned_state()

    bad_json, update = run_tool("update_request", "{not json", state, context(), None)
    bad_value, _ = run_tool("search_wardrobe", '{"query": ""}', state, context(), None)
    unknown, _ = run_tool("get_weather", "{}", state, context(), None)

    assert "invalid arguments" in bad_json["error"] and update == {}
    assert "invalid arguments" in bad_value["error"]
    assert "unknown tool" in unknown["error"]


def test_update_request_replans_with_the_change():
    state = planned_state()

    result, update = run_tool(
        "update_request", json.dumps({"must_include": [JEANS_ID]}), state, context(), None
    )

    assert update["slots"].must_include == [JEANS_ID]
    assert all(JEANS.id in {p.id for p in o.pieces} for o in update["plan"].outfits)
    assert result["outfits"][0]["pieces"][1]["id"] == JEANS_ID
    assert update["advice"] is not None  # re-planned with the weather


def test_show_items_puts_them_in_front_of_the_user():
    result, update = run_tool(
        "show_items",
        json.dumps({"item_ids": [JEANS_ID, "ffffffff"]}),
        planned_state(),
        context(),
        None,
    )

    assert update["found"] == [JEANS]
    assert result == {"shown": ["medium blue jeans"], "unknown_ids": ["ffffffff"]}


def test_tools_stop_at_the_budget():
    calls = [
        call("search_wardrobe", query="jeans") | {"id": f"c{n}"} for n in range(MAX_TOOL_CALLS + 1)
    ]
    state = {
        **planned_state(),
        "react": [{"role": "assistant", "content": "", "tool_calls": calls}],
    }

    update = tools_node(state, context=context())

    results = [json.loads(m["content"]) for m in update["react"][1:]]
    assert update["tool_calls"] == MAX_TOOL_CALLS
    assert [m["tool_call_id"] for m in update["react"][1:]] == [c["id"] for c in calls]
    assert "budget" in results[-1]["error"]
    assert route_after_tools(update) == "done"


def test_refine_routes_on_tool_calls():
    assert (
        route_after_refine({"react": [{"role": "assistant", "tool_calls": [call("x")]}]}) == "tools"
    )
    assert route_after_refine({"react": [{"role": "assistant", "content": "done"}]}) == "done"
    assert route_after_refine({}) == "done"


def test_stylist_sees_the_follow_up_and_the_items_to_show():
    state = {
        **planned_state(found=[CHINOS]),
        "messages": [{"role": "user", "content": "my bottoms?"}],
    }

    f = facts(state)

    assert f["user_message"] == "my bottoms?"
    assert f["items_to_show"] == [{"name": "beige chinos", "category": "bottom"}]


# --- the whole graph: request, follow-up, chat ----------------------------------------


def first_turn(*later):
    """A graph whose first turn (casual, Pune, white tee) has run; `later` replies follow."""
    client = FakeClient(
        router_reply(),
        reply(json.dumps({"occasion": "casual", **TRIP})),
        stylist_reply("Your white t-shirt and beige chinos." + NO_SHOES),
        critic_reply(),
        *later,
    )
    graph = build_graph(client, SETTINGS, context())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [{"role": "user", "content": "casual day out"}]}, config)
    return graph, client, config


def say(graph, config, text):
    return graph.invoke({"messages": [{"role": "user", "content": text}]}, config)


def test_follow_up_runs_the_react_loop_and_replans():
    with_jeans = "Your white t-shirt and medium blue jeans instead." + NO_SHOES
    graph, client, config = first_turn(
        router_reply("follow_up"),
        tool_call_reply(("search_wardrobe", {"query": "blue jeans"})),
        tool_call_reply(("update_request", {"must_include": [JEANS_ID]})),
        reply("Re-planned with the blue jeans."),
        stylist_reply(with_jeans),
        critic_reply(),
    )

    out = say(graph, config, "something with my blue jeans instead")

    assert out["reply"] == with_jeans
    assert out["slots"].occasion is Occasion.CASUAL  # the request was kept, then changed
    assert out["slots"].must_include == [JEANS_ID]
    assert all(JEANS.id in {p.id for p in o.pieces} for o in out["plan"].outfits)
    refine_calls = [r for r in client.requests if "tools" in r]
    assert len(refine_calls) == 3  # two tool steps and the closing answer
    assert refine_calls[-1]["messages"][-1]["role"] == "tool"  # results went back to the model
    assert (out["tool_calls"], out["react"]) == (0, [])  # reset for the next message


def test_show_items_follow_up_reaches_the_stylist_and_passes_the_critic():
    shown = "Your bottoms: your medium blue jeans and beige chinos." + NO_SHOES
    graph, client, config = first_turn(
        router_reply("follow_up"),
        tool_call_reply(("show_items", {"item_ids": [JEANS_ID, str(CHINOS.id)[:8]]})),
        reply("Showed the bottoms."),
        stylist_reply(shown),
        critic_reply(),
    )

    out = say(graph, config, "show me my bottoms")

    assert out["reply"] == shown
    stylist_facts = json.loads(client.requests[-2]["messages"][1]["content"])
    assert [i["name"] for i in stylist_facts["items_to_show"]] == [
        "medium blue jeans",
        "beige chinos",
    ]


def test_chat_replies_without_touching_the_request():
    graph, client, config = first_turn(router_reply("chat", "You're welcome!"))
    before = len(client.requests)

    out = say(graph, config, "thanks!")

    assert out["reply"] == "You're welcome!"
    assert len(client.requests) == before + 1  # just the router
    assert out["plan"].outfits and out["slots"].occasion is Occasion.CASUAL
