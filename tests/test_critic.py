import json
from dataclasses import replace

from outfit.agents.critic import (
    GAP_TOPICS,
    NOTE_TOPICS,
    code_checks,
    critic_node,
    invented_items,
    missing,
    route_after_critic,
)
from outfit.agents.graph import build_graph
from outfit.agents.orchestrate import WEATHER_FAILED
from outfit.agents.respond import respond_node
from outfit.schemas import Category
from outfit.tools.outfits import OutfitPlan
from tests.factories import make_item
from tests.fakes import (
    FakeClient,
    critic_reply,
    fake_context,
    reply,
    router_reply,
    stylist_reply,
)
from tests.test_agent_flow import SETTINGS, TRIP, WARDROBE, planned_state, weather_fetch

GOOD = "Your white t-shirt and medium blue jeans suit the heat; you have no shoes for it yet."
INVENTED = "Wear your white t-shirt with your navy blazer; you have no shoes for it yet."
RAIN = "Rain likely: take an umbrella or a water-resistant layer."


def plan_of(state) -> OutfitPlan:
    return state["plan"]


# --- Code checks ----------------------------------------------------------------


def test_pieces_in_the_plan_pass_in_any_wording():
    plan = plan_of(planned_state())

    draft = "Your white tee and your blue jeans; your top is light and your jeans breathe."

    assert invented_items(draft, plan, []) == []


def test_a_piece_not_in_the_plan_is_caught():
    plan = plan_of(planned_state())

    assert invented_items(INVENTED, plan, []) == ["your navy blazer"]
    assert invented_items("Your red t-shirt is perfect.", plan, []) == ["your red t-shirt"]
    assert invented_items("Pair it with your shoes.", plan, []) == ["your shoes"]  # none planned


def test_suggestions_and_known_references_are_not_claims():
    plan = plan_of(planned_state())
    assumptions = ["couldn't find anything matching 'red saree' in your wardrobe"]

    draft = (
        "A light jacket would help. Your outfit works with sneakers. "
        "I couldn't find your red saree."
    )

    assert invented_items(draft, plan, assumptions) == []


def test_gaps_and_weather_notes_must_be_passed_on():
    gaps = ["no footwear suits casual"]
    notes = [RAIN, "Humid (80%): breathable cotton or linen."]

    assert missing("Your tee and jeans.", gaps, GAP_TOPICS) == gaps
    assert missing("No shoes suit this yet.", gaps, GAP_TOPICS) == []
    assert missing("Bring an umbrella.", notes, NOTE_TOPICS) == []  # humidity is optional
    assert missing("Have fun!", notes, NOTE_TOPICS) == [RAIN]


def test_accessories_are_checked_like_garments():
    belt = make_item(
        category=Category.ACCESSORY, subtype="leather belt", primary_color_name="brown"
    )
    base = plan_of(planned_state()).outfits[0]
    plan = OutfitPlan([replace(base, pieces=(*base.pieces, belt))])

    assert invented_items("Finish it with your brown leather belt.", plan, []) == []
    assert invented_items("Add your silver watch.", plan, []) == ["your silver watch"]


def test_accessory_gaps_must_be_passed_on():
    gaps = ["no sunglasses for the strong sun", "no scarf for the cold", "no tie suits formal"]

    assert missing("Your shirt and trousers.", gaps, GAP_TOPICS) == gaps
    assert missing("Sunglasses, a warm scarf and a tie would help.", gaps, GAP_TOPICS) == []


def test_a_failed_weather_check_must_be_passed_on():
    failed = f"{WEATHER_FAILED} (Could not find a place called 'Banglore'), so this ignores it"
    state = planned_state(assumptions=[failed, "you're leaving now"])

    _, _, left_out = code_checks(state, GOOD)

    assert failed in left_out  # "you're leaving now" has no topic: the rubric checks it
    assert code_checks(state, GOOD + " I couldn't get the forecast.")[2] == []


# --- The node ---------------------------------------------------------------------


def test_clean_draft_goes_to_the_rubric_and_passes():
    client = FakeClient(critic_reply())

    update = critic_node(planned_state(draft=GOOD), client=client, settings=SETTINGS)

    assert update == {"critique": []}
    payload = json.loads(client.requests[0]["messages"][1]["content"])
    assert payload["draft"] == GOOD and payload["facts"]["outfits"]
    # The rubric reasons longest, so it gets its own, larger budget.
    assert (
        client.requests[0]["max_tokens"] == SETTINGS.critic_max_tokens > SETTINGS.agent_max_tokens
    )


def test_code_issues_send_the_draft_back_without_an_llm_call():
    client = FakeClient()

    update = critic_node(planned_state(draft=INVENTED), client=client, settings=SETTINGS)

    assert update["revisions"] == 1
    assert update["critique"] == [
        "'your navy blazer' is not in `outfits`: name only the pieces given there"
    ]
    assert client.requests == []
    assert route_after_critic(update) == "revise"


def test_rubric_issues_send_the_draft_back():
    client = FakeClient(critic_reply("mention that you're leaving now"))

    update = critic_node(planned_state(draft=GOOD), client=client, settings=SETTINGS)

    assert update == {"critique": ["mention that you're leaving now"], "revisions": 1}


def test_a_broken_rubric_accepts_the_draft():
    client = FakeClient(reply("nope"), reply(json.dumps({"ok": False})))  # no issues listed

    update = critic_node(planned_state(draft=GOOD), client=client, settings=SETTINGS)

    assert update == {"critique": []}


def test_after_the_revision_an_invented_piece_drops_the_draft():
    state = planned_state(draft=INVENTED, revisions=1)

    update = critic_node(state, client=FakeClient(), settings=SETTINGS)

    assert update == {"draft": None, "critique": []}
    assert respond_node({**state, **update})["reply"].startswith("I'd wear: white t-shirt")


def test_after_the_revision_left_out_notes_are_appended():
    state = planned_state(draft="Your white t-shirt and medium blue jeans.", revisions=1)

    update = critic_node(state, client=FakeClient(), settings=SETTINGS)

    assert update["draft"].endswith("Missing: no footwear suits casual.")
    assert route_after_critic(update) == "done"


def test_no_draft_passes_straight_through():
    assert critic_node({"draft": None}, client=FakeClient(), settings=SETTINGS) == {"critique": []}


# --- The graph: one revision at most ----------------------------------------------


def run_once(*replies):
    intake = reply(json.dumps({"occasion": "casual", **TRIP}))
    client = FakeClient(router_reply(), intake, *replies)
    graph = build_graph(client, SETTINGS, fake_context(WARDROBE, fetch=weather_fetch))
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "casual day out"}]},
        {"configurable": {"thread_id": "t"}},
    )
    return out, client


def test_invented_piece_is_revised_once_then_sent():
    out, client = run_once(stylist_reply(INVENTED), stylist_reply(GOOD))

    assert out["reply"] == GOOD
    assert len(client.requests) == 4  # router, intake, stylist, stylist again; no rubric
    revision = json.loads(client.requests[3]["messages"][1]["content"])
    assert revision["previous_draft"] == INVENTED
    assert "navy blazer" in revision["fix_these_issues"][0]
    assert (out["revisions"], out["critique"]) == (0, [])  # reset for the next request


def test_still_invented_after_the_revision_falls_back_to_the_template():
    out, client = run_once(stylist_reply(INVENTED), stylist_reply(INVENTED))

    assert out["reply"].startswith("I'd wear: white t-shirt, medium blue jeans.")
    assert "navy blazer" not in out["reply"]
    assert len(client.requests) == 4  # never a second revision


def test_rubric_revision_is_not_reviewed_by_the_rubric_again():
    out, client = run_once(
        stylist_reply(GOOD), critic_reply("mention that you're leaving now"), stylist_reply(GOOD)
    )

    assert out["reply"] == GOOD
    assert len(client.requests) == 5  # router, intake, stylist, rubric, stylist
