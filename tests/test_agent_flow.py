import json
from datetime import datetime

from langgraph.types import Command

from outfit.agents.graph import build_graph
from outfit.agents.orchestrate import (
    MIN_MATCH_SCORE,
    NO_WEATHER,
    WEATHER_FAILED,
    orchestrate_node,
    resolve_references,
    wardrobe_resolver,
)
from outfit.agents.respond import respond_node, template_reply
from outfit.agents.state import Slots
from outfit.agents.stylist import facts, stylist_node
from outfit.config import Settings
from outfit.db.qdrant_store import SearchHit
from outfit.schemas import Category, Occasion
from outfit.tools.outfits import OutfitPlan
from outfit.tools.weather import Transport
from tests.factories import make_item
from tests.fakes import (
    FakeClient,
    critic_reply,
    fake_context,
    reply,
    router_reply,
    stylist_reply,
)
from tests.test_weather import PUNE, hourly_payload

SETTINGS = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")
CASUAL = [Occasion.CASUAL]
TEE = make_item(subtype="t-shirt", primary_color_name="white", formality=2, occasions=CASUAL)
JEANS = make_item(
    category=Category.BOTTOM,
    subtype="jeans",
    primary_color_name="medium blue",
    primary_color_lab=(49.0, -2.0, -8.0),
    fabric="denim",
    formality=2,
    occasions=CASUAL,
)
WARDROBE = [TEE, JEANS]
TRIP = {"city": "Pune", "start": "now", "hours_away": 3, "hours_outdoors": 1, "transport": "walk"}
HOT_DAY = "Your white t-shirt and medium blue jeans suit the heat; you have no shoes for it yet."


def weather_fetch(url, params):
    """Fake Open-Meteo: geocoding, then a warm, dry forecast for whatever hours are asked."""
    if "geocoding" in url:
        return {"results": [PUNE.model_dump()]}
    start = datetime.fromisoformat(params["start_hour"])
    return hourly_payload([f"{start:%Y-%m-%dT%H:%M}"], apparent_temperature=[29.0])


# --- orchestrate -------------------------------------------------------------


def test_orchestrate_plans_with_weather():
    context = fake_context(WARDROBE, fetch=weather_fetch)
    state = {"slots": Slots(occasion=Occasion.CASUAL, **TRIP)}

    update = orchestrate_node(state, context=context)

    assert update["weather"].place == "Pune, Maharashtra, India"
    assert update["advice"].warmth == (1, 2)
    [outfit] = update["plan"].outfits
    assert {p.id for p in outfit.pieces} == {TEE.id, JEANS.id}


def test_weather_failure_becomes_an_assumption_not_an_error_or_a_gap():
    state = {"slots": Slots(occasion=Occasion.CASUAL, **TRIP)}

    update = orchestrate_node(state, context=fake_context(WARDROBE))  # fetch always fails

    assert update["advice"] is None
    assert update["plan"].outfits  # still planned, just without weather
    assert not any("weather" in g for g in update["plan"].gaps)  # the wardrobe lacks nothing
    assert any(a.startswith(WEATHER_FAILED) for a in update["assumptions"])


def test_skip_weather_never_calls_the_weather_service():
    calls = []
    context = fake_context(WARDROBE, fetch=lambda url, params: calls.append(url))

    orchestrate_node({"slots": Slots(occasion=Occasion.CASUAL, skip_weather=True)}, context=context)

    assert calls == []


def test_planning_without_weather_says_so_once():
    context = fake_context(WARDROBE, fetch=weather_fetch)
    skipped = Slots(occasion=Occasion.CASUAL, skip_weather=True)
    given_up = "no weather check, since the trip details were unclear"  # from intake

    update = orchestrate_node({"slots": skipped}, context=context)
    after_give_up = orchestrate_node({"slots": skipped, "assumptions": [given_up]}, context=context)

    assert update["advice"] is None and update["assumptions"] == [NO_WEATHER]
    assert after_give_up["assumptions"] == [given_up]  # not a second weather note


def test_item_references_are_resolved_and_reported():
    def resolve(ref, items):
        return JEANS if "jeans" in ref else None

    ids, notes = resolve_references(["my blue jeans", "red saree"], WARDROBE, resolve)

    assert ids == {JEANS.id}
    assert notes == [
        "took 'my blue jeans' to mean your medium blue jeans",
        "couldn't find anything matching 'red saree' in your wardrobe",
    ]


def test_an_id_prefix_matches_directly():
    ids, notes = resolve_references([str(TEE.id)[:8]], WARDROBE, lambda ref, items: None)

    assert ids == {TEE.id} and notes == []


def test_must_include_and_exclude_reach_the_planner():
    other_tee = make_item(subtype="polo", primary_color_name="grey", formality=2, occasions=CASUAL)

    def resolve(ref, items):
        return {"polo": other_tee, "white tee": TEE}.get(ref)

    slots = Slots(occasion=Occasion.CASUAL, skip_weather=True, exclude=["white tee"])
    update = orchestrate_node(
        {"slots": slots}, context=fake_context([*WARDROBE, other_tee], resolve=resolve)
    )

    assert all(TEE.id not in {p.id for p in o.pieces} for o in update["plan"].outfits)
    assert "took 'white tee' to mean your white t-shirt" in update["assumptions"]


class ScoredSearch:
    """CLIP stand-in returning fixed hits; records the query."""

    def __init__(self, *hits):
        self.hits = list(hits)
        self.query = None

    def by_text(self, query, limit):
        self.query = query
        return self.hits[:limit]


def test_resolver_trusts_tags_before_clip():
    clip = ScoredSearch(SearchHit(TEE, 0.30), SearchHit(JEANS, 0.20))  # CLIP prefers the tee

    assert wardrobe_resolver(lambda: clip)("My blue jeans", WARDROBE) == JEANS
    assert clip.query is None  # one clear tag match: CLIP was not even asked


def test_resolver_breaks_tag_ties_with_clip():
    dark_tee = make_item(subtype="t-shirt", primary_color_name="black", occasions=CASUAL)
    clip = ScoredSearch(SearchHit(dark_tee, 0.27), SearchHit(TEE, 0.25))

    assert wardrobe_resolver(lambda: clip)("my tee", [TEE, dark_tee]) == dark_tee
    assert clip.query == "tee"  # filler words stripped


def test_resolver_falls_back_to_a_confident_clip_match():
    sure = ScoredSearch(SearchHit(JEANS, MIN_MATCH_SCORE + 0.05))
    unsure = ScoredSearch(SearchHit(JEANS, MIN_MATCH_SCORE - 0.05))

    assert wardrobe_resolver(lambda: sure)("something denim-ish", WARDROBE) == JEANS
    assert wardrobe_resolver(lambda: unsure)("leather jacket", WARDROBE) is None


# --- stylist -------------------------------------------------------------------


def planned_state(**extra):
    update = orchestrate_node(
        {"slots": Slots(occasion=Occasion.CASUAL, **TRIP)},
        context=fake_context(WARDROBE, fetch=weather_fetch),
    )
    return {"slots": Slots(occasion=Occasion.CASUAL, **TRIP), **update, **extra}


def test_facts_contain_only_plan_items_weather_and_assumptions():
    state = planned_state(assumptions=["you're leaving now"])

    f = facts(state)

    assert [p["name"] for p in f["outfits"][0]["pieces"]] == ["white t-shirt", "medium blue jeans"]
    assert f["weather"]["place"] == "Pune, Maharashtra, India"
    assert f["assumptions"] == ["you're leaving now"]
    assert f["request"]["occasion"] == "casual"
    assert "fix_these_issues" not in f
    json.dumps(f)  # plain JSON, ready to send


def test_stylist_sends_facts_and_returns_the_message():
    client = FakeClient(stylist_reply("  Wear your white t-shirt and medium blue jeans.  "))

    update = stylist_node(planned_state(), client=client, settings=SETTINGS)

    assert update == {"draft": "Wear your white t-shirt and medium blue jeans."}
    system, user = client.requests[0]["messages"]
    assert "Only name garments that appear in `outfits`" in system["content"]
    assert json.loads(user["content"])["outfits"][0]["option"] == 1


def test_stylist_failure_leaves_no_draft():
    client = FakeClient(reply("not json"), reply("still not json"))

    assert stylist_node(planned_state(), client=client, settings=SETTINGS) == {"draft": None}


# --- respond ---------------------------------------------------------------------


def test_respond_sends_the_draft_and_resets_the_request_counters():
    update = respond_node({"draft": "Hi!", "asks": 2, "assumptions": ["x"], "missing": []})

    assert update["reply"] == "Hi!"
    assert update["messages"] == [{"role": "assistant", "content": "Hi!"}]
    assert (update["asks"], update["assumptions"], update["draft"]) == (0, [], None)


def test_template_reply_is_built_only_from_the_plan():
    state = planned_state(assumptions=["you're leaving now"])

    text = respond_node({**state, "draft": None})["reply"]

    assert text.startswith("I'd wear: white t-shirt, medium blue jeans.")
    assert "Assumed: you're leaving now." in text


def test_template_reply_for_an_empty_plan_lists_the_gaps():
    plan = OutfitPlan(outfits=[], gaps=["no footwear suits wedding"])

    text = template_reply(plan, [], [])

    assert "couldn't put together an outfit" in text
    assert "Missing: no footwear suits wedding." in text


# --- the whole graph -------------------------------------------------------------


def test_conversation_from_question_to_recommendation():
    client = FakeClient(
        router_reply(),
        reply(json.dumps({"occasion": "casual", "start": "now"})),
        reply(
            json.dumps({"city": "Pune", "hours_away": 3, "hours_outdoors": 1, "transport": "walk"})
        ),
        stylist_reply(HOT_DAY),
        critic_reply(),
    )
    graph = build_graph(client, SETTINGS, fake_context(WARDROBE, fetch=weather_fetch))
    config = {"configurable": {"thread_id": "t"}}

    first = graph.invoke({"messages": [{"role": "user", "content": "casual day out"}]}, config)
    final = graph.invoke(Command(resume="Pune, 3 hours, 1 hour outside, walking"), config)

    assert "__interrupt__" in first
    assert final["reply"] == HOT_DAY
    assert final["slots"].transport is Transport.WALK
    assert final["plan"].outfits and final["advice"] is not None
    assert [m["role"] for m in final["messages"]] == ["user", "assistant", "user", "assistant"]
    assert final["asks"] == 0  # reset for the next request
