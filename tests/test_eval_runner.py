import json
from dataclasses import replace
from datetime import datetime

from outfit.evals.fixtures import weather_fetch
from outfit.evals.runner import format_report, run_scenario, summarize
from outfit.evals.scenarios import Scenario
from tests.fakes import FakeClient, critic_reply, fake_context, reply, router_reply, stylist_reply
from tests.test_agent_flow import SETTINGS, TRIP
from tests.test_router_refine import NO_SHOES, WARDROBE

NOW = datetime(2026, 10, 17, 16, 0)
GOOD = "Your white t-shirt and medium blue jeans." + NO_SHOES  # option 1, gaps passed on
INVENTED = "Your red leather jacket and white t-shirt." + NO_SHOES


def context_for(weather):
    return replace(fake_context(WARDROBE), fetch=weather_fetch(weather), now=NOW)


def scenario(*turns, id="s"):
    return Scenario.model_validate({"id": id, "about": "test", "turns": list(turns)})


ASK_THEN_ANSWER = scenario(
    {"say": "casual day out", "expect": {"intent": "new", "asks": True, "missing": ["city"]}},
    {
        "say": "Pune, 3 hours, 1 outside, walking",
        "expect": {"asks": False, "slots": {"city": "Pune"}, "names_top_outfit": True},
    },
    id="ask_then_answer",
)
ONE_SHOT = scenario(
    {"say": "casual day in Pune, 3 hours, 1 outside, walking", "expect": {"asks": False}},
    id="one_shot",
)
FULL_REQUEST = reply(json.dumps({"occasion": "casual", **TRIP}))


def run(s, *replies):
    return run_scenario(s, FakeClient(*replies), SETTINGS, context_for, NOW)


def test_a_clean_scenario_passes_and_answers_its_question():
    client = FakeClient(
        router_reply(),
        reply(json.dumps({"occasion": "casual"})),
        reply(json.dumps(TRIP)),
        stylist_reply(GOOD),
        critic_reply(),
    )

    result = run_scenario(ASK_THEN_ANSWER, client, SETTINGS, context_for, NOW)

    assert result.passed, [t.failures for t in result.turns]
    asked, answered = result.turns
    assert asked.record["path"] == ["router", "intake"]
    assert answered.record["path"][0] == "ask"  # the second line resumed the question
    assert answered.metrics | {"seconds": 0} == {
        "styled": True,
        "revised": False,
        "template": False,
        "retries": 0,
        "truncated": 0,
        "api_errors": 0,
        "llm_calls": 3,
        "tokens": 0,
        "seconds": 0,
    }


def test_a_scenarios_profile_reaches_the_agents():
    s = Scenario.model_validate(
        {
            "id": "home",
            "about": "test",
            "profile": {"home_city": "Pune"},
            "turns": [{"say": "casual day out", "expect": {"slots": {"city": "Pune"}}}],
        }
    )
    client = FakeClient(router_reply(), reply(json.dumps({"occasion": "casual"})))

    result = run_scenario(s, client, SETTINGS, context_for, NOW)

    assert result.passed, [t.failures for t in result.turns]


def test_a_revision_is_counted():
    result = run(
        ONE_SHOT, router_reply(), FULL_REQUEST, stylist_reply(INVENTED), stylist_reply(GOOD)
    )

    [turn] = result.turns
    assert turn.metrics["revised"] and not turn.metrics["template"]
    assert turn.record["reply"] == GOOD


def test_a_template_fallback_is_counted():
    result = run(
        ONE_SHOT, router_reply(), FULL_REQUEST, stylist_reply(INVENTED), stylist_reply(INVENTED)
    )

    [turn] = result.turns
    assert turn.metrics["revised"] and turn.metrics["template"]
    assert turn.record["reply"].startswith("I'd wear: ")


def test_a_json_retry_is_counted():
    result = run(
        ONE_SHOT,
        router_reply(),
        reply("not json"),
        FULL_REQUEST,
        stylist_reply(GOOD),
        critic_reply(),
    )

    assert result.passed and result.turns[0].metrics["retries"] == 1


def test_a_turn_that_raises_ends_the_scenario_as_failed():
    result = run(ASK_THEN_ANSWER, router_reply())  # no reply queued for intake

    assert not result.passed
    assert len(result.turns) == 1 and result.turns[0].error.startswith("IndexError")


def test_repeated_runs_show_flaky_scenarios():
    chat = scenario({"say": "hi", "expect": {"intent": "chat"}}, id="greet")
    passing = run(chat, router_reply("chat", "Hello!"))
    # The router fails twice, so the message counts as "new"; intake then has no reply
    # queued and the run errors out. Either way, a failed run.
    failing = run(chat, reply("?"), reply("??"))

    summary = summarize([passing, failing, passing])

    assert summary["by_scenario"] == {"greet": {"passed": 2, "runs": 3}}
    assert all(f.startswith("run 2, turn 1: ") for f in summary["failed"]["greet"])
    report = format_report(summary)
    assert "Flaky (passed some runs): greet 2/3" in report
    assert "Failed every run: none" in report


def test_summary_and_report():
    clean = run(
        ASK_THEN_ANSWER,
        router_reply(),
        reply(json.dumps({"occasion": "casual"})),
        reply(json.dumps(TRIP)),
        stylist_reply(GOOD),
        critic_reply(),
    )
    revised = run(
        scenario({"say": "x", "expect": {"intent": "chat"}}, id="wrong_intent"),
        router_reply(),
        FULL_REQUEST,
        stylist_reply(INVENTED),
        stylist_reply(GOOD),
    )

    summary = summarize([clean, revised])

    assert summary["scenarios"] == {"passed": 1, "total": 2}
    assert summary["checks"]["intent"] == {"agent": "router", "passed": 1, "total": 2}
    assert summary["checks"]["slots"] == {"agent": "intake", "passed": 1, "total": 1}
    assert summary["turns"]["styled_replies"] == 2 and summary["turns"]["revised"] == 1
    assert summary["failed"] == {"wrong_intent": ["turn 1: intent: wanted chat, got ['new']"]}
    report = format_report(summary)
    assert "Scenarios passed: 1/2 (50%)" in report
    assert "router       intent            1/2 (50%)" in report
    assert "critic revisions   1/2 (50%)" in report
