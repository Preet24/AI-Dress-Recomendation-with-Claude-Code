"""Run eval scenarios through the real graph and summarize the results per agent.

Each scenario gets a fresh graph, thread and context (its weather fixture, the fixed
clock), so scenarios are independent and can run in parallel. A turn answers the
previous turn's question when there was one, exactly like a person in `chat.py`.
Every turn is recorded with `run_turn`, checked with `check_results`, and measured:
critic revisions, template fallbacks, JSON retries, truncated replies, API errors,
LLM calls, tokens and time.
"""

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from statistics import mean

from openai import OpenAI

from outfit.agents.graph import build_graph
from outfit.agents.orchestrate import Context
from outfit.agents.recording import RecordingClient, run_turn
from outfit.config import Settings
from outfit.evals.checks import KINDS, CheckResult, check_results
from outfit.evals.scenarios import Scenario, Weather

logger = logging.getLogger(__name__)

JSON_NODES = {"router", "intake", "stylist", "critic"}  # chat_json: a 2nd call is a retry


@dataclass
class TurnResult:
    say: str
    record: dict
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: str | None = None  # the turn raised: the scenario stopped here

    @property
    def failures(self) -> list[str]:
        return [c.detail for c in self.checks if not c.ok] + ([self.error] if self.error else [])


@dataclass
class ScenarioResult:
    scenario: Scenario
    turns: list[TurnResult]

    @property
    def passed(self) -> bool:
        return not any(t.failures for t in self.turns)


def turn_metrics(record: dict) -> dict:
    """What a turn cost and how the reply was made, from its recording."""
    path = record["path"]
    stylist_runs = path.count("stylist")
    drafts = [
        s["update"]["draft"]
        for s in record["steps"]
        if s["node"] in ("stylist", "critic") and "draft" in s["update"]
    ]
    calls = [c for s in record["steps"] for c in s["llm"]]
    return {
        "styled": stylist_runs > 0,  # the stylist wrote this reply (not a question or chat)
        "revised": stylist_runs > 1,  # the critic sent the first draft back
        "template": stylist_runs > 0 and (not drafts or drafts[-1] is None),
        "retries": sum(1 for s in record["steps"] if s["node"] in JSON_NODES and len(s["llm"]) > 1),
        "truncated": sum(c.get("finish_reason") == "length" for c in calls),
        "api_errors": sum(bool(c.get("error")) for c in calls),
        "llm_calls": record["llm_calls"],
        "tokens": record["tokens"],
        "seconds": record["seconds"],
    }


def run_scenario(
    scenario: Scenario,
    client: OpenAI,
    settings: Settings,
    context_for: Callable[[Weather], Context],
    now: datetime,
) -> ScenarioResult:
    """Play one scenario on a fresh graph; a turn that raises ends the scenario."""
    recording = RecordingClient(client)
    context = replace(context_for(scenario.weather), profile=lambda: scenario.profile)
    graph = build_graph(recording, settings, context)
    config = {"configurable": {"thread_id": scenario.id}}
    turns: list[TurnResult] = []
    asked = False
    for turn in scenario.turns:
        try:
            result = run_turn(graph, config, turn.say, resume=asked, client=recording)
        except Exception as err:  # report it and go on with the next scenario
            logger.exception("Scenario %s failed on %r", scenario.id, turn.say)
            turns.append(TurnResult(turn.say, {}, error=f"{type(err).__name__}: {err}"))
            break
        state = graph.get_state(config).values
        turns.append(
            TurnResult(
                turn.say,
                result.record,
                check_results(turn.expect, result.record, state, now),
                turn_metrics(result.record),
            )
        )
        asked = result.question is not None
    return ScenarioResult(scenario, turns)


# --- Report ------------------------------------------------------------------------


def summarize(results: list[ScenarioResult]) -> dict:
    """Pass rates per check kind and per scenario, and the turn metrics, as JSON data."""
    passed: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for check in (c for r in results for t in r.turns for c in t.checks):
        total[check.kind] += 1
        passed[check.kind] += check.ok

    metrics = [t.metrics for r in results for t in r.turns if t.metrics]
    styled = [m for m in metrics if m["styled"]]
    seconds = [m["seconds"] for m in metrics]
    return {
        "scenarios": {"passed": sum(r.passed for r in results), "total": len(results)},
        "checks": {
            kind: {"agent": KINDS[kind], "passed": passed[kind], "total": total[kind]}
            for kind in KINDS
            if total[kind]
        },
        "turns": {
            "total": len(metrics),
            "styled_replies": len(styled),
            "revised": sum(m["revised"] for m in styled),
            "template": sum(m["template"] for m in styled),
            "json_retries": sum(m["retries"] for m in metrics),
            "truncated": sum(m["truncated"] for m in metrics),
            "api_errors": sum(m["api_errors"] for m in metrics),
            "turn_errors": sum(bool(t.error) for r in results for t in r.turns),
            "llm_calls": sum(m["llm_calls"] for m in metrics),
            "tokens": sum(m["tokens"] for m in metrics),
            "seconds_mean": round(mean(seconds), 1) if seconds else 0.0,
            "seconds_max": max(seconds, default=0.0),
        },
        "by_scenario": by_scenario(results),
        "failed": failures_by_scenario(results),
    }


def by_scenario(results: list[ScenarioResult]) -> dict[str, dict[str, int]]:
    """Runs and passes per scenario id (more than one run with --repeat)."""
    counts: dict[str, dict[str, int]] = {}
    for r in results:
        c = counts.setdefault(r.scenario.id, {"passed": 0, "runs": 0})
        c["runs"] += 1
        c["passed"] += r.passed
    return counts


def failures_by_scenario(results: list[ScenarioResult]) -> dict[str, list[str]]:
    """Failures per scenario id; prefixed with the run number when it ran more than once."""
    runs = Counter(r.scenario.id for r in results)
    seen: Counter[str] = Counter()
    failed: dict[str, list[str]] = {}
    for r in results:
        seen[r.scenario.id] += 1
        prefix = f"run {seen[r.scenario.id]}, " if runs[r.scenario.id] > 1 else ""
        for n, t in enumerate(r.turns, 1):
            for f in t.failures:
                failed.setdefault(r.scenario.id, []).append(f"{prefix}turn {n}: {f}")
    return failed


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"


def format_report(summary: dict) -> str:
    """The summary as a short plain-text report."""
    s, t = summary["scenarios"], summary["turns"]
    lines = [f"Scenarios passed: {pct(s['passed'], s['total'])}", "", "Checks by agent:"]
    for kind, c in summary["checks"].items():
        lines.append(f"  {c['agent']:<12} {kind:<17} {pct(c['passed'], c['total'])}")
    lines += [
        "",
        f"Turns: {t['total']} ({t['styled_replies']} styled replies)",
        f"  critic revisions   {pct(t['revised'], t['styled_replies'])}",
        f"  template fallbacks {pct(t['template'], t['styled_replies'])}",
        f"  JSON retries {t['json_retries']} · truncated replies {t['truncated']} · "
        f"API errors {t['api_errors']} · turn errors {t['turn_errors']}",
        f"  LLM calls {t['llm_calls']} · tokens {t['tokens']} · "
        f"{t['seconds_mean']}s per turn (max {t['seconds_max']}s)",
    ]
    flaky = [
        f"{sid} {c['passed']}/{c['runs']}"
        for sid, c in summary["by_scenario"].items()
        if 0 < c["passed"] < c["runs"]
    ]
    never = [sid for sid, c in summary["by_scenario"].items() if c["passed"] == 0]
    if any(c["runs"] > 1 for c in summary["by_scenario"].values()):
        lines += ["", f"Flaky (passed some runs): {', '.join(flaky) or 'none'}"]
        lines.append(f"Failed every run: {', '.join(never) or 'none'}")
    if summary["failed"]:
        lines += ["", "Failures:"]
        for scenario_id, failures in summary["failed"].items():
            lines.append(f"  {scenario_id}")
            lines += [f"    - {f}" for f in failures]
    return "\n".join(lines)
