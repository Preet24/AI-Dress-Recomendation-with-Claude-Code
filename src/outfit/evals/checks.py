"""Check one recorded turn against its scenario expectations.

Works on what `recording.run_turn` captured (the per-node updates, since respond
resets the per-turn fields) plus the graph state after the turn (slots and plan
survive it). `check_results` gives one result per check, passed or not, so a report
can show accuracy per kind; `check_turn` is just the failures, in plain words.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from outfit.agents.state import Slots
from outfit.evals.scenarios import Expect
from outfit.schemas import ClothingItem
from outfit.tools.outfits import OutfitPlan, label
from outfit.tools.weather import parse_start

FLOAT_TOLERANCE = 0.05

# Check kind -> the agent it measures, for the report.
KINDS = {
    "intent": "router",
    "asks": "intake",
    "missing": "intake",
    "slots": "intake",
    "plan": "orchestrate",
    "tools": "refine",
    "found": "refine",
    "reply": "stylist",
    "names_top_outfit": "stylist",
}


@dataclass(frozen=True)
class CheckResult:
    kind: str  # a key of KINDS
    ok: bool
    detail: str  # what was checked, or why it failed


def updates(record: dict, node: str) -> list[dict]:
    """State updates of every run of `node` in the turn, in order."""
    return [s["update"] for s in record["steps"] if s["node"] == node]


def tools_called(record: dict) -> list[str]:
    """Tool names refine asked for, in order."""
    names = []
    for step in record["steps"]:
        if step["node"] != "refine":
            continue
        for call in step["llm"]:
            for tool_call in (call.get("response") or {}).get("tool_calls") or []:
                names.append(tool_call["function"]["name"])
    return names


def mentions(text: str, term: str) -> bool:
    """Whole-word, case-insensitive; "a|b" matches either."""
    return any(
        re.search(rf"\b{re.escape(alt.strip())}\b", text, re.IGNORECASE) for alt in term.split("|")
    )


def slot_matches(name: str, actual: Any, wanted: Any, now: datetime) -> bool:
    if actual is None or wanted is None:
        return actual is wanted
    if name == "start":
        try:
            return parse_start(str(actual), now) == parse_start(str(wanted), now)
        except ValueError:
            return False
    if isinstance(actual, float | int) and not isinstance(actual, bool):
        if isinstance(wanted, list):
            return wanted[0] <= actual <= wanted[1]
        return abs(actual - wanted) <= FLOAT_TOLERANCE
    if isinstance(actual, tuple | list):
        return list(actual) == list(wanted)
    if isinstance(actual, str):  # also the str enums; a list = any of these
        options = wanted if isinstance(wanted, list) else [wanted]
        return actual.lower() in {str(o).lower() for o in options}
    return actual == wanted


def names_piece(reply: str, item: ClothingItem) -> bool:
    """The reply names a piece by its subtype's last word ("pumps") or its color's first
    ("champagne"): the stylist may say "champagne heels" for "champagne slingback heel pumps"."""
    words = (item.subtype.split()[-1], item.primary_color_name.split()[0])
    return any(mentions(reply, w) for w in words)


def check_results(expect: Expect, record: dict, state: dict, now: datetime) -> list[CheckResult]:
    """One result per expectation: `record` from run_turn, `state` the graph values after it."""
    results: list[CheckResult] = []

    def check(kind: str, ok: bool, failure: str, passed: str) -> None:
        results.append(CheckResult(kind, ok, passed if ok else failure))

    reply = record.get("reply") or ""
    plan: OutfitPlan | None = state.get("plan")
    outfits = [[label(p).lower() for p in o.pieces] for o in plan.outfits] if plan else []

    if expect.intent is not None:
        routed = [u.get("intent") for u in updates(record, "router")]
        check(
            "intent",
            routed == [expect.intent],
            f"intent: wanted {expect.intent}, got {routed or 'no router step'}",
            f"intent {expect.intent}",
        )

    if expect.asks is not None:
        question = record.get("question")
        got = f"question {question!r}" if question else "a reply"
        check(
            "asks",
            (question is not None) == expect.asks,
            f"asks: wanted {'a question' if expect.asks else 'a reply'}, got {got}",
            "asked" if expect.asks else "replied",
        )

    if expect.missing:
        intakes = updates(record, "intake")
        missing = intakes[-1].get("missing", []) if intakes else []
        lacking = [m for m in expect.missing if m not in missing]
        check(
            "missing",
            not lacking,
            f"missing: should ask for {lacking}, asked for {missing}",
            f"asked for {expect.missing}",
        )

    slots: Slots = state.get("slots") or Slots()
    for name, wanted in expect.slots.items():
        actual = getattr(slots, name)
        check(
            "slots",
            slot_matches(name, actual, wanted, now),
            f"slot {name}: wanted {wanted!r}, got {actual!r}",
            f"slot {name}",
        )

    if expect.tools:
        called = tools_called(record)
        lacking = [t for t in expect.tools if t not in called]
        check(
            "tools",
            not lacking,
            f"tools: {lacking} not called (called {called})",
            f"called {expect.tools}",
        )

    for term in expect.reply_mentions:
        check("reply", mentions(reply, term), f"reply does not mention {term!r}", term)
    for term in expect.reply_avoids:
        check("reply", not mentions(reply, term), f"reply mentions {term!r}", f"no {term}")

    for term in expect.plan_includes:
        ok = bool(outfits) and all(any(term in name for name in o) for o in outfits)
        check("plan", ok, f"not every outfit has {term!r}: {outfits}", f"all have {term}")
    for term in expect.plan_excludes:
        ok = not any(term in name for o in outfits for name in o)
        check("plan", ok, f"an outfit has {term!r}: {outfits}", f"none has {term}")

    if expect.found_min is not None:
        shown = max((len(u.get("found", [])) for u in updates(record, "tools")), default=0)
        check(
            "found",
            shown >= expect.found_min,
            f"found: wanted at least {expect.found_min} items shown, got {shown}",
            f"{shown} shown",
        )

    if expect.names_top_outfit:
        if not plan or not plan.outfits:
            check("names_top_outfit", False, "names_top_outfit: the plan has no outfit", "")
        else:
            unnamed = [label(p) for p in plan.outfits[0].pieces if not names_piece(reply, p)]
            check(
                "names_top_outfit",
                not unnamed,
                f"reply does not name option 1's {unnamed}",
                "named option 1",
            )
    return results


def check_turn(expect: Expect, record: dict, state: dict, now: datetime) -> list[str]:
    """Failures of one turn, in plain words; empty means it passed."""
    return [r.detail for r in check_results(expect, record, state, now) if not r.ok]
