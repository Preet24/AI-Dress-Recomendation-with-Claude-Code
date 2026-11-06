"""Respond node: send the stylist's draft, or a plain template reply built from the plan.

The template is the safety net: used when the stylist fails, or when the critic
still finds an invented piece after one revision (it then drops the draft). Less
charming, never wrong, because every word of it comes straight from the plan.
"""

from outfit.agents.state import ChatState
from outfit.tools.outfits import OutfitPlan, label


def template_reply(plan: OutfitPlan | None, notes: list[str], assumptions: list[str]) -> str:
    """A factual reply assembled from the plan, weather notes and assumptions."""
    lines = []
    if plan and plan.outfits:
        best, *others = plan.outfits
        lines.append("I'd wear: " + ", ".join(label(p) for p in best.pieces) + ".")
        lines += [f"- {reason}" for reason in best.reasons]
        lines += [f"- Note: {warning}" for warning in best.warnings]
        for n, outfit in enumerate(others, 2):
            lines.append(f"Option {n}: " + ", ".join(label(p) for p in outfit.pieces) + ".")
    else:
        lines.append("I couldn't put together an outfit for this from your wardrobe.")
    lines += notes
    if plan:
        lines += [f"Missing: {gap}." for gap in plan.gaps]
    if assumptions:
        lines.append("Assumed: " + "; ".join(assumptions) + ".")
    return "\n".join(lines)


def respond_node(state: ChatState) -> dict:
    """Final reply; resets the per-request counters so the next request starts clean."""
    advice = state.get("advice")
    reply = state.get("draft") or template_reply(
        state.get("plan"), advice.notes if advice else [], state.get("assumptions", [])
    )
    return {
        "reply": reply,
        "messages": [{"role": "assistant", "content": reply}],
        "asks": 0,
        "missing": [],
        "assumptions": [],
        "critique": [],
        "revisions": 0,
        "draft": None,
        "found": [],
        "react": [],
        "tool_calls": 0,
    }
