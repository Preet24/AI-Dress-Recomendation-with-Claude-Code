"""Critic agent: reflection on the stylist's draft, with a hard limit of one revision.

1. Code checks (free, deterministic): every wardrobe piece the draft calls "your ..."
   is in the facts the stylist got; every wardrobe gap and every rain / UV / wind
   note is passed on.
2. LLM rubric (DeepSeek JSON, first pass only, and only if the code checks pass):
   answers the request, passes on warnings, does not contradict the facts, right tone.
3. Issues on the first pass go back to the stylist once (`critique`, `revisions`).
   After the revision nothing is sent back again: a draft that still invents a piece
   is dropped (respond falls back to the template), and notes or gaps it still leaves
   out are appended to it verbatim.

The checks are word-level heuristics, tuned to be strict about "your <garment>" (the
stylist's convention for the user's own pieces, see stylist.md) and lenient elsewhere,
so a suggestion for a gap ("a light jacket would help") is never flagged.
"""

import json
import logging
import re
from collections.abc import Iterable

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from outfit.agents.llm import LLMError, chat_json, load_prompt
from outfit.agents.state import ChatState
from outfit.agents.stylist import facts
from outfit.config import Settings
from outfit.schemas import Category
from outfit.tools.outfits import OutfitPlan, label

logger = logging.getLogger(__name__)

# Words that name a garment. A "your <...> garment" phrase must match the facts.
GARMENT_NOUNS = frozenset(
    "shirt t-shirt tee top blouse tunic polo sweater sweatshirt hoodie cardigan vest tank "
    "camisole crop kurta kurti jeans trousers pants chinos shorts skirt leggings joggers "
    "palazzo salwar churidar dhoti dress gown jumpsuit romper saree sari lehenga anarkali "
    "jacket blazer coat overcoat parka raincoat windbreaker shrug shawl dupatta stole scarf "
    "shoes shoe sneakers trainers heels pumps sandals boots loafers flats slippers "
    "juttis mojaris footwear outerwear bottoms bottom tops "
    # accessories (tools/accessories.KIND_WORDS)
    "belt tie necktie bowtie bag handbag purse clutch tote backpack satchel sunglasses shades "
    "hat cap beanie fedora muffler watch smartwatch jewellery jewelry necklace earrings "
    "bracelet bangles ring pendant jhumkas anklet brooch".split()
)
# Generic words: fine whenever the plan has a piece of that kind.
GENERIC_NOUNS: dict[str, frozenset[Category]] = {
    "top": frozenset({Category.TOP, Category.ETHNIC_TOP}),
    "bottom": frozenset({Category.BOTTOM, Category.ETHNIC_BOTTOM}),
    "shoe": frozenset({Category.FOOTWEAR}),
    "footwear": frozenset({Category.FOOTWEAR}),
    "outerwear": frozenset({Category.OUTERWEAR}),
}
SYNONYMS = {"tee": "t-shirt", "tshirt": "t-shirt", "pant": "trouser", "sari": "saree"}
COLOR_WORDS = frozenset(
    "black white grey gray silver red maroon burgundy wine pink blush rose magenta orange "
    "rust peach coral yellow mustard gold golden green olive mint teal turquoise blue navy "
    "cobalt sky indigo purple violet lavender lilac mauve plum brown tan beige cream ivory "
    "khaki camel champagne nude off-white charcoal".split()
)
# A "your ..." phrase ends at its garment noun, at most this many words after "your",
# and never crosses one of these words (so "your outfit works with shoes" is no claim).
PHRASE_WORDS = 4
PHRASE_BREAKS = frozenset(
    "and or with but for to the a an is are was will would can could should it "
    "that this so as if".split()
)

# Plan text -> words the draft must contain (as word prefixes) to count as passed on.
GAP_TOPICS = (
    ("no footwear", ("shoe", "footwear", "heel", "sneaker", "sandal", "boot", "flat", "loafer")),
    ("no outerwear", ("jacket", "coat", "layer", "outerwear", "blazer", "cardigan", "shawl")),
    # tools/accessories.missing
    ("no sunglasses", ("sunglass", "shades")),
    ("no scarf", ("scarf", "stole", "muffler", "shawl")),
    ("no tie", ("tie",)),
)
# Assumptions the draft must pass on (orchestrate.WEATHER_FAILED), checked like notes.
ASSUMPTION_TOPICS = (("couldn't check the weather", ("weather", "forecast")),)
NOTE_TOPICS = (
    ("rain", ("rain", "umbrella", "wet", "shower")),
    ("uv ", ("uv", "sun")),
    ("windy", ("wind", "gust", "breez", "flowy")),
    ("flowy", ("flowy", "flow", "catch", "long", "wheel")),
)

WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
CLAUSE = re.compile(r"[.,;:!?()\n\"]")


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    issues: list[str] = Field(default_factory=list, max_length=5)  # instructions to the stylist

    @model_validator(mode="after")
    def issues_when_not_ok(self) -> "CriticVerdict":
        if not self.ok and not self.issues:
            raise ValueError("ok is false, so list the issues to fix")
        return self


# --- Code checks --------------------------------------------------------------


def stem(word: str) -> str:
    """Rough singular, applied to both the draft and the facts: heels -> heel."""
    if word.endswith(("sses", "shes", "ches")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    return SYNONYMS.get(word, word)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def stems(text: str) -> set[str]:
    """Stems of every word, plus the parts of hyphenated ones (t-shirt -> t, shirt)."""
    result = set()
    for word in words(text):
        result.add(stem(word))
        result.update(stem(part) for part in word.split("-"))
    return result


def your_phrases(draft: str) -> list[tuple[str, list[str], str]]:
    """(phrase, words before the noun, noun stem) for each "your ... <garment>"."""
    found = []
    for clause in CLAUSE.split(draft.lower()):
        tokens = words(clause)
        for i, token in enumerate(tokens):
            if token != "your":
                continue
            for j in range(i + 1, min(i + 1 + PHRASE_WORDS, len(tokens))):
                if tokens[j] in PHRASE_BREAKS:
                    break
                if tokens[j] in GARMENT_NOUNS:
                    found.append((" ".join(tokens[i : j + 1]), tokens[i + 1 : j], stem(tokens[j])))
                    break
    return found


def invented_items(draft: str, plan: OutfitPlan | None, known_texts: Iterable[str]) -> list[str]:
    """ "your ..." phrases that match no piece in the plan and nothing else in the facts."""
    pieces = [p for o in plan.outfits for p in o.pieces] if plan else []
    known = [stems(label(p)) for p in pieces] + [stems(t) for t in known_texts]
    categories = {p.category for p in pieces}
    invented = []
    for phrase, before, noun in your_phrases(draft):
        colors = {stem(w) for w in before if w in COLOR_WORDS}
        if noun in GENERIC_NOUNS and not colors and GENERIC_NOUNS[noun] & categories:
            continue
        if not any(noun in k and colors <= k for k in known):
            invented.append(phrase)
    return invented


def mentions(draft: str, prefixes: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(p)}", draft, re.IGNORECASE) for p in prefixes)


def missing(draft: str, texts: Iterable[str], topics) -> list[str]:
    """Texts (gaps or notes) with a known topic that the draft does not bring up."""
    result = []
    for text in texts:
        topic = next((said for trigger, said in topics if trigger in text.lower()), None)
        if topic is not None and not mentions(draft, topic):
            result.append(text)
    return result


def code_checks(state: ChatState, draft: str) -> tuple[list[str], list[str], list[str]]:
    """(invented "your ..." phrases, gaps left out, weather notes left out). A failed
    weather check counts as a weather note."""
    plan = state.get("plan")
    advice = state.get("advice")
    assumptions = state.get("assumptions", [])
    warnings = [w for o in plan.outfits for w in o.warnings] if plan else []
    shown = [label(i) for i in state.get("found", [])]
    known = [*assumptions, *(plan.gaps if plan else []), *warnings, *shown]
    return (
        invented_items(draft, plan, known),
        missing(draft, plan.gaps if plan else [], GAP_TOPICS),
        missing(draft, advice.notes if advice else [], NOTE_TOPICS)
        + missing(draft, assumptions, ASSUMPTION_TOPICS),
    )


def issues_for(invented: list[str], gaps: list[str], notes: list[str]) -> list[str]:
    """The code-check findings as instructions to the stylist."""
    return (
        [f"'{p}' is not in `outfits`: name only the pieces given there" for p in invented]
        + [f"pass on this wardrobe gap: {g}" for g in gaps]
        + [f"pass on this weather note: {n}" for n in notes]
    )


# --- LLM rubric -----------------------------------------------------------------


def rubric(state: ChatState, draft: str, client: OpenAI, settings: Settings) -> list[str]:
    """Issues the model finds (empty = ok). A failed call counts as ok: the code checks
    have passed, so the draft is safe to send."""
    payload = {"facts": facts({**state, "critique": []}), "draft": draft}
    messages = [
        {"role": "system", "content": load_prompt("critic", CriticVerdict)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        verdict = chat_json(
            client, settings, messages, CriticVerdict, max_tokens=settings.critic_max_tokens
        )
    except LLMError as exc:
        logger.warning("Critic rubric failed, accepting the draft: %s", exc)
        return []
    return [] if verdict.ok else verdict.issues


# --- Node and routing -------------------------------------------------------------


def critic_node(state: ChatState, *, client: OpenAI, settings: Settings) -> dict:
    draft = state.get("draft")
    if not draft:  # the stylist failed: respond uses the template
        return {"critique": []}
    invented, gaps, notes = code_checks(state, draft)

    if not state.get("revisions"):
        issues = issues_for(invented, gaps, notes) or rubric(state, draft, client, settings)
        if issues:
            logger.info("Critic sends the draft back: %s", issues)
            return {"critique": issues, "revisions": 1}
        return {"critique": []}

    # After the one revision: never loop again; make the draft safe instead.
    if invented:
        logger.warning("Draft still names pieces not in the plan %s; using the template", invented)
        return {"draft": None, "critique": []}
    if gaps or notes:
        logger.info("Appending what the revised draft left out: %s", gaps + notes)
        draft = "\n".join([draft, *notes, *(f"Missing: {g}." for g in gaps)])
    return {"draft": draft, "critique": []}


def route_after_critic(state: ChatState) -> str:
    return "revise" if state.get("critique") else "done"
