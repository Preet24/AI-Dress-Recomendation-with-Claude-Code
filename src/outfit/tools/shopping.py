"""Should I buy this? Score a candidate garment against the wardrobe, in plain rules.

For every situation the user dresses for (an occasion, at a saved place's dress code
if they have one, in each typical weather of their home city) the planner runs with
and without the candidate. From that:

- goes with / clashes: the user's pieces it would be worn with, by color harmony;
- new outfits: distinct outfits that include it and score at least `GOOD_OUTFIT` (an
  outfit good for several situations, or with other shoes, counts once);
- unlocks: situations with no good outfit before, and one with it;
- fixes: wardrobe gaps (no footwear, no warm outerwear) it closes;
- accessories never change the outfits (tools/accessories.py): for one, "new outfits"
  are the good outfits it would be added to, fixes are the accessory gaps it closes
  ("no sunglasses for the strong sun"), and it is a buy only with a fix;
- looks like: the closest same-category piece by CLIP ("looks alike", not "goes with");
- verdict: skip / maybe / buy, every one with its reasons.

Pure functions: the caller supplies the wardrobe, the CLIP look-alikes and the weather
kinds (tools/climate.py), so tests need no store, model or network.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from outfit.schemas import Category, ClothingItem, Occasion
from outfit.tools.accessories import choose_accessories
from outfit.tools.climate import WeatherKind, typical_advice
from outfit.tools.color_harmony import pair_harmony, swatch
from outfit.tools.outfits import (
    ACCESSORY_SLOTS,
    BOTTOMS,
    NO_BASE,
    TOPS,
    Outfit,
    OutfitPlan,
    OutfitRequest,
    all_outfits,
    label,
)

DEFAULT_OCCASIONS = (Occasion.CASUAL, Occasion.OFFICE, Occasion.PARTY)
GOOD_OUTFIT = 0.7  # outfits scoring below this are not counted as wearable
GOES_WITH = 0.75  # pair harmony at or above: "goes with"
CLASHES = 0.55  # below: "clashes"
SHOW_PAIRS = 5
# CLIP similarity of same-category cutouts, measured on the real 20-item wardrobe: the
# same garment re-shot (mirrored, darker, warmer light, rotated + cropped) scored
# 0.82-0.99 (median 0.93-0.98); different garments at most 0.79 (tan chinos vs black
# trousers), similar shirts 0.70-0.78. A real shop photo differs more than these edits,
# so DUPLICATE errs towards missing a duplicate rather than crying wolf.
DUPLICATE = 0.85  # "very close to": counts against buying
SIMILAR = 0.75  # "looks similar to": mentioned only
BUY_OUTFITS = 6  # this many new outfits makes a buy even without an unlock

Verdict = Literal["buy", "maybe", "skip"]


@dataclass(frozen=True)
class Situation:
    name: str  # "office" or the saved place's name, "Office"
    occasion: Occasion
    formality: tuple[int, int] | None = None  # a saved place's dress code


@dataclass(frozen=True)
class Pairing:
    item: ClothingItem
    score: float  # pair harmony 0..1, or CLIP similarity for look-alikes
    reason: str


@dataclass
class ShoppingReport:
    candidate: ClothingItem
    verdict: Verdict
    reasons: list[str]
    goes_with: list[Pairing] = field(default_factory=list)
    clashes: list[Pairing] = field(default_factory=list)
    color_note: str = ""  # "goes with 7 of your 9 bottoms"
    new_outfits: int = 0
    unlocks: list[str] = field(default_factory=list)  # "office on hot days", "party in any weather"
    unlocked: int = 0  # situation-and-weather pairs behind `unlocks`
    fixes: list[str] = field(default_factory=list)  # gaps it closes
    looks_like: Pairing | None = None
    situations: list[str] = field(default_factory=list)  # what was checked
    owned: bool = False  # this very photo is already in the wardrobe


def partners(candidate: ClothingItem) -> tuple[set[Category], str]:
    """Categories it is worn with, and what to call them."""
    if candidate.category in TOPS:
        return BOTTOMS, "bottoms"
    if candidate.category in BOTTOMS:
        return TOPS, "tops"
    if candidate.category is Category.ONE_PIECE:
        return {Category.FOOTWEAR, Category.OUTERWEAR}, "shoes and outerwear"
    return TOPS | BOTTOMS | {Category.ONE_PIECE}, "tops, bottoms and one-pieces"


def pairings(candidate: ClothingItem, wardrobe: list[ClothingItem]) -> list[Pairing]:
    """Its partners in the wardrobe, best match first."""
    categories, _ = partners(candidate)
    mine = swatch(candidate)
    pairs = []
    for item in wardrobe:
        if item.category in categories:
            harmony = pair_harmony(mine, swatch(item))
            pairs.append(Pairing(item, round(harmony.score, 3), harmony.reason))
    return sorted(pairs, key=lambda p: -p.score)


def situations(
    occasions: list[Occasion], places: list[tuple[str, Occasion, tuple[int, int] | None]]
) -> list[Situation]:
    """Saved places first (with their dress codes), then ticked occasions no place covers;
    the defaults when there are neither."""
    found = [Situation(name, occasion, formality) for name, occasion, formality in places]
    covered = {s.occasion for s in found}
    found += [Situation(str(o), o) for o in occasions if o not in covered]
    return found or [Situation(str(o), o) for o in DEFAULT_OCCASIONS]


def outfit_key(outfit: Outfit, candidate: ClothingItem) -> frozenset:
    """Outfits that differ only in other shoes or outerwear count once."""
    return frozenset(
        p.id for p in outfit.pieces if p.category not in ACCESSORY_SLOTS or p.id == candidate.id
    )


def good(plan: OutfitPlan) -> list[Outfit]:
    return [o for o in plan.outfits if o.score >= GOOD_OUTFIT]


Baseline = dict[tuple[Situation, WeatherKind], OutfitPlan]


def situation_request(
    taste: OutfitRequest, situation: Situation, kind: WeatherKind
) -> OutfitRequest:
    return replace(
        taste,
        occasion=situation.occasion,
        formality=situation.formality,
        advice=typical_advice(kind),
    )


def where(situation: Situation, kind: WeatherKind) -> str:
    return f"{situation.name} on {kind} days"


def and_join(words: Sequence[str]) -> str:
    return " and ".join([", ".join(words[:-1]), words[-1]] if len(words) > 1 else words)


def group_where(
    found: Sequence[tuple[Situation, WeatherKind]], kinds: list[WeatherKind]
) -> list[str]:
    """One phrase per situation: "wedding in any weather", "office on hot and rainy days"."""
    by_situation: dict[Situation, list[str]] = {}
    for situation, kind in found:
        by_situation.setdefault(situation, []).append(str(kind))
    return [
        f"{s.name} in any weather"
        if len(kinds) > 1 and len(found_kinds) == len(kinds)
        else f"{s.name} on {and_join(found_kinds)} days"
        for s, found_kinds in by_situation.items()
    ]


def baseline(
    wardrobe: list[ClothingItem],
    checks: list[Situation],
    kinds: list[WeatherKind],
    taste: OutfitRequest | None = None,
) -> Baseline:
    """Every outfit the wardrobe makes in each situation and weather."""
    taste = taste or OutfitRequest()
    return {
        (s, k): all_outfits(wardrobe, situation_request(taste, s, k)) for s in checks for k in kinds
    }


def accessories_of(items: list[ClothingItem]) -> list[ClothingItem]:
    return [i for i in items if i.category is Category.ACCESSORY]


def accessory_effect(
    candidate: ClothingItem, others: list[ClothingItem], plan: OutfitPlan, request: OutfitRequest
) -> tuple[list[Outfit], list[str]]:
    """The good outfits an accessory would be added to, and the accessory gaps it closes."""
    owned = accessories_of(others)
    joined, closed = [], {}
    for outfit in good(plan):
        before = choose_accessories(owned, outfit.pieces, request)
        after = choose_accessories([*owned, candidate], outfit.pieces, request)
        if candidate.id in after.ids:
            joined.append(outfit)
            closed |= dict.fromkeys(g for g in before.gaps if g not in after.gaps)
    return joined, list(closed)


def accessory_gaps(
    wardrobe: list[ClothingItem], before: Baseline, taste: OutfitRequest | None = None
) -> dict[tuple[Situation, WeatherKind], list[str]]:
    """Accessory gaps of each situation's good outfits (`baseline()` never has them)."""
    taste = taste or OutfitRequest()
    owned = accessories_of(wardrobe)
    found = {}
    for (situation, kind), plan in before.items():
        request = situation_request(taste, situation, kind)
        gaps = {
            g: None for o in good(plan) for g in choose_accessories(owned, o.pieces, request).gaps
        }
        if gaps:
            found[(situation, kind)] = list(gaps)
    return found


def check_candidate(
    candidate: ClothingItem,
    wardrobe: list[ClothingItem],
    checks: list[Situation],
    kinds: list[WeatherKind],
    look_alikes: Sequence[tuple[ClothingItem, float]] = (),
    taste: OutfitRequest | None = None,
    before: Baseline | None = None,
) -> ShoppingReport:
    """The report for one candidate. `taste` carries the profile's taste fields; `before`
    is `baseline()` of the wardrobe without it, when the caller checks many candidates."""
    taste = taste or OutfitRequest()
    others = [i for i in wardrobe if i.id != candidate.id]
    before = before or baseline(others, checks, kinds, taste)
    new_keys: set[frozenset] = set()  # one outfit worn in several situations counts once
    unlocks: list[tuple[Situation, WeatherKind]] = []
    fixes: dict[str, None] = {}  # gaps it closes, once each, in order ("nothing fits" = unlock)
    checked: list[str] = []
    for situation in checks:
        for kind in kinds:
            request = situation_request(taste, situation, kind)
            checked.append(where(situation, kind))
            plan_before = before[(situation, kind)]
            if candidate.category is Category.ACCESSORY:
                joined, closed = accessory_effect(candidate, others, plan_before, request)
                new_keys |= {outfit_key(o, candidate) for o in joined}
                fixes |= dict.fromkeys(closed)
                continue
            after = all_outfits([*others, candidate], request)
            with_it = [o for o in good(after) if candidate.id in o.ids]
            new_keys |= {outfit_key(o, candidate) for o in with_it}
            if with_it and not good(plan_before):
                unlocks.append((situation, kind))
            if with_it:
                fixes |= dict.fromkeys(
                    g for g in plan_before.gaps if g not in after.gaps and not g.startswith(NO_BASE)
                )

    pairs = pairings(candidate, others)
    _, partner_name = partners(candidate)
    goes = [p for p in pairs if p.score >= GOES_WITH]
    clashes = [p for p in reversed(pairs) if p.score < CLASHES]
    color_note = (
        f"{candidate.primary_color_name} goes with {len(goes)} of your {len(pairs)} {partner_name}"
        if pairs
        else f"you have no {partner_name} to wear it with"
    )
    looks_like = closest_look_alike(candidate, wardrobe, look_alikes)

    report = ShoppingReport(
        candidate=candidate,
        verdict="maybe",
        reasons=[],
        goes_with=goes[:SHOW_PAIRS],
        clashes=clashes[:SHOW_PAIRS],
        color_note=color_note,
        new_outfits=len(new_keys),
        unlocks=group_where(unlocks, kinds),
        unlocked=len(unlocks),
        fixes=list(fixes),
        looks_like=looks_like,
        situations=checked,
        owned=any(i.id == candidate.id for i in wardrobe),
    )
    report.verdict, report.reasons = verdict(report)
    return report


def closest_look_alike(
    candidate: ClothingItem,
    wardrobe: list[ClothingItem],
    look_alikes: Sequence[tuple[ClothingItem, float]],
) -> Pairing | None:
    """The most similar piece of the same category; the item itself if this very photo
    was already added (item ids come from the photo)."""
    owned = next((i for i in wardrobe if i.id == candidate.id), None)
    if owned is not None:
        return Pairing(owned, 1.0, "this photo is already in your wardrobe")
    same = [(i, s) for i, s in look_alikes if i.category is candidate.category]
    if not same:
        return None
    item, score = max(same, key=lambda pair: pair[1])
    return Pairing(item, round(score, 3), "looks alike (CLIP)")


def verdict(report: ShoppingReport) -> tuple[Verdict, list[str]]:
    name = label(report.candidate)
    if report.owned:
        return "skip", [f"your {name} is already in your wardrobe (same photo)"]
    duplicate = report.looks_like is not None and report.looks_like.score >= DUPLICATE
    reasons = []
    if duplicate:
        reasons.append(f"it is very close to your {label(report.looks_like.item)}")
    elif report.looks_like is not None and report.looks_like.score >= SIMILAR:
        reasons.append(f"it looks similar to your {label(report.looks_like.item)}")
    if report.unlocks:
        reasons.append(f"it makes outfits possible for {', '.join(report.unlocks)}")
    if report.fixes:
        reasons.append(f"it fills a gap ({'; '.join(report.fixes)})")
    accessory = report.candidate.category is Category.ACCESSORY
    if report.new_outfits:
        reasons.append(
            f"it goes with {report.new_outfits} of your outfits"
            if accessory
            else f"it adds {report.new_outfits} outfit(s) across what you dress for"
        )
    else:
        reasons.append(
            f"none of your outfits calls for the {name}"
            if accessory
            else f"no outfit you'd wear includes the {name}"
        )
    reasons.append(report.color_note)

    if report.new_outfits == 0:
        return "skip", reasons
    if duplicate:
        return ("maybe" if report.unlocks or report.fixes else "skip"), reasons
    if accessory:  # it never makes an outfit possible: worth buying only for a gap
        return ("buy" if report.fixes else "maybe"), reasons
    if report.unlocks or report.fixes or report.new_outfits >= BUY_OUTFITS:
        return "buy", reasons
    return "maybe", reasons
