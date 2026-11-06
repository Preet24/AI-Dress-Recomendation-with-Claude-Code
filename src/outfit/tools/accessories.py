"""Accessories: the belt, tie, bag, sunglasses ... that finish a chosen outfit.

Accessories never decide the outfit: they are chosen after it, at most one per kind,
so the outfits and their ranking stay what tools/outfits.py made them.

- The kind comes from the subtype words ("leather belt", "silk tie", "tote bag"); the
  last word that names a kind wins ("belt bag" is a bag).
- Hard rules per kind: a belt needs trousers, jeans or chinos; a tie needs a shirt and
  an occasion that allows formality 4+; a dupatta needs an ethnic piece; sunglasses,
  hats and scarves come only when the weather calls for them. For every kind: nothing
  flowy when the weather or ride rules it out, no hat or scarf much warmer than the
  weather.
- Score = half color harmony with the pieces it sits next to (a tie with the shirt, a
  belt with the bottom, anything else with the whole outfit), half fit (occasion,
  formality and fabric as for shoes, but not occasion or formality for sunglasses;
  plus belt-and-shoe leather), plus taste.
- A missing accessory is a gap only when it is called for: sunglasses in strong sun, a
  scarf in the cold, a tie for a formal occasion when the user owns ties.
- At home (lounge) nothing is added and nothing is missing, unless the user asked for it.
"""

from dataclasses import dataclass, field, replace
from enum import StrEnum
from uuid import UUID

from outfit.schemas import Category, ClothingItem, Pattern
from outfit.tools.color_harmony import pair_harmony, swatch
from outfit.tools.match import stem, tokens
from outfit.tools.outfits import (
    AT_HOME,
    BOTTOMS,
    HARMONY_WEIGHT,
    MIN_ADDON_FIT,
    PATTERN_CLASH,
    Outfit,
    OutfitPlan,
    OutfitRequest,
    label,
    piece_fit,
    taste,
)


class Kind(StrEnum):
    BELT = "belt"
    TIE = "tie"
    BAG = "bag"
    SUNGLASSES = "sunglasses"
    HAT = "hat"
    SCARF = "scarf"
    DUPATTA = "dupatta"
    WATCH = "watch"
    JEWELLERY = "jewellery"


def _words(*words: str) -> frozenset[str]:
    return frozenset(stem(w) for w in words)


KIND_WORDS: dict[Kind, frozenset[str]] = {
    Kind.BELT: _words("belt"),
    Kind.TIE: _words("tie", "necktie", "bowtie"),
    Kind.BAG: _words(
        "bag", "handbag", "purse", "clutch", "tote", "backpack", "satchel", "sling", "potli"
    ),
    Kind.SUNGLASSES: _words("sunglasses", "shades", "aviators"),
    Kind.HAT: _words("hat", "cap", "beanie", "fedora"),
    Kind.SCARF: _words("scarf", "stole", "muffler", "shawl"),
    Kind.DUPATTA: _words("dupatta", "chunni"),
    Kind.WATCH: _words("watch", "smartwatch"),
    Kind.JEWELLERY: _words(
        "jewellery", "jewelry", "necklace", "earrings", "bracelet", "bangles", "ring",
        "chain", "pendant", "jhumkas", "anklet", "brooch",
    ),
}  # fmt: skip

BELTED_BOTTOMS = _words("trousers", "jeans", "chinos", "pants", "slacks", "shorts")
BELTLESS_BOTTOMS = _words("track", "joggers", "leggings", "sweatpants", "pyjamas", "palazzo")
NOT_TIE_SHIRTS = _words("polo")
BROWN_LEATHER = frozenset({"brown", "tan", "cognac", "chocolate", "camel", "chestnut", "coffee"})

WEATHER_KINDS = {Kind.SUNGLASSES, Kind.HAT, Kind.SCARF}  # only when the weather calls for them
# worn outside and taken off indoors, so the occasion and its formality don't judge them
# (on the real wardrobe, sunglasses for the office came with "too casual")
OUTDOOR_ONLY = {Kind.SUNGLASSES}
WARM_KINDS = {Kind.HAT, Kind.SCARF}
TIE_FORMALITY = 4  # a tie only when the occasion's range reaches this
FORMAL_TIE = 4  # ...and is missed (a gap) only when the range starts here (formal, wedding)
COLD_C = 10  # felt temperature below which a scarf or a warm hat is called for
WARM_SLACK = 1  # a hat or scarf may be this much warmer than the weather's range
LEATHER_MATCH = 0.1  # belt and shoes in the same leather color
LEATHER_CLASH = 0.2  # black with brown


@dataclass(frozen=True)
class Choice:
    item: ClothingItem
    kind: Kind
    score: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()  # rules a requested accessory breaks


@dataclass(frozen=True)
class Finish:
    """The accessories chosen for one outfit, and the ones it is missing."""

    chosen: list[Choice] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def ids(self) -> frozenset[UUID]:
        return frozenset(c.item.id for c in self.chosen)


def word_parts(text: str) -> list[str]:
    """Content words with hyphenated ones split ("sling-bag" -> sling, bag), in order."""
    return [stem(part) for word in tokens(text) for part in word.split("-")]


def kind_of(item: ClothingItem) -> Kind | None:
    """Which accessory an item is, from its subtype's last kind word (None: not one)."""
    if item.category is not Category.ACCESSORY:
        return None
    for word in reversed(word_parts(item.subtype)):
        for kind, words in KIND_WORDS.items():
            if word in words:
                return kind
    return None


def takes_belt(item: ClothingItem) -> bool:
    words = set(word_parts(item.subtype))
    return (
        item.category is Category.BOTTOM
        and bool(words & BELTED_BOTTOMS)
        and not words & BELTLESS_BOTTOMS
    )


def is_shirt(item: ClothingItem) -> bool:
    """A shirt a tie goes with: "shirt" as a word, so not a t-shirt, sweatshirt or polo."""
    words = set(tokens(item.subtype))
    return item.category is Category.TOP and "shirt" in words and not words & NOT_TIE_SHIRTS


def leather(item: ClothingItem) -> str | None:
    words = set(item.primary_color_name.lower().replace("-", " ").split())
    if "black" in words:
        return "black"
    return "brown" if words & BROWN_LEATHER else None


def called_for(request: OutfitRequest) -> dict[Kind, str]:
    """Weather kinds the outing calls for, with why ("the strong sun")."""
    advice = request.advice
    if advice is None:
        return {}
    why: dict[Kind, str] = {}
    cold = advice.effective_temp < COLD_C
    if advice.sunny:
        why[Kind.SUNGLASSES] = why[Kind.HAT] = "the strong sun"
    elif advice.riding:
        why[Kind.SUNGLASSES] = "the ride"
    if cold:
        why[Kind.SCARF] = "the cold"
        why.setdefault(Kind.HAT, "the cold")
    return why


def problems(
    item: ClothingItem, kind: Kind, pieces: tuple[ClothingItem, ...], request: OutfitRequest
) -> list[str]:
    """Why an accessory does not belong with this outfit (empty if it does)."""
    found = []
    if kind is Kind.BELT and not any(takes_belt(p) for p in pieces):
        found.append(f"{label(item)} needs trousers, jeans or chinos")
    if kind is Kind.TIE:
        if request.formality_range[1] < TIE_FORMALITY:
            found.append(f"{label(item)} is too dressy for {request.occasion or 'this'}")
        if not any(is_shirt(p) for p in pieces):
            found.append(f"{label(item)} needs a shirt")
    if kind is Kind.DUPATTA and not any(
        p.category in {Category.ETHNIC_TOP, Category.ETHNIC_BOTTOM} for p in pieces
    ):
        found.append(f"{label(item)} goes with ethnic wear")
    if kind in WEATHER_KINDS and kind not in called_for(request):
        found.append(f"{label(item)} is not needed in this weather")
    if advice := request.advice:
        if advice.avoid_flowy and item.flowy:
            found.append(f"{label(item)} is flowy, which the weather or ride rules out")
        if kind in WARM_KINDS and item.warmth > advice.warmth[1] + WARM_SLACK:
            found.append(f"{label(item)} is too warm for this weather")
    return found


def partners(kind: Kind, pieces: tuple[ClothingItem, ...]) -> list[ClothingItem]:
    """The pieces an accessory's color is judged against."""
    if kind is Kind.TIE:
        return [p for p in pieces if is_shirt(p)]
    if kind is Kind.BELT:
        return [p for p in pieces if p.category in BOTTOMS]
    return list(pieces)


def score_accessory(
    item: ClothingItem, kind: Kind, pieces: tuple[ClothingItem, ...], request: OutfitRequest
) -> tuple[float, list[str]]:
    """Score 0..1 for an accessory with this outfit, with the reasons."""
    judged_by = (
        replace(request, occasion=None, formality=(1, 5)) if kind in OUTDOOR_ONLY else request
    )
    fit, notes = piece_fit(item, judged_by)
    others = partners(kind, pieces) or list(pieces)
    pairs = [pair_harmony(swatch(item), swatch(p)) for p in others]
    harmony = sum(p.score for p in pairs) / len(pairs) if pairs else 1.0
    reasons = [f"{label(item)} with {label(others[0])}: {pairs[0].reason}"] if pairs else []
    if kind is Kind.BELT:
        belt = leather(item)
        for shoe in (p for p in pieces if p.category is Category.FOOTWEAR):
            if belt and (shoes := leather(shoe)):
                if belt == shoes:
                    fit += LEATHER_MATCH
                    reasons.append(f"{label(item)} matches the {label(shoe)}")
                else:
                    fit -= LEATHER_CLASH
                    notes.append(f"{label(item)} with {label(shoe)}: belt and shoes differ")
    if kind is Kind.TIE and item.pattern is not Pattern.SOLID:
        busy = [p for p in others if p.pattern is not Pattern.SOLID]
        if busy:
            fit -= PATTERN_CLASH
            notes.append(f"patterned {label(item)} competes with {label(busy[0])}")
    bonus, taste_notes = taste(item, request)
    fit = min(max(fit, 0.0), 1.0)
    score = HARMONY_WEIGHT * harmony + (1 - HARMONY_WEIGHT) * fit + bonus
    return round(min(max(score, 0.0), 1.0), 3), reasons + notes + taste_notes


def choose_accessories(
    accessories: list[ClothingItem], pieces: tuple[ClothingItem, ...], request: OutfitRequest
) -> Finish:
    """At most one accessory per kind for this outfit, and the called-for ones missing.

    Requested accessories (`must_include`) are always chosen; the rules they break
    become warnings.
    """
    at_home = request.occasion in AT_HOME  # only what the user asked for, nothing missing
    best: dict[Kind, Choice] = {}
    owned: set[Kind] = set()
    for item in accessories:
        kind = kind_of(item)
        if kind is None or item.id in request.exclude:
            continue
        if at_home and item.id not in request.must_include:
            continue
        owned.add(kind)
        broken = problems(item, kind, pieces, request)
        requested = item.id in request.must_include
        if broken and not requested:
            continue
        score, reasons = score_accessory(item, kind, pieces, request)
        if score < MIN_ADDON_FIT and not requested:
            continue
        choice = Choice(item, kind, score, tuple(reasons), tuple(broken if requested else ()))
        current = best.get(kind)
        if current is None or (requested, score) > (
            current.item.id in request.must_include,
            current.score,
        ):
            best[kind] = choice
    gaps = [] if at_home else missing(set(best), owned, pieces, request)
    return Finish(list(best.values()), gaps)


def missing(
    chosen: set[Kind], owned: set[Kind], pieces: tuple[ClothingItem, ...], request: OutfitRequest
) -> list[str]:
    """Gaps for accessories this outfit calls for but did not get."""
    gaps = []
    advice = request.advice
    if advice and advice.sunny and Kind.SUNGLASSES not in chosen:
        gaps.append("no sunglasses for the strong sun")
    if advice and advice.effective_temp < COLD_C and Kind.SCARF not in chosen:
        gaps.append("no scarf for the cold")
    if (
        Kind.TIE in owned
        and Kind.TIE not in chosen
        and request.formality_range[0] >= FORMAL_TIE
        and any(is_shirt(p) for p in pieces)
    ):
        gaps.append(f"no tie suits {request.occasion or 'this request'}")
    return gaps


def finish_outfit(
    outfit: Outfit, accessories: list[ClothingItem], request: OutfitRequest
) -> tuple[Outfit, list[str]]:
    """The outfit with its accessories added after its pieces (its score unchanged)."""
    finish = choose_accessories(accessories, outfit.pieces, request)
    if not finish.chosen:
        return outfit, finish.gaps
    chosen = sorted(finish.chosen, key=lambda c: list(Kind).index(c.kind))
    reasons = [r for c in chosen for r in c.reasons if r not in c.warnings]
    finished = replace(
        outfit,
        pieces=outfit.pieces + tuple(c.item for c in chosen),
        reasons=outfit.reasons + tuple(reasons),
        warnings=outfit.warnings + tuple(w for c in chosen for w in c.warnings),
    )
    return finished, finish.gaps


def finish_plan(plan: OutfitPlan, items: list[ClothingItem], request: OutfitRequest) -> OutfitPlan:
    """Accessories for every outfit in a `suggest_outfits` plan, and the gaps they leave."""
    accessories = [i for i in items if i.category is Category.ACCESSORY]
    outfits, gaps = [], list(plan.gaps)
    for outfit in plan.outfits:
        finished, missed = finish_outfit(outfit, accessories, request)
        outfits.append(finished)
        gaps += [g for g in missed if g not in gaps]
    return OutfitPlan(outfits, gaps)
