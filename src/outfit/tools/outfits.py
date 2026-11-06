"""Outfit assembly: wardrobe items + occasion + weather advice -> ranked outfits.

An outfit is a base (top + bottom, or a one-piece), plus footwear when a suitable
pair exists, plus outerwear when the weather calls for it.

- Hard rules apply to base garments: occasion, formality range, and the weather's
  warmth range and no-flowy rule. Footwear and outerwear that miss the occasion or
  formality are only penalized: a wardrobe often has few of them, and an outfit
  with imperfect shoes beats no outfit.
- The tagger marks only a few occasions per garment, so a base garment tagged for a
  near occasion (`NEAR_OCCASIONS`, e.g. an office shirt for a party) passes when its
  formality fits, at a small penalty. Bottoms may be one warmth point above the
  weather's range (the tagger rates nearly all full-length bottoms 3), also penalized.
  At formal and wedding, a plain (solid, not casual) bottom may be one point less dressy
  than the range, at a small penalty: the top sets the tone. At lounge, bottoms need
  the lounge tag themselves, and no shoes are added (`AT_HOME`).
- When no base fits, the gap names the closest misses and the rules they broke.
- Soft rules give each piece a fit score (fabrics the weather advises against or
  for, light colors in heat or sun) and the outfit penalties (two busy patterns,
  pieces of very different formality).
- Taste (the user's profile) is a separate, uncapped term: main colors they like or
  avoid, and their preferred fit for tops and bottoms. Avoided colors sink an outfit
  but never ban it.
- History (the wear log, tools/feedback.py) is another uncapped term: a pairing the
  user loved rises, one they didn't want again sinks, garments worn in the last few
  days sink a little.
- Final score = half color harmony (tools/color_harmony), half fit, plus taste and
  history.

Items the user asks for (`must_include`) skip the hard rules; anything they break
is reported as a warning instead. Results are kept varied: one item appears in at
most `MAX_REPEATS` suggestions. Missing pieces are reported as gaps.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from uuid import UUID

from outfit.schemas import Category, ClothingItem, Fit, Occasion, Pattern
from outfit.tools.color_harmony import outfit_harmony, swatch
from outfit.tools.weather import WeatherAdvice

# Formality range (1-5) an occasion calls for when the request gives none.
OCCASION_FORMALITY: dict[Occasion, tuple[int, int]] = {
    Occasion.LOUNGE: (1, 2),
    Occasion.SPORTS: (1, 2),
    Occasion.CASUAL: (1, 3),
    Occasion.TRAVEL: (1, 3),
    Occasion.OFFICE: (3, 4),
    Occasion.PARTY: (3, 5),
    Occasion.INDIAN_FESTIVE: (3, 5),
    Occasion.FORMAL: (4, 5),
    Occasion.WEDDING: (4, 5),
}

# Occasions whose garments also pass for this one when their formality fits.
NEAR_OCCASIONS: dict[Occasion, frozenset[Occasion]] = {
    Occasion.LOUNGE: frozenset({Occasion.CASUAL}),
    Occasion.SPORTS: frozenset(),
    Occasion.CASUAL: frozenset({Occasion.TRAVEL, Occasion.PARTY, Occasion.OFFICE}),
    Occasion.TRAVEL: frozenset({Occasion.CASUAL, Occasion.LOUNGE}),
    Occasion.OFFICE: frozenset({Occasion.CASUAL, Occasion.FORMAL}),
    Occasion.PARTY: frozenset({Occasion.CASUAL, Occasion.OFFICE, Occasion.FORMAL}),
    Occasion.INDIAN_FESTIVE: frozenset({Occasion.WEDDING, Occasion.PARTY}),
    Occasion.FORMAL: frozenset({Occasion.OFFICE, Occasion.WEDDING, Occasion.PARTY}),
    Occasion.WEDDING: frozenset({Occasion.FORMAL, Occasion.INDIAN_FESTIVE, Occasion.PARTY}),
}

TOPS = {Category.TOP, Category.ETHNIC_TOP}
BOTTOMS = {Category.BOTTOM, Category.ETHNIC_BOTTOM}
# Base garments that pass only with the occasion's own tag: jeans or chinos tagged
# casual aren't for lounging or sleeping, though a casual t-shirt is.
OWN_TAG_ONLY: dict[Occasion, set[Category]] = {Occasion.LOUNGE: BOTTOMS}
# Occasions spent at home: no shoes or accessories are added, and none are missing.
AT_HOME = frozenset({Occasion.LOUNGE})
LOUNGE_HINT = "joggers, shorts or pyjamas tagged lounge would do"
# optional, added to a base (accessories by tools/accessories, after the outfit is chosen)
ACCESSORY_SLOTS = {Category.FOOTWEAR, Category.OUTERWEAR, Category.ACCESSORY}

SLOT_CANDIDATES = 10  # best-fitting items kept per slot before combining
MAX_REPEATS = 2
MIN_ADDON_FIT = 0.6  # shoes / outerwear below this are left out (and reported as a gap)
SINGLE_PIECE_HARMONY = 0.85  # a lone one-piece has nothing to clash with
BROKEN_RULE = 0.15  # fit lost per hard rule a requested item breaks
HARMONY_WEIGHT = 0.5

OCCASION_MISS = 0.3  # footwear / outerwear not tagged for the occasion
NEAR_OCCASION = 0.1  # base garment tagged only for a near occasion
BOTTOM_WARMTH_SLACK = 1  # bottoms may be this much warmer than the weather's range
WARM_BOTTOM = 0.1
CLOSEST_MISSES = 2  # garments per empty slot named in the "nothing fits" gap
NO_BASE = "no outfit fits this occasion and weather"  # that gap's start
# At formal / wedding, plain trousers or a skirt may be this much less dressy than the
# range (the top must meet it): from a photo the tagger rates plain dark trousers 3.
DRESSY_FROM = 4  # range minimum where the slack applies
BOTTOM_FORMALITY_SLACK = 1
LESS_DRESSY_BOTTOM = 0.1
CASUAL_BOTTOM_WORDS = frozenset(
    {"jeans", "chinos", "shorts", "joggers", "cargo", "cargos", "leggings", "sweatpants",
     "trackpants", "pyjama", "pyjamas", "pajamas"}
)  # fmt: skip
FORMALITY_STEP = 0.15  # per point outside the formality range
AVOID_FABRIC = 0.2
PREFER_FABRIC = 0.05
LIGHT_COLOR = 0.05
PATTERN_CLASH = 0.15  # per busy pattern beyond the first
FORMALITY_SPREAD_OK = 2
FORMALITY_SPREAD_STEP = 0.1
# Taste, per piece, added to the score outside the capped fit (most pieces fit at 1.0)
LIKED_COLOR = 0.03  # nudges between otherwise close outfits
AVOIDED_COLOR = 0.08  # sinks an outfit below the alternatives, but never bans it
PREFERRED_FIT = 0.02
SUITS_COLORING = 0.02  # from the photo analysis: weaker than the user's own likes
GO_EASY_COLOR = 0.03
# History, from the wear log (tools/feedback.py), outside the capped fit like taste
LIKED_PAIR = 0.05  # a pairing the user loved
DISLIKED_PAIR = 0.1  # "not again", or its colours didn't go: sinks it, never bans it
WORN_RECENTLY = 0.05  # per base garment worn in the last few days (variety)


@dataclass(frozen=True)
class OutfitRequest:
    occasion: Occasion | None = None
    formality: tuple[int, int] | None = None  # default: from the occasion, else any
    advice: WeatherAdvice | None = None
    must_include: frozenset[UUID] = frozenset()
    exclude: frozenset[UUID] = frozenset()
    limit: int = 5
    # Taste, from the profile (lower-case color words)
    fit_tops: Fit | None = None
    fit_bottoms: Fit | None = None
    colors_liked: frozenset[str] = frozenset()
    colors_avoided: frozenset[str] = frozenset()
    colors_suit: frozenset[str] = frozenset()  # colouring analysis; own likes / avoids win
    colors_go_easy: frozenset[str] = frozenset()
    # History, from the wear log (`pairing` keys)
    liked_pairs: frozenset[frozenset[UUID]] = frozenset()
    disliked_pairs: frozenset[frozenset[UUID]] = frozenset()
    recent: frozenset[UUID] = frozenset()  # base garments worn in the last few days

    @property
    def formality_range(self) -> tuple[int, int]:
        if self.formality is not None:
            return self.formality
        return OCCASION_FORMALITY.get(self.occasion, (1, 5)) if self.occasion else (1, 5)


@dataclass(frozen=True)
class Outfit:
    pieces: tuple[ClothingItem, ...]
    score: float
    harmony: float
    fit: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ids(self) -> frozenset[UUID]:
        return frozenset(p.id for p in self.pieces)


@dataclass(frozen=True)
class OutfitPlan:
    outfits: list[Outfit]
    gaps: list[str] = field(default_factory=list)  # what the wardrobe is missing


def label(item: ClothingItem) -> str:
    return f"{item.primary_color_name} {item.subtype}"


# --- Hard rules (base garments) ---------------------------------------------


def dressy_bottom_slack(item: ClothingItem, request: OutfitRequest) -> int:
    """How far below the range a bottom may be: a solid, non-casual bottom at a dressy
    range (`DRESSY_FROM`+) gets `BOTTOM_FORMALITY_SLACK`; anything else 0."""
    words = set(item.subtype.lower().replace("-", " ").split())
    plain = (
        item.category in BOTTOMS
        and item.pattern is Pattern.SOLID
        and "denim" not in item.fabric.lower()
        and not words & CASUAL_BOTTOM_WORDS
    )
    return BOTTOM_FORMALITY_SLACK if plain and request.formality_range[0] >= DRESSY_FROM else 0


def formality_fits(item: ClothingItem, request: OutfitRequest) -> bool:
    low, high = request.formality_range
    return low - dressy_bottom_slack(item, request) <= item.formality <= high


def near_occasion(item: ClothingItem, request: OutfitRequest) -> bool:
    """Not tagged for the occasion, but for a near one, at a fitting formality."""
    occasion = request.occasion
    return (
        occasion is not None
        and occasion not in item.occasions
        and item.category not in OWN_TAG_ONLY.get(occasion, set())
        and bool(NEAR_OCCASIONS[occasion] & set(item.occasions))
        and formality_fits(item, request)
    )


def warmth_ceiling(item: ClothingItem, advice: WeatherAdvice) -> int:
    slack = BOTTOM_WARMTH_SLACK if item.category in BOTTOMS else 0
    return advice.warmth[1] + slack


def base_problems(item: ClothingItem, request: OutfitRequest) -> list[str]:
    """Why a base garment does not fit the request (empty if it fits)."""
    problems = []
    low, high = request.formality_range
    if (
        request.occasion
        and request.occasion not in item.occasions
        and not near_occasion(item, request)
    ):
        problems.append(f"{label(item)} is not tagged for {request.occasion}")
    if not formality_fits(item, request):
        problems.append(f"{label(item)} has formality {item.formality}, wanted {low}-{high}")
    if advice := request.advice:
        w_low, w_high = advice.warmth
        if not w_low <= item.warmth <= warmth_ceiling(item, advice):
            problems.append(
                f"{label(item)} has warmth {item.warmth}, the weather calls for {w_low}-{w_high}"
            )
        if advice.avoid_flowy and item.flowy:
            problems.append(f"{label(item)} is flowy, which the weather or ride rules out")
    return problems


# --- Soft rules -------------------------------------------------------------


def piece_fit(item: ClothingItem, request: OutfitRequest) -> tuple[float, list[str]]:
    """Fit score 0..1 for one piece, with the reasons for anything that moved it."""
    score, notes = 1.0, []
    if item.category in ACCESSORY_SLOTS:
        if request.occasion and request.occasion not in item.occasions:
            score -= OCCASION_MISS
            notes.append(f"{label(item)} is not really for {request.occasion}")
        low, high = request.formality_range
        off = max(low - item.formality, item.formality - high, 0)
        if off:
            score -= FORMALITY_STEP * off
            notes.append(
                f"{label(item)} is {'too dressy' if item.formality > high else 'too casual'}"
            )
    else:
        if near_occasion(item, request):
            score -= NEAR_OCCASION
            notes.append(
                f"{label(item)} is not tagged for {request.occasion}, but is dressy enough"
            )
        if dressy_bottom_slack(item, request) and item.formality < request.formality_range[0]:
            score -= LESS_DRESSY_BOTTOM
            notes.append(f"{label(item)} is a little less dressy, so the top sets the tone")
    if advice := request.advice:
        if item.category in BOTTOMS and advice.warmth[1] < item.warmth <= warmth_ceiling(
            item, advice
        ):
            score -= WARM_BOTTOM
            notes.append(f"{label(item)} is a bit warm for this weather")
        fabric = item.fabric.lower()
        if fabric in advice.avoid_fabrics:
            score -= AVOID_FABRIC
            notes.append(f"{fabric} {item.subtype} is not ideal in this weather")
        elif fabric in advice.prefer_fabrics:
            score += PREFER_FABRIC
        if (
            advice.prefer_light_colors
            and item.category is not Category.FOOTWEAR
            and swatch(item).light
        ):
            score += LIGHT_COLOR
    return min(max(score, 0.0), 1.0), notes


def color_is(item: ClothingItem, colors: frozenset[str]) -> str | None:
    """The first of `colors` that names the main color ("navy" in "navy blue"), if any."""
    words = set(item.primary_color_name.lower().replace("-", " ").split())
    return next((c for c in sorted(colors) if set(c.split()) <= words), None)


def taste(item: ClothingItem, request: OutfitRequest) -> tuple[float, list[str]]:
    """Score adjustment from the user's likes, with a reason for each."""
    score, notes = 0.0, []
    if liked := color_is(item, request.colors_liked):
        score += LIKED_COLOR
        notes.append(f"{label(item)} is in {liked}, a color you like")
    if avoided := color_is(item, request.colors_avoided):
        score -= AVOIDED_COLOR
        notes.append(f"{label(item)} is in {avoided}, a color you usually avoid")
    if not (liked or avoided):  # the user's own choice beats the analysis
        if suits := color_is(item, request.colors_suit):
            score += SUITS_COLORING
            notes.append(f"{label(item)} is in {suits}, which suits your colouring")
        elif go_easy := color_is(item, request.colors_go_easy):
            score -= GO_EASY_COLOR
            notes.append(
                f"{label(item)} is in {go_easy}, which your colouring suggests going easy on"
            )
    preferred = (
        request.fit_tops
        if item.category in TOPS
        else request.fit_bottoms
        if item.category in BOTTOMS
        else None
    )
    if preferred is not None and item.fit is preferred:
        score += PREFERRED_FIT
        notes.append(f"{label(item)} has the {preferred} fit you prefer")
    return score, notes


def pairing(pieces: Sequence[ClothingItem]) -> frozenset[UUID] | None:
    """What makes an outfit "the same one" for the wear log: its top + bottom, or a
    one-piece with its shoes. None when there is no such pair."""
    halves = [p for p in pieces if p.category in TOPS | BOTTOMS]
    if len(halves) >= 2:
        return frozenset(p.id for p in halves)
    one = [p for p in pieces if p.category in {Category.ONE_PIECE, Category.FOOTWEAR}]
    if len(one) >= 2 and any(p.category is Category.ONE_PIECE for p in one):
        return frozenset(p.id for p in one)
    return None


def history(pieces: Sequence[ClothingItem], request: OutfitRequest) -> tuple[float, list[str]]:
    """Score adjustment from the wear log, with a reason for each."""
    score, notes = 0.0, []
    key = pairing(pieces)
    names = " with ".join(label(p) for p in pieces if key and p.id in key)
    if key in request.disliked_pairs:
        score -= DISLIKED_PAIR
        notes.append(f"you didn't want your {names} again")
    elif key in request.liked_pairs:
        score += LIKED_PAIR
        notes.append(f"you loved your {names} last time")
    for p in pieces:
        if p.id in request.recent and p.category not in ACCESSORY_SLOTS:
            score -= WORN_RECENTLY
            notes.append(f"you wore your {label(p)} in the last few days")
    return score, notes


def outfit_penalties(pieces: tuple[ClothingItem, ...]) -> tuple[float, list[str]]:
    penalty, notes = 0.0, []
    clothes = [p for p in pieces if p.category is not Category.FOOTWEAR]
    busy = [p for p in clothes if p.pattern is not Pattern.SOLID]
    if len(busy) > 1:
        penalty += PATTERN_CLASH * (len(busy) - 1)
        notes.append(f"{len(busy)} patterned pieces compete ({', '.join(map(label, busy))})")
    spread = max(p.formality for p in pieces) - min(p.formality for p in pieces)
    if spread > FORMALITY_SPREAD_OK:
        penalty += FORMALITY_SPREAD_STEP * (spread - FORMALITY_SPREAD_OK)
        notes.append("the pieces differ a lot in how dressy they are")
    return penalty, notes


# --- Assembly ---------------------------------------------------------------


def candidates(
    items: list[ClothingItem], categories: set[Category], request: OutfitRequest, base: bool
) -> list[tuple[ClothingItem, float, list[str]]]:
    """Items for one slot, best fit first: (item, fit, notes). Base items must pass the
    hard rules unless the user asked for them."""
    slot = []
    for item in items:
        if item.category not in categories or item.id in request.exclude:
            continue
        problems = base_problems(item, request) if base else []
        if problems and item.id not in request.must_include:
            continue
        fit, notes = piece_fit(item, request)
        fit = max(fit - BROKEN_RULE * len(problems), 0.0)  # requested despite the rules
        slot.append((item, fit, problems + notes))
    slot.sort(key=lambda c: (c[0].id not in request.must_include, -c[1]))
    return slot[:SLOT_CANDIDATES]


def closest_misses(
    items: list[ClothingItem], categories: set[Category], name: str, request: OutfitRequest
) -> str:
    """'no bottom fits: black trousers has warmth 4, the weather calls for 1-2' for the
    garments nearest to fitting (`name` is singular: "top")."""
    rejected = sorted(
        (
            (i, base_problems(i, request))
            for i in items
            if i.category in categories and i.id not in request.exclude
        ),
        key=lambda r: len(r[1]),
    )
    if not rejected:
        return f"no {name}s in the wardrobe"
    misses = [
        f"{label(i)} " + " and ".join(p.removeprefix(f"{label(i)} ") for p in problems)
        for i, problems in rejected[:CLOSEST_MISSES]
    ]
    return f"no {name} fits: " + "; ".join(misses)


def addons(slot: list, request: OutfitRequest) -> list:
    """Shoes / outerwear good enough to add (requested items always are)."""
    return [c for c in slot if c[1] >= MIN_ADDON_FIT or c[0].id in request.must_include]


def needs_outerwear(request: OutfitRequest) -> int | None:
    """Minimum outerwear warmth if the weather calls for outerwear, else None."""
    advice = request.advice
    return advice.outerwear_min_warmth if advice and advice.needs_outerwear else None


def score_outfit(
    chosen: tuple[tuple[ClothingItem, float, list[str]], ...], request: OutfitRequest
) -> Outfit:
    pieces = tuple(c[0] for c in chosen)
    if len(pieces) > 1:
        result = outfit_harmony(list(pieces))
        harmony, color_note = result.score, result.reason
    else:
        harmony, color_note = SINGLE_PIECE_HARMONY, "a single piece, nothing to clash with"
    penalty, outfit_notes = outfit_penalties(pieces)
    fit = max(sum(c[1] for c in chosen) / len(chosen) - penalty, 0.0)
    tastes = [taste(p, request) for p in pieces] + [history(pieces, request)]
    score = HARMONY_WEIGHT * harmony + (1 - HARMONY_WEIGHT) * fit + sum(t[0] for t in tastes)
    score = min(max(score, 0.0), 1.0)
    piece_notes = [n for c in chosen for n in c[2]] + [n for t in tastes for n in t[1]]
    warnings = [n for c in chosen if c[0].id in request.must_include for n in c[2]]
    reasons = [f"colors: {color_note}", *outfit_notes]
    reasons += [n for n in piece_notes if n not in warnings]
    return Outfit(pieces, round(score, 3), harmony, round(fit, 3), tuple(reasons), tuple(warnings))


def suggest_outfits(items: list[ClothingItem], request: OutfitRequest) -> OutfitPlan:
    """The best outfits for the request, varied, with gaps in the wardrobe reported."""
    plan = all_outfits(items, request)
    return OutfitPlan(pick_varied(plan.outfits, request.limit, request.must_include), plan.gaps)


def all_outfits(items: list[ClothingItem], request: OutfitRequest) -> OutfitPlan:
    """Every outfit that passes the rules, best first, without the variety cap or limit
    (the shopping assistant counts them)."""
    tops = candidates(items, TOPS, request, base=True)
    bottoms = candidates(items, BOTTOMS, request, base=True)
    one_pieces = candidates(items, {Category.ONE_PIECE}, request, base=True)
    gaps: list[str] = []
    for_what = request.occasion or "this request"

    shoes: list = []
    if request.occasion not in AT_HOME:
        shoes = addons(candidates(items, {Category.FOOTWEAR}, request, base=False), request)
        if not shoes:
            gaps.append(f"no footwear suits {for_what}")

    outer_min = needs_outerwear(request)
    outers: list = []
    if outer_min is not None:
        warm_enough = [
            c
            for c in candidates(items, {Category.OUTERWEAR}, request, base=False)
            if c[0].warmth >= outer_min or c[0].id in request.must_include
        ]
        outers = addons(warm_enough, request)
        if not outers:
            gaps.append(f"no outerwear with warmth {outer_min}+ suits {for_what}")

    bases = list(product(tops, bottoms)) + [(o,) for o in one_pieces]
    if not bases:
        why = [
            closest_misses(items, categories, name, request)
            for name, categories, slot in (("top", TOPS, tops), ("bottom", BOTTOMS, bottoms))
            if not slot
        ]
        if request.occasion in AT_HOME:
            why.append(LOUNGE_HINT)
        gaps.append(NO_BASE + (f" ({'; '.join(why)})" if why else ""))
        return OutfitPlan([], gaps)

    combos = product(bases, shoes or [None], outers or [None])
    # requested accessories are added later (tools/accessories.finish_plan)
    must = request.must_include - {i.id for i in items if i.category is Category.ACCESSORY}
    outfits = []
    for base, shoe, outer in combos:
        chosen = tuple(c for c in (*base, shoe, outer) if c is not None)
        ids = {c[0].id for c in chosen}
        if not must <= ids:
            continue
        outfits.append(score_outfit(chosen, request))
    if not outfits:
        gaps.append("the requested items cannot be combined into one outfit")
    outfits.sort(key=lambda o: -o.score)
    return OutfitPlan(outfits, gaps)


def pick_varied(outfits: list[Outfit], limit: int, keep: frozenset[UUID]) -> list[Outfit]:
    """Best outfits first, but no base garment (other than requested ones) more than
    MAX_REPEATS times. Shoes and outerwear may repeat: wardrobes have few of them."""
    used: Counter[UUID] = Counter()
    picked = []
    for outfit in outfits:
        base_ids = {p.id for p in outfit.pieces if p.category not in ACCESSORY_SLOTS} - keep
        if any(used[i] >= MAX_REPEATS for i in base_ids):
            continue
        picked.append(outfit)
        used.update(base_ids)
        if len(picked) == limit:
            break
    return picked
