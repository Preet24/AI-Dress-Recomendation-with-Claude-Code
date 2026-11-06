"""What should I buy? Try common staples against the wardrobe and suggest the best few.

There is no photo of a suggestion, so each staple is a fixed set of tags (`Staple`, one
row of `staples.toml`) in a few usual colors. For one staple:

1. colors the user avoids or should go easy on are left out, and so are colors in
   which they already own that kind of piece (same category and subtype word);
2. the color that goes with most of the user's pieces is chosen (liked colors win
   ties), and the report's color note says why ("white goes with 15 of your 20 ...");
3. `tools/shopping.check_candidate` scores it like a photographed candidate.

Suggestions are picked greedily: the best staple is added to the wardrobe and the rest
are scored again, so a second pair of shoes is not suggested for the gap the first
pair already fills. A staple is picked only when its verdict is "buy"; shoes and
outerwear also need an unlock or a gap fixed, because every outfit with a new pair of
shoes counts as new even when the user already has shoes for it. When no single staple
unlocks anything but some situation has no outfit at all, a top and a bottom are tried
together (`best_pair`: "sports" needs both a sports tee and track pants).

`missing` lists what the wardrobe lacks today: situations with no good outfit, and the
planner's gaps with where they occur. Pure functions; no LLM, store or network.
"""

import tomllib
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, field_validator

from outfit.schemas import Category, ClothingItem, Fit, Length, Occasion, Pattern, Scale
from outfit.tools.climate import WeatherKind
from outfit.tools.outfits import (
    ACCESSORY_SLOTS,
    BOTTOMS,
    NO_BASE,
    TOPS,
    OutfitRequest,
    all_outfits,
    color_is,
    label,
)
from outfit.tools.shopping import (
    GOES_WITH,
    Baseline,
    ShoppingReport,
    Situation,
    baseline,
    check_candidate,
    good,
    group_where,
    pairings,
    situation_request,
)

STAPLES_PATH = Path(__file__).with_name("staples.toml")  # the catalogue
STAPLE_NAMESPACE = UUID("3b7e9d21-8c4f-4a6e-b1d2-7f0a5c9e2b48")
MAX_SUGGESTIONS = 5

# Typical CIELAB values as the wardrobe's photos measure them (desaturated, not paint chips)
COLORS: dict[str, tuple[float, float, float]] = {
    "white": (94.0, 0.0, 2.0),
    "black": (14.0, 0.0, 0.0),
    "grey": (58.0, 0.0, 0.0),
    "navy": (24.0, 4.0, -24.0),
    "beige": (78.0, 3.0, 14.0),
    "tan": (62.0, 8.0, 24.0),
    "brown": (36.0, 12.0, 20.0),
    "olive": (46.0, -8.0, 24.0),
    "light blue": (78.0, -4.0, -14.0),
    "blue": (45.0, 2.0, -30.0),
    "maroon": (30.0, 30.0, 10.0),
}


class Staple(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Category
    subtype: str
    colors: tuple[str, ...]  # keys of COLORS, most usual first
    fabric: str
    formality: int = Scale
    warmth: int = Scale
    occasions: tuple[Occasion, ...]
    length: Length = Length.FULL
    fit: Fit = Fit.REGULAR
    flowy: bool = False
    gender: Literal["man", "woman"] | None = None  # None = everyone

    @field_validator("colors")
    @classmethod
    def known_colors(cls, colors: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [c for c in colors if c not in COLORS]
        if unknown or not colors:
            raise ValueError(f"colors must be some of {sorted(COLORS)}, got {list(colors)}")
        return colors

    def item(self, color: str) -> ClothingItem:
        """The staple as a garment; its id is stable and never a real item's."""
        slug = f"{self.subtype}-{color}".replace(" ", "-")
        return ClothingItem(
            id=uuid5(STAPLE_NAMESPACE, slug),
            image_path=Path("staples") / f"{slug}.png",  # no picture
            category=self.category,
            subtype=self.subtype,
            primary_color_name=color,
            primary_color_lab=COLORS[color],
            pattern=Pattern.SOLID,
            fabric=self.fabric,
            length=self.length,
            fit=self.fit,
            formality=self.formality,
            warmth=self.warmth,
            flowy=self.flowy,
            occasions=list(self.occasions),
        )


def load_staples(path: Path = STAPLES_PATH) -> tuple[Staple, ...]:
    with path.open("rb") as f:
        return tuple(Staple.model_validate(row) for row in tomllib.load(f)["staple"])


STAPLES = load_staples()


@dataclass
class Missing:
    no_outfit: list[str] = field(default_factory=list)  # "office on hot days"
    gaps: dict[str, list[str]] = field(default_factory=dict)  # gap -> where it occurs


@dataclass
class Suggestion:
    staple: Staple
    report: ShoppingReport  # against the wardrobe plus the suggestions before it
    after: list[str] = field(default_factory=list)  # labels of the earlier suggestions
    together_with: str | None = None  # the other half of a top + bottom pair


def for_gender(gender: str | None, staples: tuple[Staple, ...] = STAPLES) -> list[Staple]:
    """Staples for this gender; everything when it is unset or non-binary."""
    if gender not in ("man", "woman"):
        return list(staples)
    return [s for s in staples if s.gender in (None, gender)]


def garment_words(subtype: str) -> set[str]:
    """ "Canvas Sneakers" -> {"canvas", "sneaker"}; "t-shirt" -> {"t", "shirt"}."""
    words = subtype.lower().replace("-", " ").split()
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}


def owns_kind(item: ClothingItem, wardrobe: list[ClothingItem]) -> bool:
    """Whether the wardrobe has this kind of piece in this color ("white sneakers")."""
    words = garment_words(item.subtype)
    color = frozenset({item.primary_color_name})
    return any(
        w.category is item.category
        and words <= garment_words(w.subtype)
        and color_is(w, color) is not None
        for w in wardrobe
    )


def pick_color(
    staple: Staple, wardrobe: list[ClothingItem], taste: OutfitRequest
) -> ClothingItem | None:
    """The staple in the color that goes with most of the wardrobe, or None if every
    usual color is avoided or already owned."""
    banned = taste.colors_avoided | taste.colors_go_easy
    options = [
        staple.item(c)
        for c in staple.colors
        if c not in banned and not owns_kind(staple.item(c), wardrobe)
    ]
    if not options:
        return None

    def rank(item: ClothingItem) -> tuple[int, bool, float]:
        pairs = pairings(item, wardrobe)
        goes = sum(p.score >= GOES_WITH for p in pairs)
        mean = sum(p.score for p in pairs) / len(pairs) if pairs else 0.0
        return goes, item.primary_color_name in taste.colors_liked, mean

    return max(options, key=rank)  # first of equals: the most usual color


def missing(
    before: Baseline,
    kinds: list[WeatherKind],
    accessories: dict[tuple[Situation, WeatherKind], list[str]] | None = None,
) -> Missing:
    """What the wardrobe lacks, from its `shopping.baseline()` and, optionally, its
    `shopping.accessory_gaps()`. The "nothing fits" gap becomes a `no_outfit` entry,
    with the closest misses of its first weather."""
    accessories = accessories or {}
    no_outfit: list[tuple[Situation, WeatherKind]] = []
    why: dict[Situation, str] = {}
    gaps: dict[str, list[tuple[Situation, WeatherKind]]] = {}
    for (situation, kind), plan in before.items():
        if not good(plan):
            no_outfit.append((situation, kind))
        for gap in [*plan.gaps, *accessories.get((situation, kind), [])]:
            if gap.startswith(NO_BASE):
                why.setdefault(situation, gap.removeprefix(NO_BASE).strip())
            else:
                gaps.setdefault(gap, []).append((situation, kind))
    empty = list(dict.fromkeys(s for s, _ in no_outfit))  # group_where's order
    grouped = zip(empty, group_where(no_outfit, kinds), strict=True)
    return Missing(
        no_outfit=[f"{text} {why.get(s, '')}".strip() for s, text in grouped],
        gaps={gap: group_where(found, kinds) for gap, found in gaps.items()},
    )


def worth_suggesting(report: ShoppingReport) -> bool:
    if report.verdict != "buy":
        return False
    if report.candidate.category in ACCESSORY_SLOTS:
        return bool(report.unlocks or report.fixes)
    return True


def gain(staple: Staple, report: ShoppingReport, checks: list[Situation]) -> tuple[int | bool, ...]:
    """Unlocks, then gaps fixed, then clothes before accessories (a coat for the cold
    matters more than sunglasses), then new outfits; ties go to the staple made for more
    of the situations (running shoes over sneakers for sports)."""
    made_for = sum(s.occasion in staple.occasions for s in checks)
    garment = staple.category is not Category.ACCESSORY
    return report.unlocked, len(report.fixes), garment, report.new_outfits, made_for


def best_pair(
    staples: list[Staple],
    owned: list[ClothingItem],
    checks: list[Situation],
    kinds: list[WeatherKind],
    taste: OutfitRequest,
    before: Baseline,
) -> list[tuple[Staple, ShoppingReport]]:
    """A top and a bottom that together unlock the most empty situations, each scored
    with the other one owned."""
    empty = [(s, k) for (s, k), plan in before.items() if not good(plan)]
    if not empty:
        return []
    tops = [s for s in staples if s.category in TOPS]
    bottoms = [s for s in staples if s.category in BOTTOMS]
    best: tuple[int, ClothingItem, ClothingItem, Staple, Staple] | None = None
    for top_staple, bottom_staple in product(tops, bottoms):
        top = pick_color(top_staple, owned, taste)
        bottom = top and pick_color(bottom_staple, [*owned, top], taste)
        if top is None or bottom is None:
            continue
        both = [*owned, top, bottom]
        unlocked = sum(
            any(top.id in o.ids and bottom.id in o.ids for o in good(all_outfits(both, request)))
            for request in (situation_request(taste, s, k) for s, k in empty)
        )
        if unlocked and (best is None or unlocked > best[0]):
            best = unlocked, top, bottom, top_staple, bottom_staple
    if best is None:
        return []
    _, top, bottom, top_staple, bottom_staple = best
    return [
        (top_staple, check_candidate(top, [*owned, bottom], checks, kinds, taste=taste)),
        (bottom_staple, check_candidate(bottom, [*owned, top], checks, kinds, taste=taste)),
    ]


def suggest_purchases(
    wardrobe: list[ClothingItem],
    checks: list[Situation],
    kinds: list[WeatherKind],
    taste: OutfitRequest | None = None,
    gender: str | None = None,
    limit: int = MAX_SUGGESTIONS,
    staples: tuple[Staple, ...] = STAPLES,
) -> list[Suggestion]:
    """Up to `limit` staples, best first, each scored on top of the ones before it.

    A staple not worth suggesting in one round is dropped: adding pieces rarely makes
    another one more useful, and re-scoring every staple each round is the slow part.
    """
    taste = taste or OutfitRequest()
    everything = for_gender(gender, staples)
    remaining = list(everything)
    owned = list(wardrobe)
    picked: list[Suggestion] = []

    def take(staple: Staple, report: ShoppingReport, together_with: str | None = None) -> None:
        earlier = [label(s.report.candidate) for s in picked]
        picked.append(Suggestion(staple, report, earlier, together_with))
        owned.append(report.candidate)

    while len(picked) < limit:
        before = baseline(owned, checks, kinds, taste)
        scored: list[tuple[Staple, ShoppingReport]] = []
        for staple in remaining:
            item = pick_color(staple, owned, taste)
            if item is None:
                continue
            report = check_candidate(item, owned, checks, kinds, taste=taste, before=before)
            if worth_suggesting(report):
                scored.append((staple, report))
        best = max(scored, key=lambda pair: gain(*pair, checks), default=None)  # first of equals
        if (best is None or best[1].unlocked == 0) and len(picked) + 2 <= limit:
            unused = [s for s in everything if s not in {p.staple for p in picked}]
            pair = best_pair(unused, owned, checks, kinds, taste, before)
            if pair:
                (first, first_report), (second, second_report) = pair
                take(first, first_report, label(second_report.candidate))
                take(second, second_report, label(first_report.candidate))
                # a new situation opened up: shoes for it now count, so try everything again
                remaining = [s for s in unused if s not in (first, second)]
                continue
        if best is None:
            break
        take(*best)
        remaining = [s for s, _ in scored if s is not best[0]]
    return picked
