"""Color harmony between garments, from their measured CIELAB color and color name.

Lightness L* and chroma C* (how colorful) come from LAB. Hue is measured on the
artist's RYB color wheel that styling rules use (red-green, blue-orange and
yellow-purple are complements), not as the LAB hue angle: in LAB, red and green
are only ~96 degrees apart. Then:

- Neutrals (black, white, grey, beige, navy, denim...) go with everything. Whether a
  garment is neutral comes mainly from its color *name*: real photos are strongly
  desaturated (on the test wardrobe the most colorful item had C=19, and lavender
  and blush pink measured C<12), so a chroma threshold alone would call pastels
  neutral. Very low chroma is the fallback for unnamed greys.
- Two colors are scored by hue distance: analogous (close) and complementary
  (opposite) are good, an in-between gap of about 90 degrees is the weakest.
  Hue only counts as much as both colors are saturated: for washed-out colors the
  measured hue is noisy, so their score blends toward a gentle "soft colors" value.
- Some light-dark contrast between pieces helps; two loud colors that are not
  analogous are penalized; more than two non-neutral colors in one outfit is too.

Scores are 0..1. Every rule is a named constant, so the outcome is explainable.
"""

import colorsys
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from itertools import combinations

from outfit.ingestion.color import lab_to_rgb
from outfit.schemas import ClothingItem

NEUTRAL_WORDS = {
    "black",
    "white",
    "ivory",
    "cream",
    "grey",
    "gray",
    "charcoal",
    "silver",
    "beige",
    "tan",
    "camel",
    "khaki",
    "taupe",
    "sand",
    "stone",
    "nude",
    "champagne",
    "brown",
    "chocolate",
    "navy",
    "denim",
}
NEUTRAL_FABRICS = {"denim"}  # blue jeans act as a neutral whatever their measured hue
NEUTRAL_CHROMA = 6.0  # below this, a garment reads as grey whatever it is called
SOFT_CHROMA = 10.0  # at or below this, the measured hue is ignored (photo pastels)
FULL_CHROMA = 30.0  # chroma at which hue rules apply fully
LOUD_CHROMA = 50.0
LIGHT_L = 65.0  # "light color" for hot or sunny weather

# HSV hue -> RYB wheel hue (red 0, orange 60, yellow 120, green 180, blue 240, purple 300).
RGB_TO_RYB = [(0, 0), (30, 60), (60, 120), (120, 180), (240, 240), (300, 300), (360, 360)]
# Score by RYB hue distance in degrees (0..180), linearly interpolated between points.
HUE_CURVE = [(0, 0.85), (30, 0.85), (60, 0.5), (90, 0.4), (120, 0.6), (150, 0.75), (180, 0.8)]
SOFT_PAIR = 0.7  # two washed-out colors: hue too unreliable to judge
NEUTRAL_PAIR = 0.75
NEUTRAL_WITH_COLOR = 0.85  # the classic: neutral base, one color
CONTRAST_FULL_L = 40.0  # lightness gap that earns the full contrast bonus
CONTRAST_BONUS = 0.1
FLAT_PENALTY = 0.05  # no light-dark contrast at all
LOUD_PENALTY = 0.15
MAX_COLORS = 2  # non-neutral colors per outfit before a penalty
EXTRA_COLOR_PENALTY = 0.1


class Relation(StrEnum):
    NEUTRALS = "neutrals"
    NEUTRAL_BASE = "neutral base"
    TONE_ON_TONE = "tone on tone"
    ANALOGOUS = "analogous"
    COMPLEMENTARY = "complementary"
    TRIADIC = "triadic"
    AWKWARD = "awkward hue gap"
    SOFT = "soft colors"


@dataclass(frozen=True)
class Swatch:
    """A garment's color as harmony sees it."""

    name: str
    lightness: float  # L*, 0 black .. 100 white
    chroma: float  # C*, 0 grey .. ~130 vivid
    hue: float  # RYB wheel degrees, 0..360
    neutral: bool

    @property
    def light(self) -> bool:
        return self.lightness >= LIGHT_L


@dataclass(frozen=True)
class PairHarmony:
    score: float
    relation: Relation
    reason: str


@dataclass(frozen=True)
class OutfitHarmony:
    score: float
    pairs: list[PairHarmony]
    colors: list[str]  # the non-neutral color names
    reason: str


def swatch(item: ClothingItem) -> Swatch:
    lightness, a, b = item.primary_color_lab
    chroma = math.hypot(a, b)
    words = set(re.findall(r"[a-z]+", item.primary_color_name.lower()))
    neutral = (
        bool(words & NEUTRAL_WORDS)
        or item.fabric.lower() in NEUTRAL_FABRICS
        or chroma < NEUTRAL_CHROMA
    )
    return Swatch(item.primary_color_name, lightness, chroma, ryb_hue((lightness, a, b)), neutral)


def interpolate(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear lookup in sorted (x, y) points; clamps outside the range."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


@cache  # pure, and the slow part of scoring: outfit search asks again for every combination
def ryb_hue(lab: tuple[float, float, float]) -> float:
    """Hue of a LAB color on the RYB artist's wheel, degrees 0..360."""
    r, g, b = lab_to_rgb(lab)
    hsv_hue, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return interpolate(hsv_hue * 360, RGB_TO_RYB) % 360


def hue_distance(h1: float, h2: float) -> float:
    """Shortest angle between two hues, 0..180."""
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def hue_score(distance: float) -> float:
    return interpolate(distance, HUE_CURVE)


def hue_relation(distance: float) -> Relation:
    if distance <= 30:
        return Relation.ANALOGOUS
    if distance >= 150:
        return Relation.COMPLEMENTARY
    if 105 <= distance < 150:
        return Relation.TRIADIC
    return Relation.AWKWARD


def pair_harmony(a: Swatch, b: Swatch) -> PairHarmony:
    """How well two garment colors go together, 0..1, with the reason."""
    gap_l = abs(a.lightness - b.lightness)
    contrast = CONTRAST_BONUS * min(gap_l / CONTRAST_FULL_L, 1.0)
    if gap_l < 8:
        contrast -= FLAT_PENALTY

    if a.neutral and b.neutral:
        base, relation, reason = NEUTRAL_PAIR, Relation.NEUTRALS, "two neutrals"
    elif a.neutral or b.neutral:
        color = b if a.neutral else a
        base, relation = NEUTRAL_WITH_COLOR, Relation.NEUTRAL_BASE
        reason = f"neutral {(a if a.neutral else b).name} lets the {color.name} stand out"
    else:
        distance = hue_distance(a.hue, b.hue)
        # How far to trust the measured hue: fully only when both colors are saturated.
        weakest = min(a.chroma, b.chroma)
        trust = min(max((weakest - SOFT_CHROMA) / (FULL_CHROMA - SOFT_CHROMA), 0.0), 1.0)
        base = (1 - trust) * SOFT_PAIR + trust * hue_score(distance)
        relation = hue_relation(distance) if trust >= 0.5 else Relation.SOFT
        if relation is Relation.ANALOGOUS and gap_l < 15:
            relation = Relation.TONE_ON_TONE
        reason = f"{a.name} and {b.name}: {relation}"
        if min(a.chroma, b.chroma) >= LOUD_CHROMA and distance > 30:
            base -= LOUD_PENALTY
            reason += ", both loud"

    if gap_l >= CONTRAST_FULL_L / 2:
        reason += ", good light-dark contrast"
    elif gap_l < 8:
        reason += ", little contrast"
    return PairHarmony(round(min(max(base + contrast, 0.0), 1.0), 3), relation, reason)


def outfit_harmony(items: list[ClothingItem]) -> OutfitHarmony:
    """Mean pairwise harmony of all pieces, minus a penalty for too many colors."""
    swatches = [swatch(item) for item in items]
    if len(swatches) < 2:
        raise ValueError("An outfit needs at least two pieces to judge color harmony")
    pairs = [pair_harmony(a, b) for a, b in combinations(swatches, 2)]
    colors = list(dict.fromkeys(s.name for s in swatches if not s.neutral))
    score = sum(p.score for p in pairs) / len(pairs)
    extra = max(len(colors) - MAX_COLORS, 0)
    score -= EXTRA_COLOR_PENALTY * extra
    if not colors:
        reason = "all neutrals"
    elif len(colors) == 1:
        reason = f"neutrals with a {colors[0]} accent"
    else:
        reason = f"{len(colors)} colors: {', '.join(colors)}"
    if extra:
        reason += f" (more than {MAX_COLORS} colors is busy)"
    return OutfitHarmony(round(min(max(score, 0.0), 1.0), 3), pairs, colors, reason)
