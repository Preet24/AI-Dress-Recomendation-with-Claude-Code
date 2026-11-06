"""Which wardrobe pieces a photo of the user wearing an outfit shows (local CLIP only).

Each garment cut from the photo (`worn` segmentation: upper / lower, or full) is
compared with the wardrobe pieces of the same kind, then with the pieces the user
logged. A worn photo scores lower than a re-shot garment (the person is small in the
frame, the light differs): on the user's first 4 photos the right piece scored
0.73-0.82 and a wrong one up to 0.79, and dark trousers were only 0.01 apart. So no
score decides alone: the logged piece is kept unless another piece beats it by
`CLEAR_WIN`, anything under `NOT_IN_WARDROBE` is called unknown, and the user confirms.
Starting values, to tune on more photos.
"""

from uuid import UUID

import numpy as np

from outfit.schemas import Category, ClothingItem
from outfit.tools.outfits import label
from outfit.wear_log import GarmentMatch, PhotoMatch, Verdict

NOT_IN_WARDROBE = 0.70  # best match below this: probably not a wardrobe piece
CLEAR_WIN = 0.03  # another piece must beat the logged one by this much to doubt it

REGION_CATEGORIES: dict[str, frozenset[Category]] = {
    "upper": frozenset({Category.TOP, Category.ETHNIC_TOP, Category.OUTERWEAR}),
    "lower": frozenset({Category.BOTTOM, Category.ETHNIC_BOTTOM}),
    "full": frozenset({Category.ONE_PIECE}),
}
REGION_WORDS = {"upper": "top", "lower": "bottom", "full": "outfit"}


def ranked_matches(
    region: str, vector: np.ndarray, wardrobe: list[ClothingItem], vectors: np.ndarray
) -> list[tuple[float, ClothingItem]]:
    """Wardrobe pieces of the region's kind, most similar first (`vectors`: unit rows)."""
    if not wardrobe:
        return []
    kinds = REGION_CATEGORIES[region]
    pairs = zip((vectors @ vector).tolist(), wardrobe, strict=True)
    return sorted(((s, i) for s, i in pairs if i.category in kinds), key=lambda p: -p[0])


def verdict(
    best: tuple[float, ClothingItem] | None,
    mine: tuple[float, ClothingItem] | None,
    word: str,
) -> tuple[Verdict, str]:
    if best is None or best[0] < NOT_IN_WARDROBE:
        if mine is None:
            return Verdict.UNKNOWN, f"this {word} may not be in your wardrobe"
        return Verdict.UNKNOWN, (
            f"this {word} doesn't look like your {label(mine[1])} or anything else in your wardrobe"
        )
    if mine is None:
        return Verdict.EXTRA, f"this {word} looks like your {label(best[1])}, not in this outfit"
    if best[0] - mine[0] > CLEAR_WIN:
        return Verdict.UNSURE, (
            f"this {word} looks more like your {label(best[1])} "
            f"than your {label(mine[1])}: which was it?"
        )
    return Verdict.YOURS, f"your {label(mine[1])}: yes"


def match_garment(
    region: str,
    vector: np.ndarray,
    wardrobe: list[ClothingItem],
    vectors: np.ndarray,
    logged: set[UUID],
) -> GarmentMatch:
    """One cutout against the wardrobe and the entry's pieces."""
    ranked = ranked_matches(region, vector, wardrobe, vectors)
    best = ranked[0] if ranked else None
    mine = next((pair for pair in ranked if pair[1].id in logged), None)
    found, message = verdict(best, mine, REGION_WORDS[region])
    return GarmentMatch(
        region=region,
        verdict=found,
        logged=mine[1].id if mine else None,
        best=best[1].id if best else None,
        score=round(best[0], 3) if best else 0.0,
        logged_score=round(mine[0], 3) if mine else None,
        message=message,
    )


def match_photo(
    found: list[tuple[str, np.ndarray]],
    wardrobe: list[ClothingItem],
    vectors: np.ndarray,
    logged: list[ClothingItem],
) -> PhotoMatch:
    """Every cutout of the photo, and the logged pieces none of them could show."""
    ids = {i.id for i in logged}
    garments = [match_garment(r, v, wardrobe, vectors, ids) for r, v in found]
    shown = {g.logged for g in garments}
    return PhotoMatch(garments=garments, unchecked=[i.id for i in logged if i.id not in shown])
