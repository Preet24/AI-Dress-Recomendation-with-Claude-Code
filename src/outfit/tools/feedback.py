"""What the wear log means for the next suggestions, in plain rules.

- Pairings (top + bottom, or a one-piece with its shoes; `outfits.pairing`): the
  latest rating of a pairing decides. "Loved it" makes it a liked pair; "not again",
  or "the colours didn't go", a disliked one; "fine" clears both.
- Recent: base garments worn in the last `RECENT_DAYS` days rank a little lower.
- Warmth: once "too cold" outnumbers "too warm" by `WARMTH_REPORTS` (or the other way
  round), the weather's warmth range shifts one point for this user, and the reply says
  so. Only outings with a forecast are shifted.
- Too formal / too casual / uncomfortable are kept in the log but not used yet.

Pure: the caller passes the log, the wardrobe and today's date. Nothing is ever banned:
the scores move by the constants in tools/outfits.py, each with a reason.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from outfit.schemas import ClothingItem
from outfit.tools.outfits import ACCESSORY_SLOTS, pairing
from outfit.tools.weather import WeatherAdvice
from outfit.wear_log import Issue, Rating, WearEntry

RECENT_DAYS = 7  # worn on or after today minus this many days
WARMTH_REPORTS = 2  # "too cold" this many more times than "too warm" -> one point warmer


@dataclass(frozen=True)
class Feedback:
    liked_pairs: frozenset[frozenset[UUID]] = frozenset()
    disliked_pairs: frozenset[frozenset[UUID]] = frozenset()
    recent: frozenset[UUID] = frozenset()
    too_cold: int = 0  # entries that said so
    too_warm: int = 0

    @property
    def warmth_bias(self) -> int:
        """+1: dress one point warmer; -1: one cooler; 0: as the weather says."""
        diff = self.too_cold - self.too_warm
        return 1 if diff >= WARMTH_REPORTS else -1 if diff <= -WARMTH_REPORTS else 0

    def request_fields(self, recent: bool = True) -> dict[str, Any]:
        """The history fields of `OutfitRequest`; `recent=False` for a future trip."""
        return {
            "liked_pairs": self.liked_pairs,
            "disliked_pairs": self.disliked_pairs,
            "recent": self.recent if recent else frozenset(),
        }

    def warmth_note(self) -> str | None:
        """The assumption to pass on when the warmth range was shifted."""
        if not self.warmth_bias:
            return None
        way = "warmer" if self.warmth_bias > 0 else "lighter"
        return (
            f"you've found outfits too cold {times(self.too_cold)} and too warm "
            f"{times(self.too_warm)}, so this picks one warmth point {way}"
        )


def times(n: int) -> str:
    return {0: "never", 1: "once", 2: "twice"}.get(n, f"{n} times")


def feedback(entries: Sequence[WearEntry], items: Sequence[ClothingItem], today: date) -> Feedback:
    """The wear log turned into ranking fields. Pieces no longer in the wardrobe are
    skipped; the latest entry for a pairing wins."""
    by_id = {i.id: i for i in items}
    liked: set[frozenset[UUID]] = set()
    disliked: set[frozenset[UUID]] = set()
    recent: set[UUID] = set()
    since = today - timedelta(days=RECENT_DAYS)
    for entry in sorted(entries, key=lambda e: (e.worn_on, e.created_at)):
        pieces = [by_id[i] for i in entry.pieces if i in by_id]
        if since <= entry.worn_on <= today:
            recent |= {p.id for p in pieces if p.category not in ACCESSORY_SLOTS}
        key = pairing(pieces)
        if key is None:
            continue
        liked.discard(key)
        disliked.discard(key)
        if entry.rating is Rating.NOT_AGAIN or Issue.COLOURS in entry.issues:
            disliked.add(key)
        elif entry.rating is Rating.LOVED:
            liked.add(key)
    return Feedback(
        liked_pairs=frozenset(liked),
        disliked_pairs=frozenset(disliked),
        recent=frozenset(recent),
        too_cold=sum(Issue.TOO_COLD in e.issues for e in entries),
        too_warm=sum(Issue.TOO_WARM in e.issues for e in entries),
    )


def shifted(advice: WeatherAdvice | None, bias: int) -> WeatherAdvice | None:
    """The advice with its warmth range (and outerwear's minimum) moved by `bias`, kept
    within 1-5."""
    if advice is None or not bias:
        return advice

    def clamp(n: int) -> int:
        return min(max(n + bias, 1), 5)

    low, high = advice.warmth
    outer = advice.outerwear_min_warmth
    return advice.model_copy(
        update={
            "warmth": (clamp(low), clamp(high)),
            "outerwear_min_warmth": clamp(outer) if outer is not None else None,
        }
    )
