"""Variety check for a multi-day plan: does it repeat the same clothes?

Looks only at the base garments (tops, bottoms, one-pieces): shoes, outerwear and
accessories are meant to repeat. Two slots on the same day may share an outfit (a
day out, then a party), so everything is counted in distinct days.

- `same outfit`: the same base (top + bottom, or the same one-piece) on 2+ days.
- `over limit`: a garment worn on more days than its re-wear limit (`Wears`).
- `two days running`: the same top or one-piece on consecutive days. Only a note:
  the planner avoids it when the packed pieces leave a choice.

Problems are the first two; `scripts/check_variety.py` exits 1 when there are any.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from outfit.schemas import Category, ClothingItem
from outfit.tools.outfits import ACCESSORY_SLOTS, TOPS, label
from outfit.tools.packing import PackingPlan, Wears, days_text

Kind = Literal["same outfit", "over limit", "two days running"]
PROBLEMS: frozenset[Kind] = frozenset({"same outfit", "over limit"})


@dataclass(frozen=True)
class Repeat:
    kind: Kind
    pieces: tuple[str, ...]  # labels, "white oxford shirt"
    days: tuple[int, ...]
    limit: int | None = None  # for "over limit"

    @property
    def problem(self) -> bool:
        return self.kind in PROBLEMS

    def __str__(self) -> str:
        what = " + ".join(self.pieces)
        when = days_text(self.days)
        if self.kind == "over limit":
            return f"over limit: {what} on {when} (limit {self.limit})"
        return f"{self.kind}: {what} on {when}"


def base_pieces(pieces: tuple[ClothingItem, ...]) -> tuple[ClothingItem, ...]:
    return tuple(p for p in pieces if p.category not in ACCESSORY_SLOTS)


def check_variety(plan: PackingPlan, wears: Wears) -> list[Repeat]:
    """Every repeat in the plan: problems first, then notes, each by first day."""
    outfit_days: dict[frozenset[UUID], set[int]] = defaultdict(set)
    outfit_names: dict[frozenset[UUID], tuple[str, ...]] = {}
    piece_days: dict[UUID, set[int]] = defaultdict(set)
    items: dict[UUID, ClothingItem] = {}
    for day in plan.days:
        if day.outfit is None:
            continue
        base = base_pieces(day.outfit.pieces)
        if not base:
            continue
        key = frozenset(p.id for p in base)
        outfit_days[key].add(day.slot.day)
        outfit_names.setdefault(key, tuple(label(p) for p in base))
        for p in base:
            piece_days[p.id].add(day.slot.day)
            items[p.id] = p

    found = [
        Repeat("same outfit", outfit_names[key], tuple(sorted(days)))
        for key, days in outfit_days.items()
        if len(days) >= 2
    ]
    for pid, days in piece_days.items():
        item = items[pid]
        limit = wears.limit(item)
        if limit is not None and len(days) > limit:
            found.append(Repeat("over limit", (label(item),), tuple(sorted(days)), limit))
        if item.category in TOPS | {Category.ONE_PIECE}:
            running = sorted(d for d in days if d + 1 in days)
            if running:
                run_days = tuple(sorted({*running, *(d + 1 for d in running)}))
                found.append(Repeat("two days running", (label(item),), run_days))
    return sorted(found, key=lambda r: (not r.problem, r.days, r.kind))
