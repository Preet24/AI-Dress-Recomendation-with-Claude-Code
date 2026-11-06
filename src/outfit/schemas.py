"""Wardrobe data model. CLAUDE.md's "ClothingItem schema" section is the source of truth."""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field

# Namespace for deterministic item ids, so re-ingesting a photo updates instead of duplicating.
ITEM_ID_NAMESPACE = UUID("6f1c2b0e-5a7d-4e8b-9c3f-2d4a1b6e8f90")


class Category(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    ONE_PIECE = "one_piece"
    OUTERWEAR = "outerwear"
    FOOTWEAR = "footwear"
    ETHNIC_TOP = "ethnic_top"
    ETHNIC_BOTTOM = "ethnic_bottom"
    ACCESSORY = "accessory"


class Pattern(StrEnum):
    SOLID = "solid"
    STRIPED = "striped"
    CHECKED = "checked"
    FLORAL = "floral"
    PRINTED = "printed"
    OTHER = "other"


class Length(StrEnum):
    CROPPED = "cropped"
    SHORT = "short"
    KNEE = "knee"
    MIDI = "midi"
    FULL = "full"


class Fit(StrEnum):
    SLIM = "slim"
    REGULAR = "regular"
    RELAXED = "relaxed"
    OVERSIZED = "oversized"


class Occasion(StrEnum):
    CASUAL = "casual"
    OFFICE = "office"
    FORMAL = "formal"
    SPORTS = "sports"
    PARTY = "party"
    INDIAN_FESTIVE = "indian_festive"
    WEDDING = "wedding"
    TRAVEL = "travel"
    LOUNGE = "lounge"


Scale = Field(ge=1, le=5)


class ClothingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    image_path: Path  # cleaned, background-removed PNG
    category: Category
    subtype: str = Field(min_length=1)  # free text, e.g. "kurta", "jeans", "blazer"
    primary_color_name: str = Field(min_length=1)
    primary_color_lab: tuple[float, float, float]  # CIELAB (L*, a*, b*), computed locally
    secondary_colors: list[str] = Field(default_factory=list)
    pattern: Pattern
    fabric: str  # best guess: cotton, denim, linen, wool, silk, synthetic...
    length: Length
    fit: Fit
    formality: int = Scale  # 1 = gym, 5 = black tie / wedding
    warmth: int = Scale  # 1 = very light, 5 = heavy winter
    flowy: bool  # matters for wind
    occasions: list[Occasion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def stable_item_id(photo_bytes: bytes, region: str) -> UUID:
    """Deterministic id for one garment in one photo: same photo + region -> same id."""
    digest = hashlib.sha256(photo_bytes).hexdigest()
    return uuid5(ITEM_ID_NAMESPACE, f"{digest}:{region}")
