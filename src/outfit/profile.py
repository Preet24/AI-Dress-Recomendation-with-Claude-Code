"""The user's profile: who they are and the places they go (docs/features_to_add.md).

One profile per install, kept in `data/profile.json`. Every field is optional and every
field changes something the app does; nothing is stored "just in case". The optional
photo is kept next to the profile (`profile_photo.jpg`, re-encoded without EXIF) and
leaves the machine only when the user asks for a colouring analysis.
"""

import io
import logging
from datetime import time
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator

from outfit.schemas import Fit, Occasion
from outfit.tools.weather import Transport

logger = logging.getLogger(__name__)

MAX_PLACES = 20
MAX_COLORS = 12
PHOTO_MAX_SIDE = 1024  # the kept copy; the analysis sends a smaller one
PHOTO_NAME = "profile_photo.jpg"


class Gender(StrEnum):
    MAN = "man"
    WOMAN = "woman"
    NON_BINARY = "non_binary"


class Build(StrEnum):
    SLIM = "slim"
    AVERAGE = "average"
    ATHLETIC = "athletic"
    BROAD = "broad"
    CURVY = "curvy"
    PLUS_SIZE = "plus_size"


class DressCode(StrEnum):
    FORMAL = "formal"
    BUSINESS = "business"
    BUSINESS_CASUAL = "business_casual"
    SMART_CASUAL = "smart_casual"
    CASUAL = "casual"


# A place's dress code replaces the occasion's default formality range.
DRESS_CODE_FORMALITY: dict[DressCode, tuple[int, int]] = {
    DressCode.FORMAL: (4, 5),
    DressCode.BUSINESS: (4, 4),
    DressCode.BUSINESS_CASUAL: (3, 4),
    DressCode.SMART_CASUAL: (3, 3),
    DressCode.CASUAL: (2, 3),
}


def short_text(max_length: int):
    return Field(default=None, min_length=1, max_length=max_length)


def clean_colors(colors: list[str]) -> list[str]:
    """Lower-case, trimmed, no blanks or repeats, order kept."""
    cleaned = (c.strip().lower() for c in colors)
    return list(dict.fromkeys(c for c in cleaned if c))


class Colouring(BaseModel):
    """The colour analysis, as editable text; the photo is never used again after it."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=300)  # "warm undertone, medium depth, ..."
    suits: list[str] = Field(default_factory=list, max_length=MAX_COLORS)
    go_easy: list[str] = Field(default_factory=list, max_length=MAX_COLORS)

    @field_validator("suits", "go_easy")
    @classmethod
    def _clean_colors(cls, colors: list[str]) -> list[str]:
        return clean_colors(colors)


class SavedPlace(BaseModel):
    """A place the user goes often, with its dress code and usual trip."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=40, examples=["Office"])
    occasion: Occasion
    dress_code: DressCode | None = None  # None = the occasion's usual formality
    start: time | None = None  # usual start, e.g. 09:00
    hours_away: float | None = Field(default=None, gt=0, le=24)
    hours_outdoors: float | None = Field(default=None, ge=0, le=24)
    transport: Transport | None = None

    @field_validator("name")
    @classmethod
    def strip_name(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("a place needs a name")
        return name.strip()

    def formality(self) -> tuple[int, int] | None:
        """The dress code's range, or None to keep the occasion's default."""
        return DRESS_CODE_FORMALITY[self.dress_code] if self.dress_code else None


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Gender | None = None  # reply wording, shopping suggestions
    home_city: str | None = short_text(80)  # default city when none is said
    height_cm: int | None = Field(default=None, ge=100, le=230)  # fit tips
    build: Build | None = None  # fit tips
    waist: str | None = short_text(20)  # shopping only, e.g. "32" or "81 cm"
    top_size: str | None = short_text(10)  # shopping only, e.g. "M"
    fit_tops: Fit | None = None  # ranking: preferred fits score a little higher
    fit_bottoms: Fit | None = None
    colors_liked: list[str] = Field(default_factory=list, max_length=MAX_COLORS)
    colors_avoided: list[str] = Field(default_factory=list, max_length=MAX_COLORS)  # never banned
    places: list[SavedPlace] = Field(default_factory=list, max_length=MAX_PLACES)
    colouring: Colouring | None = None  # from the photo analysis, editable; ranking
    occasions: list[Occasion] = Field(default_factory=list)  # shopping: what they dress for

    @field_validator("colors_liked", "colors_avoided")
    @classmethod
    def _clean_colors(cls, colors: list[str]) -> list[str]:
        return clean_colors(colors)

    @field_validator("places")
    @classmethod
    def unique_place_names(cls, places: list[SavedPlace]) -> list[SavedPlace]:
        names = [p.name.lower() for p in places]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"place names must be unique: {', '.join(repeated)}")
        return places

    @field_validator("occasions")
    @classmethod
    def _unique_occasions(cls, occasions: list[Occasion]) -> list[Occasion]:
        return list(dict.fromkeys(occasions))

    def taste_fields(self) -> dict[str, Any]:
        """The taste fields of `tools.outfits.OutfitRequest`, from this profile."""
        colouring = self.colouring
        return {
            "fit_tops": self.fit_tops,
            "fit_bottoms": self.fit_bottoms,
            "colors_liked": frozenset(self.colors_liked),
            "colors_avoided": frozenset(self.colors_avoided),
            "colors_suit": frozenset(colouring.suits if colouring else ()),
            "colors_go_easy": frozenset(colouring.go_easy if colouring else ()),
        }

    def place(self, name: str) -> SavedPlace | None:
        """The saved place called `name` (any case), if there is one."""
        return next((p for p in self.places if p.name.lower() == name.strip().lower()), None)


class ProfileStore:
    """The profile as a JSON file; `path=None` keeps it in memory (tests)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._profile = Profile()
        self._photo: bytes | None = None
        self._lock = Lock()

    @property
    def photo_path(self) -> Path | None:
        return self.path.with_name(PHOTO_NAME) if self.path else None

    def photo(self) -> bytes | None:
        """The kept photo (JPEG), if there is one."""
        with self._lock:
            if self.photo_path is None:
                return self._photo
            return self.photo_path.read_bytes() if self.photo_path.is_file() else None

    def save_photo(self, data: bytes) -> bytes:
        """Keep a re-encoded copy: upright, at most PHOTO_MAX_SIDE, no EXIF (GPS, camera)."""
        jpeg = reencode(data, PHOTO_MAX_SIDE)
        with self._lock:
            if self.photo_path is None:
                self._photo = jpeg
            else:
                self.photo_path.parent.mkdir(parents=True, exist_ok=True)
                self.photo_path.write_bytes(jpeg)
        return jpeg

    def delete_photo(self) -> bool:
        """Remove the photo; the colouring text stays. False if there was none."""
        with self._lock:
            if self.photo_path is None:
                had, self._photo = self._photo is not None, None
                return had
            if not self.photo_path.is_file():
                return False
            self.photo_path.unlink()
            return True

    def load(self) -> Profile:
        with self._lock:
            if self.path is None or not self.path.is_file():
                return self._profile.model_copy(deep=True)
            return Profile.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, profile: Profile) -> Profile:
        """Replace the whole profile; the file is swapped in at once, never half-written."""
        with self._lock:
            if self.path is None:
                self._profile = profile.model_copy(deep=True)
                return profile
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(".tmp")
            partial.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
            partial.replace(self.path)
            logger.info("Saved the profile to %s", self.path)
            return profile


def reencode(data: bytes, max_side: int) -> bytes:
    """A JPEG copy turned upright and shrunk to `max_side`, with no metadata kept."""
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)  # no exif= argument: nothing carried over
    return buffer.getvalue()
