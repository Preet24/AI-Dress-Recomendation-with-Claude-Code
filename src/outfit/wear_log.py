"""The wear log: outfits the user wore, with their rating ("I wore this").

One JSON file per install (`Settings.wear_log_path`, gitignored), written like the
profile: a temp file swapped in, so it is never half-written; `path=None` keeps it in
memory (tests). The log only records; how it nudges outfit ranking is a separate step
(`docs/features_to_add.md` §5). Deleting a garment leaves its entries: the API shows
such pieces as removed.

An entry may have a photo of the user wearing the outfit: a re-encoded copy (upright,
no EXIF) in `worn/` next to the log file, named by the entry id, with the local CLIP
match report (`PhotoMatch`, from `tools/worn_match.py`) kept on the entry.
"""

import logging
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from outfit.profile import reencode
from outfit.schemas import Occasion

logger = logging.getLogger(__name__)

MAX_NOTE = 500
PHOTO_MAX_SIDE = 1024  # = Settings.max_side, so segmentation sees the kept copy as it is
PHOTO_DIR = "worn"


class Rating(StrEnum):
    LOVED = "loved"
    FINE = "fine"
    NOT_AGAIN = "not_again"


class Issue(StrEnum):
    """What was off; any number of them."""

    TOO_WARM = "too_warm"
    TOO_COLD = "too_cold"
    TOO_FORMAL = "too_formal"
    TOO_CASUAL = "too_casual"
    COLOURS = "colours"  # the colours didn't go together
    UNCOMFORTABLE = "uncomfortable"


Source = Literal["chat", "packing", "other"]  # where the outfit was suggested


class Verdict(StrEnum):
    """What one garment cut from the photo turned out to be."""

    YOURS = "yours"  # the logged piece is the best match
    UNSURE = "unsure"  # another piece of the wardrobe clearly matches better: ask
    EXTRA = "extra"  # matches a wardrobe piece that isn't in the entry
    UNKNOWN = "unknown"  # nothing in the wardrobe looks like it


class GarmentMatch(BaseModel):
    region: Literal["upper", "lower", "full"]
    verdict: Verdict
    logged: UUID | None  # the entry's piece this garment was compared with
    best: UUID | None  # the closest wardrobe piece of the same kind
    score: float  # CLIP similarity of `best`
    logged_score: float | None
    message: str  # plain words for the user


class PhotoMatch(BaseModel):
    garments: list[GarmentMatch]
    unchecked: list[UUID]  # logged pieces the photo can't show (shoes, accessories, ...)


class LookScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=200)


class LookScores(BaseModel):
    """DeepSeek's view of the outfit as worn in the photo (`outfit/look.py`), never the person.

    Shown in History; it changes neither the rating nor the ranking."""

    fit: LookScore
    proportion: LookScore  # lengths and proportions of the pieces together
    colour: LookScore  # the colours on the user (their colouring, if known)
    coordination: LookScore  # how the pieces go together
    occasion: LookScore  # right for the occasion
    polish: LookScore  # overall finish: pressed, tucked, shoes, accessories
    tips: list[str] = Field(default_factory=list, max_length=2)


class WearEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    worn_on: date
    occasion: Occasion | None = None
    pieces: list[UUID] = Field(min_length=1)  # what was actually worn
    rating: Rating
    issues: list[Issue] = Field(default_factory=list)
    note: str = Field("", max_length=MAX_NOTE)  # for the user; not used in ranking
    felt: tuple[float, float] | None = None  # feels-like range of the outing, °C, if known
    source: Source = "other"
    created_at: datetime = Field(default_factory=datetime.now)
    photo: bool = False  # a photo of the user wearing it is kept
    match: PhotoMatch | None = None  # what the photo showed, for the user to confirm
    look: LookScores | None = None  # only after "Ask DeepSeek about this look"

    @model_validator(mode="after")
    def _clean(self) -> "WearEntry":
        self.pieces = list(dict.fromkeys(self.pieces))
        self.issues = list(dict.fromkeys(self.issues))
        self.note = self.note.strip()
        if Issue.TOO_WARM in self.issues and Issue.TOO_COLD in self.issues:
            raise ValueError("an outfit can't be both too warm and too cold")
        if Issue.TOO_FORMAL in self.issues and Issue.TOO_CASUAL in self.issues:
            raise ValueError("an outfit can't be both too formal and too casual")
        return self


class WearLog(BaseModel):
    entries: list[WearEntry] = Field(default_factory=list)


class WearLogStore:
    """The wear log as a JSON file; `path=None` keeps it in memory (tests)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._log = WearLog()
        self._photos: dict[UUID, bytes] = {}
        self._lock = Lock()

    def photo_path(self, entry_id: UUID) -> Path | None:
        return self.path.with_name(PHOTO_DIR) / f"{entry_id}.jpg" if self.path else None

    def _read(self) -> WearLog:
        if self.path is None or not self.path.is_file():
            return self._log.model_copy(deep=True)
        return WearLog.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _write(self, log: WearLog) -> None:
        if self.path is None:
            self._log = log.model_copy(deep=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".tmp")
        partial.write_text(log.model_dump_json(indent=2), encoding="utf-8")
        partial.replace(self.path)

    def entries(self) -> list[WearEntry]:
        """Newest first: by the day worn, then by when it was logged."""
        with self._lock:
            entries = self._read().entries
        return sorted(entries, key=lambda e: (e.worn_on, e.created_at), reverse=True)

    def get(self, entry_id: UUID) -> WearEntry | None:
        return next((e for e in self.entries() if e.id == entry_id), None)

    def add(self, entry: WearEntry) -> WearEntry:
        with self._lock:
            log = self._read()
            log.entries.append(entry)
            self._write(log)
        logger.info("Logged %s outfit worn on %s", entry.rating, entry.worn_on)
        return entry

    def replace(self, entry: WearEntry) -> bool:
        """Swap in a changed entry (same id); False if there was none with that id."""
        with self._lock:
            log = self._read()
            ids = [e.id for e in log.entries]
            if entry.id not in ids:
                return False
            log.entries[ids.index(entry.id)] = entry
            self._write(log)
        return True

    def delete(self, entry_id: UUID) -> bool:
        """Remove one entry and its photo; False if there was none with that id."""
        with self._lock:
            log = self._read()
            kept = [e for e in log.entries if e.id != entry_id]
            if len(kept) == len(log.entries):
                return False
            self._write(WearLog(entries=kept))
        self.delete_photo(entry_id)
        return True

    # --- Photos ---------------------------------------------------------------

    def save_photo(self, entry_id: UUID, data: bytes) -> bytes:
        """Keep a re-encoded copy (upright, at most PHOTO_MAX_SIDE, no EXIF); returns it."""
        jpeg = reencode(data, PHOTO_MAX_SIDE)
        with self._lock:
            if (path := self.photo_path(entry_id)) is None:
                self._photos[entry_id] = jpeg
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpeg)
        return jpeg

    def photo(self, entry_id: UUID) -> bytes | None:
        with self._lock:
            if (path := self.photo_path(entry_id)) is None:
                return self._photos.get(entry_id)
            return path.read_bytes() if path.is_file() else None

    def delete_photo(self, entry_id: UUID) -> bool:
        """Remove the photo file only (the entry's `photo`/`match` are the caller's)."""
        with self._lock:
            if (path := self.photo_path(entry_id)) is None:
                return self._photos.pop(entry_id, None) is not None
            if not path.is_file():
                return False
            path.unlink()
            return True
