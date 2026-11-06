"""Wear-log endpoints: "I wore this", with a rating; list, correct and delete entries.

Free: no DeepSeek call, except `POST /worn/{id}/look` (one, when the user asks). The
pieces must be in the wardrobe when logged; a garment deleted later stays in its
entries and is listed under `removed`. An entry's photo stays on this machine: it is cut
out and compared with the wardrobe by CLIP (`tools/worn_match.py`), and the report says
which pieces it shows; the user fixes the pieces with `PATCH /worn/{id}`, which checks
the photo again. Only "look" sends a re-encoded copy (no EXIF) to DeepSeek.
"""

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, Query, Response, UploadFile
from pydantic import BaseModel, Field, ValidationError

from outfit.api.errors import BadWearLog, NotFound
from outfit.api.items import ItemOut, photo_suffix
from outfit.api.services import Services, get_services
from outfit.look import score_look
from outfit.profile import reencode
from outfit.schemas import ClothingItem, Occasion
from outfit.tools.worn_match import match_photo
from outfit.wear_log import (
    MAX_NOTE,
    PHOTO_MAX_SIDE,
    Issue,
    LookScores,
    PhotoMatch,
    Rating,
    Source,
    WearEntry,
)

router = APIRouter(prefix="/worn", tags=["worn"])
ServicesDep = Annotated[Services, Depends(get_services)]


class WornIn(BaseModel):
    pieces: list[UUID] = Field(min_length=1)  # what was actually worn
    rating: Rating  # required, even with a photo
    issues: list[Issue] = []
    note: str = Field("", max_length=MAX_NOTE)
    worn_on: date | None = None  # default: today
    occasion: Occasion | None = None
    felt: tuple[float, float] | None = None  # feels-like range of the outing, if known
    source: Source = "other"


class WornOut(BaseModel):
    id: UUID
    worn_on: date
    occasion: Occasion | None
    rating: Rating
    issues: list[Issue]
    note: str
    felt: tuple[float, float] | None
    source: Source
    created_at: datetime
    pieces: list[ItemOut]  # still in the wardrobe
    removed: int  # pieces deleted from the wardrobe since
    photo_url: str | None  # the kept photo of the user wearing it
    match: PhotoMatch | None  # what the photo showed
    look: LookScores | None  # DeepSeek's scores for the outfit, if the user asked


class PiecesIn(BaseModel):
    pieces: list[UUID] = Field(min_length=1)  # what was actually worn, after the check


def today(s: Services) -> date:
    return (s.context.now or datetime.now()).date()


def worn_out(entry: WearEntry, s: Services) -> WornOut:
    items = [s.store.get(i) for i in entry.pieces]
    return WornOut(
        **entry.model_dump(exclude={"pieces", "photo"}),
        pieces=[ItemOut.of(i) for i in items if i is not None],
        removed=sum(i is None for i in items),
        photo_url=f"/worn/{entry.id}/photo" if entry.photo else None,
    )


def stored_entry(entry_id: UUID, s: Services) -> WearEntry:
    if (entry := s.wear_log.get(entry_id)) is None:
        raise NotFound(f"No logged outfit {entry_id}")
    return entry


def in_wardrobe(pieces: list[UUID], s: Services) -> None:
    if missing := [str(i)[:8] for i in pieces if s.store.get(i) is None]:
        raise BadWearLog(f"not in your wardrobe: {', '.join(missing)}")


def check_photo(entry: WearEntry, jpeg: bytes, s: Services) -> PhotoMatch:
    """Cut the photo's garments out and match them with the wardrobe (CPU, local)."""
    with s.ingest_lock:
        found = s.pipeline().worn_vectors(jpeg)
    wardrobe: list[ClothingItem] = []
    rows = []
    for item in s.store.list_items():
        if (vector := s.store.vector(item.id)) is not None:
            wardrobe.append(item)
            rows.append(vector)
    vectors = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
    logged = [i for i in (s.store.get(p) for p in entry.pieces) if i is not None]
    return match_photo(found, wardrobe, vectors, logged)


@router.post("", status_code=201)
def log_worn(body: WornIn, s: ServicesDep) -> WornOut:
    """I wore this: the pieces worn, a rating and what was off."""
    worn_on = body.worn_on or today(s)
    if worn_on > today(s):
        raise BadWearLog(f"{worn_on} is in the future: log an outfit after wearing it")
    in_wardrobe(body.pieces, s)
    try:
        entry = WearEntry(**body.model_dump(exclude={"worn_on"}), worn_on=worn_on)
    except ValidationError as err:
        raise BadWearLog(err.errors()[0]["msg"].removeprefix("Value error, ")) from err
    return worn_out(s.wear_log.add(entry), s)


@router.get("")
def list_worn(s: ServicesDep, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[WornOut]:
    """Logged outfits, newest first."""
    return [worn_out(e, s) for e in s.wear_log.entries()[:limit]]


@router.patch("/{entry_id}")
def correct_pieces(entry_id: UUID, body: PiecesIn, s: ServicesDep) -> WornOut:
    """Set what was actually worn (after the photo check); the photo is checked again."""
    entry = stored_entry(entry_id, s)
    in_wardrobe(body.pieces, s)
    entry = entry.model_copy(update={"pieces": list(dict.fromkeys(body.pieces))})
    if entry.photo and (jpeg := s.wear_log.photo(entry.id)) is not None:
        entry.match = check_photo(entry, jpeg, s)
    s.wear_log.replace(entry)
    return worn_out(entry, s)


@router.delete("/{entry_id}", status_code=204)
def delete_worn(entry_id: UUID, s: ServicesDep) -> Response:
    """Remove the entry and its photo."""
    if not s.wear_log.delete(entry_id):
        raise NotFound(f"No logged outfit {entry_id}")
    return Response(status_code=204)


@router.put("/{entry_id}/photo")
def upload_photo(entry_id: UUID, photo: UploadFile, s: ServicesDep) -> WornOut:
    """A photo of you wearing it: kept here (no EXIF) and matched locally, never sent."""
    entry = stored_entry(entry_id, s)
    max_mb = s.settings.max_upload_mb
    data = photo.file.read(int(max_mb * 1024 * 1024) + 1)
    photo_suffix(data, max_mb)  # checks it really is a JPEG / PNG / WEBP under the limit
    # Checked first: a photo with no garment in it (SegmentationError, 422) is not kept.
    match = check_photo(entry, reencode(data, PHOTO_MAX_SIDE), s)
    s.wear_log.save_photo(entry.id, data)
    # Look scores were about the old photo.
    entry = entry.model_copy(update={"photo": True, "match": match, "look": None})
    s.wear_log.replace(entry)
    return worn_out(entry, s)


@router.post("/{entry_id}/look")
def ask_about_look(entry_id: UUID, s: ServicesDep) -> LookScores:
    """Send a copy of the photo to DeepSeek once (1 call): scores for the outfit as worn."""
    entry = stored_entry(entry_id, s)
    if (photo := s.wear_log.photo(entry.id)) is None:
        raise NotFound("No photo for this outfit; add one first")
    pieces = [i for i in (s.store.get(p) for p in entry.pieces) if i is not None]
    colouring = s.profile.load().colouring
    look = score_look(photo, pieces, entry.occasion, colouring, s.client, s.settings)
    s.wear_log.replace(entry.model_copy(update={"look": look}))
    return look


@router.get("/{entry_id}/photo")
def read_photo(entry_id: UUID, s: ServicesDep) -> Response:
    if (photo := s.wear_log.photo(stored_entry(entry_id, s).id)) is None:
        raise NotFound("No photo for this outfit")
    return Response(photo, media_type="image/jpeg")


@router.delete("/{entry_id}/photo", status_code=204)
def delete_photo(entry_id: UUID, s: ServicesDep) -> Response:
    """Remove the photo and its match report; the entry stays."""
    entry = stored_entry(entry_id, s)
    if not s.wear_log.delete_photo(entry.id) and not entry.photo:
        raise NotFound("No photo for this outfit")
    s.wear_log.replace(entry.model_copy(update={"photo": False, "match": None}))
    return Response(status_code=204)
