"""Wardrobe endpoints: list, get, cutout image, photo upload, re-tag, tag edits, delete.

Uploads run the same `Pipeline.ingest_image` as `scripts/ingest_folder.py`; edits and
re-tagging use the same `review` functions as `scripts/review_wardrobe.py`, so only
the tag fields (`TagResult`) are editable.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, create_model

from outfit.api.errors import BadUpload, NotFound
from outfit.api.services import Services, get_services
from outfit.db.qdrant_store import ItemFilter, WardrobeStore
from outfit.ingestion.background import SegMode
from outfit.ingestion.review import EditError, apply_field_edits, remove_item, retag_with_note
from outfit.ingestion.tagger import TagResult
from outfit.schemas import Category, ClothingItem, Occasion
from outfit.tools.outfits import label

router = APIRouter(prefix="/items", tags=["wardrobe"])
ServicesDep = Annotated[Services, Depends(get_services)]


class ItemOut(ClothingItem):
    """A wardrobe item as the API shows it: the cutout as a URL, never a local path."""

    image_path: Path = Field(exclude=True)
    name: str  # what the agents call it, e.g. "lavender skirt"
    image_url: str

    @classmethod
    def of(cls, item: ClothingItem) -> "ItemOut":
        return cls(**dict(item), name=label(item), image_url=f"/items/{item.id}/image")


# Every tag field, all optional: send only what changes. Built from TagResult so the
# editable fields can't drift; values are validated by apply_field_edits.
TagEdits = create_model(
    "TagEdits",
    __config__=ConfigDict(extra="forbid"),
    **{name: (f.annotation | None, None) for name, f in TagResult.model_fields.items()},
)


class RetagNote(BaseModel):
    note: str = Field(min_length=1, max_length=500, examples=["this is a skirt, not shorts"])


PHOTO_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}  # what ingest_folder takes


def get_item(store: WardrobeStore, item_id: UUID) -> ClothingItem:
    item = store.get(item_id)
    if item is None:
        raise NotFound(f"No item {item_id}")
    return item


def cutout_of(item: ClothingItem) -> Path:
    if not item.image_path.is_file():
        raise NotFound(f"The cutout of item {item.id} is missing")
    return item.image_path


def photo_suffix(data: bytes, max_mb: float) -> str:
    """File extension for an uploaded photo, after checking the bytes really are one
    (the client's filename and content type are not trusted)."""
    if len(data) > max_mb * 1024 * 1024:
        raise BadUpload(f"The photo is larger than {max_mb:g} MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            image.load()  # decode it all: verify() alone passes a truncated JPEG
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise BadUpload("Not a readable image") from err
    if image_format not in PHOTO_FORMATS:
        raise BadUpload(f"Unsupported image type {image_format}; send JPEG, PNG or WEBP")
    return PHOTO_FORMATS[image_format]


def photo_path(raw_dir: Path, filename: str | None, data: bytes, suffix: str) -> Path:
    """Where to keep an upload: its own name, made safe (no folders, no odd characters),
    with a content hash added if a different photo already has that name."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename or "").stem).strip("-")[:60] or "photo"
    path = raw_dir / f"{stem}{suffix}"
    if path.exists() and path.read_bytes() != data:
        path = raw_dir / f"{stem}-{hashlib.sha256(data).hexdigest()[:8]}{suffix}"
    return path


@router.get("")
def list_items(
    s: ServicesDep,
    category: Annotated[list[Category] | None, Query()] = None,
    occasion: Annotated[list[Occasion] | None, Query()] = None,
) -> list[ItemOut]:
    """The wardrobe, optionally only some categories or occasions (repeat to allow several)."""
    filters = ItemFilter(categories=category or (), occasions=occasion or ())
    order = list(Category)  # tops, bottoms, one-pieces, ... as the schema lists them
    items = sorted(s.store.list_items(filters), key=lambda i: (order.index(i.category), label(i)))
    return [ItemOut.of(item) for item in items]


@router.get("/{item_id}")
def read_item(item_id: UUID, s: ServicesDep) -> ItemOut:
    return ItemOut.of(get_item(s.store, item_id))


@router.get("/{item_id}/image", response_class=FileResponse)
def item_image(item_id: UUID, s: ServicesDep) -> FileResponse:
    """The background-removed cutout (PNG)."""
    return FileResponse(cutout_of(get_item(s.store, item_id)), media_type="image/png")


@router.post("", status_code=201)
def upload_photo(
    photo: UploadFile, s: ServicesDep, mode: Annotated[SegMode, Form()] = SegMode.FLAT
) -> list[ItemOut]:
    """Add the garments in one photo: `flat` = a garment alone, `worn` = worn by a person
    (may yield a top and a bottom). Re-uploading a photo updates its items."""
    max_mb = s.settings.max_upload_mb
    data = photo.file.read(int(max_mb * 1024 * 1024) + 1)
    suffix = photo_suffix(data, max_mb)
    with s.ingest_lock:
        path = photo_path(s.settings.raw_dir, photo.filename, data, suffix)
        is_new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            items = s.pipeline().ingest_image(path, mode)
        except Exception:
            if is_new:  # nothing was stored, so don't keep the photo either
                path.unlink(missing_ok=True)
            raise
    return [ItemOut.of(item) for item in items]


@router.post("/{item_id}/retag")
def retag_item(item_id: UUID, body: RetagNote, s: ServicesDep) -> ItemOut:
    """Re-tag the cutout with the owner's correction as guidance (one DeepSeek call)."""
    with s.ingest_lock:
        item = get_item(s.store, item_id)
        cutout_of(item)
        item = retag_with_note(item, body.note, s.client, s.settings)
        s.store.update_tags(item)
    return ItemOut.of(item)


@router.patch("/{item_id}")
def edit_item(item_id: UUID, edits: TagEdits, s: ServicesDep) -> ItemOut:  # type: ignore[valid-type]
    """Correct tags, e.g. `{"category": "bottom", "subtype": "skirt"}`."""
    changes: dict[str, Any] = edits.model_dump(exclude_unset=True)
    if not changes:
        raise EditError("No edits given")
    item = apply_field_edits(get_item(s.store, item_id), changes)
    s.store.update_tags(item)
    return ItemOut.of(item)


@router.delete("/{item_id}", status_code=204)
def delete_item(item_id: UUID, s: ServicesDep) -> Response:
    """Remove the item and its cutout file."""
    remove_item(s.store, get_item(s.store, item_id))
    return Response(status_code=204)
