"""Correcting wardrobe items the pipeline got wrong (used by scripts/review_wardrobe.py).

Three ways to fix an item:
- replace it by ingesting a photo: the same photo in another mode, or a new one;
- re-tag the same cutout with the owner's note as guidance (one DeepSeek call);
- set tag fields directly, e.g. ``category=one_piece length=midi``.

Only the tags are editable. The id, cutout, CLIP vector, color LAB and
`created_at` stay owned by the pipeline.
"""

import shlex
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

from outfit.config import Settings
from outfit.db.qdrant_store import WardrobeStore
from outfit.ingestion.background import GarmentRegion, SegMode
from outfit.ingestion.pipeline import Pipeline
from outfit.ingestion.tagger import TagResult, tag_garment
from outfit.schemas import ClothingItem

EDITABLE_FIELDS = tuple(TagResult.model_fields)
LIST_FIELDS = {"secondary_colors", "occasions"}


class EditError(ValueError):
    """A field edit the user typed is unknown or invalid."""


def describe(item: ClothingItem) -> str:
    """One-line summary of an item's tags."""
    occasions = ", ".join(item.occasions)
    return (
        f"{item.primary_color_name} {item.subtype} [{item.category}] · {item.pattern} "
        f"{item.fabric} · length={item.length} fit={item.fit} · formality={item.formality} "
        f"warmth={item.warmth} flowy={item.flowy} · {occasions}"
    )


def changes(old: ClothingItem, new: ClothingItem) -> dict[str, tuple[Any, Any]]:
    """Tag fields whose value differs: name -> (old, new)."""
    return {
        name: (getattr(old, name), getattr(new, name))
        for name in EDITABLE_FIELDS
        if getattr(old, name) != getattr(new, name)
    }


def parse_field_edits(text: str) -> dict[str, Any]:
    """Parse ``key=value`` pairs; quote values with spaces, comma-separate lists.

    >>> parse_field_edits('category=one_piece subtype="slip dress" occasions=party,formal')
    {'category': 'one_piece', 'subtype': 'slip dress', 'occasions': ['party', 'formal']}
    """
    try:
        tokens = shlex.split(text)
    except ValueError as err:  # e.g. an unclosed quote
        raise EditError(str(err)) from err
    if not tokens:
        raise EditError("No edits given")
    edits: dict[str, Any] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise EditError(f"Expected key=value, got {token!r}")
        if key not in EDITABLE_FIELDS:
            raise EditError(f"Unknown field {key!r}; editable: {', '.join(EDITABLE_FIELDS)}")
        if key in LIST_FIELDS:
            edits[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            edits[key] = value
    return edits


def apply_field_edits(item: ClothingItem, edits: dict[str, Any]) -> ClothingItem:
    """The item with `edits` applied, validated like any ClothingItem."""
    try:
        return ClothingItem.model_validate({**item.model_dump(), **edits})
    except ValidationError as err:
        raise EditError(str(err)) from err


def region_of(item: ClothingItem) -> GarmentRegion:
    """Region the cutout came from, read from its `<stem>_<region>.png` name."""
    suffix = item.image_path.stem.rsplit("_", 1)[-1]
    return GarmentRegion(suffix) if suffix in set(GarmentRegion) else GarmentRegion.WHOLE


def retag_with_note(
    item: ClothingItem, note: str, client: OpenAI, settings: Settings
) -> ClothingItem:
    """Re-tag the item's cutout, telling DeepSeek the owner's correction. Not saved."""
    with Image.open(item.image_path) as image:
        tags = tag_garment(image, region_of(item), client, settings, note=note)
    return item.model_copy(update=tags.model_dump())


def remove_item(store: WardrobeStore, item: ClothingItem) -> None:
    """Delete the item and its cutout file."""
    store.delete([item.id])
    item.image_path.unlink(missing_ok=True)


def replace_with_photo(
    pipeline: Pipeline, item: ClothingItem, photo: Path, mode: SegMode
) -> list[ClothingItem]:
    """Ingest `photo` in place of `item`; returns the new items.

    The new photo is ingested first, so if it fails the old item is kept. The old
    item is then removed, unless re-ingesting the same photo and region recreated
    it under the same id.
    """
    new_items = pipeline.ingest_image(photo, mode)
    if item.id not in {new.id for new in new_items}:
        remove_item(pipeline.store, item)
    return new_items
