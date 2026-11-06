"""Review wardrobe items one by one and fix the ones the pipeline got wrong.

For each item the cutout opens in your image viewer and you choose:
  k  keep it as is
  r  replace it with a photo: the same photo in another mode, or a new one you took
  t  describe it in text; DeepSeek re-tags the same cutout using your note
  f  set fields directly, e.g.  category=one_piece length=midi subtype="slip dress"
  d  delete it
  q  quit

Examples:
    python scripts/review_wardrobe.py               # every item
    python scripts/review_wardrobe.py --match dress # items whose subtype/file contains "dress"
"""

import argparse
import logging
from pathlib import Path

from PIL import Image

from outfit.config import Settings
from outfit.ingestion.background import SegmentationError, SegMode
from outfit.ingestion.pipeline import Pipeline
from outfit.ingestion.review import (
    EDITABLE_FIELDS,
    EditError,
    apply_field_edits,
    changes,
    describe,
    parse_field_edits,
    remove_item,
    replace_with_photo,
    retag_with_note,
)
from outfit.ingestion.tagger import TaggingError
from outfit.schemas import ClothingItem

MENU = "[k]eep  [r]eplace with photo  [t]ext note  [f]ields  [d]elete  [q]uit > "

logger = logging.getLogger("review_wardrobe")


class Quit(Exception):
    pass


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError as err:
        raise Quit from err


def show_value(value: object) -> str:
    """Enums and lists as plain text ("party, formal"), not their reprs."""
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "(none)"
    return str(value)


def confirm_save(pipeline: Pipeline, old: ClothingItem, new: ClothingItem) -> bool:
    """Show what changed and save on 'y'; True if saved."""
    diff = changes(old, new)
    if not diff:
        logger.info("  Nothing changed.")
        return False
    for name, (before, after) in diff.items():
        logger.info("  %s: %s -> %s", name, show_value(before), show_value(after))
    if ask("  Save these changes? [y/N] > ").lower() != "y":
        return False
    pipeline.store.update_tags(new)
    logger.info("  Saved.")
    return True


def replace(pipeline: Pipeline, item: ClothingItem) -> bool:
    raw = ask("  Photo path (drag the file here; empty to cancel) > ").strip('"')
    if not raw:
        return False
    photo = Path(raw)
    if not photo.is_file():
        logger.warning("  No such file: %s", photo)
        return False
    mode_text = ask("  Mode: [f]lat (garment alone) or [w]orn (on a person) > ").lower()
    mode = SegMode.WORN if mode_text.startswith("w") else SegMode.FLAT
    try:
        new_items = replace_with_photo(pipeline, item, photo, mode)
    except (SegmentationError, TaggingError) as exc:
        logger.warning("  Could not ingest %s, item kept: %s", photo.name, exc)
        return False
    for new in new_items:
        logger.info("  Stored: %s", describe(new))
    logger.info("  Review the new item(s) on the next run if they still look wrong.")
    return True


def review(pipeline: Pipeline, item: ClothingItem, show: bool) -> None:
    """Prompt until the item is kept, replaced or deleted."""
    if show:
        with Image.open(item.image_path) as image:
            image.show()
    while True:
        choice = ask(MENU).lower()[:1]
        if choice in ("k", ""):
            return
        if choice == "q":
            raise Quit
        if choice == "d":
            if ask("  Delete this item? [y/N] > ").lower() == "y":
                remove_item(pipeline.store, item)
                logger.info("  Deleted.")
                return
        elif choice == "r":
            if replace(pipeline, item):
                return
        elif choice == "t":
            note = ask("  Describe the garment (e.g. black polka-dot midi slip dress) > ")
            if not note:
                continue
            try:
                new = retag_with_note(item, note, pipeline.client, pipeline.settings)
            except TaggingError as exc:
                logger.warning("  Re-tagging failed: %s", exc)
                continue
            if confirm_save(pipeline, item, new):
                return
        elif choice == "f":
            logger.info("  Editable: %s", ", ".join(EDITABLE_FIELDS))
            text = ask("  Edits (key=value ...) > ")
            if not text:
                continue
            try:
                new = apply_field_edits(item, parse_field_edits(text))
            except EditError as exc:
                logger.warning("  %s", exc)
                continue
            if confirm_save(pipeline, item, new):
                return
        else:
            logger.warning("  Unknown choice %r", choice)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", help="only items whose subtype or file name contains this")
    parser.add_argument("--no-show", action="store_true", help="don't open cutouts in a viewer")
    args = parser.parse_args()
    # Libraries only report warnings; this script's own messages are the conversation.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    pipeline = Pipeline.from_settings(Settings())
    items = sorted(pipeline.store.list_items(), key=lambda i: str(i.image_path))
    if args.match:
        needle = args.match.lower()
        items = [i for i in items if needle in f"{i.subtype} {i.image_path.name}".lower()]
    if not items:
        logger.info("No items to review.")
        return

    try:
        for n, item in enumerate(items, 1):
            logger.info("\n(%d/%d) %s\n  %s", n, len(items), item.image_path.name, describe(item))
            review(pipeline, item, show=not args.no_show)
    except (Quit, KeyboardInterrupt):
        logger.info("\nStopped.")
    logger.info("Wardrobe holds %d item(s).", pipeline.store.count())


if __name__ == "__main__":
    main()
