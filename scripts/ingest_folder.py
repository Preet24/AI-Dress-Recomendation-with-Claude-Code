"""Ingest every image in a folder into the wardrobe (segment, tag, color, embed, store).

Each garment cutout is sent to the DeepSeek API for tagging; everything else runs
locally on CPU. Re-running on the same photos updates their items, e.g.:
    python scripts/ingest_folder.py data/wardrobe_raw --mode worn
"""

import argparse
import logging
from pathlib import Path

from outfit.config import Settings
from outfit.ingestion.background import SegmentationError, SegMode
from outfit.ingestion.pipeline import Pipeline
from outfit.ingestion.tagger import TaggingError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

logger = logging.getLogger("ingest_folder")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mode", type=SegMode, choices=list(SegMode), default=SegMode.FLAT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        logger.warning("No images in %s", args.folder)
        return

    pipeline = Pipeline.from_settings(Settings())
    stored, failed = 0, []
    for path in photos:
        try:
            stored += len(pipeline.ingest_image(path, args.mode))
        except (SegmentationError, TaggingError) as exc:
            logger.warning("%s skipped: %s", path.name, exc)
            failed.append(path.name)

    logger.info(
        "Stored %d item(s) from %d photo(s); %d skipped; wardrobe now holds %d item(s)",
        stored,
        len(photos) - len(failed),
        len(failed),
        pipeline.store.count(),
    )


if __name__ == "__main__":
    main()
