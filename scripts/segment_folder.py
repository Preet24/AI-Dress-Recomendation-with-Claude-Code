"""Segment every image in a folder and save the garment cutouts as PNGs.

Manual check for the segmentation step, e.g.:
    python scripts/segment_folder.py "test images" --mode worn
"""

import argparse
import logging
from pathlib import Path

from outfit.config import Settings
from outfit.ingestion.background import SegmentationError, SegMode, save_cutouts, segment_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

logger = logging.getLogger("segment_folder")


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mode", type=SegMode, choices=list(SegMode), default=SegMode.FLAT)
    parser.add_argument("--out", type=Path, default=settings.clean_dir)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for path in sorted(args.folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            cutouts = segment_file(path, args.mode, settings)
        except SegmentationError as exc:
            logger.warning("%s: %s", path.name, exc)
            continue
        saved = save_cutouts(cutouts, args.out, path.stem)
        logger.info("%s -> %s", path.name, ", ".join(p.name for p in saved))


if __name__ == "__main__":
    main()
