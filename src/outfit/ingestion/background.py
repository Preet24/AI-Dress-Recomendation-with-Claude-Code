"""Clothing segmentation: cut garments out of photos with rembg ONNX models on CPU.

Two modes:
- FLAT: a garment on its own (flat-lay, hanger, product shot) -> one WHOLE cutout,
  using a general foreground model; thin parts such as a hanger hook are trimmed.
- WORN: garments on a person -> UPPER / LOWER cutouts, or one FULL cutout for a
  one-piece, using rembg's `u2net_cloth_seg` model.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from outfit.config import Settings

logger = logging.getLogger(__name__)

CPU_PROVIDERS = ["CPUExecutionProvider"]
MASK_THRESHOLD = 128  # mask value (0-255) above which a pixel counts as garment
KEEP_COMPONENT_RATIO = 0.2  # keep blobs >= 20% of the largest, e.g. both shoes of a pair
MAX_HOLE_RATIO = 0.001  # fill enclosed holes smaller than 0.1% of the image
FULL_DOMINANCE = 0.5  # WORN: a one-piece if "full" covers >= half of all garment pixels
CROP_PAD_RATIO = 0.02  # padding around the cropped garment, relative to the long side
# FLAT: trim parts thinner than ~2x this (relative to the long side), e.g. a hanger hook.
# On 4 real hanger photos 0.006 already removed every hook while keeping >= 99.8% of the
# garment; larger values only nibbled garment edges.
THIN_PART_RATIO = 0.008


class SegMode(StrEnum):
    FLAT = "flat"
    WORN = "worn"


class GarmentRegion(StrEnum):
    WHOLE = "whole"
    UPPER = "upper"
    LOWER = "lower"
    FULL = "full"


# Order of the masks returned by rembg's u2net_cloth_seg session.
CLOTH_SEG_REGIONS = (GarmentRegion.UPPER, GarmentRegion.LOWER, GarmentRegion.FULL)


class SegmentationError(RuntimeError):
    """No garment could be found in the image."""


class MaskSession(Protocol):
    """The part of a rembg session we use; lets tests pass a fake."""

    def predict(self, img: Image.Image) -> list[Image.Image]: ...


@dataclass(frozen=True)
class Cutout:
    region: GarmentRegion
    image: Image.Image  # RGBA, transparent background, cropped to the garment
    area_ratio: float  # garment pixels / image pixels


def load_image(path: Path | BinaryIO, max_side: int) -> Image.Image:
    """Open a photo upright (EXIF rotation applied), as RGB, long side <= max_side."""
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img).convert("RGB")
    upright.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return upright


@cache
def get_session(model_name: str) -> MaskSession:
    """Load a rembg ONNX session once per model, pinned to CPU."""
    from rembg import new_session  # deferred: importing rembg is slow

    logger.info("Loading segmentation model %s on CPU", model_name)
    return new_session(model_name, providers=CPU_PROVIDERS)


def model_for(mode: SegMode, settings: Settings) -> str:
    return settings.worn_model if mode is SegMode.WORN else settings.flat_model


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop speckles and fill pinholes in a boolean mask."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
    keep = np.flatnonzero(sizes >= KEEP_COMPONENT_RATIO * sizes.max()) + 1
    return fill_small_holes(np.isin(labels, keep), max_hole=int(MAX_HOLE_RATIO * mask.size))


def fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill background regions enclosed by the mask that have <= max_hole pixels."""
    holes, count = ndimage.label(~mask)
    if count == 0:
        return mask
    sizes = ndimage.sum_labels(~mask, holes, index=np.arange(1, count + 1))
    small = np.flatnonzero(sizes <= max_hole) + 1
    touching_border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    return mask | np.isin(holes, np.setdiff1d(small, touching_border))


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def remove_thin_parts(mask: np.ndarray, radius: int) -> np.ndarray:
    """Cut away parts thinner than ~2*radius that stick out of the garment (a hanger hook).

    An opening keeps only the thick body; the mask is then limited to `radius` around
    that body, so the garment's own edges come back but a hook further out does not.
    A mask with no thick body at all (a belt, a necklace) is returned unchanged.
    """
    body = ndimage.binary_opening(mask, structure=disk(radius))
    if not body.any():
        return mask
    return mask & ndimage.binary_dilation(clean_mask(body), structure=disk(radius))


def merge_worn_regions(masks: dict[GarmentRegion, np.ndarray]) -> dict[GarmentRegion, np.ndarray]:
    """Decide between one one-piece (FULL) and separate UPPER / LOWER garments.

    The cloth model often labels part of a dress as "upper", so when "full"
    dominates, all three masks are merged into one FULL garment.
    """
    total = sum(int(m.sum()) for m in masks.values())
    full = int(masks[GarmentRegion.FULL].sum())
    if total and full / total >= FULL_DOMINANCE:
        union = masks[GarmentRegion.UPPER] | masks[GarmentRegion.LOWER] | masks[GarmentRegion.FULL]
        return {GarmentRegion.FULL: clean_mask(union)}
    if full:
        logger.debug("Dropping minor 'full' region (%.0f%% of garment pixels)", 100 * full / total)
    return {region: masks[region] for region in (GarmentRegion.UPPER, GarmentRegion.LOWER)}


def cut_out(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply a boolean mask as alpha and crop to the garment plus a little padding."""
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    ys, xs = np.nonzero(mask)
    pad = int(CROP_PAD_RATIO * max(image.size))
    box = (
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + 1 + pad, image.width),
        min(int(ys.max()) + 1 + pad, image.height),
    )
    return rgba.crop(box)


def segment(
    image: Image.Image, mode: SegMode, session: MaskSession, min_area_ratio: float
) -> list[Cutout]:
    """Segment the garments in an RGB image into transparent, cropped cutouts."""
    raw = session.predict(image)
    regions = CLOTH_SEG_REGIONS if mode is SegMode.WORN else (GarmentRegion.WHOLE,)
    if len(raw) != len(regions):
        raise ValueError(
            f"{mode} mode expects {len(regions)} mask(s) from the model, got {len(raw)}; "
            "check that the configured model matches the mode"
        )

    masks = {
        region: clean_mask(np.asarray(m.convert("L")) >= MASK_THRESHOLD)
        for region, m in zip(regions, raw, strict=True)
    }
    if mode is SegMode.WORN:
        masks = merge_worn_regions(masks)
    else:
        radius = max(1, round(THIN_PART_RATIO * max(image.size)))
        masks = {region: remove_thin_parts(m, radius) for region, m in masks.items()}

    cutouts = []
    for region, mask in masks.items():
        area_ratio = float(mask.mean())
        if area_ratio < min_area_ratio:
            logger.debug("Skipping %s region: %.1f%% of image", region, 100 * area_ratio)
            continue
        cutouts.append(Cutout(region, cut_out(image, mask), area_ratio))

    if not cutouts:
        raise SegmentationError(f"No garment covering >= {min_area_ratio:.0%} of the image")
    return cutouts


def segment_file(path: Path, mode: SegMode, settings: Settings) -> list[Cutout]:
    """Load a photo and segment it with the model configured for `mode`."""
    image = load_image(path, settings.max_side)
    session = get_session(model_for(mode, settings))
    return segment(image, mode, session, settings.min_area_ratio)


def save_cutouts(cutouts: list[Cutout], out_dir: Path, stem: str) -> list[Path]:
    """Save cutouts as `<stem>_<region>.png` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cutout in cutouts:
        path = out_dir / f"{stem}_{cutout.region}.png"
        cutout.image.save(path)
        paths.append(path)
    return paths
