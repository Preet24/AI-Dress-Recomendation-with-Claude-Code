from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from outfit.config import Settings
from outfit.ingestion.background import (
    GarmentRegion,
    SegmentationError,
    SegMode,
    get_session,
    load_image,
    save_cutouts,
    segment,
    segment_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
H, W = 200, 100
MIN_AREA = 0.02


class FakeSession:
    """Stands in for a rembg session: returns pre-built masks."""

    def __init__(self, *masks: np.ndarray) -> None:
        self.masks = masks

    def predict(self, img: Image.Image) -> list[Image.Image]:
        return [Image.fromarray(m) for m in self.masks]


def blank() -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


def rect(top: int, left: int, bottom: int, right: int, mask: np.ndarray | None = None):
    mask = blank() if mask is None else mask
    mask[top:bottom, left:right] = 255
    return mask


@pytest.fixture
def photo() -> Image.Image:
    return Image.new("RGB", (W, H), (120, 60, 30))


def test_flat_returns_one_transparent_cutout(photo):
    [cutout] = segment(photo, SegMode.FLAT, FakeSession(rect(50, 20, 150, 80)), MIN_AREA)

    assert cutout.region is GarmentRegion.WHOLE
    assert cutout.image.mode == "RGBA"
    assert cutout.area_ratio == pytest.approx(100 * 60 / (H * W))
    alpha = np.asarray(cutout.image.getchannel("A"))
    assert alpha[0, 0] == 0  # padding around the garment is transparent
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255


def test_cutout_is_cropped_to_garment_with_padding(photo):
    [cutout] = segment(photo, SegMode.FLAT, FakeSession(rect(50, 20, 150, 80)), MIN_AREA)

    pad = int(0.02 * H)
    assert cutout.image.size == (60 + 2 * pad, 100 + 2 * pad)


def test_speckles_are_removed(photo):
    mask = rect(50, 20, 150, 80)
    rect(5, 90, 8, 93, mask)  # 3x3 speckle far from the garment

    [cutout] = segment(photo, SegMode.FLAT, FakeSession(mask), MIN_AREA)

    assert cutout.image.size == (60 + 8, 100 + 8)  # crop ignores the speckle


def test_two_similar_blobs_are_both_kept(photo):
    # e.g. a pair of shoes: neither should be dropped as noise
    mask = rect(20, 10, 80, 90)
    rect(120, 10, 180, 90, mask)

    [cutout] = segment(photo, SegMode.FLAT, FakeSession(mask), MIN_AREA)

    assert cutout.area_ratio == pytest.approx(2 * 60 * 80 / (H * W))


def test_pinholes_are_filled(photo):
    mask = rect(50, 20, 150, 80)
    mask[99:101, 49:51] = 0

    [cutout] = segment(photo, SegMode.FLAT, FakeSession(mask), MIN_AREA)

    assert cutout.area_ratio == pytest.approx(100 * 60 / (H * W))


def test_flat_trims_a_hanger_hook_but_keeps_the_garment(photo):
    mask = rect(50, 20, 150, 80)
    rect(20, 49, 50, 51, mask)  # 2 px wide hook rising 30 px above the shirt

    [cutout] = segment(photo, SegMode.FLAT, FakeSession(mask), MIN_AREA)

    # all of the shirt; of the 60 hook pixels only a stub within the 2 px radius remains
    shirt = 100 * 60
    assert shirt <= round(cutout.area_ratio * H * W) <= shirt + 2 * 3
    assert cutout.image.height <= 100 + 3 + 8  # the crop no longer reaches up the hook


def test_flat_keeps_a_garment_that_is_all_thin(photo):
    mask = rect(20, 49, 180, 51)  # e.g. a belt photographed lengthwise

    [cutout] = segment(photo, SegMode.FLAT, FakeSession(mask), MIN_AREA / 10)

    assert cutout.area_ratio == pytest.approx(160 * 2 / (H * W))


def test_worn_masks_are_not_trimmed(photo):
    upper = rect(20, 20, 90, 80)
    rect(90, 49, 100, 51, upper)  # a thin strap the cloth model labelled "upper"

    [cutout] = segment(photo, SegMode.WORN, FakeSession(upper, blank(), blank()), MIN_AREA)

    assert cutout.area_ratio == pytest.approx((70 * 60 + 10 * 2) / (H * W))


def test_worn_top_and_bottom_become_separate_cutouts(photo):
    session = FakeSession(rect(20, 20, 90, 80), rect(100, 25, 190, 75), blank())

    cutouts = segment(photo, SegMode.WORN, session, MIN_AREA)

    assert [c.region for c in cutouts] == [GarmentRegion.UPPER, GarmentRegion.LOWER]


def test_worn_dress_becomes_one_full_cutout(photo):
    # cloth model labels the bodice "upper" and the rest "full"
    session = FakeSession(rect(20, 20, 60, 80), blank(), rect(60, 20, 190, 80))

    [cutout] = segment(photo, SegMode.WORN, session, MIN_AREA)

    assert cutout.region is GarmentRegion.FULL
    assert cutout.area_ratio == pytest.approx(170 * 60 / (H * W))


def test_worn_drops_regions_below_min_area(photo):
    session = FakeSession(rect(20, 20, 120, 80), rect(150, 50, 153, 53), blank())

    cutouts = segment(photo, SegMode.WORN, session, MIN_AREA)

    assert [c.region for c in cutouts] == [GarmentRegion.UPPER]


@pytest.mark.parametrize("mask", [blank(), rect(10, 10, 13, 13)], ids=["empty", "tiny"])
def test_no_garment_raises(photo, mask):
    with pytest.raises(SegmentationError):
        segment(photo, SegMode.FLAT, FakeSession(mask), MIN_AREA)


def test_model_mode_mismatch_raises(photo):
    with pytest.raises(ValueError, match="worn mode expects 3"):
        segment(photo, SegMode.WORN, FakeSession(rect(50, 20, 150, 80)), MIN_AREA)


def test_load_image_applies_exif_rotation_and_downscales(tmp_path):
    path = tmp_path / "sideways.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° clockwise to display
    Image.new("RGB", (2000, 1000)).save(path, exif=exif)

    img = load_image(path, max_side=500)

    assert img.mode == "RGB"
    assert img.size == (250, 500)


def test_save_cutouts_writes_pngs(photo, tmp_path):
    session = FakeSession(rect(20, 20, 90, 80), rect(100, 25, 190, 75), blank())
    cutouts = segment(photo, SegMode.WORN, session, MIN_AREA)

    paths = save_cutouts(cutouts, tmp_path / "clean", "outfit1")

    assert [p.name for p in paths] == ["outfit1_upper.png", "outfit1_lower.png"]
    assert all(Image.open(p).mode == "RGBA" for p in paths)


@pytest.mark.slow
def test_real_flat_model_segments_fixture_on_cpu():
    settings = Settings()

    [cutout] = segment_file(FIXTURES / "red_tshirt_flat.png", SegMode.FLAT, settings)

    session = get_session(settings.flat_model)
    assert session.inner_session.get_providers() == ["CPUExecutionProvider"]
    assert 0.2 < cutout.area_ratio < 0.5  # the t-shirt covers ~1/3 of the fixture
