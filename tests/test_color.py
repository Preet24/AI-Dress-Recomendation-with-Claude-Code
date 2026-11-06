from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from outfit.ingestion.color import dominant_colors, garment_pixels, primary_color_lab

FIXTURES = Path(__file__).parent / "fixtures"


def two_tone(width: int = 10, split: int = 7) -> Image.Image:
    """Red left part, navy right part, with a transparent green border that must be ignored."""
    img = Image.new("RGBA", (width + 2, 6), (0, 255, 0, 0))
    for x in range(1, width + 1):
        for y in range(1, 5):
            img.putpixel((x, y), (200, 20, 30, 255) if x <= split else (20, 30, 90, 255))
    return img


def test_transparent_pixels_are_ignored():
    pixels = garment_pixels(two_tone())

    assert len(pixels) == 40
    assert not np.any(np.all(pixels == (0, 255, 0), axis=1))


def test_dominant_colors_sorted_by_share_with_correct_rgb():
    clusters = dominant_colors(two_tone(), k=2)

    assert [c.rgb for c in clusters] == [(200, 20, 30), (20, 30, 90)]
    assert [c.share for c in clusters] == pytest.approx([0.7, 0.3])


def test_k_is_capped_by_number_of_distinct_colors():
    solid = Image.new("RGBA", (5, 5), (255, 255, 255, 255))

    [cluster] = dominant_colors(solid, k=4)

    assert cluster.share == 1.0
    assert cluster.lab == pytest.approx((100.0, 0.0, 0.0), abs=0.01)


def test_fully_transparent_image_is_rejected():
    with pytest.raises(ValueError, match="no opaque pixels"):
        dominant_colors(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))


def test_subsampling_is_deterministic():
    img = two_tone(width=200, split=120)

    assert dominant_colors(img, max_pixels=100) == dominant_colors(img, max_pixels=100)


def test_red_tshirt_cutout_is_red():
    # The fixture is a raw photo; make it a cutout by hiding the off-white background.
    rgba = np.asarray(Image.open(FIXTURES / "red_tshirt_flat.png").convert("RGBA")).copy()
    rgba[rgba[..., :3].min(axis=2) > 200, 3] = 0

    lightness, a, b = primary_color_lab(Image.fromarray(rgba))

    assert a > 30  # strongly toward red
    assert a > abs(b)
    assert 20 < lightness < 80
