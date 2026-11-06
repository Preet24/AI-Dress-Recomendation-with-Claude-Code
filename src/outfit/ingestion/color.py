"""Dominant garment colors, computed locally from the cutout's opaque pixels.

Clustering runs in CIELAB, where Euclidean distance roughly matches perceived
color difference, so a shadowed fold and a lit fold of the same shirt land in
one cluster more often than they would in RGB.
"""

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

ALPHA_THRESHOLD = 128  # pixels at least this opaque belong to the garment
MAX_PIXELS = 20_000  # subsample larger cutouts; plenty for a few clusters
DEFAULT_K = 4

Lab = tuple[float, float, float]


@dataclass(frozen=True)
class ColorCluster:
    lab: Lab  # CIELAB (L*, a*, b*), D65
    rgb: tuple[int, int, int]
    share: float  # fraction of garment pixels in this cluster


def garment_pixels(image: Image.Image) -> np.ndarray:
    """RGB values (n, 3) uint8 of the garment: opaque pixels only."""
    rgba = np.asarray(image.convert("RGBA"))
    opaque = rgba[..., 3] >= ALPHA_THRESHOLD
    return rgba[opaque][:, :3]


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (n, 3) -> CIELAB (n, 3) float."""
    return rgb2lab(rgb.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)


def lab_to_rgb(lab: np.ndarray) -> tuple[int, int, int]:
    rgb = lab2rgb(np.asarray(lab, dtype=float).reshape(1, 1, 3)).reshape(3)
    r, g, b = (int(round(c)) for c in np.clip(rgb * 255, 0, 255))
    return r, g, b


def dominant_colors(
    image: Image.Image, k: int = DEFAULT_K, max_pixels: int = MAX_PIXELS, seed: int = 0
) -> list[ColorCluster]:
    """Cluster the garment's pixels in LAB; largest cluster first. Deterministic for a seed."""
    pixels = garment_pixels(image)
    if len(pixels) == 0:
        raise ValueError("Image has no opaque pixels to take colors from")
    if len(pixels) > max_pixels:
        pick = np.random.default_rng(seed).choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[pick]

    lab = to_lab(pixels)
    n_clusters = min(k, len(np.unique(pixels, axis=0)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(lab)
    shares = np.bincount(kmeans.labels_, minlength=n_clusters) / len(lab)

    clusters = [
        ColorCluster(lab=(float(c[0]), float(c[1]), float(c[2])), rgb=lab_to_rgb(c), share=float(s))
        for c, s in zip(kmeans.cluster_centers_, shares, strict=True)
    ]
    return sorted(clusters, key=lambda c: c.share, reverse=True)


def primary_color_lab(image: Image.Image, k: int = DEFAULT_K) -> Lab:
    """LAB of the largest color cluster: the value stored as `primary_color_lab`."""
    return dominant_colors(image, k)[0].lab
