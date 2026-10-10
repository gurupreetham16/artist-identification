"""Handcrafted descriptors: HSV histograms + LBP texture (58-d) and HOG shape (324-d)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from .images import letterbox

HSV_LBP_DIM = 58
HOG_DIM = 324


def hsv_lbp(img: Image.Image, max_side: int = 256) -> np.ndarray:
    """16 bins x 3 HSV channels + 10 uniform-LBP bins, on an aspect-preserving image
    (no padding, so padding is never measured)."""
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1:
        img = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BICUBIC)
    hsv = np.asarray(img.convert("HSV"))
    parts = []
    for c in range(3):
        hist, _ = np.histogram(hsv[..., c], bins=16, range=(0, 256))
        parts.append(hist / max(hist.sum(), 1))
    gray = np.asarray(img.convert("L"))
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    parts.append(hist / max(hist.sum(), 1))
    return np.concatenate(parts).astype(np.float32)


def hog_feat(img: Image.Image) -> np.ndarray:
    """Grayscale 128-px aspect-preserving canvas, 9 orientations, 32-px cells, 2x2 blocks."""
    gray = letterbox(img.convert("L"), 128)
    v = hog(np.asarray(gray), orientations=9, pixels_per_cell=(32, 32),
            cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return v.astype(np.float32)


def handcrafted(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    return hsv_lbp(img), hog_feat(img)
