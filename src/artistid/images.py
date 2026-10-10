"""Image helpers that need only PIL and numpy (no torch)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

PAD = (124, 116, 104)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize proportionally so the long side equals `size`, then pad to a square."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    fill = PAD if img.mode == "RGB" else 128
    canvas = Image.new(img.mode, (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def augment(img: Image.Image, rng: np.random.Generator, crop_scale=(0.8, 1.0),
            max_rot=5.0, jitter=0.05) -> Image.Image:
    """Mild augmentation: area crop, small rotation, small brightness/contrast change.
    No flips and no hue shifts, because palette and composition are artist cues."""
    w, h = img.size
    f = float(rng.uniform(*crop_scale)) ** 0.5
    cw, ch = max(8, int(w * f)), max(8, int(h * f))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = img.rotate(float(rng.uniform(-max_rot, max_rot)), resample=Image.BICUBIC, fillcolor=PAD)
    img = ImageEnhance.Brightness(img).enhance(1.0 + float(rng.uniform(-jitter, jitter)))
    img = ImageEnhance.Contrast(img).enhance(1.0 + float(rng.uniform(-jitter, jitter)))
    return img


def to_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)


def multi_views(img: Image.Image, size: int = 224, local_resize: int = 448, grid: int = 2) -> np.ndarray:
    """One global letterboxed view plus grid x grid local tiles cut from a higher-resolution
    copy of the painting. Returns float32 array [V, 3, size, size]."""
    views = [letterbox(img, size)]
    if grid > 0:
        w, h = img.size
        s = local_resize / max(w, h)
        big = img.resize((max(grid, int(round(w * s))), max(grid, int(round(h * s)))), Image.BICUBIC)
        bw, bh = big.size
        for gy in range(grid):
            for gx in range(grid):
                box = (gx * bw // grid, gy * bh // grid, (gx + 1) * bw // grid, (gy + 1) * bh // grid)
                views.append(letterbox(big.crop(box), size))
    return np.stack([to_array(v) for v in views])
