"""Manifest-driven dataset.

Manifest CSV columns:
  path    image path relative to paths.image_root
  artist  artist name
  role    known | unknown_dev | unknown_cal | unknown_test
  split   train | val | cal | test for known rows (empty for unknown rows)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import handcrafted
from .images import augment, load_rgb, multi_views


def read_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna({"split": ""})
    missing = {"path", "artist", "role", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return df


def known_labels(df: pd.DataFrame) -> list[str]:
    return sorted(df.loc[df.role == "known", "artist"].unique())


class ArtDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict[str, int], cfg: dict, train: bool):
        self.paths = df["path"].tolist()
        self.artists = df["artist"].tolist()
        self.label_map = label_map
        self.root = cfg["paths"].get("image_root", "")
        self.v = cfg["views"]
        self.aug = cfg["augment"]
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_rgb(os.path.join(self.root, self.paths[i]))
        if self.train:
            rng = np.random.default_rng()
            img = augment(img, rng, tuple(self.aug["crop_scale"]), self.aug["max_rotation_deg"], self.aug["jitter"])
        views = multi_views(img, self.v["size"], self.v["local_resize"], self.v["local_grid"])
        hsv, hog = handcrafted(img)  # same (augmented) painting for every branch
        y = self.label_map.get(self.artists[i], -1)  # -1 for unknown artists
        return (torch.from_numpy(views), torch.from_numpy(hsv), torch.from_numpy(hog),
                torch.tensor(y, dtype=torch.long))
