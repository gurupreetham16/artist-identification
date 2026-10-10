from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ArtDataset


def pick_amp_dtype():
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_context(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def make_loader(df, label_map, cfg, train: bool, batch_size: int, shuffle: bool = False) -> DataLoader:
    nw = int(cfg["train"]["num_workers"])
    return DataLoader(ArtDataset(df, label_map, cfg, train=train), batch_size=batch_size, shuffle=shuffle,
                      num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    """Returns logits [N,C], fused features [N,D], labels [N] (-1 for unknown artists)."""
    model.eval()
    logits, feats, ys = [], [], []
    for views, hsv, hog, y in loader:
        with amp_context(amp_dtype):
            lg, fused = model(views.to(device, non_blocking=True), hsv.to(device), hog.to(device),
                              return_features=True)
        logits.append(lg.float().cpu())
        feats.append(fused.float().cpu())
        ys.append(y)
    return (torch.cat(logits).numpy(), torch.cat(feats).numpy(), torch.cat(ys).numpy())
