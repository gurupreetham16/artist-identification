from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import HOG_DIM, HSV_LBP_DIM


class ArtistNet(nn.Module):
    """DINOv2 (last N blocks trainable) averaged over views, fused with HSV/LBP and HOG
    branches (each L2-normalised, handcrafted branches weighted), then dropout + linear head."""

    def __init__(self, n_classes: int, encoder_name: str = "facebook/dinov2-base",
                 trainable_blocks: int = 2, grad_ckpt: bool = False, hand_weight: float = 0.25,
                 dropout: float = 0.25, encoder: nn.Module | None = None):
        super().__init__()
        if encoder is None:
            from transformers import Dinov2Model
            encoder = Dinov2Model.from_pretrained(encoder_name)
        self.encoder = encoder
        dim = self.encoder.config.hidden_size
        for p in self.encoder.parameters():
            p.requires_grad = False
        if trainable_blocks > 0:
            for blk in self.encoder.encoder.layer[-trainable_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.encoder.layernorm.parameters():
                p.requires_grad = True
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.hand_weight = hand_weight
        self.feature_dim = dim + HSV_LBP_DIM + HOG_DIM
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_dim, n_classes)

    def embed(self, views: torch.Tensor, hsv: torch.Tensor, hog: torch.Tensor) -> torch.Tensor:
        b, v = views.shape[:2]
        cls = self.encoder(pixel_values=views.flatten(0, 1)).pooler_output  # [B*V, D]
        deep = F.normalize(cls.view(b, v, -1).float().mean(1), dim=-1)
        w = self.hand_weight
        return torch.cat([deep, w * F.normalize(hsv.float(), dim=-1), w * F.normalize(hog.float(), dim=-1)], dim=-1)

    def forward(self, views, hsv, hog, return_features: bool = False):
        fused = self.embed(views, hsv, hog)
        logits = self.head(self.drop(fused))
        return (logits, fused) if return_features else logits


def trainable_state(model: nn.Module) -> dict:
    """Only the weights that training changes (head + unfrozen encoder parts)."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}
