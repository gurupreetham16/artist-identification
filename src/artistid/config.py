from __future__ import annotations

import json
from pathlib import Path

import yaml


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"unknown config section '{k}' in '{dotted}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key '{dotted}'")
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """overrides look like 'views.local_grid=0' or 'rejection.mix_grid=[0.4,0.5]'."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config (or the config_used.json saved in a run folder)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    return apply_overrides(cfg, overrides)
