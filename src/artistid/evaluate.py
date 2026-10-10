"""Fit the rejection gate and evaluate.

  python -m artistid.evaluate --run runs/v3_2/seed106            # validation / development only
  python -m artistid.evaluate --run runs/v3_2/seed106 --final    # also touches the test split (do this once)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import rejection as rej
from .config import load_config
from .data import known_labels, read_manifest
from .engine import make_loader, pick_amp_dtype, predict
from .metrics import classification_metrics, per_artist_f1
from .model import ArtistNet


def get_split(name, df, label_map, cfg, model, device, amp, run: Path, reuse: bool):
    cache = run / f"preds_{name}.npz"
    if reuse and cache.exists():
        z = np.load(cache)
        return z["logits"], z["feats"], z["y"]
    loader = make_loader(df, label_map, cfg, train=False, batch_size=cfg["train"]["eval_batch_size"])
    logits, feats, y = predict(model, loader, device, amp)
    np.savez(cache, logits=logits, feats=feats, y=y)
    return logits, feats, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--final", action="store_true", help="also evaluate the held-out test split")
    ap.add_argument("--reuse", action="store_true", help="reuse cached predictions in the run folder")
    args = ap.parse_args()

    run = Path(args.run)
    # default: the exact config the run was trained with; --set can override evaluation-only
    # settings such as rejection.prototypes_per_artist=1
    cfg = load_config(args.config or (run / "config_used.json"), args.set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp_dtype()
    df = read_manifest(cfg["paths"]["manifest"])
    artists = known_labels(df)
    label_map = {a: i for i, a in enumerate(artists)}
    known = df[df.role == "known"]

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert ck["artists"] == artists, "manifest artists differ from the trained model's labels"
    model = ArtistNet(len(artists), cfg["encoder"]["name"], 0, False, cfg["features"]["hand_weight"],
                      cfg["train"]["dropout"])
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    print(f"loaded best.pt (epoch {ck['epoch']}, val macro-F1 {ck['val_macro_f1']:.4f})")

    def go(name, frame):
        return get_split(name, frame, label_map, cfg, model, device, amp, run, args.reuse)

    tr = go("train", known[known.split == "train"])
    va = go("val", known[known.split == "val"])
    ca = go("cal", known[known.split == "cal"])
    ud = go("unk_dev", df[df.role == "unknown_dev"])
    uc = go("unk_cal", df[df.role == "unknown_cal"])

    rc = cfg["rejection"]
    T = rej.fit_temperature(ca[0], ca[2], rc["temperature_grid"])
    protos = rej.fit_prototypes(tr[1], tr[2], len(artists), rc["prototypes_per_artist"], seed=cfg["seed"])
    sel = rej.select_mix(va[0], va[1], ud[0], ud[1], protos, T, rc["mix_grid"], rc["target_far"])
    mix = sel["mix"]
    thr = rej.threshold_for_far(rej.gate_scores(uc[0], uc[1], protos, T, mix), rc["target_far"])

    def rej_report(known_split, unk_split):
        ks = rej.gate_scores(known_split[0], known_split[1], protos, T, mix)
        us = rej.gate_scores(unk_split[0], unk_split[1], protos, T, mix)
        return rej.rejection_metrics(ks, known_split[0].argmax(1) == known_split[2], us, thr)

    result = {
        "temperature": T, "mix": mix, "threshold": thr, "prototypes_per_artist": rc["prototypes_per_artist"],
        "val_classification": classification_metrics(va[0], va[2], T),
        "cal_classification": classification_metrics(ca[0], ca[2], T),
        "dev_rejection (val known vs unknown_dev)": rej_report(va, ud),
        "cal_rejection (cal known vs unknown_cal)": rej_report(ca, uc),
        "val_per_artist_f1": per_artist_f1(va[0], va[2], artists),
    }

    if args.final:
        te = go("test", known[known.split == "test"])
        ut = go("unk_test", df[df.role == "unknown_test"])
        result["TEST_classification"] = classification_metrics(te[0], te[2], T)
        result["TEST_rejection"] = rej_report(te, ut)
        result["TEST_per_artist_f1"] = per_artist_f1(te[0], te[2], artists)
        te_df = known[known.split == "test"].reset_index(drop=True)
        pd.DataFrame({"path": te_df.path, "artist": te_df.artist,
                      "pred": [artists[i] for i in te[0].argmax(1)],
                      "correct": te[0].argmax(1) == te[2]}).to_csv(run / "test_predictions.csv", index=False)

    name = "results_final.json" if args.final else "results_dev.json"
    (run / name).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if "per_artist" not in k}, indent=2))
    print(f"saved {run / name}")


if __name__ == "__main__":
    main()
