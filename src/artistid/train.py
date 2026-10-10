"""Train the V3.2 model.  python -m artistid.train --config configs/v3_2.yaml --seed 106"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .config import load_config
from .data import known_labels, read_manifest
from .engine import amp_context, make_loader, pick_amp_dtype, predict
from .metrics import classification_metrics
from .model import ArtistNet, trainable_state


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_factor(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup)))


def warm_start_head(model, loader, n_classes, cfg, device, amp_dtype) -> None:
    """Initialise the linear head from a converged logistic regression on frozen features."""
    _, feats, y = predict(model, loader, device, amp_dtype)
    ws = cfg["warm_start"]
    lr = LogisticRegression(C=ws["C"], max_iter=ws["max_iter"])
    lr.fit(feats, y)
    if int(np.max(lr.n_iter_)) >= ws["max_iter"]:
        print("warning: warm-start logistic regression hit max_iter; raise warm_start.max_iter")
    # lr.classes_ lists only the artists present in the data; place each one's weights in its own
    # row and give any absent artist a strongly negative bias.
    w = torch.zeros(n_classes, feats.shape[1])
    b = torch.full((n_classes,), -10.0)
    coef = torch.tensor(lr.coef_, dtype=torch.float32)
    icpt = torch.tensor(lr.intercept_, dtype=torch.float32)
    if len(lr.classes_) == 2:  # sklearn returns a single row for two classes
        coef, icpt = torch.cat([-coef, coef]), torch.cat([-icpt, icpt])
    for row, c in enumerate(lr.classes_):
        w[int(c)], b[int(c)] = coef[row], icpt[row]
    if len(lr.classes_) < n_classes:
        print(f"warning: only {len(lr.classes_)}/{n_classes} artists in the warm-start data")
    with torch.no_grad():
        model.head.weight.copy_(w)
        model.head.bias.copy_(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_2.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. views.local_grid=0")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt")
    ap.add_argument("--max-train", type=int, help="debug: use only N training images")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg["seed"]
    out = Path(args.out or Path(cfg["paths"]["out_dir"]) / f"seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_amp_dtype()
    print(f"device={device} amp={amp_dtype}")

    df = read_manifest(cfg["paths"]["manifest"])
    artists = known_labels(df)
    label_map = {a: i for i, a in enumerate(artists)}
    known = df[df.role == "known"]
    tr, va = known[known.split == "train"], known[known.split == "val"]
    if args.max_train:  # debug subset that still covers every artist
        per_artist = max(2, math.ceil(args.max_train / len(artists)))
        tr = tr.sample(frac=1.0, random_state=seed).groupby("artist").head(per_artist)
    tcfg = cfg["train"]
    diag = tr.sample(n=min(tcfg["diag_train_images"], len(tr)), random_state=seed)
    print(f"{len(artists)} artists | train {len(tr)} | val {len(va)} | diag {len(diag)}")

    train_loader = make_loader(tr, label_map, cfg, train=True, batch_size=tcfg["batch_size"], shuffle=True)
    eval_bs = tcfg["eval_batch_size"]
    train_eval_loader = make_loader(tr, label_map, cfg, train=False, batch_size=eval_bs)
    val_loader = make_loader(va, label_map, cfg, train=False, batch_size=eval_bs)
    diag_loader = make_loader(diag, label_map, cfg, train=False, batch_size=eval_bs)

    model = ArtistNet(len(artists), cfg["encoder"]["name"], cfg["encoder"]["trainable_blocks"],
                      cfg["encoder"]["gradient_checkpointing"], cfg["features"]["hand_weight"],
                      tcfg["dropout"]).to(device)

    counts = np.bincount(tr.artist.map(label_map).values, minlength=len(artists)).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.mean()  # square-root inverse-frequency class weights
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                    label_smoothing=tcfg["label_smoothing"])

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": tcfg["lr_head"]},
         {"params": enc_params, "lr": tcfg["lr_encoder"]}], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_factor(e, tcfg["warmup_epochs"], tcfg["max_epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    history, best_f1, since_best, start = [], -1.0, 0, 0
    latest = out / "latest.pt"
    if latest.exists() and not args.fresh:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best_f1, since_best, start = ck["history"], ck["best_f1"], ck["since_best"], ck["epoch"] + 1
        print(f"resumed from epoch {start}")
    else:
        print("warm-starting the head with logistic regression on frozen features ...")
        warm_start_head(model, train_eval_loader, len(artists), cfg, device, amp_dtype)

    accum = tcfg["grad_accum_steps"]
    for epoch in range(start, tcfg["max_epochs"]):
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        running, steps = 0.0, 0
        for i, (views, hsv, hog, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)):
            with amp_context(amp_dtype):
                logits = model(views.to(device, non_blocking=True), hsv.to(device), hog.to(device))
            loss = criterion(logits.float(), y.to(device))
            scaler.scale(loss / accum).backward()
            running += float(loss)
            steps += 1
            if (i + 1) % accum == 0 or i + 1 == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        sched.step()

        v_logits, _, v_y = predict(model, val_loader, device, amp_dtype)
        d_logits, _, d_y = predict(model, diag_loader, device, amp_dtype)
        val = classification_metrics(v_logits, v_y)
        dg = classification_metrics(d_logits, d_y)
        rec = {"epoch": epoch + 1, "train_loss": running / max(steps, 1), "val_macro_f1": val["macro_f1"],
               "val_accuracy": val["accuracy"], "val_nll": val["nll"], "diag_train_macro_f1": dg["macro_f1"],
               "gap": dg["macro_f1"] - val["macro_f1"], "seconds": time.time() - t0}
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, since_best = val["macro_f1"], 0
            torch.save({"model": trainable_state(model), "epoch": epoch + 1, "val_macro_f1": best_f1,
                        "artists": artists}, out / "best.pt")
        else:
            since_best += 1
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "best_f1": best_f1,
                    "since_best": since_best, "epoch": epoch}, latest)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if epoch + 1 >= tcfg["min_epochs"] and since_best >= tcfg["patience"]:
            print(f"early stop at epoch {epoch + 1}; best val macro-F1 {best_f1:.4f}")
            break
    print(f"done. best validation macro-F1 {best_f1:.4f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
