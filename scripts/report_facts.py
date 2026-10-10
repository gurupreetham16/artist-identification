"""Print a compact summary of a finished run for the write-up.

  python scripts/report_facts.py --run runs/v3_2/seed106
Paste the output back. Reads history.json, config_used.json and results_*.json from the run folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)

    hist = load(run / "history.json") or []
    cfg = load(run / "config_used.json") or {}
    out = {"run": str(run)}

    if hist:
        best = max(hist, key=lambda r: r["val_macro_f1"])
        out["training"] = {
            "epochs_run": len(hist),
            "best_epoch": best["epoch"],
            "best_val_macro_f1": round(best["val_macro_f1"], 4),
            "best_val_accuracy": round(best["val_accuracy"], 4),
            "diag_train_macro_f1_at_best": round(best["diag_train_macro_f1"], 4),
            "train_minus_val_f1_at_best": round(best["gap"], 4),
            "mean_seconds_per_epoch": round(sum(r["seconds"] for r in hist) / len(hist), 1),
            "total_hours": round(sum(r["seconds"] for r in hist) / 3600, 2),
        }
    if cfg:
        out["config"] = {
            "encoder": cfg["encoder"]["name"], "trainable_blocks": cfg["encoder"]["trainable_blocks"],
            "views": f"1 global + {cfg['views']['local_grid'] ** 2} local" if cfg["views"]["local_grid"] else "global only",
            "batch x accum": f"{cfg['train']['batch_size']} x {cfg['train']['grad_accum_steps']}",
            "lr_head/lr_encoder": [cfg["train"]["lr_head"], cfg["train"]["lr_encoder"]],
            "prototypes_per_artist": cfg["rejection"]["prototypes_per_artist"],
            "seed": cfg.get("seed"),
        }

    for name in ("results_dev.json", "results_final.json"):
        r = load(run / name)
        if not r:
            out[name] = "missing - run artistid.evaluate first"
            continue
        slim = {k: v for k, v in r.items() if "per_artist" not in k}
        for key in ("val_per_artist_f1", "TEST_per_artist_f1"):
            if key in r:
                ranked = sorted(r[key].items(), key=lambda kv: kv[1])
                slim[key + "_weakest5"] = {k: round(v, 3) for k, v in ranked[:5]}
                slim[key + "_strongest5"] = {k: round(v, 3) for k, v in ranked[-5:]}
        out[name] = slim

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
