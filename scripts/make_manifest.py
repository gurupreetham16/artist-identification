"""Build data/manifest.csv from a folder layout  <root>/<artist>/<image files>.

Known artists get a stratified train/val/cal/test split (default 60/15/10/15). Artists listed in
--unknown-dev / --unknown-cal / --unknown-test become the three disjoint unknown cohorts.

  python scripts/make_manifest.py --root data/images \
      --unknown-dev "caspar-david-friedrich,george-stubbs" \
      --unknown-cal "benjamin-west,franz-marc" \
      --unknown-test "balthus,toyen"

This script does NOT de-duplicate. If you already have the V3 audited manifest / index.json from
your Drive folder, convert that instead so the split matches the V3 report exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--unknown-dev", default="")
    ap.add_argument("--unknown-cal", default="")
    ap.add_argument("--unknown-test", default="")
    ap.add_argument("--fractions", default="0.60,0.15,0.10,0.15", help="train,val,cal,test")
    ap.add_argument("--seed", type=int, default=106)
    args = ap.parse_args()

    root = Path(args.root)
    fr = [float(x) for x in args.fractions.split(",")]
    assert abs(sum(fr) - 1) < 1e-6 and len(fr) == 4
    unk = {}
    for role, names in (("unknown_dev", args.unknown_dev), ("unknown_cal", args.unknown_cal),
                        ("unknown_test", args.unknown_test)):
        for n in filter(None, (s.strip() for s in names.split(","))):
            unk[n] = role

    rng = np.random.default_rng(args.seed)
    rows = []
    for adir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in adir.rglob("*") if f.suffix.lower() in EXT)
        rel = [str(f.relative_to(root)).replace("\\", "/") for f in files]
        if adir.name in unk:
            rows += [(r, adir.name, unk[adir.name], "") for r in rel]
            continue
        order = rng.permutation(len(rel))
        n = len(rel)
        cuts = np.cumsum([int(round(f * n)) for f in fr[:3]])
        for rank, idx in enumerate(order):
            split = "train" if rank < cuts[0] else "val" if rank < cuts[1] else "cal" if rank < cuts[2] else "test"
            rows.append((rel[idx], adir.name, "known", split))

    df = pd.DataFrame(rows, columns=["path", "artist", "role", "split"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["role", "split"]).size())
    print(f"wrote {args.out} ({len(df)} images, {df[df.role == 'known'].artist.nunique()} known artists)")


if __name__ == "__main__":
    main()
