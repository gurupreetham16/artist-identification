"""Unpack the V3.1 Drive shards and build data/manifest.csv.

Inputs (all downloaded from your Drive folder into data/):
  data/wikiart-39-v31/images-*.zip   image shards (images/<hash>.jpg inside)
  data/wikiart-39-v31/index.json     shard list with sha256 checksums
  data/image_quality.csv             image_id, artist, path, usable, ...
  data/artists.csv                   artist, role, ...
  data/per_artist_support.csv        (optional) train/val/calibration/test counts per known artist

Outputs:
  data/images/<artist>/<image_id>.jpg
  data/manifest.csv                  path, artist, role, split

Only images marked usable=True are extracted. Known artists are split with the per-artist counts from
per_artist_support.csv when they match the usable image count, otherwise 60/15/10/15. The
assignment of WHICH image goes to which split is a fresh seeded shuffle, so it matches the V3 counts
but not the exact V3 images; that is why V3 test scores are not directly reusable.

  python scripts/prepare_v31_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_MAP = {
    "known": "known",
    "unknown_dev": "unknown_dev",
    "unknown_cal": "unknown_cal",
    "unknown_calibration": "unknown_cal",
    "unknown_test": "unknown_test",
}
SPLITS = ["train", "val", "cal", "test"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_counts(n: int, support_row, fractions) -> list[int]:
    """Counts for train/val/cal/test summing to n."""
    if support_row is not None:
        c = [int(support_row[k]) for k in ("train", "val", "calibration", "test")]
        if sum(c) == n:
            return c
    f = fractions
    tr, va, ca = int(round(f[0] * n)), int(round(f[1] * n)), int(round(f[2] * n))
    return [tr, va, ca, n - tr - va - ca]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/wikiart-39-v31")
    ap.add_argument("--meta", default="data", help="folder with image_quality.csv and artists.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=106)
    ap.add_argument("--fractions", default="0.60,0.15,0.10,0.15")
    ap.add_argument("--no-verify", action="store_true", help="skip shard checksum verification")
    args = ap.parse_args()

    shards, meta, out = Path(args.shards), Path(args.meta), Path(args.out)
    fractions = [float(x) for x in args.fractions.split(",")]

    index = json.loads((shards / "index.json").read_text(encoding="utf-8"))
    if not args.no_verify:
        print("verifying shard checksums ...")
        bad = []
        for ch in index["chunks"]:
            p = shards / ch["name"]
            if not p.exists():
                bad.append(f"{ch['name']} (missing)")
            elif sha256_file(p) != ch["sha256"]:
                bad.append(f"{ch['name']} (checksum mismatch: incomplete or corrupted download)")
        if bad:
            sys.exit("shard problems:\n  " + "\n  ".join(bad) + "\nRe-download those files and run again.")
        print(f"  {len(index['chunks'])} shards OK")

    q = pd.read_csv(meta / "image_quality.csv")
    a = pd.read_csv(meta / "artists.csv")
    q = q[q["usable"].astype(str) == "True"].drop_duplicates("image_id")
    a["role"] = a["role"].map(ROLE_MAP)
    if a["role"].isna().any():
        sys.exit("artists.csv has a role I don't recognise: " + str(sorted(set(pd.read_csv(meta / 'artists.csv')['role']))))
    roles = dict(zip(a.artist, a.role))
    unknown_artists = sorted(set(q.artist) - set(roles))
    if unknown_artists:
        sys.exit(f"artists in image_quality.csv but not in artists.csv: {unknown_artists[:5]}")
    support = None
    sp = meta / "per_artist_support.csv"
    if sp.exists():
        support = pd.read_csv(sp).set_index("artist")

    rng = np.random.default_rng(args.seed)
    rows = []
    for artist, g in q.groupby("artist"):
        ids = sorted(g.image_id)
        role = roles[artist]
        if role != "known":
            rows += [(i, artist, role, "") for i in ids]
            continue
        row = support.loc[artist] if support is not None and artist in support.index else None
        counts = split_counts(len(ids), row, fractions)
        order = [ids[k] for k in rng.permutation(len(ids))]
        bounds = np.cumsum([0] + counts)
        for s, name in enumerate(SPLITS):
            rows += [(i, artist, "known", name) for i in order[bounds[s]:bounds[s + 1]]]

    plan = pd.DataFrame(rows, columns=["image_id", "artist", "role", "split"])
    plan["member"] = "images/" + plan.image_id + ".jpg"
    plan["path"] = plan.artist + "/" + plan.image_id + ".jpg"
    wanted = dict(zip(plan.member, plan.path))

    img_root = out / "images"
    done, seen = 0, set()
    print(f"extracting {len(wanted)} usable images ...")
    for ch in index["chunks"]:
        with zipfile.ZipFile(shards / ch["name"]) as zf:
            for member in zf.namelist():
                rel = wanted.get(member)
                if rel is None or member in seen:
                    continue
                seen.add(member)
                dest = img_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                done += 1
        print(f"  {ch['name']}: {done}/{len(wanted)}")

    missing = plan[~plan.member.isin(seen)]
    if len(missing):
        print(f"WARNING: {len(missing)} usable images were not found in any shard "
              f"(first few: {missing.member.head(3).tolist()}). They are left out of the manifest.")
    plan = plan[plan.member.isin(seen)]

    plan[["path", "artist", "role", "split"]].to_csv(out / "manifest.csv", index=False)
    print("\nmanifest summary:")
    print(plan.groupby(["role", "split"]).size().to_string())
    print(f"\nknown artists: {plan[plan.role == 'known'].artist.nunique()}  total images: {len(plan)}")
    print(f"wrote {out / 'manifest.csv'}; images in {img_root}")


if __name__ == "__main__":
    main()
