# Artist identification (V3.2)

DINOv2 fine-tuning fused with HSV/LBP and HOG features, multi-view inputs (one global view plus
four high-resolution local tiles), and unknown-artist rejection with several prototypes per artist.
This is an extension of the V1 to V3 study (Project 32, PES University) and of Chen & Deng,
Stanford CS229 (2018). It is a fresh implementation written from the V1-V3 report, not a copy of
the original private code, so numbers will not match V3 exactly until you reuse the V3 manifest.

## Setup (Windows PowerShell, RTX 4060 8 GB)

```powershell
python -m venv artist-env
artist-env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

## Data

Put images under `data/images/<artist>/...` and build the manifest (random stratified 60/15/10/15
split for known artists, three disjoint unknown cohorts):

```powershell
python scripts/make_manifest.py --root data/images `
  --unknown-dev "name1,name2" --unknown-cal "name3,name4" --unknown-test "name5,name6"
```

For results comparable with V3, convert your audited V3 manifest to the same columns instead
(`path, artist, role, split`; roles `known | unknown_dev | unknown_cal | unknown_test`). The helper
script does no de-duplication.

## Train and evaluate

```powershell
python -m artistid.train --config configs/v3_2.yaml --seed 106
python -m artistid.evaluate --run runs/v3_2/seed106            # validation / development only
python -m artistid.evaluate --run runs/v3_2/seed106 --final    # touches the test split: do this once
```

Training saves `best.pt` and `latest.pt` every epoch and resumes automatically if you re-run the
same command. If you hit CUDA out-of-memory, set `train.batch_size=2` and `train.grad_accum_steps=16`
(`--set train.batch_size=2 --set train.grad_accum_steps=16`). On Windows, if the data loader hangs,
use `--set train.num_workers=0`.

## Ablations (one change at a time, judge on validation)

| Run | How |
|---|---|
| V3 baseline (ViT-S, global view) | `--set encoder.name=facebook/dinov2-small --set views.local_grid=0 --set train.batch_size=8 --set train.grad_accum_steps=4` |
| ViT-B only | `--set views.local_grid=0` |
| Multi-view only | `--set encoder.name=facebook/dinov2-small` |
| ViT-B + multi-view | defaults |
| Single prototype gate | evaluate with `--set rejection.prototypes_per_artist=1` (no retraining) |

Run seeds 106, 107, 108 and report mean and spread. Compare paired predictions with
`artistid.metrics.mcnemar_exact`.

## Protocol

Mix, temperature and threshold are chosen on validation, calibration and unknown-development data
only. The test split is read only with `--final`. Since the V3 test set has already been inspected,
a fresh confirmation set is needed for any new claim of improvement.

## Not implemented

CutMix/mixup, a second backbone (ConvNeXt/CLIP), and automatic duplicate screening. The code has
not been run against your dataset or on a GPU by the author of this scaffold; run the unit tests
first, then a short smoke run with `--max-train 64 --set train.max_epochs=1`.
