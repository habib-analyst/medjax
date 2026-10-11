# medjax

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/medjax.svg)]

**A JAX-native medical imaging toolkit: functional, jit-able transforms, NIfTI I/O, losses, metrics, sliding-window inference, and Flax NNX models — MONAI for the JAX/TPU era.**

MONAI is PyTorch-only. `medjax` is the alternative for the JAX/TPU stack, and it is not a port: every transform is a pure function that composes under `jax.jit` and `jax.vmap`; models use Flax NNX (no `batch_stats` plumbing, GroupNorm instead of BatchNorm).

## Quickstart

```bash
# Not on PyPI yet — install from source:
git clone https://github.com/habib-analyst/medjax && cd medjax && pip install -e .
```

```python
from medjax import data as mjdata, losses, models, recipes

d = mjdata.load_medmnist("breastmnist")            # JAX arrays, channel-last
model = models.TinyCNN(in_channels=1, num_classes=2, seed=0)
hist = recipes.quick_train(                        # AdamW + cosine + warmup, jitted
    model, losses.ce_loss, d["train"], d["val"], num_classes=2, epochs=5,
)
print("val_acc:", hist["val_acc"][-1])
```

Expected output (breastmnist, 5 epochs, CPU, from our measured parity run):

```
val_acc: 0.7308
```

## Measured results

All measured on CPU in this repo's `eval/` (see `PAPER.md` for the full writeup).

| Experiment | Result |
|---|---|
| JAX vs PyTorch parity (TinyCNN, same arch/epochs/seed, 5ep) | breast: JAX **0.7308** vs torch **0.7308** (exact) · pneumonia: **0.8526** vs **0.8478** (Δ 0.005) · derma: **0.6748** vs **0.6903** (Δ 0.016) |
| jit'd transform pipeline (128×128×64 vol, best-of-5) | **2.14×** vs same code eager · **7.57×** vs numpy/scipy reference |
| 3D UNet smoke (synthetic BraTS-style, 48³) | DiceCE ↓ **1.4631 → 1.1717** in 6 steps · held-out fg-Dice **0.2202** |
| Tests | **40/40** green |

Notes: the jit benchmark compares the same transform pipeline under three implementations; sampling conventions differ between the JAX and numpy/scipy versions (documented in `eval/`). No TPU runs measured — CPU only so far.

## How it works

| Module | What |
|---|---|
| `medjax.transforms` | pure, jit-able: `normalize`, `rescale_intensity`, `clip` (windowing), `resize`, `resample_to_spacing`, `flip`/`random_flip`, `rot90`/`random_rot90`, `crop_center`, `pad_to`, `orientation_flip_to_ras` |
| `medjax.io` | `load_nifti` → `(array, meta)` with spacing/affine/orientation; `save_nifti` |
| `medjax.losses` | `dice_loss`, `ce_loss`, `dice_ce_loss`, `focal_loss` |
| `medjax.metrics` | `dice_score`, `iou_score`, `accuracy`, `mean_dice` |
| `medjax.models` | TinyCNN / UNet / SegResNet, 2D+3D, Flax NNX |
| `medjax.inference` | sliding-window inference for volumes |
| `medjax.data` | MedMNIST `.npz` loaders (JAX arrays, no torch dependency) |
| `medjax.recipes` | `quick_train`: AdamW + cosine + warmup, fully jitted |
| `medjax.dataloader` | Grain recipes (optional dep) |
| `medjax.checkpoint` | Orbax checkpointing (optional dep) |

Optional dependencies are lazy: `medjax.transforms` needs only JAX; Grain/Orbax load only when imported.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
