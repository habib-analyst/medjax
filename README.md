# medjax — JAX-native medical imaging toolkit

**MONAI is PyTorch-only. This is the JAX/TPU-native alternative.**

`pip install medjax` → functional, jit-able transforms · NIfTI I/O · DiceCE/Dice/focal losses · segmentation metrics · sliding-window inference · UNet/SegResNet in Flax NNX · Grain dataloader recipes · Orbax checkpointing · MedMNIST loaders out of the box.

Not a MONAI port — JAX-idiomatic: every transform is a pure function that composes under `jax.jit` and `jax.vmap`; models use Flax NNX (no `batch_stats` plumbing, GroupNorm instead of BatchNorm).

## 30-second quickstart

```python
from medjax import data as mjdata, losses, models, recipes

d = mjdata.load_medmnist("breastmnist")          # JAX arrays, channel-last
model = models.TinyCNN(in_channels=1, num_classes=2, seed=0)
hist = recipes.quick_train(                        # AdamW + cosine + warmup, jitted
    model, losses.ce_loss, d["train"], d["val"], num_classes=2, epochs=5,
)
print("val_acc:", hist["val_acc"][-1])
```

Or the CLI-free script: `python examples/quickstart.py --dataset breastmnist --epochs 5`

## Why JAX for medical imaging?

- `jit` the *whole* pipeline — preprocessing included, not just the model
- `pmap`/`pjit` scale to TPU pods; PyTorch medical training hits ceilings on large volumes
- Functional transforms: no hidden state, reproducible by construction

## Measured numbers (CPU, this repo's `scripts/`)

| Experiment | Result |
|---|---|
| JAX vs PyTorch parity (TinyCNN, same arch/epochs/seed, 5ep) | breast: JAX **0.7308** vs torch **0.7308** (exact) · pneumonia: **0.8526** vs **0.8478** (Δ 0.005) · derma: **0.6748** vs **0.6903** (Δ 0.016) |
| jit'd transform pipeline (128×128×64 vol, best-of-5) | **2.14×** vs same code eager · **7.57×** vs numpy/scipy reference |
| 3D UNet smoke (synthetic BraTS-style, 48³) | DiceCE ↓ **1.4631** → **1.1717** in 6 steps · held-out fg-Dice **0.2202** |
| Tests | **40/40** green |

See [`hidden_files/papers/medjax/PAPER.md`](https://github.com/habib-analyst/medjax) for the full writeup.

## API map

| Module | What |
|---|---|
| `medjax.transforms` | pure, jit-able: `normalize`, `rescale_intensity`, `clip` (windowing), `resize`, `resample_to_spacing`, `flip`/`random_flip`, `rot90`/`random_rot90`, `crop_center`, `pad_to`, `orientation_flip_to_ras` |
| `medjax.io` | `load_nifti` → `(array, meta)` with spacing/affine/orientation; `save_nifti` |
| `medjax.losses` | `dice_loss`, `ce_loss`, `dice_ce_loss`, `focal_loss` |
| `medjax.metrics` | `dice_score`, `iou_score`, `accuracy`, `mean_dice` |
| `medjax.models` | `TinyCNN`, `UNet` (2D/3D), `SegResNet` (2D/3D) — Flax NNX |
| `medjax.inference` | `sliding_window_inference` with Gaussian blending |
| `medjax.data` | `load_medmnist` → JAX arrays; `minibatches` |
| `medjax.recipes` | `quick_train` (one-call training), `default_optimizer` |
| `medjax.dataloader` | Grain recipes *(optional: `pip install grain`)* |
| `medjax.checkpoint` | Orbax save/restore *(optional: `pip install orbax-checkpoint`)* |

## Design notes

- **Channel-last everywhere** (`(*spatial, C)`), matching Flax. Convert at I/O edges with `ensure_channel_first/last`.
- Stochastic transforms take an explicit `key` — no global RNG state.
- `dataloader` and `checkpoint` are optional imports: the core never requires Grain/Orbax.

## Limitations (honest)

- 2D paths are battle-tested (MedMNIST); 3D is smoke-tested on synthetics — real BraTS validation is future work.
- CPU-tested; TPU/GPU paths are API-compatible but unmeasured here.
- NIfTI only for now (no DICOM series reader yet).

## License

MIT — Habib Ur Rehman, 2026.
