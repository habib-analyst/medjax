#!/bin/bash
# Full medjax evaluation pipeline. Resumable: train_medmnist.py skips datasets
# with existing JSONs. Writes EVALS_DONE on completion.
set -u
cd "$(dirname "$0")"
VENV=.venv/bin/python
TORCHVENV=../tta-calib/.venv/bin/python
LOG=eval/run_all.log
mkdir -p eval

{
echo "=== $(date -u) JAX training ==="
$VENV scripts/train_medmnist.py --datasets breastmnist,pneumoniamnist,dermamnist --epochs 5 --out eval/results_jax

echo "=== $(date -u) PyTorch parity ==="
$TORCHVENV scripts/parity_pytorch.py --datasets breastmnist,pneumoniamnist,dermamnist --epochs 5 --out eval/results_torch

echo "=== $(date -u) transform benchmark ==="
$VENV scripts/bench_transforms.py

echo "=== $(date -u) 3D UNet smoke ==="
$VENV scripts/unet_3d_smoke.py

echo "=== $(date -u) ALL EVALS COMPLETE ==="
touch EVALS_DONE
} >> "$LOG" 2>&1
