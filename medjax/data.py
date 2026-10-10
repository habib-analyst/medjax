"""MedMNIST loaders returning JAX arrays, channel-last.

Reads the official ``.npz`` files (same cache dir as the ``medmnist``
package, ``~/.medmnist`` by default) — no re-download, no torch.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

MEDMNIST_NCLASSES = {
    "pathmnist": 9, "chestmnist": 14, "dermamnist": 7, "octmnist": 4,
    "pneumoniamnist": 2, "retinamnist": 5, "breastmnist": 2, "bloodmnist": 8,
    "tissuemnist": 8, "organamnist": 11, "organcmnist": 11, "organsmnist": 11,
}

DEFAULT_ROOT = os.path.expanduser("~/.medmnist")


def _load_npz(name: str, root: str = DEFAULT_ROOT) -> dict[str, np.ndarray]:
    path = os.path.join(root, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — install the `medmnist` package and download "
            f"'{name}' once, or point root= at your .npz cache."
        )
    return dict(np.load(path))


def load_medmnist(
    name: str, root: str = DEFAULT_ROOT, normalize: bool = True
) -> dict[str, dict[str, jax.Array]]:
    """Load a MedMNIST ``.npz`` -> ``{"train"/"val"/"test": {"images", "labels"}}``.

    Images: ``(N, H, W, C)`` float32 in ``[0, 1]``; labels: ``(N,)`` int32.
    """
    raw = _load_npz(name, root)
    out: dict[str, dict[str, jax.Array]] = {}
    for split in ("train", "val", "test"):
        imgs = raw[f"{split}_images"].astype(np.float32)
        if imgs.ndim == 3:
            imgs = imgs[..., None]  # add channel dim -> (N, H, W, C)
        if normalize:
            imgs = imgs / 255.0
        labels = raw[f"{split}_labels"].astype(np.int32).reshape(-1)
        out[split] = {"images": jnp.asarray(imgs), "labels": jnp.asarray(labels)}
    return out


def num_classes(name: str) -> int:
    """Number of classes for a MedMNIST dataset."""
    return MEDMNIST_NCLASSES[name]


def minibatches(
    key: jax.Array,
    images: jax.Array,
    labels: jax.Array,
    batch_size: int,
    shuffle: bool = True,
):
    """Yield ``{"images", "labels"}`` minibatch dicts."""
    n = images.shape[0]
    idx = jnp.arange(n)
    if shuffle:
        idx = jax.random.permutation(key, idx)
    for start in range(0, n, batch_size):
        sel = idx[start : start + batch_size]
        yield {"images": images[sel], "labels": labels[sel]}


def describe(name: str, root: str = DEFAULT_ROOT) -> dict[str, Any]:
    """One-line-per-split summary (shapes, classes) — for READMEs and logs."""
    d = load_medmnist(name, root)
    return {
        split: {
            "images": tuple(v["images"].shape),
            "classes": sorted(set(int(c) for c in np.asarray(v["labels"]))),
        }
        for split, v in d.items()
    }
