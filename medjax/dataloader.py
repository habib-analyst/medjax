"""Grain dataloader recipes (optional dependency).

Grain is Google's JAX-native dataloader. These helpers build standard
train/val ``grain.DataLoader``\\ s over in-memory JAX arrays with keyed
shuffling and on-the-fly augmentation via :mod:`medjax.transforms`.
Importing this module without ``grain`` installed raises a clear error —
the rest of medjax never needs it.
"""

from __future__ import annotations

from typing import Callable, Sequence

try:
    import grain.python as grain
    _GRAIN_OK = True
except ImportError:  # pragma: no cover
    grain = None  # type: ignore
    _GRAIN_OK = False

import jax
import numpy as np


def _require_grain():
    if not _GRAIN_OK:
        raise ImportError(
            "medjax.dataloader needs the `grain` package: pip install grain"
        )


class _ArraySource(grain.RandomAccessDataSource):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self._images, self._labels = images, labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return {"images": self._images[idx], "labels": self._labels[idx]}


class _Augment(grain.MapTransform):
    def __init__(self, augment_fn: Callable, seed: int):
        self._fn, self._seed = augment_fn, seed

    def map(self, element):
        key = jax.random.PRNGKey(self._seed + int(element.get("_idx", 0)))
        element["images"] = np.asarray(self._fn(key, jax.numpy.asarray(element["images"])))
        return element


def make_loader(
    images,
    labels,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    augment_fn: Callable | None = None,
    num_workers: int = 0,
    drop_remainder: bool = True,
):
    """Build a ``grain.DataLoader`` over ``(images, labels)`` numpy/JAX arrays.

    ``augment_fn`` is a ``(key, image) -> image`` callable built from
    :mod:`medjax.transforms` (e.g. ``lambda k, x: transforms.random_flip(k, x, (0,))``).
    """
    _require_grain()
    images_np, labels_np = np.asarray(images), np.asarray(labels)
    ds = _ArraySource(images_np, labels_np)
    ops: list = []
    if shuffle:
        ops.append(grain.transforms.Shuffle(seed=seed))
    if augment_fn is not None:
        ops.append(_Augment(augment_fn, seed))
    ops.append(grain.transforms.Batch(batch_size, drop_remainder=drop_remainder))
    return grain.DataLoader(
        data_source=ds,
        operations=ops,
        worker_count=num_workers,
        read_options=grain.ReadOptions(num_threads=4),
    )


def train_val_loaders(
    train: dict,
    val: dict,
    batch_size: int,
    seed: int = 0,
    augment_fn: Callable | None = None,
):
    """Paired train (shuffled + augmented) / val loaders over split dicts."""
    train_dl = make_loader(
        train["images"], train["labels"], batch_size,
        shuffle=True, seed=seed, augment_fn=augment_fn,
    )
    val_dl = make_loader(
        val["images"], val["labels"], batch_size,
        shuffle=False, seed=seed,
    )
    return train_dl, val_dl
