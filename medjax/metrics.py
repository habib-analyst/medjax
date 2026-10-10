"""Segmentation / classification metrics as pure functions.

``preds``/``targets`` are integer label maps ``(*spatial)`` (batch-leading
``(B, *spatial)`` also fine); ``num_classes`` selects the label set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _cm(preds: jax.Array, targets: jax.Array, num_classes: int) -> jax.Array:
    """Confusion matrix ``(C, C)``: rows = target, cols = pred."""
    p = preds.reshape(-1).astype(jnp.int32)
    t = targets.reshape(-1).astype(jnp.int32)
    idx = t * num_classes + p
    return jnp.bincount(idx, length=num_classes * num_classes).reshape(num_classes, num_classes)


def dice_score(
    preds: jax.Array, targets: jax.Array, num_classes: int, include_background: bool = True
) -> jax.Array:
    """Per-class Dice ``(C,)``."""
    cm = _cm(preds, targets, num_classes)
    tp = jnp.diag(cm).astype(jnp.float32)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    dice = 2 * tp / jnp.maximum(denom, 1e-8)
    return dice if include_background else dice[1:]


def iou_score(
    preds: jax.Array, targets: jax.Array, num_classes: int, include_background: bool = True
) -> jax.Array:
    """Per-class IoU/Jaccard ``(C,)``."""
    cm = _cm(preds, targets, num_classes)
    tp = jnp.diag(cm).astype(jnp.float32)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    iou = tp / jnp.maximum(union, 1e-8)
    return iou if include_background else iou[1:]


def accuracy(preds: jax.Array, targets: jax.Array) -> jax.Array:
    """Pixel (or sample) accuracy."""
    return jnp.mean(preds.astype(jnp.int32) == targets.astype(jnp.int32))


def mean_dice(preds, targets, num_classes, include_background: bool = False) -> jax.Array:
    """Mean Dice over foreground classes — the headline segmentation number."""
    return jnp.mean(
        dice_score(preds, targets, num_classes, include_background=include_background)
    )
