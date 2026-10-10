"""Segmentation losses as pure functions: ``loss(logits, targets) -> scalar``.

Conventions: ``logits`` channel-last ``(*spatial, C)`` (raw scores),
``targets`` integer label map ``(*spatial)`` with values in ``[0, C)``.
All losses are ``jax.jit``- and ``jax.vmap``-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _one_hot(targets: jax.Array, num_classes: int) -> jax.Array:
    return jax.nn.one_hot(targets.astype(jnp.int32), num_classes)


def dice_loss(
    logits: jax.Array,
    targets: jax.Array,
    smooth: float = 1e-5,
    include_background: bool = True,
    squared: bool = False,
) -> jax.Array:
    """Soft Dice loss, averaged over classes and batch (batch = leading dim)."""
    probs = jax.nn.softmax(logits, axis=-1)
    y = _one_hot(targets, logits.shape[-1])
    if not include_background:
        probs, y = probs[..., 1:], y[..., 1:]
    reduce = tuple(range(probs.ndim - 1))  # batch + spatial
    inter = jnp.sum(probs * y, axis=reduce)
    denom = (
        jnp.sum(probs ** 2 + y ** 2, axis=reduce)
        if squared
        else jnp.sum(probs + y, axis=reduce)
    )
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - jnp.mean(dice)


def ce_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Softmax cross-entropy, mean over batch+spatial."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    y = _one_hot(targets, logits.shape[-1])
    return -jnp.mean(jnp.sum(y * logp, axis=-1))


def dice_ce_loss(
    logits: jax.Array,
    targets: jax.Array,
    ce_weight: float = 1.0,
    smooth: float = 1e-5,
    include_background: bool = True,
) -> jax.Array:
    """DiceCE = dice_loss + ce_weight * ce_loss (MONAI's workhorse default)."""
    return dice_loss(
        logits, targets, smooth=smooth, include_background=include_background
    ) + ce_weight * ce_loss(logits, targets)


def focal_loss(
    logits: jax.Array, targets: jax.Array, gamma: float = 2.0, alpha: float = 0.25
) -> jax.Array:
    """Sigmoid focal loss per class, for multi-label / imbalanced problems."""
    y = _one_hot(targets, logits.shape[-1]).astype(jnp.float32)
    p = jax.nn.sigmoid(logits)
    pt = jnp.where(y == 1, p, 1 - p)
    w = jnp.where(y == 1, alpha, 1 - alpha)
    return jnp.mean(-w * (1 - pt) ** gamma * jnp.log(jnp.maximum(pt, 1e-8)))
