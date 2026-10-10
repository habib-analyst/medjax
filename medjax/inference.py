"""Sliding-window inference for large volumes.

``sliding_window_inference`` tiles a ``(*spatial, C)`` volume (or a batch)
into overlapping ROIs, runs a jitted ``model`` on each, and blends with a
Gaussian importance map — the standard trick for seamless large-volume
segmentation, here as one pure-ish function.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def _gaussian_importance(roi: Sequence[int]) -> jax.Array:
    grids = jnp.meshgrid(
        *[jnp.linspace(-1, 1, s) for s in roi], indexing="ij"
    )
    dist2 = sum(g**2 for g in grids)
    w = jnp.exp(-4.0 * dist2)
    return w / jnp.max(w)


def _tile_starts(length: int, roi: int, overlap: float) -> list[int]:
    stride = max(1, int(roi * (1 - overlap)))
    starts = list(range(0, max(1, length - roi + 1), stride))
    if starts[-1] + roi < length:
        starts.append(length - roi)
    return starts


def sliding_window_inference(
    volume: jax.Array,
    roi_size: Sequence[int],
    model: Callable[[jax.Array], jax.Array],
    overlap: float = 0.5,
    batch_size: int = 4,
) -> jax.Array:
    """Run ``model`` over ``volume`` in overlapping windows.

    ``volume``: ``(*spatial, C)`` or ``(B, *spatial, C)``; ``model`` maps a
    batch of windows ``(b, *roi, C)`` to ``(b, *roi, C_out)`` (should be
    jitted by the caller). Returns ``(*spatial, C_out)`` (batch dim squeezed
    if the input was unbatched).
    """
    batched = volume.ndim == len(roi_size) + 2
    if not batched:
        volume = volume[None]
    B = volume.shape[0]
    spatial = volume.shape[1:-1]
    C = volume.shape[-1]
    roi = tuple(min(r, s) for r, s in zip(roi_size, spatial))

    grids = [_tile_starts(s, r, overlap) for s, r in zip(spatial, roi)]
    positions = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(roi), -1).T

    # probe output channels with one window
    probe_in = volume[0:1, *[slice(p, p + r) for p, r in zip(positions[0], roi)], :]
    C_out = model(probe_in).shape[-1]

    importance = _gaussian_importance(roi)
    out = jnp.zeros((B, *spatial, C_out), dtype=jnp.float32)
    weight = jnp.zeros((B, *spatial, 1), dtype=jnp.float32)

    for b in range(B):
        for i in range(0, len(positions), batch_size):
            chunk = positions[i : i + batch_size]
            wins = jnp.stack(
                [
                    jax.lax.dynamic_slice(
                        volume[b], [*[p for p in pos], 0], [*roi, C]
                    )
                    for pos in chunk
                ]
            )
            preds = model(wins)  # (b, *roi, C_out)
            for pos, pred in zip(chunk, preds):
                sl = (b, *[slice(p, p + r) for p, r in zip(pos, roi)])
                out = out.at[sl].add(pred * importance[..., None])
                weight = weight.at[(b, *[slice(p, p + r) for p, r in zip(pos, roi)], slice(None))].add(
                    importance[..., None]
                )
    out = out / jnp.maximum(weight, 1e-8)
    return out if batched else out[0]
