"""Functional, jit-able image transforms for medical imaging.

Design rules (JAX-idiomatic, deliberately unlike MONAI's stateful API):
- every transform is a pure function ``(x, ...) -> x`` (or ``(key, x, ...) -> x``
  for stochastic ones) — no hidden state, no in-place mutation;
- spatial args (axes, sizes, spacings) are Python values, so every transform
  composes under ``jax.jit`` and ``jax.vmap``;
- channel convention is **channel-last** (``..., C``), matching Flax NNX.
  Use :func:`ensure_channel_first` / :func:`ensure_channel_last` at I/O edges.

Spatial dims convention: ``x`` has shape ``(*spatial, C)`` where
``*spatial`` is 2D or 3D.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# intensity
# ---------------------------------------------------------------------------

def normalize(x: jax.Array, mean: float | None = None, std: float | None = None) -> jax.Array:
    """Standardize ``x``: ``(x - mean) / std`` (statistics of ``x`` if omitted)."""
    m = jnp.mean(x) if mean is None else mean
    s = jnp.std(x) if std is None else std
    return (x - m) / jnp.maximum(s, 1e-8)


def rescale_intensity(
    x: jax.Array, out_min: float = 0.0, out_max: float = 1.0
) -> jax.Array:
    """Min-max rescale of ``x`` into ``[out_min, out_max]``."""
    lo, hi = jnp.min(x), jnp.max(x)
    x01 = (x - lo) / jnp.maximum(hi - lo, 1e-8)
    return x01 * (out_max - out_min) + out_min


def clip(x: jax.Array, a_min: float, a_max: float) -> jax.Array:
    """Clip intensities to ``[a_min, a_max]`` (e.g. CT windowing)."""
    return jnp.clip(x, a_min, a_max)


def zscore_per_channel(x: jax.Array) -> jax.Array:
    """Per-channel standardization over the spatial dims (channel-last)."""
    spatial = tuple(range(x.ndim - 1))
    m = jnp.mean(x, axis=spatial, keepdims=True)
    s = jnp.std(x, axis=spatial, keepdims=True)
    return (x - m) / jnp.maximum(s, 1e-8)


# ---------------------------------------------------------------------------
# geometry — all jit-able
# ---------------------------------------------------------------------------

def resize(x: jax.Array, spatial_size: Sequence[int], method: str = "linear") -> jax.Array:
    """Resize spatial dims to ``spatial_size`` (channels untouched)."""
    out_shape = tuple(spatial_size) + (x.shape[-1],)
    return jax.image.resize(x, out_shape, method=method)


def resample_to_spacing(
    x: jax.Array,
    in_spacing: Sequence[float],
    out_spacing: Sequence[float],
    method: str = "linear",
) -> jax.Array:
    """Resample to a target voxel ``out_spacing`` (spacing-aware, like MONAI's
    ``Spacingd`` — but a pure function). ``in_spacing``/``out_spacing`` are in
    mm per axis, ordered like the spatial dims of ``x``."""
    # pure-Python arithmetic: keeps shapes static so the transform jits
    zoom = [float(i) / float(o) for i, o in zip(in_spacing, out_spacing)]
    new_spatial = tuple(int(round(s * z)) for s, z in zip(x.shape[:-1], zoom))
    return resize(x, new_spatial, method=method)


def flip(x: jax.Array, axes: Sequence[int]) -> jax.Array:
    """Flip spatial ``axes`` (e.g. ``axes=(0,)`` mirrors the first axis)."""
    return jnp.flip(x, axis=tuple(axes))


def rot90(x: jax.Array, k: int, axes: tuple[int, int]) -> jax.Array:
    """Rotate 90° ``k`` times in the plane ``axes``."""
    return jnp.rot90(x, k=k, axes=axes)


def random_flip(
    key: jax.Array, x: jax.Array, axes: Sequence[int], prob: float = 0.5
) -> jax.Array:
    """Flip spatial ``axes`` with probability ``prob`` (per call, keyed)."""
    do = jax.random.bernoulli(key, prob)
    return jax.lax.cond(do, lambda a: flip(a, axes), lambda a: a, x)


def random_rot90(
    key: jax.Array, x: jax.Array, axes: tuple[int, int]
) -> jax.Array:
    """Random 0/90/180/270° rotation in the plane ``axes``."""
    k = jax.random.randint(key, (), 0, 4)
    return jnp.rot90(x, k=k, axes=axes)


def crop_center(x: jax.Array, spatial_size: Sequence[int]) -> jax.Array:
    """Center-crop spatial dims to ``spatial_size``."""
    starts = [(s - t) // 2 for s, t in zip(x.shape[:-1], spatial_size)]
    slices = tuple(slice(st, st + t) for st, t in zip(starts, spatial_size))
    return x[slices + (slice(None),)]


def pad_to(x: jax.Array, spatial_size: Sequence[int], mode: str = "constant") -> jax.Array:
    """Zero-pad (or edge-pad) spatial dims up to ``spatial_size``."""
    pads = [(0, max(0, t - s)) for s, t in zip(x.shape[:-1], spatial_size)]
    widths = [(b, a) for b, a in pads]  # (before, after); pad at the end
    return jnp.pad(x, widths + [(0, 0)], mode=mode)


def crop_or_pad(x: jax.Array, spatial_size: Sequence[int]) -> jax.Array:
    """Center-crop or zero-pad spatial dims to exactly ``spatial_size``."""
    x = crop_center(x, tuple(min(s, t) for s, t in zip(x.shape[:-1], spatial_size)))
    return pad_to(x, spatial_size)


# ---------------------------------------------------------------------------
# channel / label helpers
# ---------------------------------------------------------------------------

def ensure_channel_last(x: jax.Array, channel_dim: int = 0) -> jax.Array:
    """Move the channel axis to the last position."""
    if channel_dim < 0:
        channel_dim = x.ndim + channel_dim
    axes = [a for a in range(x.ndim) if a != channel_dim] + [channel_dim]
    return jnp.transpose(x, axes)


def ensure_channel_first(x: jax.Array) -> jax.Array:
    """Move the channel axis (last) to the front — for PyTorch interop."""
    axes = [x.ndim - 1] + list(range(x.ndim - 1))
    return jnp.transpose(x, axes)


def to_one_hot(y: jax.Array, num_classes: int) -> jax.Array:
    """Integer label map ``(*spatial)`` -> one-hot ``(*spatial, C)``."""
    return jax.nn.one_hot(y.astype(jnp.int32), num_classes)


def orientation_flip_to_ras(
    x: jax.Array, orientation: str
) -> tuple[jax.Array, list[int]]:
    """Reorient a volume given as an orientation code like ``"LPI"`` towards
    RAS, returning ``(reoriented, flips_applied)``.

    This is the lightweight, metadata-free cousin of nibabel's
    ``as_closest_canonical``: the caller supplies the current orientation
    code (from :func:`medjax.io.load_nifti` metadata); each axis whose code
    differs from RAS ``("R", "A", "S")`` gets flipped.
    """
    ras = ("R", "A", "S")
    flip_axes, applied = [], []
    for i, code in enumerate(orientation[: x.ndim - 1]):
        if code != ras[i]:
            flip_axes.append(i)
            applied.append(i)
    if flip_axes:
        x = flip(x, flip_axes)
    return x, applied
