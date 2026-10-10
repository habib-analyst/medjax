"""NIfTI I/O with metadata, returning JAX arrays.

``load_nifti`` returns ``(array, meta)`` where ``meta`` carries everything a
spacing/orientation-aware pipeline needs: ``spacing`` (mm), ``affine``,
``orientation`` (e.g. ``"RAS"``), and ``shape``. Arrays are channel-last
with an explicit channel dim: ``(*spatial, C)``.
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import nibabel as nib
import numpy as np


def load_nifti(path: str) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Load a NIfTI file -> ``(array, meta)``.

    ``array`` is a JAX array, channel-last with explicit channel dim.
    ``meta`` = ``{"spacing", "affine", "orientation", "shape", "dtype"}``.
    """
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    if data.ndim == 3:
        data = data[..., None]  # add channel dim
    # channel-first (C, *spatial) -> channel-last (*spatial, C)
    if data.ndim == 4 and data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
        data = np.moveaxis(data, 0, -1)
    zooms = img.header.get_zooms()[:3]
    orientation = "".join(nib.aff2axcodes(img.affine))
    meta = {
        "spacing": tuple(float(z) for z in zooms),
        "affine": np.asarray(img.affine),
        "orientation": orientation,
        "shape": tuple(int(s) for s in data.shape),
        "dtype": str(data.dtype),
        "path": os.path.abspath(path),
    }
    return jnp.asarray(data), meta


def save_nifti(array: jnp.ndarray, meta: dict[str, Any], path: str) -> str:
    """Save a channel-last array as NIfTI, reusing ``meta["affine"]``."""
    arr = np.asarray(array)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    affine = meta.get("affine", np.eye(4))
    img = nib.Nifti1Image(arr, affine)
    nib.save(img, path)
    return os.path.abspath(path)


def save_numpy_as_nifti(
    array: np.ndarray, spacing: tuple[float, float, float], path: str
) -> str:
    """Save a raw numpy volume with an isotropic-spacing affine (for synthetics)."""
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(np.asarray(array), affine), path)
    return os.path.abspath(path)
