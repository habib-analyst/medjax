"""Orbax checkpointing for Flax NNX models (optional dependency).

Saves/restores the full NNX state (params) of a model. Importing without
``orbax-checkpoint`` installed raises a clear error — the rest of medjax
never needs it.
"""

from __future__ import annotations

import os

try:
    import orbax.checkpoint as ocp
    _ORBAX_OK = True
except ImportError:  # pragma: no cover
    ocp = None  # type: ignore
    _ORBAX_OK = False

from flax import nnx
import jax


def _require_orbax():
    if not _ORBAX_OK:
        raise ImportError(
            "medjax.checkpoint needs `orbax-checkpoint`: pip install orbax-checkpoint"
        )


def save_checkpoint(path: str, model: nnx.Module) -> str:
    """Save NNX model state to ``path`` (creates the directory)."""
    _require_orbax()
    os.makedirs(path, exist_ok=True)
    state = nnx.state(model)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(os.path.join(path, "state"), state)
    return os.path.abspath(path)


def restore_checkpoint(path: str, model: nnx.Module) -> nnx.Module:
    """Restore NNX model state from ``path`` into ``model`` (in place)."""
    _require_orbax()
    abstract = nnx.state(model)
    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(os.path.join(path, "state"), abstract)
    nnx.update(model, restored)
    return model


def latest_checkpoint(dirpath: str) -> str | None:
    """Return the most recently modified checkpoint dir, if any."""
    if not os.path.isdir(dirpath):
        return None
    cands = [
        os.path.join(dirpath, d)
        for d in os.listdir(dirpath)
        if os.path.isdir(os.path.join(dirpath, d))
    ]
    return max(cands, key=os.path.getmtime) if cands else None
