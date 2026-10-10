"""medjax — a JAX-native medical imaging toolkit.

"MONAI for the JAX/TPU era": functional, jit-able transforms, NIfTI I/O,
segmentation losses/metrics, Flax NNX models, sliding-window inference,
MedMNIST loaders, Grain dataloader recipes, and Orbax checkpointing.

Everything is a pure function wherever possible; transforms compose and jit.
Submodules are imported lazily so a missing optional dependency never breaks
the core (``medjax.transforms`` only needs JAX).
"""

__version__ = "0.1.0"

_SUBMODULES = {
    "transforms", "io", "losses", "metrics", "models",
    "inference", "data", "recipes", "dataloader", "checkpoint",
}

__all__ = ["__version__"] + sorted(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib
        return importlib.import_module(f"medjax.{name}")
    raise AttributeError(f"module 'medjax' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
