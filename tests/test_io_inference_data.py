import jax
import jax.numpy as jnp
import numpy as np
import pytest

from medjax import io as mjio
from medjax import inference, data as mjdata
from medjax import models


def test_nifti_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    vol = (rng.random((16, 16, 16)) * 100).astype(np.float32)
    p = str(tmp_path / "test.nii.gz")
    mjio.save_numpy_as_nifti(vol, (1.5, 1.5, 2.0), p)
    arr, meta = mjio.load_nifti(p)
    assert arr.shape == (16, 16, 16, 1)
    assert meta["spacing"] == (1.5, 1.5, 2.0)
    assert meta["orientation"] == "RAS"
    assert np.allclose(np.asarray(arr)[..., 0], vol, atol=1e-4)


def test_nifti_save_load_meta(tmp_path):
    arr = jnp.zeros((10, 10, 10, 2))
    meta = {"affine": np.eye(4)}
    p = str(tmp_path / "m.nii.gz")
    mjio.save_nifti(arr, meta, p)
    arr2, meta2 = mjio.load_nifti(p)
    assert arr2.shape == (10, 10, 10, 2)


def test_sliding_window_identity():
    model = jax.jit(lambda x: x * 2.0)
    vol = jnp.ones((24, 24, 1))
    out = inference.sliding_window_inference(vol, (16, 16), model, overlap=0.5)
    assert out.shape == (24, 24, 1)
    # interior should be ~2.0 (Gaussian blending is exact away from borders)
    assert abs(float(out[12, 12, 0]) - 2.0) < 0.05


def test_sliding_window_with_unet():
    m = models.UNet(1, 2, features=(8, 16), spatial_dims=2, seed=0)
    jm = jax.jit(m)
    vol = jax.random.normal(jax.random.PRNGKey(0), (40, 40, 1))
    out = inference.sliding_window_inference(vol, (32, 32), jm, overlap=0.25, batch_size=2)
    assert out.shape == (40, 40, 2)


def test_medmnist_loader_shapes():
    d = mjdata.load_medmnist("breastmnist")
    assert d["train"]["images"].shape[1:] == (28, 28, 1)
    assert d["train"]["labels"].ndim == 1
    assert mjdata.num_classes("breastmnist") == 2


def test_medmnist_values_normalized():
    d = mjdata.load_medmnist("pneumoniamnist")
    imgs = d["train"]["images"]
    assert float(jnp.min(imgs)) >= 0.0 and float(jnp.max(imgs)) <= 1.0


def test_minibatches_cover():
    key = jax.random.PRNGKey(0)
    xs = jnp.zeros((10, 4))
    ys = jnp.arange(10)
    seen = []
    for b in mjdata.minibatches(key, xs, ys, batch_size=4, shuffle=False):
        seen.extend([int(v) for v in b["labels"]])
    assert sorted(seen) == list(range(10))
