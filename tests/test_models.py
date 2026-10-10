import jax
import jax.numpy as jnp

from medjax import models


def test_tinycnn_forward():
    m = models.TinyCNN(in_channels=1, num_classes=2, seed=0)
    x = jnp.zeros((2, 28, 28, 1))
    out = jax.jit(m)(x)
    assert out.shape == (2, 2)


def test_tinycnn_3ch_7class():
    m = models.TinyCNN(in_channels=3, num_classes=7, seed=1)
    out = m(jnp.zeros((4, 28, 28, 3)))
    assert out.shape == (4, 7)


def test_unet_2d_forward():
    m = models.UNet(1, 3, features=(8, 16), spatial_dims=2, seed=0)
    out = jax.jit(m)(jnp.zeros((1, 32, 32, 1)))
    assert out.shape == (1, 32, 32, 3)


def test_unet_3d_forward():
    m = models.UNet(1, 2, features=(8, 16), spatial_dims=3, seed=0)
    out = m(jnp.zeros((1, 16, 16, 16, 1)))
    assert out.shape == (1, 16, 16, 16, 2)


def test_segresnet_2d_forward():
    m = models.SegResNet(1, 2, features=(8, 16), spatial_dims=2, seed=0)
    out = jax.jit(m)(jnp.zeros((1, 32, 32, 1)))
    assert out.shape == (1, 32, 32, 2)


def test_segresnet_3d_forward():
    m = models.SegResNet(1, 2, features=(8, 16), spatial_dims=3, seed=0)
    out = m(jnp.zeros((1, 16, 16, 16, 1)))
    assert out.shape == (1, 16, 16, 16, 2)


def test_models_deterministic():
    m = models.TinyCNN(1, 2, seed=0)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 28, 28, 1))
    assert jnp.allclose(m(x), m(x))
