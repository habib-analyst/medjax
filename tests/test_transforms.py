import jax
import jax.numpy as jnp
import numpy as np
import pytest

from medjax import transforms as T


@pytest.fixture
def vol():
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (32, 32, 3))  # 2D, channel-last


def test_normalize_zero_mean_unit_var(vol):
    out = T.normalize(vol)
    assert abs(float(jnp.mean(out))) < 1e-5
    assert abs(float(jnp.std(out)) - 1.0) < 1e-5


def test_rescale_intensity_range(vol):
    out = T.rescale_intensity(vol, 0.0, 1.0)
    assert float(jnp.min(out)) >= 0.0 and float(jnp.max(out)) <= 1.0 + 1e-6


def test_clip_windowing(vol):
    out = T.clip(vol, -1.0, 1.0)
    assert float(jnp.min(out)) >= -1.0 and float(jnp.max(out)) <= 1.0


def test_resize_shape(vol):
    out = T.resize(vol, (16, 16))
    assert out.shape == (16, 16, 3)


def test_resample_to_spacing_doubles(vol):
    out = T.resample_to_spacing(vol, (2.0, 2.0), (1.0, 1.0))
    assert out.shape == (64, 64, 3)


def test_flip_involution(vol):
    assert jnp.allclose(T.flip(T.flip(vol, (0,)), (0,)), vol)


def test_random_flip_deterministic_key(vol):
    k = jax.random.PRNGKey(1)
    assert jnp.allclose(T.random_flip(k, vol, (0,)), T.random_flip(k, vol, (0,)))


def test_rot90_shape_3d():
    x = jnp.zeros((10, 20, 5, 2))
    assert T.rot90(x, 1, (0, 1)).shape == (20, 10, 5, 2)


def test_crop_center(vol):
    assert T.crop_center(vol, (16, 16)).shape == (16, 16, 3)


def test_pad_to(vol):
    out = T.pad_to(vol, (40, 40))
    assert out.shape == (40, 40, 3)


def test_crop_or_pad_exact(vol):
    assert T.crop_or_pad(vol, (24, 24)).shape == (24, 24, 3)
    assert T.crop_or_pad(vol, (40, 40)).shape == (40, 40, 3)


def test_channel_conventions():
    x = jnp.zeros((3, 16, 16))  # channel-first
    assert T.ensure_channel_last(x, channel_dim=0).shape == (16, 16, 3)
    y = jnp.zeros((16, 16, 3))
    assert T.ensure_channel_first(y).shape == (3, 16, 16)


def test_to_one_hot():
    y = jnp.array([[0, 1], [2, 0]])
    oh = T.to_one_hot(y, 3)
    assert oh.shape == (2, 2, 3)
    assert jnp.allclose(oh.sum(-1), 1.0)


def test_orientation_flip():
    x = jnp.zeros((8, 8, 8, 1))
    out, applied = T.orientation_flip_to_ras(x, "LPI")
    assert applied == [0, 1, 2] and out.shape == x.shape  # L->R, P->A, I->S
    out2, applied2 = T.orientation_flip_to_ras(x, "RAS")
    assert applied2 == []


def test_jit_compose(vol):
    # the headline property: a whole pipeline jits
    def pipeline(x):
        x = T.rescale_intensity(x)
        x = T.resize(x, (16, 16))
        return T.zscore_per_channel(x)

    f = jax.jit(pipeline)
    out = f(vol)
    assert out.shape == (16, 16, 3)
    # and vmaps
    batched = jax.vmap(pipeline)(jnp.stack([vol, vol]))
    assert batched.shape == (2, 16, 16, 3)


def test_random_augment_pipeline_jits(vol):
    def aug(key, x):
        x = T.random_flip(key, x, (0,))
        return T.resize(x, (16, 16))

    f = jax.jit(aug)
    assert f(jax.random.PRNGKey(0), vol).shape == (16, 16, 3)
