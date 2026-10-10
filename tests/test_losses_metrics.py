import jax
import jax.numpy as jnp
import numpy as np
import pytest

from medjax import losses, metrics


@pytest.fixture
def seg():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    logits = jax.random.normal(k1, (2, 16, 16, 3))
    targets = jax.random.randint(k2, (2, 16, 16), 0, 3)
    return logits, targets


def test_dice_loss_range(seg):
    logits, targets = seg
    v = losses.dice_loss(logits, targets)
    assert 0.0 <= float(v) <= 1.0


def test_dice_loss_perfect():
    targets = jnp.array([[[0, 1], [1, 0]]])
    logits = jnp.array([[[[10.0, -10.0], [-10.0, 10.0]],
                         [[-10.0, 10.0], [10.0, -10.0]]]])
    assert float(losses.dice_loss(logits, targets)) < 0.01


def test_dice_ce_combines(seg):
    logits, targets = seg
    d = float(losses.dice_loss(logits, targets))
    c = float(losses.ce_loss(logits, targets))
    dc = float(losses.dice_ce_loss(logits, targets, ce_weight=1.0))
    assert abs(dc - (d + c)) < 1e-5


def test_focal_loss_positive(seg):
    logits, targets = seg
    assert float(losses.focal_loss(logits, targets)) > 0


def test_losses_jit(seg):
    logits, targets = seg
    f = jax.jit(losses.dice_ce_loss)
    assert float(f(logits, targets)) >= 0


def test_dice_score_perfect():
    p = jnp.array([0, 1, 1, 0, 2])
    t = jnp.array([0, 1, 1, 0, 2])
    d = metrics.dice_score(p, t, num_classes=3)
    assert jnp.allclose(d, 1.0)


def test_dice_score_miss():
    p = jnp.zeros(10, dtype=jnp.int32)
    t = jnp.ones(10, dtype=jnp.int32)
    d = metrics.dice_score(p, t, num_classes=2, include_background=False)
    assert float(d[0]) < 1e-6


def test_iou_bounds():
    p = jnp.array([0, 1, 1, 2])
    t = jnp.array([0, 1, 2, 2])
    iou = metrics.iou_score(p, t, num_classes=3)
    assert bool(jnp.all(iou >= 0)) and bool(jnp.all(iou <= 1))


def test_accuracy():
    assert float(metrics.accuracy(jnp.array([0, 1, 1]), jnp.array([0, 1, 0]))) == pytest.approx(2 / 3)


def test_mean_dice_foreground():
    p = jnp.array([0, 1, 1, 0])
    t = jnp.array([0, 1, 1, 0])
    assert float(metrics.mean_dice(p, t, num_classes=2)) == pytest.approx(1.0)
