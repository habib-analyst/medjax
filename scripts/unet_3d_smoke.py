"""3D UNet smoke test: synthetic BraTS-style volumes, verify loss decreases.

Generates N=8 synthetic 48^3 volumes: random smooth background + 1-3 Gaussian
blobs as the "tumor" label. Trains a small 3D UNet with DiceCE for a few
steps and asserts the loss goes down. Writes eval/unet_3d_smoke.json.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from scipy.ndimage import gaussian_filter

from medjax import losses, metrics, models


def synth_volume(key, size=48, n_blobs=2):
    k1, k2 = jax.random.split(key)
    bg = np.asarray(jax.random.normal(k1, (size, size, size))) * 0.3
    bg = gaussian_filter(bg, 2.0)
    label = np.zeros((size, size, size), np.int32)
    rng = np.random.default_rng(int(k2[0]))
    for _ in range(n_blobs):
        c = rng.integers(8, size - 8, 3)
        r = rng.integers(4, 9)
        zz, yy, xx = np.ogrid[:size, :size, :size]
        blob = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2) < r * r
        label[blob] = 1
    img = bg + label * 1.5 + np.asarray(jax.random.normal(k2, bg.shape)) * 0.1
    return img[..., None].astype(np.float32), label


def main():
    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    imgs, labels = zip(*[synth_volume(k) for k in keys])
    X = jnp.asarray(np.stack(imgs))          # (8, 48, 48, 48, 1)
    Y = jnp.asarray(np.stack(labels))        # (8, 48, 48, 48)
    print("data:", X.shape, Y.shape, flush=True)

    model = models.UNet(1, 2, features=(8, 16, 32), spatial_dims=3, seed=0)
    graphdef, params = nnx.split(model)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, xb, yb):
        def _loss(p):
            m = nnx.merge(graphdef, p)
            return losses.dice_ce_loss(m(xb), yb)
        loss, grads = jax.value_and_grad(_loss)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    first, last = None, None
    for i in range(6):
        params, opt_state, loss = step(params, opt_state, X[:4], Y[:4])
        loss = float(loss)
        first = loss if first is None else first
        last = loss
        print(f"step {i}: dicece={loss:.4f}", flush=True)

    m = nnx.merge(graphdef, params)
    preds = jnp.argmax(m(X[4:]), -1)
    dice = float(metrics.mean_dice(preds, Y[4:], num_classes=2))
    res = {
        "steps": 6, "loss_first": round(first, 4), "loss_last": round(last, 4),
        "loss_decreased": last < first,
        "heldout_mean_dice_fg": round(dice, 4),
        "volume": "(48,48,48,1)", "wall_s": round(time.time() - t0, 1),
    }
    os.makedirs("eval", exist_ok=True)
    with open("eval/unet_3d_smoke.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    assert last < first, "3D UNet smoke FAILED: loss did not decrease"


if __name__ == "__main__":
    main()
