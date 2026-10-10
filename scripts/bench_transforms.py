"""Benchmark: medjax jit'd transform pipeline vs eager numpy/scipy equivalents.

Pipeline (on a 128x128x64x1 synthetic volume): rescale -> resample to 1mm
spacing -> zscore per channel -> random flip. Times best-of-5 after warmup.
Writes eval/bench_transforms.json.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import zoom as scipy_zoom

from medjax import transforms as T


def medjax_pipeline(x):
    x = T.rescale_intensity(x)
    x = T.resample_to_spacing(x, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    x = T.zscore_per_channel(x)
    return x


def numpy_pipeline(x):
    x = np.asarray(x)
    lo, hi = x.min(), x.max()
    x = (x - lo) / max(hi - lo, 1e-8)
    x = scipy_zoom(x[..., 0], 2.0, order=1)[..., None].astype(np.float32)
    m, s = x.mean(), x.std()
    return (x - m) / max(s, 1e-8)


def bench(fn, arg, warmup=2, iters=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(arg)) if isinstance(arg, jax.Array) else fn(arg)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(arg)
        if isinstance(out, jax.Array):
            jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    key = jax.random.PRNGKey(0)
    vol = jax.random.normal(key, (128, 128, 64, 1))
    vol_np = np.asarray(vol)

    jit_pipe = jax.jit(medjax_pipeline)
    t_jit = bench(jit_pipe, vol)
    t_eager_jax = bench(medjax_pipeline, vol)   # un-jitted, same code
    t_numpy = bench(numpy_pipeline, vol_np)

    # correctness: same output shape, close values
    a = np.asarray(jit_pipe(vol))
    b = numpy_pipeline(vol_np)
    assert a.shape == b.shape, (a.shape, b.shape)
    maxdiff = float(np.abs(a - b).max())

    res = {
        "volume": "(128,128,64,1)",
        "medjax_jit_s": round(t_jit, 4),
        "medjax_eager_s": round(t_eager_jax, 4),
        "numpy_scipy_s": round(t_numpy, 4),
        "speedup_jit_vs_numpy": round(t_numpy / t_jit, 2),
        "speedup_jit_vs_eager": round(t_eager_jax / t_jit, 2),
        "max_abs_diff": round(maxdiff, 6),
    }
    os.makedirs("eval", exist_ok=True)
    with open("eval/bench_transforms.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
