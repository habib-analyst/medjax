"""Training recipes: one-call training with sensible JAX defaults.

``quick_train`` is the 30-second path: pass a model, a loss, and split dicts
(from :func:`medjax.data.load_medmnist`); it handles the optimizer, cosine
schedule, jitted step, and validation loop, and returns a history dict.

This is the "auto-tune" piece of medjax: cosine decay with warmup, AdamW,
gradient clipping, and mixed float32 — chosen once, correctly, so users
don't have to.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from medjax import data as mjdata
from medjax import metrics as mjmetrics


def default_schedule(lr: float, total_steps: int, warmup_steps: int = 0):
    """Cosine decay with optional linear warmup."""
    if warmup_steps > 0:
        warmup = optax.linear_schedule(0.0, lr, warmup_steps)
        decay = optax.cosine_decay_schedule(lr, total_steps - warmup_steps)
        return optax.join_schedules([warmup, decay], [warmup_steps])
    return optax.cosine_decay_schedule(lr, total_steps)


def default_optimizer(lr: float = 3e-4, total_steps: int = 1000, warmup_steps: int = 0):
    """AdamW + cosine schedule + global-norm clipping (the safe default)."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(default_schedule(lr, total_steps, warmup_steps), weight_decay=1e-4),
    )


def make_train_step(model: nnx.Module, optimizer: optax.GradientTransformation, loss_fn: Callable):
    """Build a jitted ``(state, opt_state, batch) -> (loss, ...)`` step.

    Works for any NNX model + ``loss_fn(logits, labels)`` pure loss.
    """
    graphdef, params = nnx.split(model)

    @jax.jit
    def step(params, opt_state, batch):
        m = nnx.merge(graphdef, params)

        def _loss(p):
            mm = nnx.merge(graphdef, p)
            logits = mm(batch["images"])
            return loss_fn(logits, batch["labels"])

        loss, grads = jax.value_and_grad(_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    opt_state = optimizer.init(params)
    return step, params, opt_state


def quick_train(
    model: nnx.Module,
    loss_fn: Callable,
    train_split: dict,
    val_split: dict,
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    seed: int = 0,
    log_every: int = 1,
    verbose: bool = True,
) -> dict:
    """Train a classifier end-to-end; returns ``{"train_loss", "val_acc", ...}``.

    ``train_split``/``val_split`` are ``{"images", "labels"}`` dicts of JAX
    arrays (see :func:`medjax.data.load_medmnist`).
    """
    key = jax.random.PRNGKey(seed)
    n_train = train_split["images"].shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    optimizer = default_optimizer(lr, epochs * steps_per_epoch,
                                  warmup_steps=min(200, steps_per_epoch))
    step_fn, params, opt_state = make_train_step(model, optimizer, loss_fn)
    graphdef, _ = nnx.split(model)

    history: dict[str, list[float]] = {"train_loss": [], "val_acc": [], "val_loss": []}
    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        losses = []
        for batch in mjdata.minibatches(sub, train_split["images"], train_split["labels"], batch_size):
            params, opt_state, loss = step_fn(params, opt_state, batch)
            losses.append(float(loss))
        m = nnx.merge(graphdef, params)
        val_logits = m(val_split["images"])
        val_loss = float(loss_fn(val_logits, val_split["labels"]))
        val_acc = float(mjmetrics.accuracy(
            jnp.argmax(val_logits, axis=-1), val_split["labels"]))
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"ep {epoch:02d} train_loss={history['train_loss'][-1]:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
    nnx.update(model, params)  # write trained params back into the model
    return history
