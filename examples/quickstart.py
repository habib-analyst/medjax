"""medjax 30-second quickstart: MedMNIST classification in JAX.

    python examples/quickstart.py [--dataset breastmnist] [--epochs 3]
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
from medjax import data as mjdata, losses, models, recipes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breastmnist")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    d = mjdata.load_medmnist(args.dataset)
    n_cls = mjdata.num_classes(args.dataset)
    n_ch = d["train"]["images"].shape[-1]
    print(f"{args.dataset}: train={d['train']['images'].shape} classes={n_cls}")

    model = models.TinyCNN(in_channels=n_ch, num_classes=n_cls, seed=0)
    hist = recipes.quick_train(
        model, losses.ce_loss, d["train"], d["val"], n_cls,
        epochs=args.epochs, batch_size=128, seed=0,
    )
    test_logits = model(d["test"]["images"])
    test_acc = float(jnp.mean(jnp.argmax(test_logits, -1) == d["test"]["labels"]))
    print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
