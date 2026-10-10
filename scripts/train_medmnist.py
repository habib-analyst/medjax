"""Train TinyCNN on MedMNIST sets in JAX; save metrics JSON for the paper.

Usage: python scripts/train_medmnist.py --datasets breastmnist,pneumoniamnist,dermamnist --epochs 5 --out eval/results_jax.json
Resumable: writes per-dataset JSONs; skips datasets already done unless --redo.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
from medjax import data as mjdata, losses, models, recipes


def train_one(dataset, epochs, seed, out_dir):
    d = mjdata.load_medmnist(dataset)
    n_cls = mjdata.num_classes(dataset)
    n_ch = d["train"]["images"].shape[-1]
    model = models.TinyCNN(in_channels=n_ch, num_classes=n_cls, seed=seed)
    t0 = time.time()
    hist = recipes.quick_train(
        model, losses.ce_loss, d["train"], d["val"], n_cls,
        epochs=epochs, batch_size=128, seed=seed, verbose=True,
    )
    test_logits = model(d["test"]["images"])
    test_acc = float(jnp.mean(jnp.argmax(test_logits, -1) == d["test"]["labels"]))
    wall = time.time() - t0
    res = {
        "dataset": dataset, "seed": seed, "epochs": epochs,
        "val_acc": hist["val_acc"][-1], "test_acc": test_acc,
        "train_loss": hist["train_loss"][-1], "wall_s": round(wall, 1),
        "backend": "jax-cpu",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_seed{seed}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"DONE {dataset}: test_acc={test_acc:.4f} wall={wall:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breastmnist,pneumoniamnist,dermamnist")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results_jax")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    all_res = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        marker = os.path.join(args.out, f"{ds}_seed{args.seed}.json")
        if os.path.exists(marker) and not args.redo:
            print(f"SKIP {ds} (done)")
            with open(marker) as f:
                all_res[ds] = json.load(f)
            continue
        all_res[ds] = train_one(ds, args.epochs, args.seed, args.out)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
