"""PyTorch parity baseline: same TinyCNN architecture, same data, same epochs.

Mirrors scripts/train_medmnist.py exactly (AdamW, cosine schedule, batch 128,
seed 0) so the paper can report JAX-vs-PyTorch numbers honestly.
Run with a torch-capable python: <torch-venv>/bin/python scripts/parity_pytorch.py
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_npz(name, root=os.path.expanduser("~/.medmnist")):
    z = np.load(os.path.join(root, f"{name}.npz"))
    out = {}
    for split in ("train", "val", "test"):
        imgs = z[f"{split}_images"].astype(np.float32) / 255.0
        if imgs.ndim == 3:
            imgs = imgs[..., None]
        # NHWC -> NCHW
        imgs = np.moveaxis(imgs, -1, 1)
        out[split] = (imgs, z[f"{split}_labels"].astype(np.int64).reshape(-1))
    return out


class TinyCNN(nn.Module):
    def __init__(self, in_ch, n_cls, width=32):
        super().__init__()
        g = min(8, width)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GroupNorm(g, width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.GroupNorm(min(8, width * 2), width * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.GroupNorm(min(8, width * 4), width * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 4, 128), nn.ReLU(),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def run(dataset, epochs, seed, batch_size=128, lr=3e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = load_npz(dataset)
    n_cls = int(d["train"][1].max()) + 1
    in_ch = d["train"][0].shape[1]
    model = TinyCNN(in_ch, n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_train = len(d["train"][0])
    steps = epochs * (n_train // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = np.random.RandomState(seed + ep).permutation(n_train)
        tl = 0.0
        nb = 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            xb = torch.from_numpy(d["train"][0][idx])
            yb = torch.from_numpy(d["train"][1][idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); nb += 1
        print(f"ep {ep:02d} train_loss={tl / nb:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for s in range(0, len(d["test"][0]), 1024):
            xb = torch.from_numpy(d["test"][0][s:s + 1024])
            yb = d["test"][1][s:s + 1024]
            pred = model(xb).argmax(1).numpy()
            correct += (pred == yb).sum(); total += len(yb)
    return {"dataset": dataset, "seed": seed, "epochs": epochs,
            "test_acc": correct / total, "wall_s": round(time.time() - t0, 1),
            "backend": "pytorch-cpu"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breastmnist,pneumoniamnist,dermamnist")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results_torch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        res[ds] = run(ds, args.epochs, args.seed)
        with open(os.path.join(args.out, f"{ds}_seed{args.seed}.json"), "w") as f:
            json.dump(res[ds], f, indent=2)
        print(f"DONE {ds}: test_acc={res[ds]['test_acc']:.4f}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
