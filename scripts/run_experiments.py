"""Run the experiment grid. Every finished run is appended to results/runs.jsonl, so the script
can be interrupted and resumed, and the notebook only has to read the log.

Protocol: the 12,000 images are pooled and split into 5 folds *by drawing* (see data.py).
Rotation r uses fold r as the test set, fold r+1 for validation (early stopping and every
hyper-parameter decision) and the remaining 3 folds for training.

Usage:  python scripts/run_experiments.py <stage> [--device mps|cpu]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hanacaraka.data import Aksara, rotation  # noqa: E402
from hanacaraka.train import DEFAULT, DEVICE, fit, predict_proba  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOG = os.path.join(ROOT, "results", "runs.jsonl")
FOLD_SEED = 0


def read_log():
    return [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []


def cfg_key(cfg, split, seed):
    return json.dumps({**DEFAULT, **cfg, "seed": seed}, sort_keys=True) + split


class Runner:
    def __init__(self, device):
        self.device = device
        self.ds = Aksara(os.path.join(ROOT, "cache"))
        self.fold = self.ds.drawing_folds(5, FOLD_SEED)
        self.y = torch.from_numpy(self.ds.y).to(device)
        self._x = {}
        log = read_log()
        self.done = {(r["stage"], r["name"], r["split"], r["seed"]) for r in log}
        self.by_cfg = {cfg_key(r["cfg"], r["split"], r["seed"]): r for r in log}

    def x(self, size):
        if size not in self._x:
            self._x = {size: self.ds.tensor(size, self.device)}  # keep one resolution in memory
        return self._x[size]

    def run(self, stage, name, cfg, split, idx, seed=0, save_preds=False, tta=0):
        """idx = (train, val, test) index arrays. `split` is a label such as 'rot0' or 'writer:theo'."""
        key = (stage, name, split, seed)
        if key in self.done:
            return
        twin = self.by_cfg.get(cfg_key(cfg, split, seed))
        if twin is not None and not save_preds:   # identical run already trained under another name
            with open(LOG, "a") as f:
                f.write(json.dumps({**twin, "stage": stage, "name": name}) + "\n")
            self.done.add(key)
            return
        tr, va, te = (torch.from_numpy(i).to(self.device) for i in idx)
        x = self.x(cfg.get("size", 64))
        model, h = fit(x[tr], self.y[tr], x[va], self.y[va], {**cfg, "seed": seed}, self.device)
        p = predict_proba(model, x[te], tta=tta)
        yt = self.ds.y[idx[2]]
        rec = dict(stage=stage, name=name, split=split, seed=seed, cfg=cfg,
                   n_train=len(idx[0]), n_val=len(idx[1]), n_test=len(idx[2]),
                   val_acc=h["best_val_acc"], val_loss=h["best_val_loss"],
                   test_acc=float((p.argmax(1) == yt).mean()),
                   test_f1=float(f1_score(yt, p.argmax(1), average="macro")),
                   best_epoch=h["best_epoch"], epochs_run=h["epochs_run"], seconds=h["seconds"], params=h["params"],
                   hist={k: h[k] for k in ("loss", "acc", "val_loss", "val_acc")})
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done.add(key)
        self.by_cfg[cfg_key(cfg, split, seed)] = rec
        if save_preds:
            os.makedirs(os.path.join(ROOT, "results", "preds"), exist_ok=True)
            np.savez_compressed(os.path.join(ROOT, "results", "preds", f"{stage}_{name}_{split.replace(':', '-')}_s{seed}.npz"),
                                idx=idx[2], proba=p.astype(np.float32))
            torch.save(model.state_dict(), os.path.join(ROOT, "results", "preds", f"{stage}_{name}_{split.replace(':', '-')}_s{seed}.pt"))
        print(f"[{stage}] {name:<28} {split:<14} seed {seed}  val {h['best_val_acc']:.4f}  test {rec['test_acc']:.4f}  "
              f"ep {h['best_epoch']}/{h['epochs_run']}  {h['seconds']:.0f}s", flush=True)

    def cv(self, stage, name, cfg, rots=range(5), seeds=(0,), **kw):
        for r in rots:
            for s in seeds:
                self.run(stage, name, cfg, f"rot{r}", rotation(self.fold, r), s, **kw)


# ------------------------------------------------------------------ stage A: architectures
def stage_a(R):
    base = dict(optimizer="adam", lr=1e-3, batch_size=64, epochs=25, patience=6)
    for m in ("linear", "mlp", "lenet5", "lecture_cnn", "kaggle_cnn", "aksaranet", "resnet"):
        R.cv("A", m, {**base, "model": m})


# ------------------------------------------------------------------ stage B: learning strategy
B_BASE = dict(model="aksaranet", optimizer="adam", lr=1e-3, batch_size=64, epochs=12, patience=4)
B_ROTS = tuple(int(i) for i in os.environ.get("B_ROTS", "0,1").split(","))   # rotations used for tuning


def stage_b1(R):  # optimiser x learning rate
    for opt in ("sgd", "momentum", "rmsprop", "adam", "adamw"):
        for lr in (1e-1, 1e-2, 1e-3, 1e-4):
            wd = 1e-2 if opt == "adamw" else 0.0
            R.cv("B1", f"{opt}|lr={lr:g}", {**B_BASE, "optimizer": opt, "lr": lr, "weight_decay": wd}, B_ROTS)


def stage_b2(R):  # batch size, regularisation, augmentation, schedule
    for bs in (16, 64, 256):
        R.cv("B2", f"batch={bs}", {**B_BASE, "batch_size": bs}, B_ROTS)
    for d in (0.0, 0.3, 0.5):
        R.cv("B3", f"dropout={d}", {**B_BASE, "model_kw": {"dropout": d}}, B_ROTS)
    R.cv("B3", "no-batchnorm", {**B_BASE, "model_kw": {"bn": False}}, B_ROTS)
    for wd in (1e-4, 1e-3):
        R.cv("B3", f"L2={wd:g}", {**B_BASE, "weight_decay": wd}, B_ROTS)
    R.cv("B3", "label-smoothing=0.1", {**B_BASE, "label_smoothing": 0.1}, B_ROTS)
    for a in ("none", "light", "medium", "strong"):
        R.cv("B4", f"augment={a}", {**B_BASE, "augment": a, "epochs": 20, "patience": 5}, B_ROTS)
    for s in ("none", "cosine", "onecycle"):
        R.cv("B5", f"schedule={s}", {**B_BASE, "schedule": s, "lr": 3e-3 if s == "onecycle" else 1e-3}, B_ROTS)


def stage_b3(R):  # capacity and input resolution
    for depth in (1, 2, 3):
        R.cv("B6", f"depth={depth}", {**B_BASE, "model_kw": {"depth": depth}}, B_ROTS)
    for name, w in (("width=x0.5", (8, 16, 32, 64)), ("width=x1", (16, 32, 64, 128)), ("width=x2", (32, 64, 128, 256))):
        R.cv("B6", name, {**B_BASE, "model_kw": {"widths": w}}, B_ROTS)
    for size in (32, 64, 128):
        R.cv("B7", f"input={size}px", {**B_BASE, "size": size}, B_ROTS)


def stage_b8(R):  # the best single-factor settings combined, with a full-length schedule
    combo = dict(model="aksaranet", optimizer="adamw", lr=3e-3, weight_decay=1e-2, batch_size=64, epochs=25,
                 patience=0, schedule="onecycle")
    for a in ("none", "light", "medium", "strong"):
        R.cv("B8", f"combined|aug={a}", {**combo, "augment": a}, B_ROTS)
    R.cv("B8", "combined|aug=strong|ls=0.1", {**combo, "augment": "strong", "label_smoothing": 0.1}, B_ROTS)


# ------------------------------------------------------------------ stage C/D/K: final model
def final_cfg():
    return json.load(open(os.path.join(ROOT, "results", "final_config.json")))


def stage_c(R):
    R.cv("C", "final", final_cfg(), range(5), seeds=(0, 1, 2), save_preds=True)


def stage_d(R):
    """Writer-independent test: hold out every writer family in turn. Validation = one drawing
    fold of the remaining writers."""
    ds, cfg = R.ds, final_cfg()
    for fam in sorted(set(ds.family)):
        held = ds.family == fam
        va = (~held) & (R.fold == 0)
        idx = (np.where(~held & ~va)[0], np.where(va)[0], np.where(held)[0])
        R.run("D", "final", cfg, f"writer:{fam}", idx, 0, save_preds=True)
        R.run("D", "kaggle_cnn", dict(model="kaggle_cnn", optimizer="adam", lr=1e-3, batch_size=64, epochs=25, patience=6),
              f"writer:{fam}", idx, 0, save_preds=True)


def stage_k(R):
    """Original Kaggle train/val/test folders, for a like-for-like comparison with the reference notebook."""
    ds = R.ds
    idx = tuple(np.where(ds.split == s)[0] for s in ("train", "val", "test"))
    for seed in (0, 1, 2):
        R.run("K", "final", final_cfg(), "kaggle", idx, seed, save_preds=True)
    # reference recipe: plain SGD (Keras default lr 0.01), batch size 1, 5 epochs, no early stopping
    ref = dict(model="kaggle_cnn", optimizer="sgd", lr=0.01, batch_size=1, epochs=5, patience=0)
    for seed in (0, 1, 2):
        R.run("K", "reference-recipe", ref, "kaggle", idx, seed, save_preds=True)


def stage_kcv(R):
    """Reference recipe under the leakage-free 5-fold protocol."""
    ref = dict(model="kaggle_cnn", optimizer="sgd", lr=0.01, batch_size=1, epochs=5, patience=0)
    R.cv("A", "kaggle_cnn+reference-recipe", ref)


def stage_l(R):
    """Leakage demonstration: the same models evaluated with a naive image-level random split
    (augmented copies of one drawing end up on both sides) instead of the drawing-level split."""
    from sklearn.model_selection import StratifiedKFold
    naive = np.empty(R.ds.n, dtype=np.int64)
    for i, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=FOLD_SEED).split(naive, R.ds.y)):
        naive[te] = i
    base = dict(optimizer="adam", lr=1e-3, batch_size=64, epochs=25, patience=6)
    for m in ("mlp", "lecture_cnn", "aksaranet"):
        for r in range(3):
            R.run("L", m, {**base, "model": m}, f"naive{r}", rotation(naive, r))


STAGES = dict(l=stage_l, b8=stage_b8, a=stage_a, b1=stage_b1, b2=stage_b2, b3=stage_b3, c=stage_c, d=stage_d, k=stage_k, kcv=stage_kcv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()
    STAGES[a.stage](Runner(a.device))
