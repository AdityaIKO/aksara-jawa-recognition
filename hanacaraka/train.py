"""Training loop with the learning-strategy knobs studied in the project."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import AUG_PRESETS, augment
from .models import build, count_params

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Reference recipe. Every experiment is described by the keys it overrides.
DEFAULT = dict(
    model="aksaranet", model_kw={}, size=64,
    optimizer="adam", lr=1e-3, momentum=0.9, weight_decay=0.0,
    batch_size=64, epochs=30, schedule="none", label_smoothing=0.0,
    augment="none", patience=8, seed=0,
)


def make_optimizer(name, params, lr, momentum, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


@torch.no_grad()
def predict_proba(model, x, batch_size=512, tta=0, aug_kw=None):
    """Class probabilities. With tta > 0 the prediction is averaged over extra augmented views."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        p = F.softmax(model(xb), 1)
        for _ in range(tta):
            p = p + F.softmax(model(augment(xb, **(aug_kw or AUG_PRESETS["light"]))), 1)
        out.append((p / (tta + 1)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def evaluate(model, x, y, batch_size=512):
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        loss += F.cross_entropy(logits, y[i:i + batch_size], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    return loss / len(x), correct / len(x)


def fit(x_tr, y_tr, x_va, y_va, cfg=None, device=DEVICE, verbose=False):
    """Train one model. x_*: float tensors (N,1,S,S) already on `device`; y_*: long tensors.

    The weights of the epoch with the best validation accuracy (ties -> lower loss) are restored,
    and training stops early after `patience` epochs without improvement.
    Returns (model, history dict).
    """
    cfg = {**DEFAULT, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build(cfg["model"], size=cfg["size"], **cfg["model_kw"]).to(device)
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["momentum"], cfg["weight_decay"])
    n, bs = len(x_tr), cfg["batch_size"]
    steps = cfg["epochs"] * ((n + bs - 1) // bs)
    if cfg["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    elif cfg["schedule"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.25)
    elif cfg["schedule"] == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.1)
    else:
        sched = None
    aug_kw = AUG_PRESETS[cfg["augment"]]
    gen = torch.Generator().manual_seed(cfg["seed"])

    hist = dict(loss=[], acc=[], val_loss=[], val_acc=[])
    best, best_state, t0 = (-1.0, float("inf"), 0), None, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        run_loss = torch.zeros((), device=device)      # accumulated on the device: no sync per step
        run_correct = torch.zeros((), device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = x_tr[idx], y_tr[idx]
            if aug_kw is not None:
                xb = augment(xb, **aug_kw)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            run_loss += loss.detach() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum()
        run_loss, run_correct = run_loss.item(), run_correct.item()
        val_loss, val_acc = evaluate(model, x_va, y_va)
        for k, v in zip(hist, (run_loss / n, run_correct / n, val_loss, val_acc)):
            hist[k].append(float(v))
        if not np.isfinite(run_loss):
            break
        if (val_acc, -val_loss) > (best[0], -best[1]):
            best = (val_acc, val_loss, epoch)
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"epoch {epoch:3d}  loss {hist['loss'][-1]:.4f} acc {hist['acc'][-1]:.4f}  "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}", flush=True)
        if cfg["patience"] and epoch - best[2] >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist.update(best_val_acc=best[0], best_val_loss=best[1], best_epoch=best[2], epochs_run=len(hist["loss"]),
                seconds=time.time() - t0, params=count_params(model))
    return model, hist
