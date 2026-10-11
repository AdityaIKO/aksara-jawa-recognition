"""Dataset loading, leakage-free splitting and on-device augmentation."""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("train", "val", "test")
META = ("label", "fname", "source", "is_copy", "drawing", "writer", "variant", "md5")

# Writer tags found in the file names, merged into families so that e.g. "mine", "mine2" and
# "mine3" (very likely the same hand) can never sit on both sides of a writer-independent split.
WRITER_FAMILY = {"mine": "mine", "mine2": "mine", "mine3": "mine", "pred": "pred", "pred3": "pred",
                 "nando": "nando", "meki": "meki", "theo": "theo"}


class Aksara:
    """All 12,000 images pooled, with the provenance parsed from the file names."""

    def __init__(self, cache_dir):
        self._dir = cache_dir
        parts = [np.load(os.path.join(cache_dir, f"{s}.npz")) for s in SPLITS]
        self.classes = [str(c) for c in parts[0]["classes"]]
        self.split = np.concatenate([[s] * len(p["label"]) for s, p in zip(SPLITS, parts)])
        for k in META:
            setattr(self, k, np.concatenate([p[k] for p in parts]))
        self.y = self.label.astype(np.int64)
        self._x = {}                                   # pixel arrays are loaded on first use, per resolution
        self.family = np.array([WRITER_FAMILY[w] for w in self.writer])
        self.n = len(self.y)

    def x(self, size=64):
        """uint8 images (N, size, size); 255 = paper, 0 = ink. Cached resolutions: 32, 64, 128."""
        if size not in self._x:
            self._x[size] = np.concatenate([np.load(os.path.join(self._dir, f"{s}.npz"))[f"x{size}"] for s in SPLITS])
        return self._x[size]

    def tensor(self, size=64, device="cpu"):
        """float images (N, 1, size, size) with ink = 1 and paper = 0, so zero padding is 'paper'."""
        return (1.0 - torch.from_numpy(self.x(size)).float().div_(255.0)).unsqueeze(1).to(device)

    def drawing_folds(self, k=5, seed=0):
        """Fold index per image. Stratified by class and grouped by drawing: all ~29 augmented
        versions of one hand-drawn character always land in the same fold."""
        fold = np.empty(self.n, dtype=np.int64)
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (_, te) in enumerate(sgkf.split(np.zeros(self.n), self.y, self.drawing)):
            fold[te] = i
        return fold


def rotation(fold, r, k=5):
    """Train / validation / test indices for rotation r of a k-fold scheme (3 / 1 / 1 folds)."""
    test, val = r % k, (r + 1) % k
    return (np.where((fold != test) & (fold != val))[0], np.where(fold == val)[0], np.where(fold == test)[0])


# ----------------------------------------------------------------------------- augmentation
AUG_PRESETS = {
    "none": None,
    "light": dict(rot=7, scale=0.08, shift=0.05, shear=5, thick=0.0),
    "medium": dict(rot=12, scale=0.15, shift=0.08, shear=10, thick=0.3),
    "strong": dict(rot=20, scale=0.25, shift=0.12, shear=15, thick=0.5),
}


def augment(x, rot=12, scale=0.15, shift=0.08, shear=10, thick=0.3):
    """Random affine warp + stroke-thickness jitter, applied to a whole batch on the device.

    x: (B, 1, H, W) with ink = 1. A fresh transform is drawn for every image at every step, so
    the network never sees exactly the same pixels twice (unlike the offline copies on disk).
    """
    b, dev = x.size(0), x.device
    u = lambda: torch.rand(b, device=dev) * 2 - 1
    a, sh = u() * math.radians(rot), u() * math.radians(shear)
    sx, sy = 1 + u() * scale, 1 + u() * scale
    theta = torch.stack([
        torch.stack([sx * torch.cos(a), -sy * torch.sin(a + sh), u() * shift * 2], 1),
        torch.stack([sx * torch.sin(a), sy * torch.cos(a + sh), u() * shift * 2], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                      mode="bilinear", padding_mode="zeros", align_corners=False)
    if thick > 0:
        p = torch.rand(b, 1, 1, 1, device=dev)
        thicker = F.max_pool2d(x, 3, 1, 1)
        thinner = -F.max_pool2d(-x, 3, 1, 1)
        x = torch.where(p < thick * 0.6, thicker, torch.where(p > 1 - thick * 0.4, thinner, x))
    return x
