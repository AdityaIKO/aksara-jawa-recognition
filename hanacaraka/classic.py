"""Conventional pipeline used as a non-neural reference: hand-crafted HOG features + SVM."""
import numpy as np


def hog(images, cell=8, bins=9, eps=1e-6):
    """Histogram of Oriented Gradients (Dalal & Triggs, 2005) for a stack of square images.

    images: (N, S, S) float array. Returns (N, (S/cell - 1)^2 * 4 * bins) features:
    unsigned gradient orientation histograms per cell, L2-normalised over 2x2 cell blocks.
    """
    x = images.astype(np.float32)
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]
    gy[:, 1:-1, :] = x[:, 2:, :] - x[:, :-2, :]
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi) / np.pi * bins          # [0, bins)
    lo = np.floor(ang).astype(np.int64) % bins
    w_hi = ang - np.floor(ang)
    n, s, _ = x.shape
    c = s // cell
    hist = np.zeros((n, c, c, bins), dtype=np.float32)
    for b in range(bins):                                           # linear interpolation between bins
        m = mag * ((lo == b) * (1 - w_hi) + (((lo + 1) % bins) == b) * w_hi)
        hist[..., b] = m.reshape(n, c, cell, c, cell).sum((2, 4))
    blocks = np.concatenate([hist[:, :-1, :-1], hist[:, :-1, 1:], hist[:, 1:, :-1], hist[:, 1:, 1:]], -1)
    blocks /= np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps)
    return blocks.reshape(n, -1)
