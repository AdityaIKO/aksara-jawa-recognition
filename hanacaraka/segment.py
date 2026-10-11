"""Word-level recognition: split a photographed word into glyphs and classify each one.

The reference notebook cut the word at every connected component, which breaks the aksara that
are written with two separate strokes (ba, nya, nga, tha ...). Here the components are treated
as an *over-segmentation* and a dynamic programme decides which neighbours belong together,
using both geometry (the gap between them) and the classifier's own confidence.
"""
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi


def otsu(gray):
    """Otsu's threshold for a uint8 image."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    w = np.cumsum(hist)
    m = np.cumsum(hist * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m[-1] * w - m * w[-1]) ** 2 / (w * (w[-1] - w))
    return int(np.nanargmax(between))


def ink_mask(image):
    """Boolean mask of pen strokes from a PIL image of dark ink on light paper."""
    gray = np.asarray(image.convert("L"))
    gray = ndi.median_filter(gray, 3)
    return gray < otsu(gray)


def primitives(mask, min_rel_height=0.2):
    """Connected components as (x0, y0, x1, y1) boxes sorted left to right, noise removed, and
    components that overlap horizontally fused (they are stacked parts of the same glyph)."""
    lab, _ = ndi.label(mask, structure=np.ones((3, 3)))
    boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndi.find_objects(lab)]
    if not boxes:
        return []
    hmax = max(b[3] - b[1] for b in boxes)
    boxes = sorted(b for b in boxes if (b[3] - b[1]) >= min_rel_height * hmax)
    merged = [boxes[0]]
    for b in boxes[1:]:
        a = merged[-1]
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        if overlap > 0.3 * min(a[2] - a[0], b[2] - b[0]):
            merged[-1] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        else:
            merged.append(b)
    return merged


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def glyph(mask, box, size=64, fill=0.72):
    """Crop a box, centre it on a square canvas (glyph spans `fill` of the side, as in the
    training images) and resample to size x size. Returns float array with ink = 1."""
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1].astype(np.float32)
    side = int(round(max(crop.shape) / fill))
    canvas = np.zeros((side, side), np.float32)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    img = Image.fromarray((canvas * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255.0


@torch.no_grad()
def _proba(model, glyphs, device):
    model.eval()
    x = torch.from_numpy(np.stack(glyphs)).unsqueeze(1).to(device)
    return torch.softmax(model(x), 1).cpu().numpy()


def read_word(model, image, classes, device="cpu", size=64, max_parts=2, gap_penalty=4.0, mode="dp"):
    """Transliterate a word image.

    mode="cc": one glyph per primitive (the reference notebook's behaviour).
    mode="dp": choose the grouping of consecutive primitives (up to `max_parts` each) that
               maximises  sum_g [ log p_max(g) - gap_penalty * (inner gap / line height) ].
    Returns (list of labels, list of boxes, list of confidences).
    """
    mask = ink_mask(image)
    prim = primitives(mask)
    if not prim:
        return [], [], []
    height = float(np.median([b[3] - b[1] for b in prim]))
    n = len(prim)
    cands = {}
    for i in range(n):
        for k in range(1, (max_parts if mode == "dp" else 1) + 1):
            if i + k > n:
                break
            gap = sum(max(0, prim[j + 1][0] - prim[j][2]) for j in range(i, i + k - 1))
            cands[(i, k)] = (union(prim[i:i + k]), gap / height)
    keys = list(cands)
    proba = _proba(model, [glyph(mask, cands[c][0], size) for c in keys], device)
    score = {c: float(np.log(p.max() + 1e-9)) - gap_penalty * cands[c][1] for c, p in zip(keys, proba)}
    best = [(0.0, None)] + [(-np.inf, None)] * n          # best[j] = (score, k) for the first j primitives
    for j in range(1, n + 1):
        for k in range(1, max_parts + 1):
            if (j - k, k) in score and best[j - k][0] + score[(j - k, k)] > best[j][0]:
                best[j] = (best[j - k][0] + score[(j - k, k)], k)
    groups, j = [], n
    while j > 0:
        k = best[j][1]
        groups.append((j - k, k))
        j -= k
    groups.reverse()
    p = {c: pr for c, pr in zip(keys, proba)}
    return ([classes[int(p[g].argmax())] for g in groups], [cands[g][0] for g in groups],
            [float(p[g].max()) for g in groups])
