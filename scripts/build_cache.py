"""Decode the 12,000 PNGs once and cache them as compact uint8 arrays.

Every image is converted to 8-bit grayscale and resized from 224x224 to 32/64/128 px.
Alongside the pixels we parse the *provenance* that is encoded in each file name:

    <label><id>.<writer>[.<variant>].png                      -> offline-augmented source file
    <label>_original_<source file>_<uuid>.png                 -> extra augmented copy of a source file

`drawing` = label + id + writer identifies one physical hand-drawn character. All files that
share a drawing are near-duplicates, so the drawing is the unit that must never be shared
between training and evaluation data.

Usage:  python scripts/build_cache.py <dataset_root> <cache_dir>
"""
import hashlib
import io
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
SIZES = (32, 64, 128)
WORKERS = 8
AUG_RE = re.compile(r"^[a-z]+_original_(.+?\.png)_[0-9a-f-]{36}\.png$")
SRC_RE = re.compile(r"^([a-z]+?)(\d+)\.([a-z]+\d*)(?:\.(r\d+|rm\d+|b))?\.png$")


def parse(fname):
    m = AUG_RE.match(fname)
    source, is_copy = (m.group(1), True) if m else (fname, False)
    label, idx, writer, variant = SRC_RE.match(source).groups()
    return source, is_copy, f"{label}{idx}.{writer}", writer, variant or "plain"


def is_placeholder(path):
    """True for an iCloud 'dataless' file whose content is not on this disk right now."""
    return bool(getattr(os.stat(path), "st_flags", 0) & 0x40000000)


def load_one(job):
    path, ci = job
    raw = open(path, "rb").read()
    img = Image.open(io.BytesIO(raw)).convert("L")
    px = [np.asarray(img.resize((s, s), Image.LANCZOS), dtype=np.uint8) for s in SIZES]
    return px, (ci, os.path.basename(path)) + parse(os.path.basename(path)) + (hashlib.md5(raw).hexdigest(),)


KEYS = ("label", "fname", "source", "is_copy", "drawing", "writer", "variant", "md5")


def jobs_for(root, split):
    classes = sorted(d for d in os.listdir(os.path.join(root, split)) if not d.startswith("."))
    return classes, [(os.path.join(root, split, c, f), ci) for ci, c in enumerate(classes)
                     for f in sorted(os.listdir(os.path.join(root, split, c))) if not f.startswith(".")]


def one_pass(root, split, cache):
    """Decode every file of `split` that is on disk right now. Returns the number still missing.

    Decoded files are checkpointed, because cloud placeholders may take several passes to arrive."""
    final, part = os.path.join(cache, f"{split}.npz"), os.path.join(cache, f"_partial_{split}.pkl")
    if os.path.exists(final):
        return 0
    classes, jobs = jobs_for(root, split)
    done = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
    todo = [j for j in jobs if j[0] not in done and not is_placeholder(j[0])]
    with ThreadPoolExecutor(WORKERS) as pool:
        for j, r in zip(todo, pool.map(load_one, todo)):
            done[j[0]] = r
    if todo:
        pickle.dump(done, open(part, "wb"))
    print(split, len(done), "/", len(jobs), flush=True)
    if len(done) < len(jobs):
        subprocess.Popen(["brctl", "download", os.path.join(root, split)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return len(jobs) - len(done)
    rows = [done[j[0]] for j in jobs]
    out = {f"x{s}": np.stack([px[k] for px, _ in rows]) for k, s in enumerate(SIZES)}
    out.update({k: np.array([m[i] for _, m in rows]) for i, k in enumerate(KEYS)})
    out["classes"] = np.array(classes)
    np.savez_compressed(final, **out)
    os.remove(part)
    print("wrote", final, flush=True)
    return 0


if __name__ == "__main__":
    root, cache = sys.argv[1], sys.argv[2]
    os.makedirs(cache, exist_ok=True)
    while sum(one_pass(root, s, cache) for s in SPLITS):
        time.sleep(30)
