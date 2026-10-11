"""Conventional baseline under the same protocol: HOG features + RBF-SVM (C chosen on the validation fold)."""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hanacaraka.classic import hog  # noqa: E402
from hanacaraka.data import Aksara, rotation  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ds = Aksara(os.path.join(ROOT, "cache"))
fold = ds.drawing_folds(5, 0)
feats = hog(1.0 - ds.x(64) / 255.0)
print("HOG features:", feats.shape, flush=True)
for r in range(5):
    tr, va, te = rotation(fold, r)
    t0, best = time.time(), None
    for C in (1, 10, 100):
        clf = SVC(C=C, kernel="rbf", gamma="scale").fit(feats[tr], ds.y[tr])
        acc = float((clf.predict(feats[va]) == ds.y[va]).mean())
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    pred = best[2].predict(feats[te])
    rec = dict(stage="A", name="hog+svm", split=f"rot{r}", seed=0, cfg=dict(model="hog+svm", C=best[1]),
               n_train=len(tr), n_val=len(va), n_test=len(te), val_acc=best[0], val_loss=None,
               test_acc=float((pred == ds.y[te]).mean()), test_f1=float(f1_score(ds.y[te], pred, average="macro")),
               best_epoch=0, epochs_run=0, seconds=time.time() - t0, params=int(feats.shape[1]), hist={})
    with open(os.path.join(ROOT, "results", "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"rot{r}  C={best[1]}  val {best[0]:.4f}  test {rec['test_acc']:.4f}  {rec['seconds']:.0f}s", flush=True)
