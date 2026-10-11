"""Train the Keras port of AksaraNet on exactly the same folds as the PyTorch experiments.

Run with the Keras environment's interpreter:
    /opt/anaconda3/envs/keras-lecture/bin/python scripts/run_keras.py [reference|final] ...
Results are appended to results/keras_runs.jsonl (resumable).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import keras  # noqa: E402

from hanacaraka_keras.aksaranet import OneCycle, build_aksaranet  # noqa: E402

LOG = os.path.join(ROOT, "results", "keras_runs.jsonl")
STRONG = dict(rot=20, scale=0.25, shift=0.12, shear=15, thick=0.5)

parts = [np.load(os.path.join(ROOT, "cache", f"{s}.npz")) for s in ("train", "val", "test")]
X = (1.0 - np.concatenate([p["x64"] for p in parts]).astype("float32") / 255.0)[..., None]   # ink = 1
Y = np.concatenate([p["label"] for p in parts]).astype("int64")
FOLD = np.load(os.path.join(ROOT, "cache", "folds_seed0.npy"))      # written by the PyTorch pipeline
Y1H = keras.utils.to_categorical(Y, 20)


def macro_f1(y, p, k=20):
    f = []
    for c in range(k):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def run(recipe, r, seed=0):
    done = {(d["name"], d["split"], d["seed"]) for d in map(json.loads, open(LOG))} if os.path.exists(LOG) else set()
    if (recipe, f"rot{r}", seed) in done:
        return
    keras.utils.set_random_seed(seed)
    te, va = FOLD == r, FOLD == (r + 1) % 5
    tr = ~te & ~va
    steps = int(np.ceil(tr.sum() / 64))
    if recipe == "reference":      # Experiment 1 recipe: Adam 1e-3, no augmentation, early stopping (patience 6)
        model, epochs, patience = build_aksaranet(), 25, 6
        model.compile(optimizer=keras.optimizers.Adam(1e-3, epsilon=1e-8), loss="categorical_crossentropy", metrics=["accuracy"])
    else:                          # final recipe: AdamW + one-cycle + strong augmentation + label smoothing
        model, epochs, patience = build_aksaranet(augment={**STRONG, "seed": seed}), 25, 25
        opt = keras.optimizers.AdamW(OneCycle(3e-3, epochs * steps), weight_decay=0.01, epsilon=1e-8)
        model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True)
    t0 = time.time()
    h = model.fit(X[tr], Y1H[tr], validation_data=(X[va], Y1H[va]), epochs=epochs, batch_size=64, shuffle=True,
                  callbacks=[stop], verbose=0).history
    pred = model.predict(X[te], batch_size=512, verbose=0).argmax(1)
    rec = dict(name=recipe, split=f"rot{r}", seed=seed, framework=f"keras {keras.__version__} ({keras.backend.backend()})",
               val_acc=float(max(h["val_accuracy"])), test_acc=float((pred == Y[te]).mean()), test_f1=macro_f1(Y[te], pred),
               best_epoch=int(np.argmax(h["val_accuracy"]) + 1), epochs_run=len(h["loss"]), seconds=time.time() - t0,
               params=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
               hist={k: [float(v) for v in h[k]] for k in ("loss", "accuracy", "val_loss", "val_accuracy")})
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[keras] {recipe:<10} rot{r} seed {seed}  val {rec['val_acc']:.4f}  test {rec['test_acc']:.4f}  "
          f"ep {rec['best_epoch']}/{rec['epochs_run']}  {rec['seconds']:.0f}s", flush=True)
    if recipe == "final" and r == 0 and seed == 0:
        model.save(os.path.join(ROOT, "results", "aksaranet_keras.keras"))


if __name__ == "__main__":
    for recipe in (sys.argv[1:] or ["reference", "final"]):
        for r in range(5):
            run(recipe, r)
