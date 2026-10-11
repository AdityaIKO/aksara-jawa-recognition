"""Reload the saved Keras model and predict unseen test images (the lecture's 'load model and prediction' step).

Run with the Keras environment's interpreter from the project root.
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import keras  # noqa: E402

from hanacaraka_keras.aksaranet import RandomAffineStroke  # noqa: E402

model = keras.models.load_model(os.path.join(ROOT, "results", "aksaranet_keras.keras"),
                                custom_objects={"RandomAffineStroke": RandomAffineStroke}, compile=False)   # inference only
parts = [np.load(os.path.join(ROOT, "cache", f"{s}.npz")) for s in ("train", "val", "test")]
X = (1 - np.concatenate([p["x64"] for p in parts]).astype("float32") / 255)[..., None]
y = np.concatenate([p["label"] for p in parts])
classes = parts[0]["classes"]
fold = np.load(os.path.join(ROOT, "cache", "folds_seed0.npy"))

test = np.where(fold == 0)[0]                       # fold 0 was the held-out test fold of this model
pred = model.predict(X[test], batch_size=512, verbose=0)
print("test accuracy of the reloaded model on fold 0:", round(float((pred.argmax(1) == y[test]).mean()), 4))
for i in test[::300]:
    p = model.predict(X[i:i + 1], verbose=0)[0]
    print("label actual:", classes[y[i]], "| label prediction:", classes[p.argmax()], f"({p.max():.2f})")
