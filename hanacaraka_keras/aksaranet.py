"""AksaraNet in Keras, written in the style of the lecture examples.

The network, the augmentation and the learning-rate schedule mirror the PyTorch implementation in
hanacaraka/models.py, hanacaraka/data.py and hanacaraka/train.py.
"""
import math

import keras
from keras import layers, ops


def build_aksaranet(size=64, n_classes=20, widths=(16, 32, 64, 128), depth=2, dropout=0.3, augment=None):
    """4 stages of [Conv 3x3 - BatchNorm - ReLU] x depth, max-pooling between stages,
    global average pooling, dropout, softmax. `augment`: optional dict for RandomAffineStroke."""
    model = keras.Sequential(name="AksaraNet")
    model.add(keras.Input(shape=(size, size, 1)))
    if augment:
        model.add(RandomAffineStroke(**augment))                      # active only while training
    for i, w in enumerate(widths):
        for _ in range(depth):
            model.add(layers.Conv2D(w, (3, 3), padding="same", use_bias=False))
            model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))   # = PyTorch momentum 0.1
            model.add(layers.Activation("relu"))
        if i < len(widths) - 1:
            model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return model


class RandomAffineStroke(layers.Layer):
    """Online augmentation: random rotation, anisotropic scale, shear and shift (one affine warp),
    then random thickening or thinning of the pen stroke. Ink = 1, paper = 0."""

    def __init__(self, rot=20, scale=0.25, shift=0.12, shear=15, thick=0.5, seed=0, **kw):
        super().__init__(**kw)
        self.rot, self.scale, self.shift, self.shear, self.thick = rot, scale, shift, shear, thick
        self.seed = seed
        self.rng = keras.random.SeedGenerator(seed)

    def call(self, x, training=False):
        if not training:
            return x
        b, w = ops.shape(x)[0], float(x.shape[1])
        u = lambda: keras.random.uniform((b,), -1.0, 1.0, seed=self.rng)
        a, sh = u() * math.radians(self.rot), u() * math.radians(self.shear)
        sx, sy = 1 + u() * self.scale, 1 + u() * self.scale
        t00, t01, t02 = sx * ops.cos(a), -sy * ops.sin(a + sh), u() * self.shift * 2
        t10, t11, t12 = sx * ops.sin(a), sy * ops.cos(a + sh), u() * self.shift * 2
        # the matrix above is in normalised [-1, 1] coordinates; convert the offsets to pixels
        a2 = (t00 + t01) * (1 - w) / 2 + t02 * w / 2 + (w - 1) / 2
        b2 = (t10 + t11) * (1 - w) / 2 + t12 * w / 2 + (w - 1) / 2
        zero = ops.zeros_like(a)
        x = ops.image.affine_transform(x, ops.stack([t00, t01, a2, t10, t11, b2, zero, zero], axis=1),
                                       interpolation="bilinear", fill_mode="constant", fill_value=0.0)
        if self.thick > 0:
            p = keras.random.uniform((b, 1, 1, 1), 0.0, 1.0, seed=self.rng)
            thicker = ops.max_pool(x, 3, 1, "same")
            thinner = -ops.max_pool(-x, 3, 1, "same")
            x = ops.where(p < self.thick * 0.6, thicker, ops.where(p > 1 - self.thick * 0.4, thinner, x))
        return x

    def get_config(self):
        return {**super().get_config(), "rot": self.rot, "scale": self.scale, "shift": self.shift,
                "shear": self.shear, "thick": self.thick, "seed": self.seed}


class OneCycle(keras.optimizers.schedules.LearningRateSchedule):
    """One-cycle learning rate (Smith & Topin): cosine warm-up from peak/25 to peak over the first
    25 % of the steps, then cosine decay to peak/250,000."""

    def __init__(self, peak, total_steps, pct_start=0.25, div=25.0, final_div=1e4):
        self.peak, self.total_steps, self.pct_start, self.div, self.final_div = peak, total_steps, pct_start, div, final_div

    def __call__(self, step):
        step = ops.cast(step, "float32")
        up = self.pct_start * self.total_steps - 1
        down = self.total_steps - 1 - up
        cos = lambda lo, hi, t: hi + (lo - hi) / 2 * (1 + ops.cos(math.pi * t))        # lo at t=0, hi at t=1
        start, end = self.peak / self.div, self.peak / (self.div * self.final_div)
        return ops.where(step <= up, cos(start, self.peak, step / up),
                         cos(self.peak, end, ops.minimum((step - up) / down, 1.0)))

    def get_config(self):
        return dict(peak=self.peak, total_steps=self.total_steps, pct_start=self.pct_start, div=self.div, final_div=self.final_div)
