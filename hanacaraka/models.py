"""Model zoo: from a linear classifier up to a small residual network.

Every model takes a (B, 1, S, S) tensor and returns 20 logits. Softmax is folded into the
cross-entropy loss.
"""
import torch
import torch.nn as nn


def _flat_dim(features, size):
    with torch.no_grad():
        return features(torch.zeros(1, 1, size, size)).flatten(1).size(1)


class Linear(nn.Sequential):
    """Multinomial logistic regression (single-layer perceptron with softmax)."""

    def __init__(self, size=64, n_classes=20):
        super().__init__(nn.Flatten(), nn.Linear(size * size, n_classes))


class MLP(nn.Sequential):
    """Fully connected network on raw pixels: S*S -> 512 -> 256 -> 20."""

    def __init__(self, size=64, n_classes=20, hidden=(512, 256), dropout=0.3):
        layers, d = [nn.Flatten()], size * size
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        super().__init__(*layers, nn.Linear(d, n_classes))


class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al., 1998): 2 x (5x5 conv + tanh + average pooling), then 120-84-20."""

    def __init__(self, size=64, n_classes=20):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                                      nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(_flat_dim(self.features, size), 120), nn.Tanh(),
                                        nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class LectureCNN(nn.Module):
    """The CNN from the lecture slides: conv16 - pool - conv32 - pool - dense64 - softmax."""

    def __init__(self, size=64, n_classes=20):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(_flat_dim(self.features, size), 64),
                                        nn.ReLU(inplace=True), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class KaggleCNN(nn.Module):
    """Re-implementation of the reference Kaggle notebook: 4 x (valid 3x3 conv + ReLU + max-pool)
    with 64-128-256-512 filters, Flatten, Dense(256), Dropout(0.2), Dense(20). No normalisation."""

    def __init__(self, size=64, n_classes=20):
        super().__init__()
        layers, c = [], 1
        for w in (64, 128, 256, 512):
            layers += [nn.Conv2d(c, w, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(_flat_dim(self.features, size), 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class AksaraNet(nn.Module):
    """VGG-style CNN proposed in this project.

    Four stages of [3x3 conv - BatchNorm - ReLU] x depth followed by 2x2 max-pooling, then global
    average pooling instead of Flatten + Dense: the classifier head has 2.6k weights instead of the
    530k in the reference model, which removes the main source of over-fitting.
    """

    def __init__(self, size=64, n_classes=20, widths=(16, 32, 64, 128), depth=2, bn=True, dropout=0.3):
        super().__init__()
        layers, c = [], 1
        for i, w in enumerate(widths):
            for _ in range(depth):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=not bn)]
                layers += [nn.BatchNorm2d(w)] if bn else []
                layers += [nn.ReLU(inplace=True)]
                c = w
            if i < len(widths) - 1:
                layers += [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(c, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.body(x) + (x if self.skip is None else self.skip(x)))


class ResNetSmall(nn.Module):
    """ResNet (He et al., 2015) scaled down for 64x64 glyphs: one BasicBlock per stage (ResNet-10)."""

    def __init__(self, size=64, n_classes=20, widths=(16, 32, 64, 128), blocks=1, dropout=0.3):
        super().__init__()
        layers = [nn.Conv2d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)]
        c = widths[0]
        for i, w in enumerate(widths):
            for b in range(blocks):
                layers.append(BasicBlock(c, w, 2 if (b == 0 and i > 0) else 1))
                c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(c, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"linear": Linear, "mlp": MLP, "lenet5": LeNet5, "lecture_cnn": LectureCNN,
          "kaggle_cnn": KaggleCNN, "aksaranet": AksaraNet, "resnet": ResNetSmall}


def build(name, size=64, **kw):
    return MODELS[name](size=size, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
