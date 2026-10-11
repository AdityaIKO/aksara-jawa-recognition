# Handwritten Javanese Script (Aksara Jawa) Recognition

Group mini project, *Introduction to Deep Learning*, Universitas Gadjah Mada.
Group: Syakirah, Balma, Nabila, Aditya Wahyu.

We classify the 20 base characters of the Javanese script (*aksara nglegena*) from handwriting and use the
classifier to transliterate handwritten words. Dataset: *Javanese Script (Aksara Jawa) Augmented* (Kaggle),
12,000 images, 20 balanced classes.

## Main findings

- The 12,000 images descend from only **420 hand-drawn characters** (about 29 augmented copies each). We therefore
  evaluate with **5-fold cross-validation grouped by drawing**. A naive random split over-estimates an MLP by 43 points.
- Our CNN (**AksaraNet**: VGG-style, BatchNorm, global average pooling, 296k parameters) reaches **99.4 %** test
  accuracy (99.6 % as a 3-seed ensemble), against 86.6 to 88.7 % for the reference Kaggle CNN (2.08 M parameters) under the same protocol.
- On writers never seen in training it stays at 98.7 to 100 %, where the reference CNN drops to 43 to 95 %.
- A segmentation step that lets the classifier decide which strokes belong together reads 7 of 8 photographed words
  (reference approach: 5 of 8).

- A **Keras 3 port** of the model (section 11 of the notebook) reaches 99.0 % on the same folds against 99.5 % for PyTorch,
  a difference inside the seed-to-seed noise.

All numbers, tables and figures are in `Hanacaraka_MiniProject.ipynb`.

## Layout

```
Hanacaraka_MiniProject.ipynb   the report notebook (executed)
hanacaraka/                    data.py  models.py  train.py  classic.py  segment.py  report.py   (PyTorch)
hanacaraka_keras/              aksaranet.py: the same network, augmentation and schedule in Keras 3
scripts/run_keras.py           trains the Keras port on the same folds; scripts/keras_predict.py reloads it
scripts/build_cache.py         decode the PNGs once into cache/*.npz and parse provenance from file names
scripts/run_experiments.py     the experiment grid (stages a, b1, b2, b3, b8, l, c, k, d, kcv); resumable
scripts/run_classic.py         HOG + SVM baseline
scripts/make_notebook.py       regenerates the notebook source
results/runs.jsonl             one line per training run (configuration, scores, learning curves)
results/final_config.json      the selected recipe
figures/                       figures used in the slides
predict_test/                  the eight word photographs shipped with the dataset
```

## Reproduce

Requirements: Python 3.11, PyTorch 2.x, torchvision, scikit-learn, scipy, pandas, matplotlib, Pillow, nbformat.

```bash
# 1. put the Kaggle folders train/ val/ test/ in <dataset_root>, then build the cache
python scripts/build_cache.py <dataset_root> cache
# 2. run the experiments (each stage appends to results/runs.jsonl and can be resumed)
for s in a b1 b2 b3 b8 l c k d kcv; do python scripts/run_experiments.py $s; done
python scripts/run_classic.py
# 3. rebuild and execute the notebook
python scripts/make_notebook.py
jupyter nbconvert --to notebook --execute --inplace Hanacaraka_MiniProject.ipynb
```

Section 11 needs a second environment with TensorFlow/Keras (kept separate so it cannot disturb PyTorch):

```bash
conda create -n keras-lecture python=3.11 -y
/opt/anaconda3/envs/keras-lecture/bin/pip install tensorflow matplotlib
/opt/anaconda3/envs/keras-lecture/bin/python scripts/run_keras.py reference final
```

The complete grid is about 230 training runs and takes roughly three hours on an Apple M2 (MPS).
