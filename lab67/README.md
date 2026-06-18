# Apples vs Tomatoes — classical image classification

Classical CV pipeline (no neural nets): load → preprocess → extract features →
train/tune → evaluate on a held-out test set.

## Layout

```
src/cvproject/
  config.py           dataset config
  data.py             loading, label encoding, train/val/test split
  preprocessing.py    artifact removal, segmentation, brightness/colour enhancement
  features.py         GLCM texture, colour stats, shape descriptors
  classification.py   SVM / decision tree / RF / kNN / logreg + GridSearchCV
  metrics.py          accuracy / precision / recall / f1, confusion matrix
  viz.py              plotting helpers
notebooks/project.ipynb   end-to-end pipeline
datasets/apples_tomatoes/ train/ and test/, each with apples/ and tomatoes/
```

## Setup

```bash
cd lab67
pip install -e .
```
