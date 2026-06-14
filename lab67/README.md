## Layout

```
src/cvproject/        installable package with the core functions
  config.py           dataset registry + the single ACTIVE switch
  data.py             loading, label encoding, train/val/test split
  preprocessing.py    artifact removal, segmentation, brightness/colour enhancement
  features.py         GLCM texture, colour stats, shape descriptors
  classification.py   SVM / decision-tree pipelines + GridSearchCV tuning
  metrics.py          accuracy / precision / recall / f1, confusion matrix
  viz.py              display helpers (implemented)
notebooks/
  project.ipynb       single end-to-end pipeline importing cvproject.*
datasets/             three datasets (potato simple/hard, apples_tomatoes)
```

The algorithmic functions are **boilerplate**: each has a signature, a detailed
docstring describing exactly what to implement, and a `# TODO` body. The display
helpers in `viz.py` are fully implemented.

## Setup

```bash
cd lab67
pip install -e ".[notebook]"      # editable install + jupyter
python -c "import cvproject; print(cvproject.__version__)"
```

## Choosing a dataset

Everything is driven by one switch. In the notebook (or any script):

```python
from cvproject import get_config
cfg = get_config("potato_simple")   # or "potato_hard" / "apples_tomatoes"
```

Available datasets are defined in `cvproject/config.py::DATASETS`.
