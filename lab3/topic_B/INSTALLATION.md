# Instalacja zależności - Lab 3

## Wymagane biblioteki

```bash
numpy>=1.24.0           # ✓ już zainstalowane
opencv-python>=4.8.0    # TODO: do zainstalowania
matplotlib>=3.7.0       # ✓ już zainstalowane (matplotlib-inline)
scikit-learn>=1.3.0     # ✓ już zainstalowane (1.8.0)
scikit-image>=0.21.0    # TODO: do zainstalowania
pandas>=2.0.0           # ✓ już zainstalowane (2.3.3)
tqdm>=4.65.0            # ✓ już zainstalowane (4.67.1)
jupyter>=1.0.0          # dla notebooka
```

## Instalacja brakujących bibliotek

### Opcja 1: PyPI (publiczne repozytorium)
```bash
pip install --index-url https://pypi.org/simple/ opencv-python scikit-image
```

### Opcja 2: Bezpośrednio z requirements.txt
```bash
cd lab3/topic_B
pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

### Opcja 3: conda (jeśli używasz Anaconda/Miniconda)
```bash
conda install -c conda-forge opencv scikit-image
```

## Weryfikacja instalacji

Po zainstalowaniu, uruchom skrypt testowy:

```bash
cd lab3/topic_B
python test_segmentation.py
```

Jeśli wszystko działa, zobaczysz:
```
Testing imports...
✓ numpy
✓ cv2 (opencv)
✓ sklearn.cluster.KMeans
✓ skimage.segmentation.slic
✓ skimage.future.graph

Found X original images
Testing with: ISIC2017_XXXXXXX.jpg
Image loaded: shape=(H, W, 3)

Testing KMeans...
✓ KMeans completed: 3 clusters

Testing SLIC...
✓ SLIC completed: XXX superpixels

Testing RAG...
✓ RAG completed: XX regions after merging

==================================================
✓ All tests passed!
==================================================
```

## Problem z pip (AWS CodeArtifact)

Jeśli pip jest skonfigurowany do używania prywatnego repozytorium AWS CodeArtifact
i nie masz dostępu, użyj flagi `--index-url` aby wskazać publiczne PyPI.

### Temporary fix
```bash
export PIP_INDEX_URL=https://pypi.org/simple/
pip install opencv-python scikit-image
```

### Permanent fix (opcjonalnie)
Edytuj `~/.pip/pip.conf` lub usuń konfigurację CodeArtifact.

## Środowisko wirtualne

Projekt używa `.venv` w katalogu głównym:
```bash
# Aktywacja (jeśli nie jest aktywne)
source .venv/bin/activate  # macOS/Linux
# lub
.venv\Scripts\activate     # Windows
```

## Jupyter Notebook

Aby uruchomić notebook:
```bash
cd lab3/topic_B
jupyter notebook segmentation_lesions.ipynb
```

Lub użyj VS Code z rozszerzeniem Jupyter.
