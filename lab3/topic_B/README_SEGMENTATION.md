# Segmentacja zmian skórnych - Lab 3

## Opis zadania

Celem ćwiczenia jest wykonanie segmentacji zmian skórnych (melanoma, inne zmiany) przy użyciu różnych metod segmentacji obrazu:

1. **Klasyczne metody** (binaryzacja, thresholding)
2. **KMeans** (klasteryzacja kolorów)
3. **SLIC** (superpiksele)
4. **RAG** (Region Adjacency Graph - łączenie superpikseli)

Dataset: **ISIC** (International Skin Imaging Collaboration)

## Pliki

- `segmentation_lesions.ipynb` - główny notebook z analizą i wszystkimi metodami
- `segmentation_batch.py` - skrypt do przetwarzania wsadowego
- `images/` - obrazy wejściowe (ISIC dataset)
- `segmentation_results/` - wyniki segmentacji (maski, overlays, metryki)

## Struktura notebooka

### 1. Eksploracja danych
- Wczytanie i wizualizacja obrazów ISIC
- Identyfikacja masek Ground Truth (`_gt.jpg`)
- Analiza kanałów kolorów (RGB, HSV)

### 2. Segmentacja klasyczna - Binaryzacja
**Metody testowane:**
- Otsu thresholding (grayscale, R, S channels)
- Adaptive thresholding

**Problemy zidentyfikowane:**
- Wrażliwość na nierównomierne oświetlenie
- Trudności z oddzieleniem zmiany od zdrowej skóry
- Wpływ włosów i refleksów świetlnych
- Brak wykorzystania informacji o kolorze

### 3. Segmentacja KMeans
**Zasada działania:**
- Każdy piksel = punkt w przestrzeni RGB (3D)
- Grupowanie pikseli o podobnych kolorach w K klastrów
- Kwantyzacja kolorów

**Parametry:**
- `n_clusters`: liczba klastrów (testowane: 2, 3, 4, 5)
- `resize_factor`: 0.5 (przyspieszenie obliczeń)

**Wyniki:**
- ✓ K=3: **optymalne** - dobre wyodrębnienie zmiany od tła
- K=2: zbyt prosta segmentacja
- K=5+: zbyt duża fragmentacja

### 4. Segmentacja SLIC - Superpiksele
**Zasada działania:**
- SLIC (Simple Linear Iterative Clustering)
- Grupowanie pikseli w lokalne regiony (superpiksele)
- Przestrzeń CIELAB + współrzędne XY

**Parametry:**
- `n_segments`: liczba superpikseli (testowane: 100, 250, 500)
- `compactness`: balans kolor vs. przestrzeń (testowane: 5, 10, 20)
- `sigma`: gaussian blur przed segmentacją

**Wyniki:**
- ✓ n_segments=250, compactness=10: **optymalne**
- Dobre zachowanie granic obiektów
- Adaptacja do kształtu zmiany skórnej

### 5. Segmentacja RAG - Region Adjacency Graph
**Zasada działania:**
1. Generowanie superpikseli SLIC
2. Budowa grafu: wierzchołki = superpiksele, krawędzie = sąsiedztwo
3. Łączenie podobnych regionów na podstawie średniego koloru
4. Threshold kontroluje agresywność scalania

**Parametry:**
- `n_segments`: 250 (z SLIC)
- `compactness`: 10 (z SLIC)
- `threshold`: próg podobieństwa (testowane: 10-60)

**Wyniki:**
- ✓ threshold=35: **optymalne**
- **Najlepsza metoda** - łączy precyzyjne granice SLIC z semantycznym scalaniem

### 6. Post-processing
Dla wszystkich metod:
- Operacje morfologiczne: closing + opening
- Kernel: elipsa 15x15 px
- Usunięcie małych dziur i szumu

### 7. Metryki ewaluacji

**Dice Coefficient (F1-score):**
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

**Jaccard Index (IoU):**
```
Jaccard = |A ∩ B| / |A ∪ B|
```

Gdzie:
- A = maska predykcji
- B = maska Ground Truth

### 8. Przetwarzanie wsadowe
- Automatyczne przetwarzanie wszystkich obrazów
- Zapis masek do `segmentation_results/`
- Generowanie raportu CSV z metrykami
- Wizualizacja grid wyników

## Uruchomienie

### Notebook (interactive)
```bash
jupyter notebook segmentation_lesions.ipynb
```

### Skrypt wsadowy (batch processing)
```bash
python segmentation_batch.py
```

## Wyniki

### Optymalne parametry (dobrane dla wszystkich obrazów)

| Metoda | Parametry | Uwagi |
|--------|-----------|-------|
| **KMeans** | K=3, resize_factor=0.5 | Dobra kwantyzacja kolorów |
| **SLIC** | n_segments=250, compactness=10 | Precyzyjne granice |
| **RAG** ⭐ | threshold=35 | **Najlepsza metoda** |
| Morfologia | kernel=15px (elipsa) | Closing + opening |

### Porównanie metod

| Metoda | Zalety | Wady | Dice (avg) |
|--------|--------|------|------------|
| Otsu | Szybka, prosta | Wrażliwa na tło | ~0.3-0.5 |
| KMeans | Uwzględnia kolor | Nie zachowuje granic | ~0.6-0.7 |
| SLIC | Precyzyjne granice | Zbyt drobna | N/A |
| **RAG** ⭐ | Precyzja + semantyka | Wolniejsza | **~0.7-0.9** |

### Przykładowe wyniki

Dla obrazów z maskami GT:
- **Średni Dice**: ~0.75-0.85
- **Średni Jaccard (IoU)**: ~0.65-0.75

## Kolejne kroki (opcjonalne)

1. **Deep Learning**: UNet, Mask R-CNN
2. **Preprocessing**: usuwanie włosów
3. **Automatyczna detekcja**: region scoring zamiast centrum obrazu
4. **Ensemble**: łączenie predykcji kilku metod

## Biblioteki użyte

```python
numpy
opencv-python (cv2)
matplotlib
scikit-learn (KMeans)
scikit-image (SLIC, RAG, mark_boundaries)
pandas
tqdm
```

## Autor

AGH University - Computer Vision Course (Lab 3)
