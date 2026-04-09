#!/usr/bin/env python3
"""
Segmentacja zmian skórnych - przetwarzanie wsadowe
Lab 3 - Computer Vision AGH

Metody:
- KMeans (klasteryzacja kolorów)
- SLIC (superpiksele)
- RAG (Region Adjacency Graph)

Autor: AGH Computer Vision Course
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.future import graph
import pandas as pd
from tqdm import tqdm


# Konfiguracja
INPUT_DIR = Path('./images')
OUTPUT_DIR = Path('./segmentation_results')
OUTPUT_DIR.mkdir(exist_ok=True)

OPTIMAL_PARAMS = {
    'kmeans': {
        'n_clusters': 3,
        'resize_factor': 0.5
    },
    'rag': {
        'n_segments': 250,
        'compactness': 10,
        'threshold': 35
    },
    'morph_kernel': 15
}


def segment_kmeans(img_rgb, n_clusters=3, resize_factor=0.5):
    """Segmentacja KMeans - kwantyzacja kolorów."""
    h, w = img_rgb.shape[:2]
    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
    img_small = cv2.resize(img_rgb, (new_w, new_h))
    
    pixels = img_small.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    labels_map = labels.reshape(new_h, new_w)
    labels_full = cv2.resize(labels_map.astype(np.float32), (w, h), 
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    return labels_full


def segment_rag(img_rgb, n_segments=250, compactness=10, threshold=30):
    """Segmentacja RAG (SLIC + merging)."""
    segments = slic(img_rgb, n_segments=n_segments, compactness=compactness, 
                    sigma=1, start_label=0)
    
    g = graph.rag_mean_color(img_rgb, segments, mode='similarity')
    merged_segments = graph.cut_threshold(segments, g, threshold)
    
    return merged_segments


def dice_coefficient(mask_pred, mask_gt):
    """Dice Coefficient (F1-score)."""
    mask_pred = (mask_pred > 127).astype(np.uint8)
    mask_gt = (mask_gt > 127).astype(np.uint8)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    if mask_pred.sum() + mask_gt.sum() == 0:
        return 1.0
    return 2.0 * intersection / (mask_pred.sum() + mask_gt.sum())


def jaccard_index(mask_pred, mask_gt):
    """Jaccard Index (IoU)."""
    mask_pred = (mask_pred > 127).astype(np.uint8)
    mask_gt = (mask_gt > 127).astype(np.uint8)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    
    if union == 0:
        return 1.0
    return intersection / union


def process_image(img_path, method='rag', params=None):
    """
    Przetwarza pojedynczy obraz.
    
    Args:
        img_path: Path do obrazu
        method: 'kmeans' lub 'rag'
        params: dict z parametrami
    
    Returns:
        mask: maska segmentacji
        metrics: dict z metrykami (jeśli GT dostępny)
    """
    # Wczytaj obraz
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Segmentacja
    if method == 'kmeans':
        labels = segment_kmeans(img_rgb, 
                                n_clusters=params['n_clusters'],
                                resize_factor=params['resize_factor'])
        center_region = labels[h//3:2*h//3, w//3:2*w//3]
        lesion_label = np.bincount(center_region.flatten()).argmax()
        mask = (labels == lesion_label).astype(np.uint8) * 255
    
    elif method == 'rag':
        merged = segment_rag(img_rgb, 
                             n_segments=params['n_segments'],
                             compactness=params['compactness'],
                             threshold=params['threshold'])
        center_region = merged[h//3:2*h//3, w//3:2*w//3]
        lesion_label = np.bincount(center_region.flatten()).argmax()
        mask = (merged == lesion_label).astype(np.uint8) * 255
    
    else:
        raise ValueError(f'Unknown method: {method}')
    
    # Post-processing: morfologia
    kernel_size = params.get('morph_kernel', 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Metryki (jeśli GT dostępny)
    metrics = {}
    gt_path = img_path.parent / img_path.name.replace('.jpg', '_gt.jpg')
    if gt_path.exists():
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        metrics['dice'] = dice_coefficient(mask, gt)
        metrics['jaccard'] = jaccard_index(mask, gt)
    
    return mask, metrics


def main():
    """Główna funkcja - przetwarzanie wsadowe wszystkich obrazów."""
    # Znajdź obrazy oryginalne
    all_files = sorted(INPUT_DIR.glob('*.jpg'))
    original_images = [f for f in all_files if '_gt' not in f.name and
                       not any(x in f.name for x in ['_l_b', '_m_b', '_s_b', '_l_l',
                                                      '_m_l', '_s_l', '_m_d', '_s_d'])]

    print(f'Segmentacja zmian skórnych - przetwarzanie wsadowe')
    print(f'=' * 60)
    print(f'Znaleziono {len(original_images)} obrazów')
    print(f'Metoda: RAG (SLIC + merging)')
    print(f'Parametry: {OPTIMAL_PARAMS["rag"]}')
    print(f'=' * 60)
    print()

    # Przetwarzaj wszystkie obrazy
    all_results = []

    for img_path in tqdm(original_images, desc='Przetwarzanie'):
        # Segmentacja RAG
        params = {**OPTIMAL_PARAMS['rag'], 'morph_kernel': OPTIMAL_PARAMS['morph_kernel']}
        mask, metrics = process_image(img_path, method='rag', params=params)

        # Zapisz maskę
        output_path = OUTPUT_DIR / f'{img_path.stem}_mask_rag.png'
        cv2.imwrite(str(output_path), mask)

        # Opcjonalnie: zapisz wizualizację (overlay)
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        overlay[mask > 0] = [100, 255, 100]
        blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        output_vis_path = OUTPUT_DIR / f'{img_path.stem}_overlay.jpg'
        cv2.imwrite(str(output_vis_path), blended_bgr)

        # Zbierz wyniki
        result = {
            'image': img_path.name,
            'has_gt': len(metrics) > 0,
            'dice': metrics.get('dice', np.nan),
            'jaccard': metrics.get('jaccard', np.nan)
        }
        all_results.append(result)

    # Zapisz raport CSV
    df_results = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / 'segmentation_metrics.csv'
    df_results.to_csv(csv_path, index=False)

    print()
    print(f'✓ Przetworzono {len(original_images)} obrazów')
    print(f'✓ Zapisano maski do: {OUTPUT_DIR}')
    print(f'✓ Raport metryk: {csv_path}')
    print()

    # Podsumowanie metryk
    df_with_gt = df_results[df_results['has_gt']]
    if len(df_with_gt) > 0:
        print(f'Podsumowanie metryk (n={len(df_with_gt)} obrazów z GT):')
        print(f'  Średni Dice:    {df_with_gt["dice"].mean():.4f} ± {df_with_gt["dice"].std():.4f}')
        print(f'  Średni Jaccard: {df_with_gt["jaccard"].mean():.4f} ± {df_with_gt["jaccard"].std():.4f}')
        print()
        print('Szczegóły:')
        print(df_with_gt[['image', 'dice', 'jaccard']].to_string(index=False))
    else:
        print('Brak obrazów z maskami Ground Truth.')


if __name__ == '__main__':
    main()
