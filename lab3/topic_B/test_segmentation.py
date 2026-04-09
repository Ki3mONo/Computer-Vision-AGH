#!/usr/bin/env python3
"""
Test segmentation pipeline - quick validation
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.future import graph

# Test imports
print("Testing imports...")
print("✓ numpy")
print("✓ cv2 (opencv)")
print("✓ sklearn.cluster.KMeans")
print("✓ skimage.segmentation.slic")
print("✓ skimage.future.graph")

# Test loading image
INPUT_DIR = Path('./images')
image_files = sorted(INPUT_DIR.glob('*.jpg'))
original_images = [f for f in image_files if '_gt' not in f.name and 
                   not any(x in f.name for x in ['_l_b', '_m_b', '_s_b', '_l_l', 
                                                  '_m_l', '_s_l', '_m_d', '_s_d'])]

print(f"\nFound {len(original_images)} original images")

if len(original_images) > 0:
    test_img_path = original_images[0]
    print(f"Testing with: {test_img_path.name}")
    
    img_bgr = cv2.imread(str(test_img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: shape={img_rgb.shape}")
    
    # Test KMeans
    print("\nTesting KMeans...")
    h, w = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, (w//2, h//2))
    pixels = img_small.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    print(f"✓ KMeans completed: {len(np.unique(labels))} clusters")
    
    # Test SLIC
    print("\nTesting SLIC...")
    segments = slic(img_rgb, n_segments=250, compactness=10, sigma=1, start_label=0)
    print(f"✓ SLIC completed: {len(np.unique(segments))} superpixels")
    
    # Test RAG
    print("\nTesting RAG...")
    g = graph.rag_mean_color(img_rgb, segments, mode='similarity')
    merged = graph.cut_threshold(segments, g, 35)
    print(f"✓ RAG completed: {len(np.unique(merged))} regions after merging")
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
    
else:
    print("ERROR: No images found in ./images directory")
