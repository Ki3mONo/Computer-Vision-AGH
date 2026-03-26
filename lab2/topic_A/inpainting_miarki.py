"""
inpainting_miarki_real.py
=========================
Skrypt dopasowany do rzeczywistych obrazów USG z przesłanego zestawu.
Miarki: niebieskie przerywane linie pomiarowe (B-R > 40, B-G > 25).

Uruchomienie:
    python3 inpainting_miarki_real.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# ── Ścieżki ────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("/mnt/user-data/uploads")
OUTPUT_DIR = Path("/mnt/user-data/outputs/inpainting_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_FILES = sorted(INPUT_DIR.glob("*.jpg"))

# ── Parametry segmentacji (skalibrowane na podstawie analizy pikseli) ──────
SEG = dict(
    b_minus_r_thresh = 30,   # B - R > X  (miarki: ~70-130 diff)
    b_minus_g_thresh = 20,   # B - G > X
    b_min_val        = 80,   # minimalna jasność kanału B
    dilate_k         = 7,    # rozmiar jądra dylatacji [px]
    dilate_iter      = 2,    # liczba iteracji dylatacji
)

# ── Parametry inpaintingu ───────────────────────────────────────────────────
INPAINT_RADIUS = 7


# ══════════════════════════════════════════════════════════════════════════════
def segment_blue_calipers(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """
    Segmentuje niebieskie miarki USG na podstawie dominacji kanału B.
    
    Strategia:
      1. Maska kolor: B - R > thresh i B - G > thresh i B > min_val
      2. Dylatacja morfologiczna – poszerz maskę
    """
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    mask = (
        (b - r > params['b_minus_r_thresh']) &
        (b - g > params['b_minus_g_thresh']) &
        (img_bgr[:, :, 0] > params['b_min_val'])
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params['dilate_k'], params['dilate_k'])
    )
    mask = cv2.dilate(mask, kernel, iterations=params['dilate_iter'])
    return mask


def apply_inpainting(img_bgr, mask, method='telea', radius=7):
    flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    return cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=flag)


def compute_metrics_no_ref(original_bgr, restored_bgr):
    """Metryki bezreferencyjna: wariancja Laplasjanu i gradient RMS."""
    def lap_var(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    def gradient_rms(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.sqrt(gx**2 + gy**2).mean())

    return {
        'LapVar_orig':     round(lap_var(original_bgr), 2),
        'LapVar_inpainted':round(lap_var(restored_bgr), 2),
        'GradRMS_orig':    round(gradient_rms(original_bgr), 2),
        'GradRMS_inpainted':round(gradient_rms(restored_bgr), 2),
    }


def save_comparison(img_bgr, mask, restored_bgr, name, out_dir):
    img_rgb  = cv2.cvtColor(img_bgr,    cv2.COLOR_BGR2RGB)
    rest_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    overlay  = img_rgb.copy()
    overlay[mask > 0] = [255, 50, 50]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, title in zip(
        axes,
        [img_rgb, overlay, rest_rgb],
        ['Oryginał (z miarkami)', 'Maska (czerwona)', 'Po inpaintingu (Telea)']
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.suptitle(name, fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = out_dir / f"{Path(name).stem}_comparison.png"
    plt.savefig(str(out_path), dpi=110, bbox_inches='tight')
    plt.close()
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  GŁÓWNA PĘTLA
# ══════════════════════════════════════════════════════════════════════════════

all_metrics = []

print(f"{'='*65}")
print(f"  Przetwarzanie {len(IMAGE_FILES)} obrazów USG")
print(f"{'='*65}")

for img_path in IMAGE_FILES:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  BŁĄD: nie można wczytać {img_path.name}")
        continue

    # 1. Segmentacja
    mask = segment_blue_calipers(img_bgr, SEG)
    coverage = 100 * float(mask.sum()) / (255.0 * mask.size)

    # 2. Inpainting – Telea (główna metoda)
    restored_telea = apply_inpainting(img_bgr, mask, 'telea', INPAINT_RADIUS)

    # 3. Inpainting – NS (do porównania)
    restored_ns = apply_inpainting(img_bgr, mask, 'ns', INPAINT_RADIUS)

    # 4. Metryki
    m = compute_metrics_no_ref(img_bgr, restored_telea)
    m['file'] = img_path.name
    m['mask_coverage_%'] = round(coverage, 3)
    all_metrics.append(m)

    # 5. Zapis wyników
    cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_telea.jpg"), restored_telea)
    cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_ns.jpg"), restored_ns)
    cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_mask.png"), mask)

    # 6. Plansza porównawcza
    save_comparison(img_bgr, mask, restored_telea, img_path.name, OUTPUT_DIR)

    print(f"  ✓ {img_path.name:30s}  "
          f"maska={coverage:.3f}%  "
          f"LapVar: {m['LapVar_orig']:.0f} → {m['LapVar_inpainted']:.0f}")

# ── Podsumowanie tabelaryczne ──────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PODSUMOWANIE METRYK (metryki bezreferencyjna)")
print(f"{'='*65}")
print(f"  {'Plik':<28} {'Maska%':>7} {'LapV↓':>8} {'GradRMS↓':>10}")
print(f"  {'-'*57}")
for m in all_metrics:
    delta_lap  = m['LapVar_inpainted'] - m['LapVar_orig']
    delta_grad = m['GradRMS_inpainted'] - m['GradRMS_orig']
    print(f"  {m['file']:<28} {m['mask_coverage_%']:>7.3f} "
          f"  {delta_lap:>+7.1f}  {delta_grad:>+9.2f}")

print(f"\n  Wyniki zapisane w: {OUTPUT_DIR.resolve()}")

# ── Zapis CSV ──────────────────────────────────────────────────────────────
import csv
csv_path = OUTPUT_DIR / "metrics_report.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
    writer.writeheader()
    writer.writerows(all_metrics)
print(f"  Raport CSV: {csv_path}")
