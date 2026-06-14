from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvproject import config, data, preprocessing, features, classification, metrics  # noqa: E402


def main() -> int:
    cfg = config.get_config("potato_simple")
    print(f"[1] dataset: {cfg['name']}  path-exists={cfg['abs_path'].exists()}")

    # --- real load of the small dataset (also exercises case-insensitive glob) ---
    X_imgs, y_str = data.load_dataset(cfg)
    imgs = X_imgs[:6]
    print(f"[2] load_dataset: {len(X_imgs)} imgs, first shape={imgs[0].shape}, dtype={imgs[0].dtype}")
    assert imgs[0].ndim == 3 and imgs[0].dtype == np.uint8

    # --- converters on a real image ---
    g = preprocessing.to_gray(imgs[0])
    q = features.quantize(imgs[0])
    b = features.binarize(imgs[0])
    fake_mask = np.zeros(imgs[0].shape[:2], np.uint8)
    fake_mask[10:-10, 10:-10] = 255
    masked = preprocessing.apply_mask(imgs[0], fake_mask)
    print(f"[3] to_gray{g.shape} quantize(max={q.max()}) binarize(uniq={np.unique(b).tolist()}) "
          f"apply_mask(corner={masked[0,0].tolist()})")
    assert g.ndim == 2 and q.max() <= 31 and set(np.unique(b)).issubset({0, 255})
    assert masked[0, 0].tolist() == [0, 0, 0]

    # --- stratified split on real labels ---
    print(f"[4] classes={sorted(set(y_str))}, counts={[int((y_str==c).sum()) for c in sorted(set(y_str))]}")
    y, le = data.encode_labels(y_str)

    # class-balanced subset (40/class) so the tiny demo has both labels
    idx = np.concatenate([np.where(y == c)[0][:40] for c in np.unique(y)])
    sub_imgs = [X_imgs[i] for i in idx]
    yy = y[idx]

    # REAL preprocessing (segmentation mask per image) + REAL feature extraction
    cfg_color = {"color": True}
    proc, masks = [], []
    for im in sub_imgs:
        p, m = preprocessing.preprocess(im, cfg_color, segment_method="otsu")
        proc.append(p)
        masks.append(m)
    X, names = features.build_feature_matrix(proc, masks=masks, use=("glcm", "color", "shape"))
    print(f"[5] build_feature_matrix (real features): X={X.shape}, n_names={len(names)} "
          f"(cols match={X.shape[1]==len(names)}, finite={np.isfinite(X).all()})")
    assert X.shape[1] == len(names) == 85 and np.isfinite(X).all()

    # --- split + tune (train/val) + evaluate on the held-out test set ---
    s = data.make_splits(X, yy, val_size=0.2, test_size=0.2)
    models = classification.build_models()
    # tune the two TASK-named models on the training set, then fit on train+val
    tuned = {}
    X_fit = np.vstack([s["X_train"], s["X_val"]])
    y_fit = np.concatenate([s["y_train"], s["y_val"]])
    for name in ("svm", "decision_tree"):
        search = classification.tune(models[name], classification.PARAM_GRIDS[name],
                                     s["X_train"], s["y_train"], cv=3)
        tuned[name] = search.best_estimator_.fit(X_fit, y_fit)
        print(f"    tuned {name}: best_cv_f1={search.best_score_:.3f} {search.best_params_}")
    preds = classification.predict_all(tuned, s["X_test"])
    table = metrics.compare_models(preds, s["y_test"])
    print(f"[6] split sizes: train={len(s['X_train'])} val={len(s['X_val'])} test={len(s['X_test'])}")
    print("[7] metrics.compare_models:")
    print(table.round(3).to_string())
    print(f"[8] report (best={table.index[0]}):")
    print(metrics.report(s["y_test"], preds[table.index[0]], target_names=list(le.classes_)))

    print("\nDRY-RUN OK — implemented plumbing chains end-to-end on real images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
