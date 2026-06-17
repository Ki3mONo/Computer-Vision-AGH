"""Parameter sweep: segmentation x feature families x classifiers.

Selection is done on a validation split carved from train; the held-out test
set is only touched for the final numbers.
"""
import sys; sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from cvproject import config, data, preprocessing as pp, features as F, classification as C, metrics as M
from cvproject.config import RNG

ALL = ("glcm", "color", "hist", "lbp", "shape")
CLFS = ("svm", "decision_tree", "random_forest", "knn", "logreg")


def col_slices():
    slices, start = {}, 0
    for fam in ALL:
        n = len(F.feature_names((fam,)))
        slices[fam] = (start, start + n)
        start += n
    return slices


SLICES = col_slices()


def cols(combo):
    idx = []
    for fam in combo:
        s, e = SLICES[fam]
        idx += list(range(s, e))
    return np.array(idx)


def build_matrix(imgs, seg, target="warm"):
    rows = []
    for im in imgs:
        p, m = pp.preprocess(im, segment_method=seg, hsv_target=target)
        rows.append(F.extract_all(p, m, use=ALL))
    return np.vstack(rows)


def eval_combo(mats, seg, combo, tr_idx, val_idx, ytr, yte):
    Xtr_full, Xte_full = mats[seg]
    c = cols(combo)
    Xtr_c, Xte_c = Xtr_full[:, c], Xte_full[:, c]
    Xt, Xv, yt, yv = Xtr_c[tr_idx], Xtr_c[val_idx], ytr[tr_idx], ytr[val_idx]

    best = None
    for name in CLFS:
        search = C.tune(C.build_models()[name], C.PARAM_GRIDS.get(name, {}), Xt, yt, cv=4)
        val_f1 = M.evaluate(yv, search.best_estimator_.predict(Xv))["f1_macro"]
        if best is None or val_f1 > best[1]:
            best = (name, val_f1, search.best_estimator_, search.best_params_)

    name, val_f1, est, params = best
    est = clone(est).fit(Xtr_c, ytr)
    test = M.evaluate(yte, est.predict(Xte_c))
    return {"name": name, "val_f1": val_f1, "test": test, "params": params, "est": est, "cols": c}


def main():
    cfg = config.get_config()
    Xtr_imgs, ytr_s, Xte_imgs, yte_s = data.load_presplit(cfg)
    y_all, le = data.encode_labels(np.concatenate([ytr_s, yte_s]))
    ytr, yte = y_all[:len(ytr_s)], y_all[len(ytr_s):]
    base = max(np.bincount(yte)) / len(yte)
    print(f"train={len(ytr)} test={len(yte)} classes={list(le.classes_)} majority-baseline acc={base:.3f}\n")

    SEGS = {"otsu": ("otsu", "warm"), "hsv": ("hsv", "warm"), "grabcut": ("grabcut", "warm")}
    mats = {}
    for nm, (seg, t) in SEGS.items():
        mats[nm] = (build_matrix(Xtr_imgs, seg, t), build_matrix(Xte_imgs, seg, t))
        print(f"built feature matrix for seg={nm}")
    print()

    tr_idx, val_idx = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=RNG)

    # ---- (a) feature-family ablation on hsv segmentation ----
    combos = [
        ("glcm",), ("color",), ("hist",), ("lbp",), ("shape",),
        ("color", "hist"), ("hist", "lbp"), ("glcm", "hist"),
        ("glcm", "color", "shape"),            # original baseline
        ("color", "hist", "glcm"),
        ("glcm", "color", "hist", "lbp", "shape"),  # everything
    ]
    rows = []
    for combo in combos:
        r = eval_combo(mats, "hsv", combo, tr_idx, val_idx, ytr, yte)
        rows.append({"features": "+".join(combo), "clf": r["name"], "val_f1": round(r["val_f1"], 3),
                     "test_acc": round(r["test"]["accuracy"], 3), "test_f1": round(r["test"]["f1_macro"], 3)})
    abl = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    print("=== (a) feature ablation (seg=hsv, selected by val_f1) ===")
    print(abl.to_string(index=False), "\n")

    best_combo = tuple(abl.iloc[0]["features"].split("+"))

    # ---- (b) segmentation comparison for the best feature combo ----
    rows = []
    for seg in SEGS:
        r = eval_combo(mats, seg, best_combo, tr_idx, val_idx, ytr, yte)
        rows.append({"seg": seg, "clf": r["name"], "val_f1": round(r["val_f1"], 3),
                     "test_acc": round(r["test"]["accuracy"], 3), "test_f1": round(r["test"]["f1_macro"], 3)})
    segdf = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    print(f"=== (b) segmentation comparison (features={'+'.join(best_combo)}) ===")
    print(segdf.to_string(index=False), "\n")

    # ---- (c) final winner: best (seg, combo) by val, full test report ----
    best_seg = segdf.iloc[0]["seg"]
    r = eval_combo(mats, best_seg, best_combo, tr_idx, val_idx, ytr, yte)
    print(f"=== (c) WINNER: seg={best_seg}  features={'+'.join(best_combo)}  clf={r['name']} ===")
    print("params:", r["params"])
    print("test metrics:", {k: round(v, 3) for k, v in r["test"].items()})
    yp = r["est"].predict(mats[best_seg][1][:, r["cols"]])
    print("\nper-class report:")
    print(M.report(yte, yp, target_names=list(le.classes_)))
    print("confusion (rows=true):")
    print(M.confusion(yte, yp))


if __name__ == "__main__":
    main()
