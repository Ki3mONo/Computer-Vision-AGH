import sys; sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvproject import config, data, preprocessing as pp, features as F, classification as C, metrics as M, viz
from cvproject.config import RNG
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.makedirs("docs/figures", exist_ok=True)
PL = ["jabłko", "pomidor"]


def prep(imgs, seg="hsv"):
    P, Mk = [], []
    for im in imgs:
        p, m = pp.preprocess(im, segment_method=seg)
        P.append(p); Mk.append(m)
    return P, Mk


cfg = config.get_config()
Xtr_i, ytr_s, Xte_i, yte_s = data.load_presplit(cfg)
Ptr, Mtr = prep(Xtr_i)
Pte, Mte = prep(Xte_i)
USE = F.DEFAULT_USE
Xtr, names = F.build_feature_matrix(Ptr, Mtr, use=USE)
Xte, _ = F.build_feature_matrix(Pte, Mte, use=USE)
ytr, le = data.encode_labels(ytr_s)
yte = le.transform(yte_s)
sp = data.make_splits(Xtr, ytr, val_size=0.2, presplit=(Xte, yte))

# --- macierz pomyłek najlepszego (wybór po walidacji) ---
tuned, valf = {}, {}
for n, m in C.build_models().items():
    e = C.tune(m, C.PARAM_GRIDS.get(n, {}), sp["X_train"], sp["y_train"]).best_estimator_
    tuned[n] = e
    valf[n] = M.evaluate(sp["y_val"], e.predict(sp["X_val"]))["f1_macro"]
best = max(valf, key=valf.get)
cm = M.confusion(sp["y_test"], tuned[best].predict(sp["X_test"]))
fig = viz.plot_confusion_matrix(cm, PL, title=f"Macierz pomyłek — {best}")
fig.savefig("docs/figures/confusion.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# --- ważność cech wg rodziny ---
rf = RandomForestClassifier(n_estimators=400, random_state=RNG).fit(Xtr, ytr)
fam = np.array([f for f in USE for _ in F.feature_names((f,))])
imp = pd.Series(rf.feature_importances_, index=fam).groupby(level=0).sum().sort_values()
etyk = {"glcm": "GLCM (tekstura)", "color": "kolor", "huehist": "hist. odcienia", "lbp": "LBP",
        "shape": "geometria", "hu": "Hu", "fourier": "Fourier", "geom": "convex/fraktal", "zernike": "Zernike"}
fig, ax = plt.subplots(figsize=(6, 3.4))
imp.rename(index=etyk).plot.barh(ax=ax, color="#4C72B0")
ax.set_xlabel("sumaryczna ważność"); ax.set_title("Ważność cech wg rodziny (Random Forest)")
fig.tight_layout(); fig.savefig("docs/figures/feat_importance.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# --- ablacja rodzin cech ---
off, s = {}, 0
for f in USE:
    k = len(F.feature_names((f,))); off[f] = list(range(s, s + k)); s += k
cols = lambda c: [i for f in c for i in off[f]]
skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
svm = C.build_models()["svm"]
zest = {"wszystko": tuple(USE), "tekstura": ("glcm", "lbp"), "kolor": ("color", "huehist"),
        "kształt": ("shape", "hu", "fourier", "geom", "zernike"),
        "tekstura+kolor": ("glcm", "lbp", "color", "huehist")}
res = {k: cross_val_score(svm, Xtr[:, cols(v)], ytr, cv=skf, scoring="f1_macro").mean() for k, v in zest.items()}
ser = pd.Series(res).sort_values()
fig, ax = plt.subplots(figsize=(6, 3.0))
ser.plot.barh(ax=ax, color="#55A868"); ax.set_xlim(0.6, 0.81)
ax.set_xlabel("f1_macro (walidacja krzyżowa)"); ax.set_title("Ablacja rodzin cech (SVM)")
fig.tight_layout(); fig.savefig("docs/figures/ablation.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# --- PCA ---
pc = PCA(n_components=2, random_state=RNG).fit_transform(StandardScaler().fit_transform(Xtr))
fig, ax = plt.subplots(figsize=(5, 3.8))
for kk in np.unique(ytr):
    selm = ytr == kk
    ax.scatter(pc[selm, 0], pc[selm, 1], label=PL[kk], alpha=0.6, s=18)
ax.legend(); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Rzut PCA cech treningowych")
fig.tight_layout(); fig.savefig("docs/figures/pca.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# --- segmentacja Otsu vs HSV ---
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
for r, i in enumerate([0, 80, 200]):
    base = pp.remove_artifacts(Xtr_i[i])
    mo = pp.segment_foreground(base, method="otsu")
    mh = pp.segment_foreground(base, method="hsv", hsv_target="warm")
    axes[r, 0].imshow(Xtr_i[i]); axes[r, 1].imshow(mo, cmap="gray"); axes[r, 2].imshow(mh, cmap="gray")
    for a in axes[r]:
        a.axis("off")
axes[0, 0].set_title("oryginał"); axes[0, 1].set_title("Otsu"); axes[0, 2].set_title("HSV")
fig.tight_layout(); fig.savefig("docs/figures/segmentation.png", dpi=120, bbox_inches="tight"); plt.close(fig)

print("FIGS DONE:", sorted(os.listdir("docs/figures")))
