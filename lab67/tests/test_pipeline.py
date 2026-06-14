from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvproject import config, data, preprocessing, features, classification, metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_get_config_resolves_and_defaults():
    cfg = config.get_config("potato_simple")
    assert cfg["name"] == "potato_simple"
    assert cfg["abs_path"].name == "Task 1 - Simple"
    assert cfg["abs_path"].is_absolute()
    # default uses ACTIVE
    assert config.get_config()["name"] == config.ACTIVE


def test_get_config_unknown_raises():
    try:
        config.get_config("does_not_exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown dataset")


def test_get_config_warns_on_missing_path(tmp_path=None):
    # Temporarily point a fake entry at a missing folder.
    config.DATASETS["__fake__"] = {
        "path": "nope", "classes": {}, "presplit": False, "color": True,
        "glob": "*.jpg", "description": "x",
    }
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.get_config("__fake__")
            assert any("not found" in str(x.message) for x in w)
    finally:
        del config.DATASETS["__fake__"]


# --------------------------------------------------------------------------- #
# data: encode_labels + make_splits
# --------------------------------------------------------------------------- #
def test_encode_labels_roundtrip():
    labels = np.array(["apple", "tomato", "apple", "tomato"])
    y, le = data.encode_labels(labels)
    assert set(y.tolist()) == {0, 1}
    assert list(le.inverse_transform([0, 1])) == sorted(set(labels))


def test_make_splits_three_way_sizes_and_stratify():
    n = 100
    X = np.arange(n * 2).reshape(n, 2)
    y = np.array([0] * 50 + [1] * 50)
    s = data.make_splits(X, y, val_size=0.2, test_size=0.2)
    # disjoint + complete
    assert len(s["X_train"]) + len(s["X_val"]) + len(s["X_test"]) == n
    # val ~20% and test ~20% of original
    assert abs(len(s["X_val"]) - 20) <= 1
    assert abs(len(s["X_test"]) - 20) <= 1
    # stratification preserved in each split
    for k in ("y_train", "y_val", "y_test"):
        frac = np.mean(s[k])
        assert 0.4 <= frac <= 0.6, (k, frac)


def test_make_splits_presplit_uses_given_test():
    X = np.arange(40).reshape(20, 2)
    y = np.array([0, 1] * 10)
    Xtest = np.full((6, 2), -1)
    ytest = np.array([0, 1, 0, 1, 0, 1])
    s = data.make_splits(X, y, val_size=0.25, presplit=(Xtest, ytest))
    assert np.array_equal(s["X_test"], Xtest)  # test set passed through untouched
    assert len(s["X_val"]) == 5  # 25% of the 20 training rows
    assert len(s["X_train"]) == 15


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_evaluate_perfect_and_keys():
    y = np.array([0, 1, 2, 0, 1, 2])
    ev = metrics.evaluate(y, y)
    assert ev["accuracy"] == 1.0 and ev["f1_macro"] == 1.0
    assert set(ev) == {
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    }


def test_evaluate_known_values():
    # 1 mistake out of 4 -> accuracy 0.75
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    ev = metrics.evaluate(y_true, y_pred)
    assert abs(ev["accuracy"] - 0.75) < 1e-9
    # recall for class 1 = 0.5, class 0 = 1.0 -> macro recall 0.75
    assert abs(ev["recall_macro"] - 0.75) < 1e-9


def test_confusion_and_compare_sorting():
    y_true = np.array([0, 0, 1, 1])
    good = np.array([0, 0, 1, 1])
    bad = np.array([1, 1, 0, 0])
    cm = metrics.confusion(y_true, good)
    assert cm.shape == (2, 2) and cm.trace() == 4
    df = metrics.compare_models({"bad": bad, "good": good}, y_true)
    assert list(df.index)[0] == "good"  # sorted by f1_macro desc
    assert df.loc["good", "accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# classification glue (fit_all / predict_all) with real sklearn estimators
# --------------------------------------------------------------------------- #
def test_fit_predict_all_glue():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(0)
    X = rng.rand(30, 4)
    y = (X[:, 0] > 0.5).astype(int)
    models = {
        "dt": Pipeline([("s", StandardScaler()),
                        ("c", DecisionTreeClassifier(random_state=0))]),
    }
    fitted = classification.fit_all(models, X, y)
    preds = classification.predict_all(fitted, X)
    assert preds["dt"].shape == (30,)
    # fit_all must not mutate the originals (clone): original still unfitted
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(models["dt"].named_steps["c"])
        raise AssertionError("fit_all mutated the input model (no clone)")
    except NotFittedError:
        pass


# --------------------------------------------------------------------------- #
# features: name builders + cv2-backed converters
# --------------------------------------------------------------------------- #
def test_glcm_names_count_and_order():
    assert len(features.GLCM_FEATURE_NAMES) == 56
    assert features.GLCM_FEATURE_NAMES[0] == "contrast_d1_a0"
    assert features.GLCM_FEATURE_NAMES[-1] == "entropy_d3_a135"


def test_feature_names_lengths_and_unknown():
    assert len(features.feature_names(("glcm",))) == 56
    assert len(features.feature_names(("color",))) == 18
    assert len(features.feature_names(("shape",))) == 11
    assert len(features.feature_names(("glcm", "color", "shape"))) == 85
    try:
        features.feature_names(("bogus",))
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown family")


def test_quantize_range_and_dtype():
    img = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)
    q = features.quantize(img, levels=32)
    assert q.dtype == np.uint8
    assert q.min() >= 0 and q.max() <= 31  # never overflows the bin count
    # a 3-channel image must be accepted (gray-converted)
    rgb = np.dstack([img, img, img])
    assert features.quantize(rgb).shape == img.shape


def test_binarize_object_is_foreground():
    # bright square (object) on dark background -> object must become 255
    img = np.zeros((20, 20), np.uint8)
    img[5:15, 5:15] = 200
    b = features.binarize(img, threshold=128)
    assert set(np.unique(b)).issubset({0, 255})
    assert b[10, 10] == 255 and b[0, 0] == 0
    # inverted case: bright background, dark object -> still object=255 (minority)
    img2 = np.full((20, 20), 200, np.uint8)
    img2[8:12, 8:12] = 10
    b2 = features.binarize(img2, threshold=128)
    assert b2[10, 10] == 255 and b2[0, 0] == 0


# --------------------------------------------------------------------------- #
# preprocessing converters/glue
# --------------------------------------------------------------------------- #
def test_uint8_converter():
    assert preprocessing._uint8(np.array([[0.0, 0.5, 1.0]])).tolist() == [[0, 127, 255]]
    u = np.array([[10, 250]], np.uint8)
    assert preprocessing._uint8(u) is u  # uint8 passes through untouched


def test_to_gray_shapes():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    g = preprocessing.to_gray(rgb)
    assert g.ndim == 2 and g.shape == (8, 8)
    assert preprocessing.to_gray(g).shape == (8, 8)  # already-gray path


def test_apply_mask_zeros_background():
    img = np.full((10, 10, 3), 128, np.uint8)
    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 2:5] = 255
    out = preprocessing.apply_mask(img, mask)
    assert out[3, 3].tolist() == [128, 128, 128]  # foreground kept
    assert out[0, 0].tolist() == [0, 0, 0]  # background zeroed
    # grayscale path
    g = np.full((10, 10), 99, np.uint8)
    og = preprocessing.apply_mask(g, mask)
    assert og[3, 3] == 99 and og[0, 0] == 0


# --------------------------------------------------------------------------- #
# feature extractors (now implemented) — fixed-length, names-aligned vectors
# --------------------------------------------------------------------------- #
def _synthetic_object():
    """RGB image with a centred bright-green ellipse on a dark background."""
    import cv2
    img = np.zeros((120, 120, 3), np.uint8)
    cv2.ellipse(img, (60, 60), (40, 25), 0, 0, 360, (40, 200, 60), -1)
    return img


def test_glcm_features_shape_and_finite():
    img = _synthetic_object()
    f = features.glcm_features(img)
    assert f.shape == (len(features.GLCM_FEATURE_NAMES),) == (56,)
    assert np.all(np.isfinite(f))


def test_color_features_shape_and_mask():
    img = _synthetic_object()
    f_all = features.color_features(img)
    assert f_all.shape == (18,) and np.all(np.isfinite(f_all))
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:80, 30:90] = 255
    f_masked = features.color_features(img, mask)
    assert f_masked.shape == (18,)
    # empty mask -> zeros, never NaN
    f_empty = features.color_features(img, np.zeros(img.shape[:2], np.uint8))
    assert np.all(f_empty == 0)
    # grayscale fallback still returns 18 finite values (uniform channel -> skew 0, not NaN)
    g = np.full((40, 40), 100, np.uint8)
    fg = features.color_features(g)
    assert fg.shape == (18,) and np.all(np.isfinite(fg))


def test_shape_features_circle_compactness():
    import cv2
    img = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(img, (100, 100), 60, (255, 255, 255), -1)
    f = features.shape_features(img)
    assert f.shape == (11,)
    names = features.SHAPE_FEATURE_NAMES
    comp = f[names.index("compactness")]
    # a filled circle is ~1.0; rasterised staircase perimeter pulls it to ~0.89
    assert 0.85 <= comp <= 1.05, comp
    ecc = f[names.index("eccentricity")]
    assert 0.85 <= ecc <= 1.0, ecc  # near-circular -> minor/major ~1
    # no contour -> all zeros, no crash
    assert np.all(features.shape_features(np.zeros((20, 20, 3), np.uint8)) == 0)


def test_extract_all_matches_feature_names():
    img = _synthetic_object()
    use = ("glcm", "color", "shape")
    v = features.extract_all(img, use=use)
    assert v.shape == (len(features.feature_names(use)),) == (85,)


def test_optional_descriptors():
    img = _synthetic_object()
    assert features.hu_moments(img).shape == (7,)
    assert features.fourier_descriptor(img, n_coeffs=16).shape == (16,)


# --------------------------------------------------------------------------- #
# preprocessing methods (now implemented)
# --------------------------------------------------------------------------- #
def test_enhance_contrast_preserves_shape():
    img = _synthetic_object()
    out = preprocessing.enhance_contrast(img, method="clahe")
    assert out.shape == img.shape and out.dtype == np.uint8
    assert preprocessing.enhance_contrast(img, method="hist_eq").shape == img.shape
    g = preprocessing.to_gray(img)
    assert preprocessing.enhance_contrast(g).shape == g.shape


def test_white_balance_neutralises_cast():
    img = _synthetic_object().copy()
    img[..., 1] = np.clip(img[..., 1].astype(int) + 40, 0, 255)  # green cast
    wb = preprocessing.white_balance(img)
    # after gray-world the three channel means should be much closer together
    before = img.reshape(-1, 3).mean(0)
    after = wb.reshape(-1, 3).mean(0)
    assert after.std() < before.std()
    assert preprocessing.white_balance(np.zeros((5, 5), np.uint8)).ndim == 2  # gray no-op


def test_remove_artifacts_variants():
    img = _synthetic_object()
    for m in ("median", "bilateral", "nlmeans"):
        out = preprocessing.remove_artifacts(img, method=m)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_segment_foreground_finds_object():
    img = _synthetic_object()  # bright object, dark background
    for method in ("otsu", "hsv", "grabcut"):
        mask = preprocessing.segment_foreground(img, method=method, hsv_target="green")
        assert set(np.unique(mask)).issubset({0, 255})
        assert mask[60, 60] == 255  # object centre is foreground
        assert mask[2, 2] == 0      # corner is background


def test_preprocess_chains_and_returns_mask():
    img = _synthetic_object()
    cfg = {"color": True}
    out, mask = preprocessing.preprocess(img, cfg, segment_method="otsu")
    assert out.shape == img.shape
    assert mask is not None and set(np.unique(mask)).issubset({0, 255})
    # without a segment step, mask is None
    out2, mask2 = preprocessing.preprocess(img, cfg, steps=("enhance_contrast",))
    assert mask2 is None


# --------------------------------------------------------------------------- #
# classification: build_models + tune (now implemented)
# --------------------------------------------------------------------------- #
def test_build_models_returns_pipelines():
    models = classification.build_models()
    assert {"svm", "decision_tree"}.issubset(models)  # the two TASK-named models
    for name, pipe in models.items():
        assert pipe.steps[0][0] == "scaler" and pipe.steps[-1][0] == "clf"


def test_tune_runs_gridsearch():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    models = classification.build_models()
    search = classification.tune(
        models["decision_tree"], classification.PARAM_GRIDS["decision_tree"],
        X, y, cv=3,
    )
    assert hasattr(search, "best_estimator_")
    assert 0.0 <= search.best_score_ <= 1.0
    assert search.best_estimator_.predict(X).shape == (40,)


# --------------------------------------------------------------------------- #
# viz helpers — guard the matplotlib/sklearn version-compat regressions
# --------------------------------------------------------------------------- #
def test_plot_confusion_matrix_int_and_float():
    import matplotlib
    matplotlib.use("Agg")
    from cvproject import viz

    cm = np.array([[8, 2], [1, 9]])
    # non-normalized must work whether cm arrives as int or float (values_format="d")
    viz.plot_confusion_matrix(cm, ["a", "b"], normalize=False)
    viz.plot_confusion_matrix(cm.astype(float), ["a", "b"], normalize=False)
    viz.plot_confusion_matrix(cm.astype(float), ["a", "b"], normalize=True)


def test_plot_feature_distributions_runs():
    import matplotlib
    matplotlib.use("Agg")
    from cvproject import viz

    X = np.random.RandomState(0).rand(30, 5)
    y = np.array(["early_blight", "healthy"] * 15)
    names = [f"f{i}" for i in range(5)]
    fig = viz.plot_feature_distributions(X, y, names, max_features=4)
    assert fig is not None


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
