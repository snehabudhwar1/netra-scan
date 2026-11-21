"""Step 2: train several statistical-ML models, compare them fairly, keep the best.

What happens, in order:
  1. Load the features made by step 1 and split the photos 80% / 20% (stratified, seed 42).
     The 20% "test" part is NOT touched until the very end.
  2. Metadata-only baseline: a small decision tree that sees only photo size/brightness
     (no retina content). This is the shortcut we found in the EDA. Real models must beat it.
  3. For every feature set (hand / emb / hand+emb) and every model, run 5-fold
     cross-validation on the 80% part and score it with QWK (quadratic weighted kappa).
  4. Fine-tune the most important setting (C) of logistic regression and SVM.
  5. Take the winner, set the safety thresholds (see below), fit it on the 80% part and
     score it ONCE on the untouched 20% test part.
  6. Save: models/netra_bundle.joblib, models/metrics.json, reports/*.csv, reports/figures/*.png

Safety thresholds learned here (all from cross-validation, not guessed):
  * refer_threshold : probability of "grade >= 2" above which we say "see an eye doctor".
                      Chosen so that at least 90% of really-referable photos are caught.
  * conf_threshold  : the model's least-confident 20% of predictions are flagged "uncertain".
  * unfamiliar threshold : photos farther from the training photos than 99.5% of them are
                      flagged "unfamiliar" (e.g. a different camera or not a retina at all).

Run:  python scripts/02_train_models.py            (about 10-20 minutes)
Quick test: python scripts/02_train_models.py --sets hand --folds 3
"""
import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from netra import data as D  # noqa: E402
from netra import metrics as M  # noqa: E402
from netra.features import feature_names  # noqa: E402
from netra.plots import plot_comparison, plot_confusion, plot_recall  # noqa: E402
from netra.quality import calibrate_thresholds  # noqa: E402

CLASSES = M.CLASSES


def make_models(n_trees=300):
    return {
        "logreg": lambda: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
        "knn": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "naive_bayes": lambda: GaussianNB(),
        "svm_rbf": lambda: SVC(C=3, kernel="rbf", probability=True, class_weight="balanced", random_state=0),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                         class_weight="balanced", random_state=0),
        "random_forest": lambda: RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3,
                                                         class_weight="balanced_subsample",
                                                         n_jobs=1, random_state=0),
    }


def make_pipe(clf, dim, n_pca):
    steps = [("scale", StandardScaler())]
    if dim > 200:  # 2048 numbers per photo is a lot: PCA keeps the strongest directions
        steps.append(("pca", PCA(n_components=n_pca, random_state=0)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def clean(o):
    """Make a structure JSON-safe (NaN -> null)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def pick_refer_threshold(y, oof, target_sens=0.90):
    p_ref = oof[:, 2:].sum(axis=1)
    yr = (y >= 2).astype(int)
    best = None
    grid = np.linspace(0.05, 0.95, 91)
    stats = []
    for t in grid:
        pred = (p_ref >= t).astype(int)
        sens = ((pred == 1) & (yr == 1)).sum() / max((yr == 1).sum(), 1)
        spec = ((pred == 0) & (yr == 0)).sum() / max((yr == 0).sum(), 1)
        stats.append((t, sens, spec))
    ok = [s for s in stats if s[1] >= target_sens]
    if ok:
        best = max(ok, key=lambda s: s[2])  # the strictest threshold that still catches >= 90%
    else:
        best = max(stats, key=lambda s: s[1] + s[2])  # fall back to Youden's J
    return float(best[0]), float(best[1]), float(best[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="aptos", help="dataset name used in step 1")
    ap.add_argument("--sets", nargs="+", default=list(D.FEATURE_SETS), choices=D.FEATURE_SETS)
    ap.add_argument("--models", nargs="+", default=None, help="subset of models to run")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca", type=int, default=128, help="PCA components for the 2048-number features")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    model_dir, report_dir = Path(args.model_dir), Path(args.report_dir)
    fig_dir = report_dir / "figures"
    for d in (model_dir, report_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ load + split
    d = D.load(args.name, args.features_dir)
    sets = [s for s in args.sets if not ("emb" in s and d["emb"] is None)]
    if len(sets) < len(args.sets):
        print("NOTE: no embeddings found, skipping feature sets that need them.")
    ok = D.valid_rows(d)
    print(f"{ok.sum()} usable photos out of {len(ok)}")
    meta = d["meta"][ok].reset_index(drop=True)
    tab = d["tab"][ok].reset_index(drop=True)
    d2 = {"tab": tab, "emb": None if d["emb"] is None else d["emb"][ok], "meta": meta}
    y = meta["label"].to_numpy()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    # PCA can never keep more directions than there are training photos in one fold
    n_pca = max(2, min(args.pca, int(len(tr) * (args.folds - 1) / args.folds) - 1))
    print(f"train {len(tr)}  |  test {len(te)}  (test is untouched until the end)")

    # ------------------------------------------------------------ quality thresholds
    qdf = pd.DataFrame({k: tab[f"q_{k}"] for k in ("sharpness", "brightness", "contrast", "fov")})
    qthr = calibrate_thresholds(qdf)
    with open(model_dir / "quality_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(qthr, f, indent=2)
    print("quality thresholds learned from the photos:", qthr)

    # ------------------------------------------------------------ metadata-only baseline
    base_cols = pd.DataFrame({"w": meta["width"], "h": meta["height"], "bright": tab["q_brightness"],
                              "contrast": tab["q_contrast"], "fill": tab["q_fill_ratio"]}).to_numpy(float)
    base = DecisionTreeClassifier(max_depth=4, random_state=42).fit(base_cols[tr], y[tr])
    base_proba = np.zeros((len(te), 5))
    base_proba[:, base.classes_.astype(int)] = base.predict_proba(base_cols[te])
    base_rep = M.full_report(y[te], base_proba, 0.5)
    always0 = float((y[te] == 0).mean())
    print(f"\nalways-guess-grade-0 accuracy : {always0:.3f}")
    print(f"metadata-only baseline (NO retina content): {M.one_line(base_rep)}")

    # ------------------------------------------------------------ compare models
    all_models = make_models(args.trees)
    names = args.models or list(all_models)
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=42)
    rows, oofs = [], {}

    def run_cv(label, fs, make_clf, X):
        t0 = time.time()
        pipe = make_pipe(make_clf(), X.shape[1], n_pca)
        oof = cross_val_predict(pipe, X[tr], y[tr], cv=skf, method="predict_proba", n_jobs=-1)
        rep = M.full_report(y[tr], oof, 0.5)
        rows.append({"feature_set": fs, "model": label, "cv_qwk": rep["qwk"], "cv_accuracy": rep["accuracy"],
                     "cv_macro_f1": rep["macro_f1"], "cv_within_one": rep["within_one_grade"],
                     "cv_referable_auc": rep["referable"]["auc"], "seconds": round(time.time() - t0, 1)})
        oofs[(fs, label)] = oof
        print(f"  {fs:9s} {label:22s} qwk={rep['qwk']:.3f} acc={rep['accuracy']:.3f} "
              f"macroF1={rep['macro_f1']:.3f}  ({time.time() - t0:.0f}s)")

    mats = {fs: D.build_matrix(d2, fs) for fs in sets}
    print(f"\nCross-validation ({args.folds} folds) on the training part:")
    for fs in sets:
        for m in names:
            try:
                run_cv(m, fs, all_models[m], mats[fs])
            except Exception as e:  # keep going if one model fails
                print(f"  {fs} {m} FAILED: {e}")
    if not rows:
        raise SystemExit("No model finished. See the messages above.")

    res = pd.DataFrame(rows).sort_values("cv_qwk", ascending=False).reset_index(drop=True)
    best_fs = res.iloc[0]["feature_set"]

    # ------------------------------------------------------------ small tuning round
    if not args.no_tune:
        print(f"\nFine-tuning C for the best feature set ({best_fs}):")
        grids = {"logreg": ("C", [0.01, 0.03, 0.3, 1.0],
                            lambda c: LogisticRegression(C=c, max_iter=3000, class_weight="balanced")),
                 "svm_rbf": ("C", [1, 10, 30],
                             lambda c: SVC(C=c, kernel="rbf", probability=True,
                                           class_weight="balanced", random_state=0))}
        for m, (par, vals, fac) in grids.items():
            if m not in names:
                continue
            for v in vals:
                try:
                    run_cv(f"{m}({par}={v})", best_fs, (lambda v=v, fac=fac: fac(v)), mats[best_fs])
                except Exception as e:
                    print(f"  tuning {m} {v} FAILED: {e}")
        res = pd.DataFrame(rows).sort_values("cv_qwk", ascending=False).reset_index(drop=True)

    res.to_csv(report_dir / "model_comparison.csv", index=False)
    print("\nTop results (cross-validated QWK):")
    print(res.head(10).round(3).to_string(index=False))

    # ------------------------------------------------------------ winner
    win = res.iloc[0]
    fs, label = win["feature_set"], win["model"]
    print(f"\nSELECTED: {label} on '{fs}' features  (cv QWK {win['cv_qwk']:.3f})")
    base_name = label.split("(")[0]
    if "(" in label:  # rebuild a tuned model
        v = float(label.split("=")[1].rstrip(")"))
        v = int(v) if v == int(v) else v
        clf_factory = {"logreg": lambda: LogisticRegression(C=v, max_iter=3000, class_weight="balanced"),
                       "svm_rbf": lambda: SVC(C=v, kernel="rbf", probability=True,
                                              class_weight="balanced", random_state=0)}[base_name]
    else:
        clf_factory = all_models[base_name]
    X = mats[fs]
    oof = oofs[(fs, label)]
    refer_t, cv_sens, cv_spec = pick_refer_threshold(y[tr], oof)
    conf_t = float(np.percentile(oof.max(axis=1), 20))
    print(f"refer threshold {refer_t:.2f} (cross-val sensitivity {cv_sens:.2f}, specificity {cv_spec:.2f})")
    print(f"confidence threshold {conf_t:.2f} (least-confident 20% are flagged 'uncertain')")

    final = make_pipe(clf_factory(), X.shape[1], n_pca).fit(X[tr], y[tr])
    # unfamiliar-photo detector
    ood_pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=min(64, X.shape[1] - 1, len(tr) - 1), random_state=0))])
    Z = ood_pipe.fit_transform(X[tr])
    k = 5
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, _ = nn.kneighbors(Z, n_neighbors=k + 1)
    train_dist = dist[:, 1:].mean(axis=1)  # leave-one-out: skip the photo itself
    ood_thr = float(np.percentile(train_dist, 99.5))
    nn = NearestNeighbors(n_neighbors=k).fit(Z)

    # ------------------------------------------------------------ the ONE look at the test set
    proba_te = final.predict_proba(X[te])
    test_rep = M.full_report(y[te], proba_te, refer_t)
    print(f"\nTEST SET (never seen during training or tuning), n={len(te)}")
    print(f"  selected model : {M.one_line(test_rep)}")
    print(f"  baseline       : {M.one_line(base_rep)}")
    print(f"  always grade 0 : acc={always0:.3f}")
    print("  recall per grade (selected):", {k_: round(v_, 2) for k_, v_ in test_rep["recall_per_grade"].items()})

    # per-source results on the test part (source = original photo size)
    src = (meta["width"].astype(str) + "x" + meta["height"].astype(str)).to_numpy()[te]
    p_ref_te = proba_te[:, 2:].sum(axis=1)
    pred_te = proba_te.argmax(axis=1)
    srows = []
    for s in pd.Series(src).value_counts().index:
        m_ = src == s
        if m_.sum() < 15:
            continue
        yy = y[te][m_]
        yr = yy >= 2
        pr = p_ref_te[m_] >= refer_t
        srows.append({"source_photo_size": s, "n_test": int(m_.sum()),
                      "share_grade0": round(float((yy == 0).mean()), 2),
                      "accuracy": round(float((pred_te[m_] == yy).mean()), 3),
                      "referable_sensitivity": round(float(pr[yr].mean()), 3) if yr.any() else None,
                      "referable_specificity": round(float((~pr[~yr]).mean()), 3) if (~yr).any() else None})
    per_src = pd.DataFrame(srows)
    per_src.to_csv(report_dir / "per_source_test.csv", index=False)
    print("\nPer source (original photo size) on the test set:")
    print(per_src.to_string(index=False))

    # ------------------------------------------------------------ save everything
    plot_confusion(test_rep["confusion_matrix"], f"Test set: {label} on {fs} features",
                   fig_dir / "confusion_matrix_test.png")
    plot_comparison(res, fig_dir / "model_comparison.png")
    plot_recall(test_rep, base_rep, fig_dir / "per_grade_recall.png")

    embed_info = d["emb_info"] if (d["emb_info"] and "emb" in fs) else None
    bundle = {
        "pipeline": final, "feature_set": fs, "hand_names": feature_names(), "classes": CLASSES,
        "model_name": label, "conf_threshold": conf_t, "refer_threshold": refer_t,
        "ood": {"scaler_pca": ood_pipe, "nn": nn, "k": k, "threshold": ood_thr},
        "embed": embed_info, "created": datetime.now(timezone.utc).isoformat(),
        "trained_with": {"sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    joblib.dump(bundle, model_dir / "netra_bundle.joblib", compress=3)
    size_mb = (model_dir / "netra_bundle.joblib").stat().st_size / 1e6
    print(f"\nsaved models/netra_bundle.joblib ({size_mb:.1f} MB)")
    if size_mb > 90:
        print("WARNING: file is over 90 MB; GitHub rejects files over 100 MB. "
              "Re-run with --trees 150 or ask for help.")

    import scipy
    with open(model_dir / "pins.txt", "w", encoding="utf-8") as f:  # same versions in Docker
        f.write(f"scikit-learn=={sklearn.__version__}\nnumpy=={np.__version__}\n"
                f"scipy=={scipy.__version__}\njoblib=={joblib.__version__}\n")

    metrics = clean({
        "created": bundle["created"], "dataset": args.name, "n_usable": int(ok.sum()),
        "split": {"train": int(len(tr)), "test": int(len(te)), "seed": 42, "stratified": True},
        "selected": {"model": label, "feature_set": fs, "cv_qwk": float(win["cv_qwk"]),
                     "refer_threshold": refer_t, "conf_threshold": conf_t,
                     "unfamiliar_threshold": ood_thr, "cv_referable_sensitivity": cv_sens,
                     "cv_referable_specificity": cv_spec},
        "test": test_rep, "baseline_metadata_only": base_rep, "always_grade0_accuracy": always0,
        "comparison": res.head(15).to_dict(orient="records"),
        "per_source_test": per_src.to_dict(orient="records"),
        "quality_thresholds": qthr, "external": None,
    })
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("saved models/metrics.json, reports/model_comparison.csv, reports/per_source_test.csv, reports/figures/*.png")
    print("Next: python scripts/03_external_validation.py")


if __name__ == "__main__":
    main()
