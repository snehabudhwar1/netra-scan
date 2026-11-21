"""Step 3: the real-world test. Score the saved model on photos from a DIFFERENT dataset.

Why this matters: a model that only works on photos like the ones it trained on cannot be
trusted in a new clinic. IDRiD comes from another eye clinic (and camera) in India, so
this number is much closer to "what would happen in real use" than the internal test.

Before running, extract IDRiD features with the SAME photo size used for APTOS:
  python scripts/01_extract_features.py --name idrid --images data/idrid/images --labels data/idrid/labels.csv --size 384

Then:
  python scripts/03_external_validation.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra import data as D  # noqa: E402
from netra import metrics as M  # noqa: E402
from netra.plots import plot_confusion  # noqa: E402
from netra.quality import load_thresholds  # noqa: E402


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="idrid")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    model_dir, report_dir = Path(args.model_dir), Path(args.report_dir)
    bundle = joblib.load(model_dir / "netra_bundle.joblib")
    fs = bundle["feature_set"]
    d = D.load(args.name, args.features_dir)

    if "emb" in fs:
        if d["emb"] is None:
            raise SystemExit("The model uses embeddings but none were extracted for this dataset.")
        want, got = (bundle["embed"] or {}).get("size"), (d["emb_info"] or {}).get("size")
        if want and got and want != got:
            raise SystemExit(f"Photo size mismatch: model trained with --size {want}, "
                             f"this dataset was extracted with --size {got}. Re-run step 1 with --size {want}.")

    ok = D.valid_rows(d)
    meta, tab = d["meta"][ok].reset_index(drop=True), d["tab"][ok].reset_index(drop=True)
    d2 = {"tab": tab, "emb": None if d["emb"] is None else d["emb"][ok], "meta": meta}
    X = D.build_matrix(d2, fs)
    y = meta["label"].to_numpy()
    print(f"{args.name}: {len(y)} usable photos, grades {dict(zip(*np.unique(y, return_counts=True)))}")

    proba = bundle["pipeline"].predict_proba(X)
    Z = bundle["ood"]["scaler_pca"].transform(X)
    dist, _ = bundle["ood"]["nn"].kneighbors(Z, n_neighbors=bundle["ood"]["k"])
    unfamiliar = dist.mean(axis=1) > bundle["ood"]["threshold"]
    rep = M.full_report(y, proba, bundle["refer_threshold"])
    print("\nEXTERNAL TEST (different clinic, never used in training):")
    print("  ", M.one_line(rep))
    print("   recall per grade:", {k: round(v, 2) for k, v in rep["recall_per_grade"].items()})

    # the same model, but only on photos the app would actually accept
    thr = load_thresholds(model_dir / "quality_thresholds.json")
    verdicts = []
    for _, r in tab.iterrows():
        # reuse the stored quality numbers instead of re-reading the photos
        m = {k: r[f"q_{k}"] for k in ("sharpness", "brightness", "contrast", "fov")}
        bad = (m["sharpness"] < thr["sharpness"][0] or m["brightness"] < thr["brightness"][0]
               or m["contrast"] < thr["contrast"][0] or m["fov"] < thr["fov"][0]
               or m["brightness"] > thr["max_brightness"][0])
        verdicts.append("reject" if bad else "pass")
    accepted = np.array(verdicts) == "pass"
    print(f"\nQuality gate would reject {int((~accepted).sum())} of {len(y)} photos "
          f"({(~accepted).mean()*100:.0f}%)")
    print(f"'Unfamiliar photo' flag would fire on {int(unfamiliar.sum())} of {len(y)} "
          f"({unfamiliar.mean()*100:.0f}%)")
    rep_acc = None
    if accepted.sum() > 20:
        rep_acc = M.full_report(y[accepted], proba[accepted], bundle["refer_threshold"])
        print("   on accepted photos only:", M.one_line(rep_acc))

    # compare with the internal test from step 2
    mf = model_dir / "metrics.json"
    metrics = json.load(open(mf, encoding="utf-8")) if mf.exists() else {}
    internal = metrics.get("test")
    if internal:
        print(f"\nINTERNAL test (APTOS)  : {M.one_line(internal)}")
        print(f"EXTERNAL test ({args.name}) : {M.one_line(rep)}")
        drop = internal["qwk"] - rep["qwk"]
        print(f"QWK change when moving to a new clinic: {-drop:+.3f}")

    metrics["external"] = clean({
        "dataset": args.name, "n": int(len(y)), "report": rep, "report_accepted_only": rep_acc,
        "quality_rejected_share": float((~accepted).mean()),
        "unfamiliar_share": float(unfamiliar.mean()),
    })
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(rep["confusion_matrix"], f"External test: {args.name} ({len(y)} photos)",
                   fig_dir / "confusion_matrix_external.png")
    print("\nsaved models/metrics.json (external block) and reports/figures/confusion_matrix_external.png")


if __name__ == "__main__":
    main()
