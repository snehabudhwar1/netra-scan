"""The whole pipeline for ONE uploaded photo:

    photo -> prepare -> quality check -> features -> model -> confidence & safety checks
          -> advice + helper pictures

The trained model lives in models/netra_bundle.joblib (made by scripts/02_train_models.py).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import __version__
from .explain import DISCLAIMER, GRADES, advice, evidence_images
from .features import extract_hand
from .preprocess import prepare
from .quality import assess, load_thresholds


class NetraModel:
    def __init__(self, model_dir="models"):
        self.dir = Path(model_dir)
        bundle_path = self.dir / "netra_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {bundle_path}. Run scripts/02_train_models.py first.")
        self.bundle = joblib.load(bundle_path)
        self.feature_set = self.bundle["feature_set"]
        self.thresholds = load_thresholds(self.dir / "quality_thresholds.json")
        self.embedder = None
        if "emb" in self.feature_set:
            from .embed import Embedder
            self.embedder = Embedder(size=self.bundle["embed"]["size"])
        self.metrics = self.bundle.get("metrics", {})
        metrics_file = self.dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                self.metrics = json.load(f)

    # ------------------------------------------------------------------ features
    def _features(self, prep):
        parts = []
        if "hand" in self.feature_set:
            hand = extract_hand(prep)
            parts.append(np.array([hand[k] for k in self.bundle["hand_names"]], dtype=np.float64))
        if "emb" in self.feature_set:
            parts.append(self.embedder([prep["rgb"]])[0].astype(np.float64))
        return np.hstack(parts)[None, :]

    def predict_features(self, X):
        """X: (N, D) -> (N, 5) probabilities, plus unfamiliarity distance per row."""
        proba = self.bundle["pipeline"].predict_proba(X)
        ood = self.bundle["ood"]
        Z = ood["scaler_pca"].transform(X)
        dist, _ = ood["nn"].kneighbors(Z, n_neighbors=ood["k"])
        return proba, dist.mean(axis=1)

    # ------------------------------------------------------------------ main entry
    def analyze(self, img_rgb, with_images=True):
        t0 = time.time()
        base = {"model_version": f"netra-{__version__}-{self.bundle['model_name']}-{self.feature_set}",
                "disclaimer": DISCLAIMER}
        prep = prepare(img_rgb)
        quality = assess(prep, self.thresholds)
        base["quality"] = {"verdict": quality["verdict"], "checks": quality["checks"],
                           "feedback": quality["feedback"]}

        if quality["verdict"] == "reject":
            return {**base, "status": "rejected",
                    "message": "This photo is not good enough to grade. " + " ".join(quality["feedback"]),
                    "seconds": round(time.time() - t0, 2)}

        X = self._features(prep)
        proba, dist = self.predict_features(X)
        proba, dist = proba[0], float(dist[0])
        grade = int(np.argmax(proba))
        confidence = float(proba[grade])
        p_ref = float(proba[2:].sum())
        referable = p_ref >= self.bundle["refer_threshold"]
        unfamiliar = dist > self.bundle["ood"]["threshold"]
        low_conf = confidence < self.bundle["conf_threshold"]

        reasons = []
        if unfamiliar:
            reasons.append("This photo looks different from the photos the model learned from.")
        if low_conf:
            reasons.append("The model is not confident about the exact grade.")
        if quality["verdict"] == "borderline":
            reasons.append("The photo quality is borderline.")
        status = "uncertain" if (unfamiliar or low_conf) else "ok"

        out = {
            **base,
            "status": status,
            "grade": grade,
            "grade_name": GRADES[grade][0],
            "grade_text": GRADES[grade][1],
            "confidence": round(confidence, 3),
            "probabilities": {GRADES[i][0]: round(float(p), 3) for i, p in enumerate(proba)},
            "referable": bool(referable),
            "referable_probability": round(p_ref, 3),
            "refer_threshold": round(float(self.bundle["refer_threshold"]), 3),
            "uncertain_reasons": reasons,
            "advice": advice(grade, referable, status),
            "seconds": round(time.time() - t0, 2),
        }
        if with_images:
            out["images"] = evidence_images(prep)
        return out
