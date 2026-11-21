"""Scores used everywhere in the project.

Why not just accuracy? Half of the photos are grade 0, so "always say grade 0" already
scores ~49%. We therefore also report:
  * QWK  (quadratic weighted kappa): the standard DR-grading score; 1.0 = perfect,
         0 = no better than chance, and being 3 grades wrong costs more than 1 grade.
  * recall per grade: of all photos that REALLY have grade g, how many did we find?
  * referable DR (grade >= 2 = "see an eye doctor"): sensitivity and specificity.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)

CLASSES = [0, 1, 2, 3, 4]
REFERABLE_FROM = 2


def referable_scores(y_true, p_ref, threshold=0.5):
    """Sensitivity / specificity / AUC for 'needs referral' (grade >= 2)."""
    y = (np.asarray(y_true) >= REFERABLE_FROM).astype(int)
    pred = (np.asarray(p_ref) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    out = {
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "threshold": float(threshold),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }
    out["auc"] = float(roc_auc_score(y, p_ref)) if len(set(y)) == 2 else float("nan")
    return out


def full_report(y_true, proba, refer_threshold=0.5):
    """Everything we report, from true grades and a (N, 5) probability matrix."""
    y_true = np.asarray(y_true)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    recall = {str(c): (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan"))
              for i, c in enumerate(CLASSES)}
    p_ref = np.asarray(proba)[:, REFERABLE_FROM:].sum(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "qwk": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=CLASSES, zero_division=0)),
        "within_one_grade": float((np.abs(pred - y_true) <= 1).mean()),
        "recall_per_grade": recall,
        "referable": referable_scores(y_true, p_ref, refer_threshold),
        "confusion_matrix": cm.tolist(),
    }


def one_line(rep):
    r = rep["referable"]
    return (f"acc={rep['accuracy']:.3f}  qwk={rep['qwk']:.3f}  macroF1={rep['macro_f1']:.3f}  "
            f"referable sens={r['sensitivity']:.3f} spec={r['specificity']:.3f}")
