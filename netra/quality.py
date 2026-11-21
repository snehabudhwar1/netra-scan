"""Step 2: is this photo good enough to grade?

Every check is a plain number measured inside the retina. The thresholds are NOT
guesses: scripts/02_train_models.py measures them on the APTOS photos
(5th/1st percentiles) and saves them to models/quality_thresholds.json.
The defaults below are only used until that file exists.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# metric -> [reject_below, warn_below]
DEFAULT_THRESHOLDS = {
    "sharpness": [20.0, 60.0],    # variance of the Laplacian (low = blurry)
    "brightness": [35.0, 60.0],   # mean green level (low = too dark)   <- from our EDA
    "contrast": [8.0, 22.0],      # std of green level (low = flat)     <- from our EDA
    "fov": [0.55, 0.70],          # how much of the round retina is visible
    "max_brightness": [220.0, 190.0],  # [reject_above, warn_above] -> overexposed
}

LABELS = {
    "sharpness": "Sharpness",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "fov": "Retina fully visible",
    "exposure": "Not overexposed",
    "retina_like": "Looks like a retina photo",
}

MESSAGES = {
    "sharpness": "The photo is blurry. Refocus and take it again.",
    "brightness": "The photo is too dark. Use more light or wait for the pupil to widen.",
    "contrast": "The photo looks flat or washed out. Check focus and clarity.",
    "fov": "Part of the retina is cut off. Re-centre the camera and retake.",
    "exposure": "The photo is overexposed. Reduce the light and retake.",
    "no_retina": "No retina could be found in this image. Please upload a retinal (fundus) photo.",
    "retina_like": "This does not look like a typical retinal photo (no dark border or unusual colours).",
}


def load_thresholds(path):
    """Read saved thresholds if the file exists, else use the defaults."""
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            saved = json.load(f)
        merged = {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}
        merged.update({k: list(v) for k, v in saved.items() if k in merged})
        return merged
    return {k: list(v) for k, v in DEFAULT_THRESHOLDS.items()}


def measure(prep):
    """Raw quality numbers for one prepared image (output of preprocess.prepare)."""
    nan = float("nan")
    if not prep["ok"]:
        return {"sharpness": nan, "brightness": nan, "contrast": nan, "fov": 0.0,
                "fill_ratio": prep["fill_ratio"], "red_green_ratio": nan}
    g, mask, rgb = prep["green"], prep["mask"], prep["rgb"]
    inner = cv2.erode(mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    if inner.sum() < 100:
        inner = mask
    lap = cv2.Laplacian(g, cv2.CV_64F)
    r_mean = float(rgb[:, :, 0][inner].mean())
    g_mean = float(rgb[:, :, 1][inner].mean())
    return {
        "sharpness": float(lap[inner].var()),
        "brightness": g_mean,
        "contrast": float(g[inner].std()),
        "fov": prep["fov"],
        "fill_ratio": prep["fill_ratio"],
        "red_green_ratio": r_mean / max(g_mean, 1e-6),
    }


def _low_check(name, value, thr):
    """Status for a 'higher is better' metric."""
    if value < thr[0]:
        return "fail"
    if value < thr[1]:
        return "warn"
    return "ok"


def assess(prep, thresholds=None):
    """Return {'verdict': good|borderline|reject, 'metrics', 'checks', 'feedback'}.

    checks   = list of dicts for the website checklist
    feedback = plain-language advice for every check that did not pass
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    m = measure(prep)
    if not prep["ok"]:
        too_dark = prep.get("orig_mean", 255) < 25
        return {"verdict": "reject", "metrics": m, "checks": [],
                "feedback": [MESSAGES["brightness"] if too_dark else MESSAGES["no_retina"]]}

    checks = []

    def add(key, label, status, value, message_key):
        checks.append({"key": key, "label": label, "status": status,
                       "value": None if value is None else round(float(value), 2),
                       "message": MESSAGES[message_key] if status != "ok" else ""})

    for key in ("sharpness", "brightness", "contrast", "fov"):
        add(key, LABELS[key], _low_check(key, m[key], thr[key]), m[key], key)

    # overexposure: higher is worse
    rej_hi, warn_hi = thr["max_brightness"]
    exp_status = "fail" if m["brightness"] > rej_hi else "warn" if m["brightness"] > warn_hi else "ok"
    add("exposure", LABELS["exposure"], exp_status, m["brightness"], "exposure")

    # is it even a retina photo? (a round photo has dark corners; retinas are red/orange)
    suspicious = prep["fill_ratio"] > 0.95 or m["red_green_ratio"] < 1.0
    add("retina_like", LABELS["retina_like"], "warn" if suspicious else "ok",
        m["red_green_ratio"], "retina_like")

    statuses = [c["status"] for c in checks]
    verdict = "reject" if "fail" in statuses else "borderline" if "warn" in statuses else "good"
    feedback = [c["message"] for c in checks if c["status"] != "ok"]
    return {"verdict": verdict, "metrics": m, "checks": checks, "feedback": feedback}


def calibrate_thresholds(q_df):
    """Learn thresholds from a table of quality numbers (columns sharpness, brightness,
    contrast). reject = 1st percentile, warn = 5th percentile of the training photos.
    Brightness/contrast rejects stay at the values found in our EDA (35 and 8).
    The visible-retina rule (fov) is NOT learned: the APTOS photos were already cropped to a
    full circle, so their percentiles (about 0.99) would reject any real photo that is even
    slightly cut off. We keep the fixed rule: reject below 55%, warn below 70% visible."""
    q = q_df.dropna()
    p = lambda col, pc: float(np.percentile(q[col], pc))
    return {
        "sharpness": [round(p("sharpness", 1), 2), round(p("sharpness", 5), 2)],
        "brightness": [35.0, round(max(p("brightness", 5), 40.0), 2)],
        "contrast": [8.0, round(max(p("contrast", 5), 10.0), 2)],
        "fov": list(DEFAULT_THRESHOLDS["fov"]),
        "max_brightness": [220.0, 190.0],
    }
