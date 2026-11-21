"""Hand-made features for the classical statistical-ML models.

Raw pixels are far too many for logistic regression / SVM (a 512x512 colour photo is
786,432 numbers, we have ~3,600 photos), so we summarise each photo with ~40 meaningful
numbers instead. This is "feature engineering".
"""
import cv2
import numpy as np
from scipy import stats
from skimage.filters import frangi

FEATURE_SIZE = 256  # features are computed on a 256x256 copy (4x faster, same information)


def _small(prep):
    s = FEATURE_SIZE
    rgb = cv2.resize(prep["rgb"], (s, s), interpolation=cv2.INTER_AREA)
    g = cv2.resize(prep["green_clahe"], (s, s), interpolation=cv2.INTER_AREA)
    m = cv2.resize(prep["mask"].astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST).astype(bool)
    return rgb, g, m


def vessel_map(g):
    """Ridge filter (Frangi): dark thin lines = blood vessels. Returns a float map 0..1."""
    ridge = frangi(g.astype(float) / 255, sigmas=(1, 2, 3), black_ridges=True)
    top = ridge.max()
    return ridge / top if top > 0 else ridge


def extract_hand(prep):
    """Return an ordered dict of ~45 numbers describing one prepared image."""
    rgb, g, m = _small(prep)
    inner = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if inner.sum() < 50:
        inner = m
    f = {}

    # 1. intensity statistics of the contrast-enhanced green channel
    v = g[m].astype(float)
    f.update(g_mean=v.mean(), g_std=v.std(), g_skew=float(stats.skew(v)),
             g_kurt=float(stats.kurtosis(v)))
    for q in (5, 25, 50, 75, 95):
        f[f"g_p{q}"] = float(np.percentile(v, q))
    hist, _ = np.histogram(v, bins=16, range=(0, 255))
    for i, h in enumerate(hist / max(hist.sum(), 1)):
        f[f"g_hist{i}"] = float(h)

    # 2. colour statistics (red / green / blue average and spread)
    for i, c in enumerate("rgb"):
        ch = rgb[:, :, i][m].astype(float)
        f[f"{c}_mean"], f[f"{c}_std"] = ch.mean(), ch.std()

    # 3. vessel density from the ridge filter
    vm = vessel_map(g)
    rv = vm[inner]
    f["vessel_mean"] = float(rv.mean())
    f["vessel_density"] = float((rv > 0.05).mean())

    # 4. bright spots (possible exudates): white top-hat keeps small bright blobs
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    f["bright_frac"] = float((tophat[inner] > 30).mean())
    f["bright_mean"] = float(tophat[inner].mean())

    # 5. dark spots (possible haemorrhages / microaneurysms): black top-hat
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    f["dark_frac"] = float((blackhat[inner] > 25).mean())
    f["dark_mean"] = float(blackhat[inner].mean())

    # 6. edge / texture density
    f["edge_density"] = float((cv2.Canny(g, 40, 100)[inner] > 0).mean())
    return {k_: float(v_) for k_, v_ in f.items()}


def feature_names():
    """The ordered list of hand-feature names (computed once on a dummy image)."""
    dummy = np.zeros((FEATURE_SIZE, FEATURE_SIZE, 3), np.uint8)
    cv2.circle(dummy, (128, 128), 110, (180, 90, 40), -1)
    cv2.circle(dummy, (100, 90), 6, (230, 200, 120), -1)
    cv2.line(dummy, (60, 120), (190, 150), (120, 40, 20), 2)
    from .preprocess import prepare
    return list(extract_hand(prepare(dummy)).keys())
