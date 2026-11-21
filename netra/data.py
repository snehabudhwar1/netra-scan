"""Reading and writing the feature tables made by scripts/01_extract_features.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = Path("features")
FEATURE_SETS = ("hand", "emb", "hand+emb")


def paths(name, features_dir=FEATURES_DIR):
    d = Path(features_dir)
    return {
        "meta": d / f"{name}_meta.csv",       # id, label, width, height
        "tab": d / f"{name}_tab.csv",         # quality numbers + hand features
        "emb": d / f"{name}_emb.npy",         # (N, 2048) pretrained-network features
        "emb_info": d / f"{name}_emb_info.json",
    }


def load(name, features_dir=FEATURES_DIR):
    """Returns dict(meta, tab, emb, emb_info). emb is None if it was not extracted."""
    p = paths(name, features_dir)
    if not p["meta"].exists() or not p["tab"].exists():
        raise FileNotFoundError(
            f"Feature files for '{name}' not found in {features_dir}/. "
            f"Run scripts/01_extract_features.py first.")
    meta = pd.read_csv(p["meta"])
    tab = pd.read_csv(p["tab"])
    emb = np.load(p["emb"]) if p["emb"].exists() else None
    info = json.load(open(p["emb_info"], encoding="utf-8")) if p["emb_info"].exists() else None
    return {"meta": meta, "tab": tab, "emb": emb, "emb_info": info}


def hand_columns(tab):
    return [c for c in tab.columns if c not in ("id", "ok", "error") and not c.startswith("q_")]


def quality_columns(tab):
    return [c for c in tab.columns if c.startswith("q_")]


def build_matrix(d, feature_set, keep=None):
    """Stack the chosen features into one (N, D) array. keep = boolean row filter."""
    tab = d["tab"]
    parts = []
    if "hand" in feature_set:
        parts.append(tab[hand_columns(tab)].to_numpy(dtype=np.float64))
    if "emb" in feature_set:
        if d["emb"] is None:
            raise ValueError("This feature set needs embeddings, but none were extracted.")
        parts.append(d["emb"].astype(np.float64))
    X = np.hstack(parts)
    return X if keep is None else X[keep]


def valid_rows(d):
    """Rows where the photo was read successfully (and, if present, embedded)."""
    ok = d["tab"]["ok"].astype(bool).to_numpy().copy()
    finite = np.isfinite(d["tab"][hand_columns(d["tab"])].to_numpy(dtype=float)).all(axis=1)
    ok = ok & finite
    if d.get("emb") is not None:
        ok = ok & np.isfinite(d["emb"]).all(axis=1)
    return ok
