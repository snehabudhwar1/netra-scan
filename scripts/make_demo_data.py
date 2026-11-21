"""Makes FAKE retina-like pictures so you can test the whole pipeline in minutes.

!! These pictures are synthetic (drawn by code). Never use results from them in your
   report. They only prove that the code runs from start to finish.

  python scripts/make_demo_data.py --out data/demo --per-grade 30
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def fake_retina(grade, rng, size=None):
    s = size or int(rng.choice([480, 640, 800]))
    h, w = s, int(s * rng.choice([1.0, 1.0, 1.33]))
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy, r = w // 2, h // 2, int(min(h, w) * 0.46)
    base = np.array([rng.integers(150, 200), rng.integers(70, 110), rng.integers(25, 60)])  # RGB orange-red
    cv2.circle(img, (cx, cy), r, tuple(int(x) for x in base[::-1]), -1)
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 1.0 - 0.35 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (r ** 2))
    img = (img * np.clip(shade, 0.4, 1.0)[..., None]).astype(np.uint8)
    # optic disc
    cv2.circle(img, (cx + r // 2, cy - r // 8), r // 9, (90, 200, 235), -1)
    # vessels: dark curvy lines from the disc
    for _ in range(14):
        pts = [(cx + r // 2, cy - r // 8)]
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(8):
            ang += rng.normal(0, 0.35)
            x0, y0 = pts[-1]
            pts.append((int(x0 + r * 0.14 * np.cos(ang)), int(y0 + r * 0.14 * np.sin(ang))))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (25, 35, 90), int(rng.integers(1, 4)))
    # lesions grow with grade: dark red spots and bright yellow spots
    for _ in range(int(grade * rng.integers(6, 14))):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * 0.8)
        cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))), int(rng.integers(2, 5)), (20, 20, 120), -1)
    for _ in range(int(max(grade - 1, 0) * rng.integers(5, 12))):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * 0.8)
        cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))), int(rng.integers(2, 5)), (80, 210, 240), -1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.6))
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)  # BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--per-grade", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for g in range(5):
        for i in range(a.per_grade):
            name = f"demo_g{g}_{i:03d}"
            cv2.imwrite(str(out / "images" / f"{name}.png"), fake_retina(g, rng))
            rows.append({"id_code": name, "diagnosis": g})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    print(f"Wrote {len(rows)} FAKE photos to {out}/images and {out}/labels.csv")


if __name__ == "__main__":
    main()
