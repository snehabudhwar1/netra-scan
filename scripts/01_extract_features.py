"""Step 1: turn every photo into numbers (features).

Two kinds of numbers are made for each photo:
  * hand features  - 36 numbers we designed (vessels, bright/dark spots, colours...)
  * embedding      - 2048 numbers from a pre-trained ResNet-50 (it reads the photo for us)
plus quality numbers (sharpness, brightness, contrast...) for the quality check.

Examples (run from the project folder):
  APTOS (training photos):
    python scripts/01_extract_features.py --name aptos --images data/aptos/processed --labels data/aptos/train_with_stats.csv
  IDRiD (external test photos):
    python scripts/01_extract_features.py --name idrid --images data/idrid/images --labels data/idrid/labels.csv
  Quick test on 40 photos first:
    python scripts/01_extract_features.py --name aptos --images ... --labels ... --limit 40

Safe to stop (Ctrl+C) and run again: finished embedding chunks are skipped.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from netra import data as D  # noqa: E402
from netra.features import extract_hand  # noqa: E402
from netra.preprocess import load_rgb, prepare  # noqa: E402
from netra.quality import measure  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_image(folder, stem):
    for ext in EXTS:
        for e in (ext, ext.upper()):
            p = folder / f"{stem}{e}"
            if p.exists():
                return p
    return None


def read_labels(path, id_col, label_col):
    """Read the labels CSV. Column names are guessed if you do not pass them."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    def guess(given, keys):
        if given:
            if given not in df.columns:
                raise SystemExit(f"Column '{given}' not found. Columns are: {list(df.columns)}")
            return given
        for k in keys:
            for lc, orig in low.items():
                if k in lc:
                    return orig
        raise SystemExit(f"Could not guess the column. Columns are: {list(df.columns)}. "
                         f"Use --id-col / --label-col.")

    idc = guess(id_col, ["id_code", "image name", "image", "id"])
    labc = guess(label_col, ["diagnosis", "retinopathy grade", "grade", "label"])
    out = pd.DataFrame({"id": df[idc].astype(str).str.strip(),
                        "label": pd.to_numeric(df[labc], errors="coerce")})
    for c in ("width", "height"):
        if c in df.columns:
            out[c] = df[c]
    out["id"] = out["id"].str.replace(r"\.(png|jpe?g|tiff?|bmp)$", "", regex=True, case=False)
    return out.dropna(subset=["label"]).astype({"label": int}).reset_index(drop=True)


def tab_one(path):
    """Worker: quality numbers + hand features for one photo (runs in parallel)."""
    try:
        prep = prepare(load_rgb(path))
        row = {"ok": bool(prep["ok"])}
        row.update({f"q_{k}": v for k, v in measure(prep).items()})
        if prep["ok"]:
            row.update(extract_hand(prep))
        return row
    except Exception as e:  # one bad photo must not stop the other thousands
        return {"ok": False, "error": str(e)[:120]}


def load_prepared(args_):
    path, size = args_
    try:
        prep = prepare(load_rgb(path), size=size)
        return prep["rgb"]
    except Exception:
        return np.zeros((size, size, 3), np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="short name for this dataset, e.g. aptos or idrid")
    ap.add_argument("--images", required=True, help="folder with the photos")
    ap.add_argument("--labels", required=True, help="CSV with photo names and grades")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N photos (quick test)")
    ap.add_argument("--size", type=int, default=384, help="photo size fed to ResNet (384 = better, 256 = faster)")
    ap.add_argument("--no-emb", action="store_true", help="skip the ResNet embeddings")
    ap.add_argument("--no-hand", action="store_true", help="skip hand features / quality (not recommended)")
    ap.add_argument("--workers", type=int, default=-1, help="CPU workers for hand features (-1 = all)")
    ap.add_argument("--features-dir", default="features")
    args = ap.parse_args()

    name = args.name + ("_limit" if args.limit else "")
    folder = Path(args.images)
    if not folder.is_dir():
        raise SystemExit(f"Image folder not found: {folder}")
    out_dir = Path(args.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = D.paths(name, out_dir)

    labels = read_labels(args.labels, args.id_col, args.label_col)
    if args.limit:  # take a stratified-ish small sample so every grade appears
        labels = labels.groupby("label").head(max(args.limit // 5, 1)).reset_index(drop=True)
    paths_ = [find_image(folder, i) for i in labels["id"]]
    missing = sum(x is None for x in paths_)
    if missing:
        print(f"WARNING: {missing} of {len(labels)} labelled photos were not found in {folder} and are skipped.")
    keep = [x is not None for x in paths_]
    labels = labels[keep].reset_index(drop=True)
    paths_ = [x for x in paths_ if x is not None]
    if not paths_:
        raise SystemExit("No photos matched the labels. Check --images and --labels (see README).")
    print(f"{len(paths_)} photos, grades: {labels['label'].value_counts().sort_index().to_dict()}")

    # original photo size (a 'source' clue). Use the CSV columns if present, else read the file header.
    if "width" not in labels or "height" not in labels:
        sizes = []
        for pth in paths_:
            with Image.open(pth) as im:
                sizes.append(im.size)
        labels["width"] = [s[0] for s in sizes]
        labels["height"] = [s[1] for s in sizes]
    labels.to_csv(p["meta"], index=False)

    # ---------------------------------------------------------------- hand + quality
    if not args.no_hand:
        t0 = time.time()
        print("Pass 1/2: quality numbers + hand features (all CPU cores)...")
        rows = Parallel(n_jobs=args.workers, batch_size=8)(
            delayed(tab_one)(pth) for pth in tqdm(paths_, desc="hand features"))
        tab = pd.DataFrame(rows)
        tab.insert(0, "id", labels["id"].values)
        if "error" in tab:
            print(f"  {int((~tab['ok']).sum())} photos failed, e.g. {tab['error'].dropna().head(3).tolist()}")
        tab.to_csv(p["tab"], index=False)
        print(f"  saved {p['tab']}  ({(time.time() - t0) / 60:.1f} min)")

    # ---------------------------------------------------------------- embeddings
    if not args.no_emb:
        from netra.embed import ARCH, EMBED_DIM, Embedder
        chunk_dir = out_dir / f"{name}_emb_chunks_{args.size}"
        chunk_dir.mkdir(exist_ok=True)
        print(f"Pass 2/2: ResNet-50 embeddings at {args.size}px (first run downloads ~100 MB of weights)...")
        emb = Embedder(size=args.size)
        chunk = 64
        n = len(paths_)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ci, start in enumerate(tqdm(range(0, n, chunk), desc="embeddings (chunks of 64)")):
                f = chunk_dir / f"chunk_{ci:05d}.npy"
                if f.exists():
                    continue
                batch = paths_[start:start + chunk]
                imgs = list(pool.map(load_prepared, [(b, args.size) for b in batch]))
                np.save(f, emb(imgs))
        allv = np.concatenate([np.load(chunk_dir / f"chunk_{ci:05d}.npy")
                               for ci in range((n + chunk - 1) // chunk)], axis=0)
        assert allv.shape == (n, EMBED_DIM), allv.shape
        np.save(p["emb"], allv)
        with open(p["emb_info"], "w", encoding="utf-8") as fh:
            json.dump({"arch": ARCH, "size": args.size, "n": n}, fh)
        print(f"  saved {p['emb']} {allv.shape}  ({(time.time() - t0) / 60:.1f} min)")

    print("Done. Next: python scripts/02_train_models.py" if args.name == "aptos"
          else "Done. Next: python scripts/03_external_validation.py")


if __name__ == "__main__":
    main()
