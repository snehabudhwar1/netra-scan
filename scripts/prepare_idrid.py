"""Prepare IDRiD for Netra Scan.

The IDRiD download has a training part (413 photos) and a testing part (103 photos). Both parts
use the SAME photo names (IDRiD_001 ...). This script copies all 516 photos into one folder and
adds a prefix (train_ / test_) so no name is used twice, and writes one labels.csv.

Run (replace the path with the folder that contains '1. Original Images'):
    python scripts/prepare_idrid.py --src "D:\\Downloads\\B. Disease Grading\\B. Disease Grading"
Result:
    data/idrid/images/train_IDRiD_001.jpg ...  and  data/idrid/labels.csv
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def read_grades(csv_path):
    """Return a table with columns name, grade from one IDRiD grading CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    name_col = df.columns[0]
    grade_col = next((c for c in df.columns if "retinopathy" in c.lower() or c.lower() == "grade"), None)
    if grade_col is None:
        raise SystemExit(f"No grade column in {csv_path}. Columns are: {list(df.columns)}")
    out = pd.DataFrame({"name": df[name_col].astype(str).str.strip(),
                        "grade": pd.to_numeric(df[grade_col], errors="coerce")})
    out = out.dropna(subset=["grade"])
    out["grade"] = out["grade"].astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder that contains '1. Original Images' and '2. Groundtruths'")
    ap.add_argument("--out", default="data/idrid", help="where to put the prepared data")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for part, prefix in (("training", "train"), ("testing", "test")):
        folders = [d for d in src.rglob("*")
                   if d.is_dir() and part in d.name.lower()
                   and any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file())]
        csvs = [f for f in src.rglob("*.csv") if part in f.name.lower()]
        if not folders or not csvs:
            raise SystemExit(f"Could not find the {part} photos or the {part} labels CSV inside {src}.\n"
                             f"Check that --src is the folder that contains '1. Original Images'.")
        photos = {f.stem: f for f in folders[0].iterdir() if f.suffix.lower() in IMG_EXT}
        grades = read_grades(csvs[0])
        copied = 0
        for _, r in grades.iterrows():
            f = photos.get(r["name"])
            if f is None:
                continue
            new_name = f"{prefix}_{f.name}"
            if not (img_out / new_name).exists():
                shutil.copy2(f, img_out / new_name)
            rows.append({"id_code": f"{prefix}_{f.stem}", "diagnosis": r["grade"], "split": prefix})
            copied += 1
        print(f"{part}: {len(photos)} photos found, {len(grades)} labels, {copied} matched")

    table = pd.DataFrame(rows)
    table.to_csv(out / "labels.csv", index=False)
    print(f"\nSaved {len(table)} photos in {img_out} and labels in {out / 'labels.csv'}")
    print("Grades:", table["diagnosis"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
