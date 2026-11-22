import csv, shutil, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
labels = ROOT / "data" / "aptos" / "train_with_stats.csv"
images = ROOT / "data" / "aptos" / "processed"
out = ROOT / "data" / "sample"
(out / "images").mkdir(parents=True, exist_ok=True)

random.seed(42)
by_grade = {}
with open(labels, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if (images / f"{row['id_code']}.png").exists():
            by_grade.setdefault(int(row["diagnosis"]), []).append(row["id_code"])

rows = []
for g in sorted(by_grade):
    for pid in random.sample(by_grade[g], 2):
        shutil.copy(images / f"{pid}.png", out / "images" / f"{pid}.png")
        rows.append((pid, g))

with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id_code", "diagnosis"])
    w.writerows(rows)

(out / "README.md").write_text(
    "# Sample data\n\n"
    "Ten example retina photos from APTOS 2019 (2 per grade, 0 to 4), already cropped to 512x512, "
    "with their grades in `labels.csv`.\n\n"
    "The full datasets are too large for GitHub. Download them here:\n\n"
    "- APTOS 2019: https://www.kaggle.com/c/aptos2019-blindness-detection\n"
    "- IDRiD: https://idrid.grand-challenge.org\n\n"
    "Put the full data in `data/aptos/` and `data/idrid/` as described in the main README.\n",
    encoding="utf-8",
)
print("Copied", len(rows), "photos to", out)