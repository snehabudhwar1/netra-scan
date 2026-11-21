"""Run this first. It checks your installation and your data folders and tells you what is missing.

  python scripts/00_check_setup.py
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print(f"Python {sys.version.split()[0]}")
    bad = 0
    for mod in ["numpy", "pandas", "scipy", "sklearn", "skimage", "cv2", "PIL", "matplotlib", "joblib",
                "tqdm", "torch", "torchvision", "fastapi", "uvicorn", "multipart"]:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {mod:12s} {getattr(m, '__version__', '')}")
        except Exception as e:
            bad += 1
            print(f"  MISSING  {mod:12s} -> run: pip install -r requirements.txt   ({e})")

    print("\nData folders (see README, step 3):")
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for name, img_dir, csv in [("APTOS (training)", "data/aptos/processed", "data/aptos/train_with_stats.csv"),
                               ("IDRiD (external test)", "data/idrid/images", "data/idrid/labels.csv")]:
        d, c = ROOT / img_dir, ROOT / csv
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in exts) if d.is_dir() else 0
        print(f"  {name:22s} photos in {img_dir}: {n:5d}   labels file {csv}: {'found' if c.exists() else 'NOT FOUND'}")

    print("\nTrained model:", "found" if (ROOT / "models/netra_bundle.joblib").exists() else "not yet (step 2 makes it)")
    print("\nEverything fine so far." if not bad else f"\n{bad} package(s) missing - fix those first.")


if __name__ == "__main__":
    main()
