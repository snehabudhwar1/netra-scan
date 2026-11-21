# Netra Scan

Retina screening assistant for **diabetic retinopathy (DR)**. A user uploads a retina (fundus) photo. The app:

1. **checks the photo quality** (blur, too dark, flat, retina cut off, not a retina),
2. **estimates the DR grade 0 to 4** with a confidence value,
3. says whether an **eye-doctor visit is advisable** (grade 2 or worse = "referable"),
4. says **"uncertain"** when the model is not sure or the photo looks unlike its training photos,
5. shows helper pictures (contrast-enhanced view, blood-vessel map) and a printable report.

> **Research prototype.** It is a screening aid, not a diagnosis. It has not been tested on patients.
> Medical use would need clinical validation and regulatory approval.

Course project for **CSET211 Statistical Machine Learning**.

## How it works

```
photo -> find retina, crop, square, 512x512 -> quality check -> features -> model -> result
                                                               |
                          hand features (~45 numbers: vessels, bright/dark spots, colours)
                          + embedding (2048 numbers from a pre-trained ResNet-50)
```

The learning is done by **statistical-ML models** from the course (logistic regression, KNN, naive Bayes,
SVM, decision tree, random forest), with PCA to shrink the 2048 numbers. The ResNet is used only as a fixed
"photo reader" and is not trained here.

**Data.** Training: APTOS 2019 (3,662 graded photos). External test: IDRiD (516 photos from a different clinic in India).
See `reports/` and `docs/dataset_comparison.md` for why these were chosen.

**Shortcut check.** In APTOS the *size of the image* hints at the grade (different cameras at different clinics).
A small tree that sees only size and brightness (no retina content) already scores ~71% accuracy. Every model is
compared against that baseline, and the external IDRiD test shows what happens in a new clinic.

## Folder map

| Path | What is inside |
|---|---|
| `netra/` | the code: preprocessing, quality check, features, embeddings, metrics, prediction |
| `scripts/` | the steps you run: `00` check setup, `01` extract features, `02` train + compare, `03` external test |
| `app/` | the website: `main.py` (FastAPI backend) and `static/` (HTML, CSS, JavaScript) |
| `models/` | the trained model and its numbers (made by step 02) |
| `reports/` | comparison tables and charts for the report |
| `data/` | your photos (not uploaded to GitHub) |
| `tests/` | `test_app.py` checks the backend |
| `Dockerfile`, `deploy/` | for putting the site online |

## Setup (Windows, PowerShell, inside VS Code)

Open the project folder in VS Code, then **Terminal > New Terminal** and run these one at a time:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/00_check_setup.py
```

If PowerShell says "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and activate again.
You know the environment is active when the prompt starts with `(.venv)`.

## Put the data in place

```
data/aptos/processed/<photo_id>.png      the 512x512 photos from the Kaggle notebook output
data/aptos/train_with_stats.csv          table with columns id_code, diagnosis, width, height, ...
data/idrid/images/IDRiD_001.jpg ...      IDRiD "disease grading" photos
data/idrid/labels.csv                    IDRiD grading labels (columns are detected automatically)
```

`python scripts/00_check_setup.py` shows how many photos and labels it finds.

## Run order

```
python scripts/01_extract_features.py --name aptos --images data/aptos/processed --labels data/aptos/train_with_stats.csv
python scripts/02_train_models.py
python scripts/01_extract_features.py --name idrid --images data/idrid/images --labels data/idrid/labels.csv
python scripts/03_external_validation.py
python -m uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000 in your browser. Add `--size 256` to both step-01 commands for a faster (a bit less accurate) run.
Step 01 can be stopped with Ctrl+C and continued later.

### Quick practice run on FAKE photos (about 3 minutes, results are meaningless)

```
python scripts/make_demo_data.py --out data/demo --per-grade 24
python scripts/01_extract_features.py --name demo --images data/demo/images --labels data/demo/labels.csv --size 224 --features-dir features_demo
python scripts/02_train_models.py --name demo --folds 3 --features-dir features_demo --model-dir models_demo --report-dir reports_demo
```

Never use numbers from the fake photos in the report.

## Tests

```
python tests/test_app.py
```

## Honest limits

- Trained on one public dataset; other cameras and clinics usually give lower scores (we measure this on IDRiD).
- Mild (grade 1) and very advanced cases are the hardest to separate.
- The quality check thresholds come from APTOS photos only.
- The helper pictures are image filters, not an explanation of the model's decision.
- Not clinically validated, not a medical device.
