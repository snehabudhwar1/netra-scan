"""Netra Scan backend (FastAPI).

  POST /api/analyze     upload one retina photo -> JSON result
  GET  /api/health      is the server and the model ready?
  GET  /api/model-info  honest numbers: how well did the model do on test data?
  GET  /                the website

Privacy: the uploaded photo is processed in memory and never written to disk or logged.
"""
import io
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from netra import __version__  # noqa: E402
from netra.explain import DISCLAIMER  # noqa: E402
from netra.predict import NetraModel  # noqa: E402

MODEL_DIR = Path(os.environ.get("NETRA_MODEL_DIR", ROOT / "models"))
MAX_BYTES = 15 * 1024 * 1024  # 15 MB upload limit
Image.MAX_IMAGE_PIXELS = 120_000_000  # refuse "decompression bomb" images

state = {"model": None, "error": None}
gate = threading.Semaphore(2)  # at most 2 analyses at the same time (memory safety)


@asynccontextmanager
async def lifespan(app):
    try:
        state["model"] = NetraModel(MODEL_DIR)
        # warm-up so the first real user does not wait for lazy initialisation
        dummy = np.zeros((256, 256, 3), np.uint8)
        state["model"].analyze(dummy, with_images=False)
    except FileNotFoundError as e:
        state["error"] = str(e)
    except Exception as e:  # keep the site up and show the reason on /api/health
        state["error"] = f"Model failed to load: {e}"
    yield


app = FastAPI(title="Netra Scan", version=__version__, lifespan=lifespan)


@app.get("/api/health")
def health():
    ready = state["model"] is not None
    return JSONResponse({"ready": ready, "error": state["error"], "version": __version__},
                        status_code=200 if ready else 503)


@app.get("/api/model-info")
def model_info():
    m = state["model"]
    if m is None:
        raise HTTPException(503, state["error"] or "Model not loaded")
    mt = m.metrics or {}
    test, base, ext = mt.get("test"), mt.get("baseline_metadata_only"), mt.get("external")

    def clean(v):  # JSON cannot hold NaN
        return None if isinstance(v, float) and v != v else v

    def pick(r):
        if not r:
            return None
        return {
            "n": r["n"], "accuracy": clean(r["accuracy"]), "qwk": clean(r["qwk"]),
            "referable_sensitivity": clean(r["referable"]["sensitivity"]),
            "referable_specificity": clean(r["referable"]["specificity"]),
            "recall_per_grade": {k: clean(v) for k, v in (r.get("recall_per_grade") or {}).items()},
        }

    return {
        "version": __version__,
        "model": m.bundle["model_name"],
        "features": m.feature_set,
        "trained_on": mt.get("dataset"),
        "internal_test": pick(test),
        "baseline_metadata_only": pick(base),
        "external_test": None if not ext else {"dataset": ext.get("dataset"), **pick(ext["report"])},
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    model = state["model"]
    if model is None:
        raise HTTPException(503, state["error"] or "The model is not loaded yet. Try again in a minute.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 15 MB. Please upload a smaller photo.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "This file could not be read as an image. Please upload a JPG, PNG or TIFF photo.")
    if max(img.size) > 5000:
        img.thumbnail((5000, 5000))
    arr = np.array(img)
    with gate:
        try:
            result = model.analyze(arr)
        except Exception:
            raise HTTPException(500, "Something went wrong while analysing this photo. Please try another one.")
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


# the website (must be mounted last so /api/* routes win)
app.mount("/", StaticFiles(directory=str(ROOT / "app" / "static"), html=True), name="static")
