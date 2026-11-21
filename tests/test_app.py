"""Smoke tests for the website backend. Run:  python tests/test_app.py"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from scripts.make_demo_data import fake_retina  # noqa: E402


def png_bytes(arr_bgr):
    rgb = arr_bgr[:, :, ::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def main():
    rng = np.random.default_rng(1)
    with TestClient(app) as c:
        r = c.get("/api/health")
        print("health:", r.status_code, r.json())
        assert r.status_code == 200, "model did not load (run scripts/02_train_models.py first)"

        info = c.get("/api/model-info").json()
        print("model-info:", info["model"], info["features"])

        good = png_bytes(fake_retina(3, rng))
        r = c.post("/api/analyze", files={"file": ("a.png", good, "image/png")})
        print("analyze good:", r.status_code)
        j = r.json()
        print({k: j[k] for k in j if k not in ("images", "quality", "probabilities", "disclaimer")})
        assert r.status_code == 200 and j["status"] in ("ok", "uncertain")
        assert set(j["images"]) == {"retina", "enhanced", "vessels"}
        assert abs(sum(j["probabilities"].values()) - 1) < 0.02

        black = png_bytes(np.zeros((300, 300, 3), np.uint8))
        r = c.post("/api/analyze", files={"file": ("b.png", black, "image/png")})
        print("analyze black image:", r.status_code, r.json().get("status"), "-", r.json().get("message", "")[:80])
        assert r.json()["status"] == "rejected"

        dark = (fake_retina(0, rng) * 0.08).astype(np.uint8)
        r = c.post("/api/analyze", files={"file": ("d.png", png_bytes(dark), "image/png")})
        print("analyze very dark photo:", r.json().get("status"), r.json()["quality"]["feedback"])

        r = c.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
        print("analyze text file:", r.status_code, r.json())
        assert r.status_code == 400

        r = c.post("/api/analyze", files={"file": ("big.png", b"0" * (16 * 1024 * 1024), "image/png")})
        print("analyze 16MB:", r.status_code)
        assert r.status_code == 413

        r = c.get("/")
        print("website:", r.status_code, "Netra Scan" in r.text)
        assert r.status_code == 200
    print("ALL APP TESTS PASSED")


if __name__ == "__main__":
    main()
