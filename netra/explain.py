"""Turns a prediction into something a health worker can read and act on.

Important honesty rules built into the wording:
  * "referable" = grade 2 or worse = "should see an eye doctor". It is a screening
    flag, NOT a diagnosis.
  * We show the model's confidence, and when it is low or the photo looks unfamiliar
    we say "uncertain, please have a doctor check" instead of pretending to know.
  * The two pictures we show (enhanced view and vessel map) come from image filters.
    They are NOT a map of what the model "looked at".
"""
import base64

import cv2
import numpy as np

from .features import vessel_map

GRADES = {
    0: ("No DR", "No signs of diabetic retinopathy were found by the model."),
    1: ("Mild", "Mild early changes may be present."),
    2: ("Moderate", "Moderate changes may be present."),
    3: ("Severe", "Severe changes may be present."),
    4: ("Proliferative", "The most advanced stage of diabetic retinopathy may be present."),
}

DISCLAIMER = ("Netra Scan is a student research prototype. It is a screening aid, not a "
              "medical diagnosis, and it has not been tested on patients. Only an eye "
              "doctor can diagnose diabetic retinopathy.")


def advice(grade, referable, status):
    """Plain-language next step. We deliberately give no treatment or timing claims."""
    if status == "uncertain":
        return ("The result is uncertain. Please have this photo checked by an eye doctor, "
                "or retake the photo in better conditions.")
    if referable or grade >= 2:
        if grade >= 3:
            return ("Please see an eye doctor as soon as possible. The model sees signs "
                    "that may need prompt attention.")
        return "Please arrange a visit to an eye doctor. The model sees signs that need a proper check."
    return ("No urgent signs were found. Keep up regular eye check-ups as your doctor "
            "advises, and keep blood sugar under control.")


def _png_data_url(img_bgr_or_gray):
    ok, buf = cv2.imencode(".jpg", img_bgr_or_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def evidence_images(prep, size=384):
    """Two helper pictures: the original (cleaned) retina and a vessel map."""
    rgb = cv2.resize(prep["rgb"], (size, size), interpolation=cv2.INTER_AREA)
    enhanced = cv2.resize(prep["green_clahe"], (size, size), interpolation=cv2.INTER_AREA)
    vm = vessel_map(cv2.resize(prep["green_clahe"], (256, 256), interpolation=cv2.INTER_AREA))
    vm_img = (np.clip(vm * 4.0, 0, 1) * 255).astype(np.uint8)  # boost so thin vessels show
    vm_img = cv2.resize(vm_img, (size, size), interpolation=cv2.INTER_LINEAR)
    return {
        "retina": _png_data_url(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
        "enhanced": _png_data_url(enhanced),
        "vessels": _png_data_url(vm_img),
    }
