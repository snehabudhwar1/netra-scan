"""Step 1 of the pipeline: find the retina, crop it, make it square, enhance it.

Order:  read photo -> find retina (mask) -> crop black border -> pad to square
        -> resize to 512x512 -> green channel -> CLAHE contrast boost

Everything here is plain NumPy / OpenCV, so you can read and explain every line.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

SIZE = 512  # every photo ends up as SIZE x SIZE pixels


def load_rgb(path):
    """Open an image file and return it as an RGB uint8 array (H x W x 3)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # respect the phone/camera rotation flag
        return np.array(im.convert("RGB"))


def retina_mask(img_rgb, thresh=15):
    """Boolean mask: True where the pixel belongs to the round retina,
    False for the black border around it."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    m = (gray > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # remove specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:  # nothing bright found
        return np.zeros(gray.shape, bool)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = labels == biggest  # keep only the largest bright blob = the retina
    return ndimage.binary_fill_holes(m)


def crop_to_mask(img_rgb, mask, pad=2):
    """Cut away the black border using the bounding box of the mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img_rgb, mask
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    return img_rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def pad_to_square(a):
    """Add black margins so height == width (no stretching)."""
    h, w = a.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s) + a.shape[2:], a.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return out


def clahe_green(green, mask, clip=2.0, tiles=8):
    """CLAHE = Contrast Limited Adaptive Histogram Equalisation on one channel.
    It boosts contrast in small tiles, which makes vessels and spots stand out."""
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(green)
    out[~mask] = 0
    return out


def prepare(img_rgb, size=SIZE):
    """Full preprocessing. Returns a dict with everything later steps need.

    keys: ok, rgb, mask, green, green_clahe, fill_ratio, fov, orig_hw, orig_mean
    """
    orig_hw = img_rgb.shape[:2]
    orig_mean = float(img_rgb.mean())  # overall brightness of the uploaded picture
    mask = retina_mask(img_rgb)
    ok = bool(mask.mean() > 0.01)  # we need at least 1% of the picture to be retina
    img, mask = crop_to_mask(img_rgb, mask)
    h, w = mask.shape
    fill_ratio = float(mask.sum() / (h * w)) if h * w else 0.0  # circle in a box ~ 0.785
    fov = min(fill_ratio / (np.pi / 4), 1.0)  # 1.0 = a full circle is visible
    img, mask = pad_to_square(img), pad_to_square(mask)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    img = img.copy()
    img[~mask] = 0
    green = np.ascontiguousarray(img[:, :, 1])
    return {
        "ok": ok,
        "rgb": img,
        "mask": mask,
        "green": green,
        "green_clahe": clahe_green(green, mask),
        "fill_ratio": fill_ratio,
        "fov": float(fov),
        "orig_hw": orig_hw,
        "orig_mean": orig_mean,
    }
