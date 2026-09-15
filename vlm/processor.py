"""Resize / normalize / optional extra crop before the VLM or detectors run.

The Java server already sends an undistorted ROI when the query mentions a
side. This module still downscales so a 1B–3B 4-bit model fits on 8 GB.
"""

from __future__ import annotations

from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# Small VLMs on Orin 8 GB typically ingest 384–672 on the long edge.
MAX_SIDE = int(__import__("os").environ.get("VLM_MAX_SIDE", "512"))


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return bgr


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_max_side(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_side:
        return bgr
    scale = max_side / float(long_edge)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def split_thirds(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    third = max(1, w // 3)
    return bgr[:, :third], bgr[:, third : 2 * third], bgr[:, 2 * third :]


def encode_jpeg(bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")
