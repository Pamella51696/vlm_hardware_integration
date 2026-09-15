"""OpenCV geometry / detection helpers. Used as a hybrid prior, not as the VLM."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


_HOG = None
_HOG_FAILED = False


def hog_enabled() -> bool:
    # Off by default: HOG on a stitched panorama often native-crashes OpenCV
    # on Windows and kills uvicorn, which then looks like "connection refused".
    return os.environ.get("VLM_USE_HOG", "0") == "1"


def _hog():
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def people_boxes(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    global _HOG, _HOG_FAILED
    if not hog_enabled():
        return []
    h, w = bgr.shape[:2]
    if h < 80 or w < 40:
        return []
    # Downscale first so detectMultiScale cannot explode on a 360 panorama.
    max_side = 320
    scale = 1.0
    small = bgr
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    if _HOG is None and not _HOG_FAILED:
        try:
            _HOG = _hog()
        except Exception:
            _HOG = None
        if _HOG is None:
            _HOG_FAILED = True
            return []
    if _HOG is None:
        return []
    try:
        rects, weights = _HOG.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
    except Exception:
        _HOG_FAILED = True
        return []
    inv = 1.0 / scale if scale else 1.0
    boxes = []
    for (x, y, bw, bh), weight in zip(rects, weights):
        if float(weight) < 0.4:
            continue
        boxes.append((int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)))
    return boxes


def hog_available() -> bool:
    if not hog_enabled():
        return False
    if _HOG is not None:
        return True
    if _HOG_FAILED:
        return False
    people_boxes(np.zeros((120, 80, 3), dtype=np.uint8))
    return _HOG is not None


def hough_lines(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    h, w = gray.shape
    # Curbs live in the lower half of a bumper / wing camera.
    lower = edges[h // 2 :, :]
    lines = cv2.HoughLinesP(
        lower, 1, np.pi / 180, threshold=40, minLineLength=max(24, w // 8), maxLineGap=12
    )
    out = []
    if lines is None:
        return out
    arr = np.asarray(lines)
    if arr.ndim == 3:
        arr = arr.reshape(-1, 4)
    elif arr.ndim == 1:
        arr = arr.reshape(1, 4)
    for x1, y1, x2, y2 in arr:
        out.append((int(x1), int(y1 + h // 2), int(x2), int(y2 + h // 2)))
    return out


def curb_evidence(bgr: np.ndarray) -> Dict[str, float]:
    """Score long near-horizontal / slightly diagonal lines typical of curbs."""
    h, w = bgr.shape[:2]
    lines = hough_lines(bgr)
    left = 0.0
    right = 0.0
    center = 0.0
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 20:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # Curb-like: not vertical, not a tiny scribble.
        if angle > 55:
            continue
        mx = 0.5 * (x1 + x2)
        score = min(1.0, length / max(w * 0.35, 1.0))
        if mx < w / 3:
            left += score
        elif mx > 2 * w / 3:
            right += score
        else:
            center += score
    total = left + right + center
    side = "unknown"
    best = 0.0
    if left >= right and left >= center and left > 0:
        side, best = "left", left
    elif right >= left and right >= center and right > 0:
        side, best = "right", right
    elif center > 0:
        side, best = "center", center
    conf = 0.0 if total <= 0 else min(0.85, 0.35 + 0.25 * best)
    return {"side": side, "score": float(total), "confidence": float(conf)}


def occupancy(bgr: np.ndarray) -> float:
    """Fraction of strong edges in the lower-center band — a crude blockage prior."""
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.45), h
    x0, x1 = int(w * 0.25), int(w * 0.75)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return float(np.count_nonzero(edges)) / float(edges.size)


def side_of_box(box: Tuple[int, int, int, int], width: int) -> str:
    x, _, bw, _ = box
    mx = x + bw / 2.0
    if mx < width / 3:
        return "left"
    if mx > 2 * width / 3:
        return "right"
    return "center"
