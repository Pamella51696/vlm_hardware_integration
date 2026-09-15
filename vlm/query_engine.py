"""Hybrid query engine: OpenCV priors + optional small VLM + JSON validator."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from geometry import curb_evidence, occupancy, people_boxes, side_of_box, hog_available as geometry_hog_available
from model import VlmBackend, try_load_backend
from processor import bgr_to_pil, decode_bgr, resize_max_side
from prompts import build_prompt
from response_parser import extract_json_object, fallback, normalize


class QueryEngine:
    def __init__(self, backend: Optional[VlmBackend] = None) -> None:
        self._backend = backend if backend is not None else try_load_backend()

    def backend_name(self) -> str:
        if self._backend is None:
            return os.environ.get("VLM_BACKEND", "hybrid") + "+opencv"
        return self._backend.name()

    def device_name(self) -> str:
        if self._backend is None:
            return "cpu"
        return self._backend.device()

    def answer(
        self,
        image_bytes: bytes,
        query: str,
        frame_id: str = "0",
        roi: str = "full",
        category: str = "",
        hint: str = "",
    ) -> Dict[str, Any]:
        bgr = resize_max_side(decode_bgr(image_bytes))
        try:
            prior = self._priors(bgr, query, roi)
        except Exception as exc:
            prior = fallback(query, f"opencv prior failed: {exc}")
        vlm_raw = None
        vlm_text = ""
        if self._backend is not None:
            pil = bgr_to_pil(bgr)
            try:
                vlm_text = self._backend.generate(pil, query, roi, hint)
                vlm_raw = normalize(extract_json_object(vlm_text), query)
            except Exception as exc:
                vlm_raw = fallback(query, f"VLM inference failed: {exc}")

        merged = _merge(query, prior, vlm_raw)
        merged["frame_id"] = frame_id
        merged["roi"] = roi.lower()
        merged["category"] = category or _category(query)
        merged["backend"] = self.backend_name()
        merged["confidence"] = round(float(merged.get("confidence", 0.0)), 3)
        if os.environ.get("VLM_DEBUG", "0") == "1":
            merged["prompt"] = build_prompt(query, hint=hint, roi=roi)
            if vlm_text:
                merged["vlm_raw"] = vlm_text[:800]
        return merged

    def _priors(self, bgr, query: str, roi: str) -> Dict[str, Any]:
        q = query.lower()
        h, w = bgr.shape[:2]
        prior: Dict[str, Any] = fallback(query, "opencv prior only")

        if any(k in q for k in ("pedestrian", "person", "people", "crossing")):
            boxes = people_boxes(bgr)
            hog_ready = geometry_hog_available()
            if boxes:
                side = side_of_box(boxes[0], w)
                prior.update(
                    {
                        "answer": "yes",
                        "object": "pedestrian",
                        "side": side,
                        "status": "present",
                        "count": len(boxes),
                        "confidence": min(0.9, 0.55 + 0.1 * len(boxes)),
                        "description": f"{len(boxes)} pedestrian-like figure(s) in the {side} of the view.",
                    }
                )
            elif hog_ready:
                prior.update(
                    {
                        "answer": "no",
                        "object": "pedestrian",
                        "status": "absent",
                        "count": 0,
                        "confidence": 0.45,
                        "description": "No pedestrian-like figure detected by the HOG prior.",
                    }
                )
            else:
                prior.update(
                    {
                        "answer": "unknown",
                        "object": "pedestrian",
                        "status": "unknown",
                        "count": 0,
                        "confidence": 0.25,
                        "description": (
                            "Pedestrian questions need a VLM backend "
                            "(set VLM_BACKEND=transformers). Hybrid mode does not run HOG "
                            "on the panorama because it can crash the Python process."
                        ),
                    }
                )

        if "curb" in q or "kerb" in q or "sidewalk" in q:
            ev = curb_evidence(bgr)
            asked_side = _asked_side(q, roi)
            found_side = ev["side"]
            present = ev["score"] > 0.35
            if asked_side != "unknown" and present:
                match = found_side == asked_side or asked_side == roi.lower()
                # If Java already cropped LEFT, treat evidence in that crop as that side.
                if roi.lower() in {"left", "right", "center"}:
                    match = True
                    found_side = roi.lower()
                prior.update(
                    {
                        "answer": "yes" if match else "no",
                        "object": "curb" if "curb" in q or "kerb" in q else "sidewalk",
                        "side": found_side,
                        "status": "present" if match else "absent",
                        "count": 1 if match else 0,
                        "confidence": ev["confidence"] if match else max(0.2, 1.0 - ev["confidence"]),
                        "description": (
                            f"Line structure consistent with a curb on the {found_side}."
                            if match
                            else f"Curb-like structure appears on the {found_side}, not {asked_side}."
                        ),
                    }
                )
            elif present:
                prior.update(
                    {
                        "answer": "yes",
                        "object": "curb",
                        "side": found_side,
                        "status": "present",
                        "count": 1,
                        "confidence": ev["confidence"],
                        "description": f"Curb-like edge structure on the {found_side}.",
                    }
                )
            else:
                prior.update(
                    {
                        "answer": "no",
                        "object": "curb",
                        "status": "absent",
                        "confidence": 0.4,
                        "description": "No strong curb-like edge in this view.",
                    }
                )

        if "clear" in q or "blocked" in q or "obstacle" in q:
            occ = occupancy(bgr)
            blocked = occ > 0.12
            if "clear" in q:
                answer = "no" if blocked else "yes"
                status = "blocked" if blocked else "clear"
            else:
                answer = "yes" if blocked else "no"
                status = "blocked" if blocked else "clear"
            prior.update(
                {
                    "answer": answer,
                    "object": "road" if "road" in q else "obstacle",
                    "status": status,
                    "side": _asked_side(q, roi),
                    "confidence": min(0.8, 0.4 + occ),
                    "description": (
                        "Lower-center band looks cluttered."
                        if blocked
                        else "Lower-center band looks relatively open."
                    ),
                }
            )

        if any(k in q for k in ("car", "vehicle", "truck", "motorcycle", "bicycle")):
            # Without a detector weights file we stay conservative.
            prior.setdefault("object", _object_name(q))
            if prior.get("object") in {"scene", "road", "curb", "pedestrian"}:
                prior["object"] = _object_name(q)
            if "answer" not in prior or prior.get("description", "").startswith("opencv"):
                prior.update(
                    {
                        "answer": "unknown",
                        "status": "unknown",
                        "confidence": 0.2,
                        "description": "No vehicle detector loaded; enable a VLM or detector model.",
                    }
                )
        return prior


def _object_name(q: str) -> str:
    for name in ("motorcycle", "bicycle", "truck", "vehicle", "car"):
        if name in q:
            return name
    return "vehicle"


def _asked_side(q: str, roi: str) -> str:
    if "left" in q:
        return "left"
    if "right" in q:
        return "right"
    if any(k in q for k in ("ahead", "front", "center")):
        return "center"
    if roi.lower() in {"left", "right", "center"}:
        return roi.lower()
    return "unknown"


def _category(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("curb", "road", "clear", "sidewalk", "blocked")):
        return "road"
    if any(k in q for k in ("pedestrian", "car", "vehicle", "bicycle", "sign")):
        return "object"
    if any(k in q for k in ("left", "right", "ahead", "where")):
        return "spatial"
    if any(k in q for k in ("obstacle", "crossing", "approaching")):
        return "safety"
    return "other"


def _merge(query: str, prior: Dict[str, Any], vlm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if vlm is None:
        prior["source"] = "opencv"
        return prior
    # Prefer VLM semantics when it is confident; otherwise keep geometry prior.
    if vlm.get("confidence", 0) >= 0.55 and vlm.get("answer") != "unknown":
        out = dict(vlm)
        out["source"] = "vlm"
        out["opencv_prior"] = {
            "answer": prior.get("answer"),
            "side": prior.get("side"),
            "confidence": prior.get("confidence"),
        }
        return out
    out = dict(prior)
    out["source"] = "fused"
    out["vlm"] = {
        "answer": vlm.get("answer"),
        "side": vlm.get("side"),
        "confidence": vlm.get("confidence"),
        "description": vlm.get("description"),
    }
    if prior.get("answer") == "unknown" and vlm.get("answer") != "unknown":
        out.update({k: vlm[k] for k in ("answer", "object", "side", "status", "count", "confidence", "description")})
        out["source"] = "vlm"
    return out
