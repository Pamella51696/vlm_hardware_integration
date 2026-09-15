"""Turn VLM text into a stable JSON dict the Java client can consume."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

ALLOWED_ANSWERS = {"yes", "no", "unknown"}
ALLOWED_SIDES = {"left", "center", "right", "unknown"}
ALLOWED_STATUS = {"clear", "blocked", "present", "absent", "unknown"}


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def normalize(raw: Dict[str, Any], query: str) -> Dict[str, Any]:
    answer = str(raw.get("answer", "unknown")).strip().lower()
    if answer in {"true", "y", "1"}:
        answer = "yes"
    if answer in {"false", "n", "0"}:
        answer = "no"
    if answer not in ALLOWED_ANSWERS:
        answer = "unknown"

    side = str(raw.get("side", "unknown")).strip().lower()
    if side in {"ahead", "front", "middle"}:
        side = "center"
    if side not in ALLOWED_SIDES:
        side = "unknown"

    status = str(raw.get("status", "unknown")).strip().lower()
    if status not in ALLOWED_STATUS:
        status = "unknown"

    obj = str(raw.get("object", "")).strip().lower() or _object_from_query(query)
    description = str(raw.get("description", "")).strip()
    if not description:
        description = f"{answer} ({obj})" if obj else answer

    out = {
        "answer": answer,
        "object": obj,
        "side": side,
        "status": status,
        "count": _as_int(raw.get("count", 0)),
        "confidence": round(_as_float(raw.get("confidence", 0.0)), 3),
        "description": description,
    }
    return out


def _object_from_query(query: str) -> str:
    q = query.lower()
    for name in (
        "pedestrian",
        "person",
        "sidewalk",
        "curb",
        "bicycle",
        "motorcycle",
        "traffic sign",
        "vehicle",
        "truck",
        "car",
        "obstacle",
        "road",
    ):
        if name in q:
            return "pedestrian" if name == "person" else name
    return "scene"


def fallback(query: str, description: str = "model did not return JSON") -> Dict[str, Any]:
    return normalize(
        {
            "answer": "unknown",
            "object": _object_from_query(query),
            "side": "unknown",
            "status": "unknown",
            "count": 0,
            "confidence": 0.1,
            "description": description,
        },
        query,
    )
