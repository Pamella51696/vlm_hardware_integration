"""Prompts that force a small VLM to emit structured JSON, not paragraphs."""

from __future__ import annotations

SCHEMA = """{
  "answer": "yes" | "no" | "unknown",
  "object": string,
  "side": "left" | "center" | "right" | "unknown",
  "status": "clear" | "blocked" | "present" | "absent" | "unknown",
  "count": integer,
  "confidence": number between 0 and 1,
  "description": short sentence
}"""

ALLOWED_QUERIES = """
Road: Is the road clear? Is there a curb? Where is the curb? Is there a sidewalk? Is the road blocked?
Objects: Is there a car? Is there a pedestrian? Is there a bicycle? Is there a motorcycle? Is there a traffic sign?
Spatial: What is on the left? What is on the right? Is the pedestrian ahead? Is the car on the left?
Safety: Is there an obstacle? Is someone crossing the road? Is there a vehicle approaching?
"""


def build_prompt(query: str, hint: str = "", roi: str = "full") -> str:
    extra = hint.strip()
    hint_line = f"Hint: {extra}\n" if extra else ""
    return (
        "You are a driving-scene assistant looking at one undistorted camera view "
        f"(ROI={roi}). Answer ONLY from the image.\n"
        f"{hint_line}"
        "Return a single JSON object with this schema and no extra text:\n"
        f"{SCHEMA}\n"
        f"User question: {query}\n"
        "If you are unsure, use answer=unknown and a low confidence."
    )
