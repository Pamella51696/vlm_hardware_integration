"""HTTP API for the companion VLM process.

The Java video server owns camera / undistortion / stitch / streaming.
This process only runs when a query arrives:

    POST /ask  multipart: image, query, frame_id, roi, category, hint
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from query_engine import QueryEngine

app = FastAPI(title="Surround-camera VLM service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = QueryEngine()


@app.get("/health")
def health():
    return {
        "ok": True,
        "backend": engine.backend_name(),
        "device": engine.device_name(),
    }


@app.get("/ask")
def ask_get(query: str = Query("", description="Question about the scene")):
    """Browser-friendly check. Real answers still need an image via POST or Java /ask."""
    if not query.strip():
        return JSONResponse(
            {
                "ok": True,
                "hint": "GET /ask?query=... only checks that the service is up. "
                "Use the Java /play page (POST) so a camera frame is attached.",
            }
        )
    return JSONResponse(
        {
            "error": "image required",
            "query": query.strip(),
            "hint": "This Python process has no camera. Open Java http://localhost:9090/play "
            "and click Ask VLM, or POST multipart image+query to /ask.",
        },
        status_code=400,
    )


@app.post("/ask")
async def ask(
    query: str = Form(...),
    image: UploadFile = File(...),
    frame_id: str = Form("0"),
    roi: str = Form("full"),
    category: str = Form(""),
    hint: str = Form(""),
):
    raw = await image.read()
    if not raw:
        return JSONResponse({"error": "empty image"}, status_code=400)
    if not query.strip():
        return JSONResponse({"error": "empty query"}, status_code=400)
    try:
        result = engine.answer(
            image_bytes=raw,
            query=query.strip(),
            frame_id=frame_id,
            roi=roi,
            category=category,
            hint=hint,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse(
            {
                "answer": "unknown",
                "object": "scene",
                "error": str(exc),
                "description": "Query failed inside the VLM service; process stayed up.",
            },
            status_code=200,
        )


def main() -> None:
    import uvicorn

    host = os.environ.get("VLM_HOST", "127.0.0.1")
    port = int(os.environ.get("VLM_PORT", "8088"))
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
