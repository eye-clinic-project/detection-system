"""FastAPI service for the RetinaCare stage-1 model.

Endpoints (the contract NestJS's AiService expects):
  GET  /health      -> 200 when the model is loaded, 503 otherwise
  GET  /model-info  -> model metadata and thresholds
  POST /predict     -> multipart field "file"; stage-1 result + Grad-CAM PNG (base64)

If AI_SERVICE_API_KEY is set, every request must send the same value in "x-api-key".
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from analysis import InvalidImageError, ModelNotLoadedError, from_env

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

API_KEY = os.getenv("AI_SERVICE_API_KEY") or None
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
# Every Grad-CAM /predict returns is also written here as PNG + JSON. Empty = do not save.
GRAD_CAM_DIR = os.getenv("GRAD_CAM_DIR", "").strip()

logger = logging.getLogger("core.main")

model = from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(model.load)
    yield


app = FastAPI(title="RetinaCare AI", version="1.0.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def root():
    return {"service": "retinacare-ai", "docs": "/docs"}


@app.get("/health", dependencies=[Depends(require_key)])
def health():
    body = {"status": "ok" if model.loaded else "model_not_loaded", "model_version": model.version, "device": str(model.device)}
    if not model.loaded:
        body["error"] = model.error
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/model-info", dependencies=[Depends(require_key)])
def model_info():
    return model.info()


@app.post("/predict", dependencies=[Depends(require_key)])
async def predict(file: UploadFile = File(...), grad_cam: bool = Query(default=True)):
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG and WEBP images are allowed")
    data = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Image must not exceed {MAX_FILE_SIZE // (1024 * 1024)} MB")

    try:
        result = await run_in_threadpool(model.predict, data, grad_cam=grad_cam)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=f"Model is not loaded: {exc}") from exc

    if GRAD_CAM_DIR and result.get("grad_cam_png_base64"):
        await run_in_threadpool(save_grad_cam, result, file.filename)
    return result


def save_grad_cam(result: dict, filename: str | None) -> None:
    """Write the overlay and the result (without the base64) to GRAD_CAM_DIR. Never fails the request."""
    try:
        out = Path(GRAD_CAM_DIR)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(filename or "upload").stem.replace(" ", "_")[:40]
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}_{result.get('grad_cam_class', 'X')}_{uuid.uuid4().hex[:6]}"
        (out / f"{name}.png").write_bytes(base64.b64decode(result["grad_cam_png_base64"]))
        meta = {k: v for k, v in result.items() if k != "grad_cam_png_base64"}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved Grad-CAM to %s", out / f"{name}.png")
    except Exception as exc:
        logger.error("Could not save Grad-CAM: %s", exc)
