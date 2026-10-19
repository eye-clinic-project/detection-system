
"""FastAPI application for uploading and analyzing images."""

from pathlib import Path
from uuid import uuid4

from analysis import analysis_image
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

IMAGE_DIR = Path(__file__).parent / "image"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/")
def home():
    """Return a message confirming that the API is running."""
    return {"message": "FastAPI is working"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image, save it, and return its analysis."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, and WEBP images are allowed",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have an image content type",
        )

    filename = f"{uuid4().hex}{extension}"
    file_path = IMAGE_DIR / filename
    size = 0

    try:
        with file_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Image must not exceed 10 MB",
                    )

                destination.write(chunk)

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    analysis_result = analysis_image(file_path)

    return {
        "result": analysis_result,
    }
