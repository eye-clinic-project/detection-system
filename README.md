# RetinaCare AI service (stage 1)

FastAPI service for the EfficientNet-B3 ensemble trained in `efficientnet-b3-ver-4.ipynb`.
Preprocessing, TTA, ensembling and thresholds match the notebook. The NestJS server calls it when `AI_MODE="remote"`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # set AI_SERVICE_API_KEY to the same value as server/.env.development
```

Put the trained weights from Kaggle (`/kaggle/working`) in `weights/`:

```
weights/b3_seed42.pth
weights/b3_seed123.pth
weights/b3_seed2024.pth
weights/ensemble_config.json   # already here (thresholds from the notebook run)
```

## Run

```bash
.venv/bin/uvicorn main:app --port 8000 --env-file .env
```

Swagger UI: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path          | Notes |
| ------ | ------------- | ----- |
| GET    | `/health`     | 200 when the weights are loaded, 503 with the reason otherwise |
| GET    | `/model-info` | Architecture, classes, thresholds, device |
| POST   | `/predict`    | multipart field `file` (JPEG/PNG/WEBP, max 10 MB). `?grad_cam=false` skips the heatmap |

All endpoints except `/` require the `x-api-key` header when `AI_SERVICE_API_KEY` is set.

`/predict` response:

```json
{
  "model_version": "effnet-b3-stage1-ens3",
  "stage1": {
    "has_dr": true, "confidence": 0.83, "dr_probability": 0.83, "threshold": 0.575,
    "status": "DETECTED", "detected": ["D"], "shown": ["D"], "top": "D",
    "findings": [{ "code": "D", "name": "Diabetic Retinopathy", "probability": 0.83, "threshold": 0.575, "detected": true }],
    "trigger_stage2": true
  },
  "stage2": null,
  "grad_cam_class": "D",
  "grad_cam_png_base64": "iVBORw0...",
  "processing_ms": 4300
}
```

`stage2` (the five-level DR grade) is `null` until a grading model is added.
Errors: 422 when the image can't be decoded or isn't usable, 503 when the model isn't loaded, 401 when the API key is wrong.
