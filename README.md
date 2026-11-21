# RetinaCare AI service (stage 1)

FastAPI service for the EfficientNet-B3 ensemble trained in `note/efficientnet-b3-ver-4.ipynb`.
Preprocessing, TTA, ensembling and thresholds match the notebook. The NestJS server calls it when `AI_MODE="remote"`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # set AI_SERVICE_API_KEY to the same value as server/.env.development
```

Put the trained weights in `src/weights/`. `ensemble_config.json` picks the model:

```json
{
  "arch": "efficientnet_b5",            // timm model name (default efficientnet_b3)
  "img_size": 456,                       // input size (default 384)
  "checkpoint": "b5v2_seed{seed}.pth",   // file name per seed (default b3_seed{seed}.pth)
  "seeds": [42, 123, 2024],
  "thresholds": { "N": 0.475, ... }
}
```

## Run

```bash
.venv/bin/python -m uvicorn main:app --app-dir src --port 8000 --env-file .env
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
  "model_version": "effnet-b5-stage1-ens3",
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
