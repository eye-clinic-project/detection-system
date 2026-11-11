"""Stage-1 inference: EfficientNet-B3 ensemble from efficientnet-b3-ver-4.ipynb.

The preprocessing, TTA, ensembling and thresholds mirror the notebook exactly
(cells 6, 7, 9, 10 and 11), so the service gives the same numbers as Kaggle.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger("core.analysis")

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertensive Retinopathy",
    "M": "Pathological Myopia",
    "O": "Other abnormalities",
}
DR_INDEX = CLASSES.index("D")

IMG_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_SIDE = 64  # smaller images cannot be a usable fundus photo

MODEL_VERSION = "effnet-b3-stage1-ens"


class InvalidImageError(ValueError):
    """The upload cannot be decoded or is not a usable fundus photo."""


class ModelNotLoadedError(RuntimeError):
    """Weights are missing or failed to load."""


# ---------- preprocessing (notebook cell 6) ----------

def crop_black_borders(rgb: np.ndarray, thr: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > thr)
    if len(ys) < 100:
        return rgb
    rgb = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=rgb.dtype)
    canvas[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = rgb
    return canvas


def enhance(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def prepare_rgb(rgb: np.ndarray, crop: bool = True) -> np.ndarray:
    if crop:
        rgb = crop_black_borders(rgb)
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def to_tensor(rgb_resized: np.ndarray) -> torch.Tensor:
    """Same as albumentations Normalize(MEAN, STD, max_pixel_value=255) + ToTensorV2."""
    x = (enhance(rgb_resized).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)


def decode_image(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidImageError("The file could not be decoded as an image")
    if min(bgr.shape[:2]) < MIN_SIDE:
        raise InvalidImageError(f"The image is too small (minimum {MIN_SIDE}px per side)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 10).mean() < 0.05:
        raise InvalidImageError("The image is almost completely black and is not a usable fundus photo")
    return rgb


# ---------- model ----------

def pick_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def build_model() -> torch.nn.Module:
    return timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(CLASSES), drop_rate=0.4, drop_path_rate=0.3)


@dataclass
class Stage1Model:
    model_dir: Path
    device: torch.device
    models: list[torch.nn.Module] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    error: str | None = None
    loaded_at: float | None = None
    _cam: GradCAM | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def loaded(self) -> bool:
        return bool(self.models) and self.error is None

    @property
    def version(self) -> str:
        return f"{MODEL_VERSION}{len(self.models) or len(self.seeds)}"

    def load(self) -> None:
        try:
            config_path = self.model_dir / "ensemble_config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing {config_path}")
            config = json.loads(config_path.read_text())
            self.thresholds = {c: float(config["thresholds"][c]) for c in CLASSES}
            self.seeds = [int(s) for s in config.get("seeds", [42, 123, 2024])]

            missing = [f"b3_seed{s}.pth" for s in self.seeds if not (self.model_dir / f"b3_seed{s}.pth").exists()]
            if missing:
                raise FileNotFoundError(f"Missing checkpoint(s) in {self.model_dir}: {', '.join(missing)}")

            models = []
            for seed in self.seeds:
                state = torch.load(self.model_dir / f"b3_seed{seed}.pth", map_location="cpu", weights_only=True)
                state = {k.removeprefix("module."): v for k, v in state.items() if k != "n_averaged"}
                model = build_model()
                model.load_state_dict(state)
                models.append(model.to(self.device).eval())

            self.models = models
            self._cam = GradCAM(model=models[0], target_layers=[models[0].blocks[-1]])
            self.error = None
            self.loaded_at = time.time()
            logger.info("Loaded %d models on %s", len(models), self.device)
        except Exception as exc:  # keep the API up so /health can explain what is wrong
            self.models = []
            self.error = str(exc)
            logger.error("Model not loaded: %s", exc)

    @torch.no_grad()
    def _tta(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
        return torch.stack([torch.sigmoid(model(v).float()) for v in views]).mean(0)

    @torch.no_grad()
    def _ensemble(self, x: torch.Tensor) -> np.ndarray:
        return torch.stack([self._tta(m, x) for m in self.models]).mean(0)[0].cpu().numpy()

    def _grad_cam_png(self, rgb: np.ndarray, x: torch.Tensor, class_idx: int) -> str:
        assert self._cam is not None
        gray = self._cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
        overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, gray, use_rgb=True)
        ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode the Grad-CAM image")
        return base64.b64encode(png.tobytes()).decode("ascii")

    def predict(self, data: bytes, *, grad_cam: bool = True, top_k: int = 3, min_prob: float = 0.15) -> dict:
        """Notebook cell 11 (analyze_rgb), shaped for the NestJS AiService contract."""
        if not self.loaded:
            raise ModelNotLoadedError(self.error or "Model is not loaded")
        started = time.perf_counter()
        rgb = prepare_rgb(decode_image(data), crop=True)
        x = to_tensor(rgb).to(self.device)

        with self._lock:  # Grad-CAM hooks are not thread-safe
            probs = self._ensemble(x)
            detected = [c for i, c in enumerate(CLASSES) if probs[i] >= self.thresholds[c]]
            has_dr = "D" in detected
            cam_idx = DR_INDEX if has_dr else int(np.argmax(probs))
            cam_png = self._grad_cam_png(rgb, x, cam_idx) if grad_cam else None

        if detected:
            status, shown = "DETECTED", detected
        else:
            status = "SUSPECTED"
            shown = [CLASSES[i] for i in np.argsort(-probs)[:top_k] if probs[i] >= min_prob]

        p_dr = float(probs[DR_INDEX])
        return {
            "model_version": self.version,
            "stage1": {
                "has_dr": has_dr,
                # confidence in the yes/no decision itself
                "confidence": round(p_dr if has_dr else 1.0 - p_dr, 4),
                "dr_probability": round(p_dr, 4),
                "threshold": self.thresholds["D"],
                "status": status,
                "detected": detected,
                "shown": shown,
                "top": CLASSES[int(np.argmax(probs))],
                "findings": [
                    {
                        "code": c,
                        "name": CLASS_NAMES[c],
                        "probability": round(float(probs[i]), 4),
                        "threshold": self.thresholds[c],
                        "detected": c in detected,
                    }
                    for i, c in enumerate(CLASSES)
                ],
                "trigger_stage2": has_dr,
            },
            # Stage 2 (five-level DR grading) is a separate model that is not in /core yet.
            "stage2": None,
            "grad_cam_class": CLASSES[cam_idx],
            "grad_cam_png_base64": cam_png,
            "processing_ms": int((time.perf_counter() - started) * 1000),
        }

    def info(self) -> dict:
        return {
            "model_version": self.version,
            "loaded": self.loaded,
            "error": self.error,
            "architecture": "efficientnet_b3 (timm), multi-label, 8 classes",
            "ensemble_seeds": self.seeds,
            "input_size": IMG_SIZE,
            "preprocessing": "crop black borders, resize 384, CLAHE (LAB L-channel), ImageNet normalize",
            "tta": ["original", "hflip", "vflip", "hvflip"],
            "classes": [{"code": c, "name": CLASS_NAMES[c], "threshold": self.thresholds.get(c)} for c in CLASSES],
            "stage2": None,
            "device": str(self.device),
            "loaded_at": self.loaded_at,
        }


def from_env() -> Stage1Model:
    model_dir = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "weights")).resolve()
    device = pick_device(os.getenv("DEVICE", "auto"))
    torch.set_grad_enabled(True)  # Grad-CAM needs autograd; inference uses no_grad explicitly
    return Stage1Model(model_dir=model_dir, device=device)
