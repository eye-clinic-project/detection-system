"""Save the Grad-CAM from a /predict response as a PNG in ./image.

Usage:
  .venv/bin/python src/save_gradcam.py photo.jpg        # calls /predict, saves the overlay + JSON
  .venv/bin/python src/save_gradcam.py response.json    # decodes a response you already saved
  pbpaste | .venv/bin/python src/save_gradcam.py -      # decodes a response from stdin

Env: CORE_URL (default http://localhost:8000); AI_SERVICE_API_KEY is read from .env.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "image"


def predict(image: Path) -> dict:
    load_dotenv(ROOT / ".env")
    url = os.getenv("CORE_URL", "http://localhost:8000") + "/predict"
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{image.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if key := os.getenv("AI_SERVICE_API_KEY"):
        headers["x-api-key"] = key
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers)) as r:
        return json.load(r)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    if arg == "-":
        response, stem = json.load(sys.stdin), "response"
    elif arg.lower().endswith(".json"):
        response, stem = json.loads(Path(arg).read_text()), Path(arg).stem
    else:
        response, stem = predict(Path(arg)), Path(arg).stem

    b64 = response.get("grad_cam_png_base64")
    if not b64:
        sys.exit("No grad_cam_png_base64 in the response (was it called with ?grad_cam=false?)")

    OUT_DIR.mkdir(exist_ok=True)
    name = f"{stem}_gradcam_{response.get('grad_cam_class', 'X')}_{time.strftime('%Y%m%d-%H%M%S')}"
    png = OUT_DIR / f"{name}.png"
    png.write_bytes(base64.b64decode(b64))
    meta = {k: v for k, v in response.items() if k != "grad_cam_png_base64"}
    (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))

    s = meta.get("stage1", {})
    print(f"saved {png.relative_to(ROOT)}")
    print(f"class={meta.get('grad_cam_class')} status={s.get('status')} detected={s.get('detected')} dr_probability={s.get('dr_probability')}")


if __name__ == "__main__":
    main()
