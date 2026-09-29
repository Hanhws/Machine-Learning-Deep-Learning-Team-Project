"""base64 PNG ↔ numpy 변환 유틸."""

from __future__ import annotations

import base64

import numpy as np

from app.ml.render import encode_png
from app.services.media import decode_png_bytes


def mask_to_b64(mask: np.ndarray, binary: bool = True) -> str:
    arr = (mask.astype(np.uint8) * 255) if binary else mask.astype(np.uint8)
    return base64.b64encode(encode_png(arr)).decode()


def b64_to_array(b64: str) -> np.ndarray:
    if "," in b64[:40] and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return decode_png_bytes(base64.b64decode(b64))
