"""세그멘테이션 오버레이 렌더링 (스트림/스냅샷 표시용)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ml.vehicle_seg import VEHICLE_CLASSES

# BGR
CLASS_COLORS = {"car": (80, 200, 80), "bus": (60, 160, 255), "truck": (220, 120, 40)}
VEHICLE_COLOR = (60, 220, 60)
DIRECTION_COLORS = [
    (214, 120, 42),  # #2a78d6 (BGR)
    (52, 104, 235),  # #eb6834
    (122, 175, 27),  # #1baf7a
    (200, 80, 200),
    (40, 200, 220),
    (120, 120, 250),
]


def hex_to_bgr(hex_color: str | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return fallback
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return (b, g, r)


class RoadLayer:
    """도로 마스크로 미리 만들어 두는 오버레이 재료(색 레이어·윤곽선).

    마스크는 거의 바뀌지 않으므로 한 번 만들어 두면 프레임마다 방향별 마스크를 다시
    만들고 윤곽선을 다시 찾는 비용을 없앨 수 있다.
    """

    def __init__(self, road_label: np.ndarray, colors: list[tuple[int, int, int]] | None = None):
        colors = colors or DIRECTION_COLORS
        n = int(road_label.max()) if road_label.size else 0
        palette = np.zeros((n + 1, 3), np.uint8)
        for d in range(1, n + 1):
            palette[d] = colors[(d - 1) % len(colors)]
        self.shape = road_label.shape
        # 도로 바깥은 어차피 칠할 것이 없으므로 마스크 바운딩 박스만 재료로 들고 있는다.
        ys, xs = np.nonzero(road_label)
        h, w = road_label.shape
        self.box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(ys) else (0, 0, w, h)
        x0, y0, x1, y1 = self.box
        win = road_label[y0:y1, x0:x1]
        self.color = palette[win]
        self.mask = win > 0
        self.contours = [
            (cv2.findContours((road_label == d).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], colors[(d - 1) % len(colors)])
            for d in range(1, n + 1)
        ]


def _blend_masked(out: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> None:
    """mask 인 픽셀에만 layer 를 alpha 로 섞는다 (out 을 제자리에서 수정)."""
    blended = cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0)
    np.copyto(out, blended, where=mask[:, :, None])


_VEHICLE_PALETTES = {
    "class": np.array([(0, 0, 0)] + [CLASS_COLORS[c] for c in VEHICLE_CLASSES], np.uint8),
    "vehicle": np.array([(0, 0, 0)] + [VEHICLE_COLOR] * len(VEHICLE_CLASSES), np.uint8),
}


def render_overlay(
    frame: np.ndarray,
    vehicle_label: np.ndarray | None,
    road_label: np.ndarray | None,
    mode: str = "class",
    direction_colors: list[tuple[int, int, int]] | None = None,
    road_alpha: float = 0.22,
    vehicle_alpha: float = 0.55,
    hud: list[str] | None = None,
    road_layer: RoadLayer | None = None,
) -> np.ndarray:
    """mode: 'class'(차종별 색) | 'vehicle'(단일 색) | 'road'(도로만) | 'none'."""
    out = frame.copy()
    layer = road_layer
    if layer is None and road_label is not None and mode != "none":
        layer = RoadLayer(road_label, direction_colors)
    if layer is not None and layer.shape != frame.shape[:2]:
        layer = None
    if layer is not None and mode != "none":
        x0, y0, x1, y1 = layer.box
        _blend_masked(out[y0:y1, x0:x1], layer.color, layer.mask, road_alpha)
        for cnts, color in layer.contours:  # 방향 경계선
            cv2.drawContours(out, cnts, -1, color, 2)

    if vehicle_label is not None and mode in ("class", "vehicle"):
        palette = _VEHICLE_PALETTES[mode]
        # 차량 라벨은 도로 마스크 밖이 이미 0 이므로 도로 바운딩 박스 안만 칠하면 된다.
        x0, y0, x1, y1 = layer.box if layer is not None else (0, 0, frame.shape[1], frame.shape[0])
        veh = np.minimum(vehicle_label[y0:y1, x0:x1], len(VEHICLE_CLASSES))
        _blend_masked(out[y0:y1, x0:x1], palette[veh], veh > 0, vehicle_alpha)

    if hud:
        y = 28
        for line in hud:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(out, (8, y - th - 8), (16 + tw, y + 6), (0, 0, 0), -1)
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += th + 16
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return buf.tobytes()


def encode_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return buf.tobytes()
