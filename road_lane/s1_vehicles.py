"""1) 클립마다 차량 세그멘테이션 + 차 없는 배경 이미지 + 차량 누적 지도.

COCO 사전학습 YOLO 세그 모델을 쓴다. 팀의 파인튜닝 모델이 생기면 --weights 로 바꾸면 된다.

출력 (outputs/s1/<clip_id>/)
  background.jpg   차량 픽셀을 뺀 프레임들의 중앙값 → 차 없는 도로 사진
  heat.npy         프레임마다 차량이 덮은 픽셀 수 누적 (uint16, 작업 해상도)
  dets.pkl         프레임별 차량 [{frame, cls, conf, box, poly}] (좌표는 작업 해상도)
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from ultralytics import YOLO

from common import OUT, ROOT, VEHICLE_CLASSES, Clip, frame_index, list_clips, work_scale

BG_MAX_FRAMES = 60


def rasterize(polys: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def process_clip(model: YOLO, clip: Clip, out_dir, imgsz: int, conf: float) -> dict:
    first = cv2.imread(str(clip.frames[0]))
    H0, W0 = first.shape[:2]
    s = work_scale(W0, H0)
    h, w = round(H0 * s), round(W0 * s)

    bg_pick = set(np.linspace(0, len(clip.frames) - 1, min(BG_MAX_FRAMES, len(clip.frames))).round().astype(int))
    stack, dets = [], []
    heat = np.zeros((h, w), np.uint16)

    for i, path in enumerate(clip.frames):
        # 목록을 한 번에 넘기면 전체가 한 배치로 GPU에 올라가므로 한 장씩 넣는다
        r = model.predict(str(path), imgsz=imgsz, conf=conf, classes=list(VEHICLE_CLASSES),
                          device="mps", verbose=False)[0]
        polys = []
        if r.masks is not None:
            for j, xy in enumerate(r.masks.xy):
                poly = (xy * s).astype(np.float32)
                polys.append(poly)
                dets.append({
                    "frame": frame_index(path),
                    "cls": VEHICLE_CLASSES[int(r.boxes.cls[j])],
                    "conf": float(r.boxes.conf[j]),
                    "box": (r.boxes.xyxy[j].cpu().numpy() * s).astype(np.float32),
                    "poly": poly,
                })
        vmask = rasterize(polys, (h, w))
        heat += vmask
        if i in bg_pick:
            img = cv2.resize(r.orig_img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            img[cv2.dilate(vmask, np.ones((7, 7), np.uint8)) > 0] = np.nan
            stack.append(img)

    bg = np.nanmedian(np.stack(stack), axis=0)
    # 모든 프레임에서 차량에 덮인 픽셀은 일반 중앙값으로 채운다
    holes = np.isnan(bg)
    if holes.any():
        plain = np.median(np.stack([np.nan_to_num(x, nan=0) for x in stack]), axis=0)
        bg[holes] = plain[holes]

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "background.jpg"), bg.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.save(out_dir / "heat.npy", heat)
    with open(out_dir / "dets.pkl", "wb") as f:
        pickle.dump({"clip_id": clip.clip_id, "size": (w, h), "orig_size": (W0, H0),
                     "frames": [frame_index(p) for p in clip.frames], "dets": dets}, f)
    return {"frames": len(clip.frames), "dets": len(dets)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolo26m-seg.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--first", nargs="*", default=[], help="먼저 처리할 clip_id 앞부분")
    ap.add_argument("--only", nargs="*", default=None, help="이 clip_id 앞부분만 처리")
    args = ap.parse_args()

    clips = list_clips()
    if args.only:
        clips = [c for c in clips if any(c.clip_id.startswith(o) for o in args.only)]
    clips.sort(key=lambda c: (not any(c.clip_id.startswith(f) for f in args.first), c.clip_id))

    model = YOLO(args.weights)
    done = 0
    t0 = time.time()
    for k, clip in enumerate(clips, 1):
        out_dir = OUT / "s1" / clip.clip_id
        if (out_dir / "dets.pkl").exists():
            continue
        t = time.time()
        info = process_clip(model, clip, out_dir, args.imgsz, args.conf)
        done += info["frames"]
        print(f"[{k}/{len(clips)}] {clip.clip_id} frames={info['frames']} dets={info['dets']} "
              f"{time.time() - t:.0f}s (누적 {done}장, {(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()
