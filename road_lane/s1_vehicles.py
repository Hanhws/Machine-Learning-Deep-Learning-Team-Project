"""1) 클립마다 차량 세그멘테이션 + 차 없는 배경 이미지 + 차량 누적 지도.

COCO 사전학습 YOLO 세그 모델을 쓴다. 팀의 파인튜닝 모델은 --weights (또는 환경변수 YOLO_WEIGHTS) 로 바꾼다.
차량은 클래스 번호가 아니라 이름(car·motorcycle·bus·truck)으로 고르므로 클래스 번호가 다른 모델도 그대로 쓴다.
가중치를 바꾸면 결과 폴더도 새로 정한다(ROAD_LANE_OUT). 한 폴더에 다른 가중치 결과가 섞이려 하면 멈춘다.

출력 (outputs/s1/<clip_id>/)
  background.jpg   차량 픽셀을 뺀 프레임들의 중앙값 → 차 없는 도로 사진
  heat.npy         프레임마다 차량이 덮은 픽셀 수 누적 (uint16, 작업 해상도)
  dets.pkl         프레임별 차량 [{frame, cls, conf, box, poly}] (좌표는 작업 해상도)
outputs/s1/weights.txt  이 폴더를 만든 가중치 파일 이름과 SHA-256
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from ultralytics import YOLO

from common import OUT, ROOT, VEHICLE_NAMES, YOLO_SHA256, Clip, frame_index, list_clips, pick_device, work_scale

BG_MAX_FRAMES = 60


def rasterize(polys: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def process_clip(model: YOLO, clip: Clip, out_dir, imgsz: int, conf: float,
                 classes: dict[int, str], device: str) -> dict:
    first = cv2.imread(str(clip.frames[0]))
    H0, W0 = first.shape[:2]
    s = work_scale(W0, H0)
    h, w = round(H0 * s), round(W0 * s)

    bg_pick = set(np.linspace(0, len(clip.frames) - 1, min(BG_MAX_FRAMES, len(clip.frames))).round().astype(int))
    stack, dets = [], []
    heat = np.zeros((h, w), np.uint16)

    for i, path in enumerate(clip.frames):
        # 목록을 한 번에 넘기면 전체가 한 배치로 GPU에 올라가므로 한 장씩 넣는다
        r = model.predict(str(path), imgsz=imgsz, conf=conf, classes=list(classes),
                          device=device, verbose=False)[0]
        polys = []
        if r.masks is not None:
            for j, xy in enumerate(r.masks.xy):
                poly = (xy * s).astype(np.float32)
                polys.append(poly)
                dets.append({
                    "frame": frame_index(path),
                    "cls": classes[int(r.boxes.cls[j])],
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
    ap.add_argument("--weights", default=os.environ.get("YOLO_WEIGHTS", str(ROOT / "weights" / "yolo26m-seg.pt")))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--first", nargs="*", default=[], help="먼저 처리할 clip_id 앞부분")
    ap.add_argument("--only", nargs="*", default=None, help="이 clip_id 앞부분만 처리")
    args = ap.parse_args()

    clips = list_clips()
    if args.only:
        clips = [c for c in clips if any(c.clip_id.startswith(o) for o in args.only)]
    clips.sort(key=lambda c: (not any(c.clip_id.startswith(f) for f in args.first), c.clip_id))

    model = YOLO(args.weights)  # yolo26m-seg.pt 가 없으면 ultralytics 가 받아 온다
    classes = {int(i): n for i, n in model.names.items() if n in VEHICLE_NAMES}
    if not classes:
        raise SystemExit(f"모델에 차량 클래스({', '.join(VEHICLE_NAMES)})가 없습니다: {model.names}")
    weights = Path(getattr(model, "ckpt_path", None) or args.weights)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if weights.name == "yolo26m-seg.pt" and sha != YOLO_SHA256:
        print("경고: yolo26m-seg.pt 가 기준 결과를 만든 파일과 다릅니다 → 결과가 달라질 수 있습니다", flush=True)
    stamp, tag = OUT / "s1" / "weights.txt", f"{weights.name} {sha}"
    if stamp.exists() and stamp.read_text().strip() != tag:
        raise SystemExit(f"{OUT} 는 다른 가중치({stamp.read_text().split()[0]})로 만든 결과 폴더입니다. "
                         "ROAD_LANE_OUT=runs/새이름 으로 새 폴더를 정하세요.")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(tag + "\n")
    device = pick_device()
    print(f"가중치 {weights.name} · 차량 클래스 {classes} · 장치 {device} · 결과 {OUT}", flush=True)

    done = 0
    t0 = time.time()
    for k, clip in enumerate(clips, 1):
        out_dir = OUT / "s1" / clip.clip_id
        if (out_dir / "dets.pkl").exists():
            continue
        t = time.time()
        info = process_clip(model, clip, out_dir, args.imgsz, args.conf, classes, device)
        done += info["frames"]
        print(f"[{k}/{len(clips)}] {clip.clip_id} frames={info['frames']} dets={info['dets']} "
              f"{time.time() - t:.0f}s (누적 {done}장, {(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()
