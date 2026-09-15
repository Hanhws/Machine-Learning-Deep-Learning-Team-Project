"""2b) 차량 앞모습/뒷모습 CNN 으로 방향별 도로의 방향 판정.

기하 규칙(우측통행: 화면 왼쪽 도로는 다가오고 오른쪽은 멀어짐)은 양쪽 도로가 다 보일 때만 쓸 수 있다.
한쪽 도로만 보이는 화면에서도 방향을 알기 위해 차량 사진으로 판정한다.

1. 자동 라벨: 기하 규칙이 양방향으로 판정했고 차로 수가 파일명과 ±1 이내인 장면에서
   다가오는 도로의 차량 = front(1), 멀어지는 도로의 차량 = rear(0)
2. ImageNet 사전학습 ResNet18 파인튜닝 (출력층만 → 전체), 카메라 단위로 train/val/test 분리
3. 비교: CLIP zero-shot (학습 없이 문장으로 판정)
4. 모든 장면의 방향별 도로에 적용: 차량 확률 평균으로 다수결

출력 outputs/s2b/
  crops/ , crops.csv       자동 라벨 차량 사진
  resnet18_direction.pt
  eval.json                test 카메라 정확도 (차량 단위, 도로 단위 다수결, 주/야간, CLIP 비교)
  test_grid.jpg            test 사진 예측 확인용
  directions.json          장면·방향별 도로의 최종 방향

build·train·eval 은 결과 파일이 이미 있으면 건너뛴다(--redo 로 다시). apply 는 매번 다시 한다.
파인튜닝 실험: --epochs-head, --epochs-full, --seed 로 학습만 바꾼다. 자동 라벨 뽑기·카메라 분할은
항상 SEED 로 고정해서 실험끼리 같은 test 카메라로 비교된다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import CLIP_ID, CLIP_REVISION, OUT, SEED, Clip, frame_index, list_clips, pick_device, seed_everything

CROP = 112
MIN_BOX_PX = 28  # 원본 해상도 기준 짧은 변
MAX_PER_CARRIAGEWAY = 120
ASSIGN_RADIUS_PX = 15
CONFIDENT = 0.15  # |평균 확률 − 0.5| 가 이보다 크고
MIN_VOTES = 8  # 차량이 이만큼 있어야 방향을 확정한다
DEVICE = pick_device()
OUT2 = OUT / "s2b"


def vehicles_by_carriageway(scene_dir, dets: list[dict]) -> dict[int, list[dict]]:
    """장면의 차량을 바닥점 기준으로 방향별 도로에 배정."""
    info = json.loads((scene_dir / "lanes.json").read_text())
    lane_map = np.load(scene_dir / "lanes.npz")["lane_map"]
    h, w = lane_map.shape
    lane_cw = {l["id"]: l["carriageway"] for l in info["lanes"]}
    dist, (iy, ix) = distance_transform_edt(lane_map == 0, return_indices=True)
    frames = set(info["frames"])
    out: dict[int, list[dict]] = {}
    for d in dets:
        if d["frame"] not in frames or len(d["poly"]) < 3:
            continue
        p = d["poly"]
        ymax = p[:, 1].max()
        fx, fy = int(np.clip(p[p[:, 1] >= ymax - 3, 0].mean(), 0, w - 1)), int(np.clip(ymax, 0, h - 1))
        if dist[fy, fx] > ASSIGN_RADIUS_PX:
            continue
        out.setdefault(lane_cw[int(lane_map[iy[fy, fx], ix[fy, fx]])], []).append(d)
    return out


def crop_vehicle(img: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray | None:
    x0, y0, x1, y1 = box * scale
    if min(x1 - x0, y1 - y0) < MIN_BOX_PX:
        return None
    pad = 0.1 * max(x1 - x0, y1 - y0)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(W, x1 + pad)), int(min(H, y1 + pad))
    return cv2.resize(img[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_AREA)


def scenes(clips: dict[str, Clip]):
    for sd in sorted(p.parent for p in (OUT / "s2").glob("*/lanes.npz")):
        clip_id = sd.name.split("__view")[0]
        if clip_id in clips:
            yield sd, clips[clip_id]


def build_crops(seed: int = SEED) -> None:
    clips = {c.clip_id: c for c in list_clips()}
    summary = pd.read_csv(OUT / "s2_summary.csv").set_index("scene_id")
    rng = np.random.default_rng(seed)
    (OUT2 / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    dets_cache: dict[str, dict] = {}
    for sd, clip in scenes(clips):
        info = json.loads((sd / "lanes.json").read_text())
        row = summary.loc[sd.name] if sd.name in summary.index else None
        trusted = (row is not None and clip.two_way and bool(info.get("geo_two_way"))
                   and bool(row.get("lane_within1", False)))
        if not trusted:
            continue
        if clip.clip_id not in dets_cache:
            with open(OUT / "s1" / clip.clip_id / "dets.pkl", "rb") as f:
                dets_cache = {clip.clip_id: pickle.load(f)}
        d = dets_cache[clip.clip_id]
        scale = d["orig_size"][0] / d["size"][0]
        paths = {frame_index(p): p for p in clip.frames}
        cw_dir = {c["id"]: c["geo_direction"] for c in info["carriageways"]}
        wanted: dict[int, list] = {}
        for cw, vs in vehicles_by_carriageway(sd, d["dets"]).items():
            if cw_dir.get(cw) is None:
                continue
            vs = [v for v in vs if min(v["box"][2] - v["box"][0], v["box"][3] - v["box"][1]) * scale >= MIN_BOX_PX]
            for i in rng.permutation(len(vs))[:MAX_PER_CARRIAGEWAY]:
                wanted.setdefault(vs[i]["frame"], []).append((cw, vs[i]))
        for fid, items in wanted.items():
            img = cv2.imread(str(paths[fid]))
            for k, (cw, v) in enumerate(items):
                crop = crop_vehicle(img, v["box"], scale)
                if crop is None:
                    continue
                name = f"{sd.name}__f{fid}__{k}.jpg"
                cv2.imwrite(str(OUT2 / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                rows.append({"file": name, "scene_id": sd.name, "camera": clip.camera_id, "hour": clip.hour,
                             "carriageway": cw, "cls": v["cls"], "label": int(cw_dir[cw] == "toward")})
        print(f"{sd.name} crops={sum(len(v) for v in wanted.values())}", flush=True)
    pd.DataFrame(rows).to_csv(OUT2 / "crops.csv", index=False)
    print(f"자동 라벨 사진 {len(rows)}장")


class Crops(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 좌우 뒤집어도 앞모습은 앞모습이다
        aug = [transforms.RandomResizedCrop(CROP, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
               transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.4, 0.4, 0.3, 0.05)] if train else []
        self.tf = transforms.Compose([transforms.ToPILImage(), *aug, transforms.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(OUT2 / "crops" / r["file"])), cv2.COLOR_BGR2RGB)
        return self.tf(img), int(r["label"])


def split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = df.copy()
    g1 = GroupShuffleSplit(1, test_size=0.3, random_state=seed)
    tr, rest = next(g1.split(df, groups=df["camera"]))
    df["split"] = "train"
    rest_df = df.iloc[rest]
    g2 = GroupShuffleSplit(1, test_size=0.5, random_state=seed)
    va, te = next(g2.split(rest_df, groups=rest_df["camera"]))
    df.loc[rest_df.index[va], "split"] = "val"
    df.loc[rest_df.index[te], "split"] = "test"
    return df


def make_model() -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


@torch.no_grad()
def predict(model: nn.Module, df: pd.DataFrame) -> np.ndarray:
    model.eval()
    probs = []
    for x, _ in DataLoader(Crops(df, train=False), batch_size=256, num_workers=0):
        probs.append(model(x.to(DEVICE)).softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def train(epochs_head: int = 2, epochs_full: int = 4, seed: int = SEED) -> None:
    df = split(pd.read_csv(OUT2 / "crops.csv"))
    df.to_csv(OUT2 / "crops.csv", index=False)
    print(df.groupby("split").agg(n=("file", "size"), cameras=("camera", "nunique"), front=("label", "mean")))
    # 출력층 초기값 · 섞는 순서 · 증강을 고정해 다시 학습했을 때의 차이를 줄인다
    seed_everything(seed)
    model = make_model().to(DEVICE)
    loader = DataLoader(Crops(df[df.split == "train"], train=True), batch_size=128, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    # 1단계: 사전학습 특징은 두고 출력층만, 2단계: 전체를 작은 학습률로
    for stage, epochs, lr in (("head", epochs_head, 1e-3), ("full", epochs_full, 1e-4)):
        for name, p in model.named_parameters():
            p.requires_grad = stage == "full" or name.startswith("fc.")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
        for ep in range(epochs):
            model.train()
            tot, n = 0.0, 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(y)
                n += len(y)
            va = df[df.split == "val"]
            acc = float(((predict(model, va) > 0.5) == va["label"].to_numpy()).mean())
            print(f"{stage} epoch {ep + 1}/{epochs} loss={tot / n:.4f} val_acc={acc:.3f}", flush=True)
            if acc > best:
                best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, OUT2 / "resnet18_direction.pt")


@torch.no_grad()
def clip_probs(df: pd.DataFrame) -> np.ndarray:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REVISION).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REVISION)
    prompts = ["a photo of the front of a vehicle with headlights and windshield, driving toward the camera",
               "a photo of the rear of a vehicle with taillights and license plate, driving away from the camera"]
    out = []
    files = df["file"].tolist()
    for i in range(0, len(files), 64):
        ims = [cv2.cvtColor(cv2.imread(str(OUT2 / "crops" / f)), cv2.COLOR_BGR2RGB) for f in files[i:i + 64]]
        inp = proc(text=prompts, images=ims, return_tensors="pt", padding=True).to(DEVICE)
        out.append(model(**inp).logits_per_image.softmax(-1)[:, 0].cpu().numpy())
    return np.concatenate(out)


def vote_accuracy(df: pd.DataFrame, prob_col: str, threshold: float = 0.5) -> float:
    g = df.groupby(["scene_id", "carriageway"]).agg(p=(prob_col, "mean"), y=("label", "first"))
    return float(((g["p"] > threshold) == g["y"].astype(bool)).mean())


def evaluate() -> None:
    df = pd.read_csv(OUT2 / "crops.csv")
    te = df[df.split == "test"].copy()
    model = make_model().to(DEVICE)
    model.load_state_dict(torch.load(OUT2 / "resnet18_direction.pt", map_location=DEVICE))
    te["p_cnn"] = predict(model, te)
    te["p_clip"] = clip_probs(te)
    y = te["label"].to_numpy()
    night = (te["hour"] >= 19) | (te["hour"] < 7)
    res = {"test_cameras": sorted(te["camera"].unique().tolist()), "n_test": len(te)}
    for name, col in (("resnet18_finetuned", "p_cnn"), ("clip_zero_shot", "p_clip")):
        p = te[col].to_numpy()
        res[name] = {
            "vehicle_acc": round(float(((p > 0.5) == y).mean()), 4),
            "vehicle_f1": round(float(f1_score(y, p > 0.5)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "carriageway_vote_acc": round(vote_accuracy(te, col), 4),
            "night_vehicle_acc": round(float(((p[night] > 0.5) == y[night]).mean()), 4) if night.any() else None,
            "day_vehicle_acc": round(float(((p[~night] > 0.5) == y[~night]).mean()), 4),
        }
    (OUT2 / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))

    sample = te.sample(min(48, len(te)), random_state=0)
    tiles = []
    for _, r in sample.iterrows():
        im = cv2.imread(str(OUT2 / "crops" / r["file"]))
        ok = (r["p_cnn"] > 0.5) == bool(r["label"])
        cv2.rectangle(im, (0, 0), (CROP - 1, CROP - 1), (0, 200, 0) if ok else (0, 0, 255), 3)
        cv2.putText(im, f"{'F' if r['p_cnn'] > 0.5 else 'R'} {r['p_cnn']:.2f}", (4, CROP - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + 8], axis=1) for i in range(0, len(tiles), 8)], axis=0)
    cv2.imwrite(str(OUT2 / "test_grid.jpg"), grid)


@torch.no_grad()
def apply_all() -> None:
    """모든 장면(한방향 포함)의 방향별 도로 방향을 차량 사진 다수결로 정한다."""
    clips = {c.clip_id: c for c in list_clips()}
    model = make_model().to(DEVICE)
    model.load_state_dict(torch.load(OUT2 / "resnet18_direction.pt", map_location=DEVICE))
    model.eval()
    norm = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    result: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    for sd, clip in scenes(clips):
        with open(OUT / "s1" / clip.clip_id / "dets.pkl", "rb") as f:
            d = pickle.load(f)
        scale = d["orig_size"][0] / d["size"][0]
        paths = {frame_index(p): p for p in clip.frames}
        info = json.loads((sd / "lanes.json").read_text())
        geo = {c["id"]: c["geo_direction"] for c in info["carriageways"]}
        result[sd.name] = {}
        imgs: dict[int, np.ndarray] = {}
        for cw, vs in vehicles_by_carriageway(sd, d["dets"]).items():
            vs = [v for v in vs if min(v["box"][2] - v["box"][0], v["box"][3] - v["box"][1]) * scale >= MIN_BOX_PX]
            batch = []
            for i in rng.permutation(len(vs))[:60]:
                v = vs[i]
                if v["frame"] not in imgs:
                    imgs[v["frame"]] = cv2.imread(str(paths[v["frame"]]))
                crop = crop_vehicle(imgs[v["frame"]], v["box"], scale)
                if crop is not None:
                    batch.append(norm(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if not batch:
                result[sd.name][cw] = {"n": 0, "p_front": None, "cnn": None, "geo": geo.get(cw), "final": geo.get(cw)}
                continue
            p = float(model(torch.stack(batch).to(DEVICE)).softmax(1)[:, 1].mean())
            cnn = "toward" if p > 0.5 else "away"
            confident = abs(p - 0.5) > CONFIDENT and len(batch) >= MIN_VOTES
            # 기하 규칙이 있으면 그것을 우선하고, 없을 때(한쪽 도로만 보임) CNN 이 확신하면 CNN 을 쓴다
            final = geo.get(cw) or (cnn if confident else None)
            result[sd.name][cw] = {"n": len(batch), "p_front": round(p, 3), "cnn": cnn, "confident": confident,
                                   "geo": geo.get(cw), "final": final}
        print(sd.name, result[sd.name], flush=True)
    (OUT2 / "directions.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    both = [(v["geo"], v["cnn"]) for s in result.values() for v in s.values() if v["geo"] and v["cnn"] and v.get("confident")]
    if both:
        print(f"기하 규칙과 CNN 일치율 (확신한 도로 {len(both)}개): {np.mean([g == c for g, c in both]):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["build", "train", "eval", "apply"], help="build train eval apply")
    ap.add_argument("--redo", action="store_true", help="결과 파일이 있어도 build·train·eval 을 다시 한다")
    ap.add_argument("--epochs-head", type=int, default=2)
    ap.add_argument("--epochs-full", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED, help="학습 시드 (자동 라벨·카메라 분할은 SEED 고정)")
    args = ap.parse_args()
    if unknown := set(args.steps) - {"build", "train", "eval", "apply"}:
        ap.error(f"모르는 단계: {sorted(unknown)}")
    done = {"build": OUT2 / "crops.csv", "train": OUT2 / "resnet18_direction.pt", "eval": OUT2 / "eval.json"}
    for step in args.steps:
        if step in done and done[step].exists() and not args.redo:
            print(f"{step}: {done[step].name} 이 있어 건너뜀 (다시 하려면 --redo)", flush=True)
            continue
        if step == "build":
            build_crops()
        elif step == "train":
            train(args.epochs_head, args.epochs_full, args.seed)
        elif step == "eval":
            evaluate()
        else:
            apply_all()


if __name__ == "__main__":
    main()
