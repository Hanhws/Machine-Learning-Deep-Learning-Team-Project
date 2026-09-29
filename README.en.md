[한국어](README.md) · **English**

# Road CCTV Congestion Monitor

**Measuring traffic congestion from the vehicle-to-road pixel area ratio, using segmentation**

A web service that segments roads and vehicles pixel-by-pixel from nationwide CCTV footage and
quantifies congestion as the ratio between the two.

```
occupancy_d  =  | vehicle ∩ road_d |  /  | road_d |
```

`road_d` is the set of pixels labelled with direction `d` in the road label map; `d=0` is the
union of all directions.

---

## Why area ratio instead of vehicle count

Counting **vehicles** saturates exactly when congestion matters most. Once traffic is dense,
occlusion makes the detector undercount, and the number of cars that fit in one frame is capped
by the field of view.

Measuring **how much of the road surface is covered** avoids both problems. Stopped vehicles still
contribute their full area, and because the value is normalised to 0–1 it stays comparable across
cameras with different framing.

---

## Two models, two different moments

Because each camera is fixed, the expensive model only has to run **once, at registration time**.

| Model | Role | Runs |
|---|---|---|
| **YOLO26s-seg** (fine-tuned) | Vehicle instance segmentation — `car` / `bus` / `truck` | **Every frame** |
| **SAM 3** | Road region segmentation. Text prompt `"road"` plus point/box prompts. Fills in the footprint of vehicles standing on the road in the snapshot | **Once, at registration** |

Once confirmed, the road mask is written to `road_mask.png` (0 = not road, 1–8 = direction) and
reused from then on. The realtime path therefore runs YOLO only — SAM is not involved.

> Without SAM 3 weights the system falls back to SAM 2.1 automatically. Only the text prompt and
> vehicle-footprint filling are lost; point, box and brush editing continue to work.

### Inference settings

| | |
|---|---|
| Input resolution | `imgsz 1280` (the final model's training resolution, set via `YOLO_IMGSZ` in `.env` — the code default is 960) with `retina_masks` |
| Confidence thresholds | YOLO `0.25` · SAM `0.30` |
| Inference cadence | 1 frame / 5 s by default · 2 FPS in realtime mode · 1 s–10 min per camera |
| Database writes | 5-second averages |
| Device | CUDA / MPS / CPU, selected automatically |

---

## Pipeline

```
① Ingest        ITS Open API live HLS · uploaded video · arbitrary stream URL
       ↓
② Road mask     SAM3 proposal → human confirms pixel-by-pixel → split by direction → saved
       ↓
③ ROI crop      Crop to the road mask bounding box (+32 px) before YOLO
       ↓
④ Occupancy     Direction × vehicle class counted in a single histogram pass
       ↓
⑤ Store & show  5-second averages to PostgreSQL · map / stats / reports · MJPEG overlay
```

### Congestion levels

Four levels by area ratio, configurable via `CONGESTION_THRESHOLDS` in `.env`.

| Free flow | Slow | Delayed | Jammed |
|---|---|---|---|
| under 8% | 8 – 15% | 15 – 25% | over 25% |

---

## Training the vehicle model

YOLO-seg was fine-tuned on the polygon annotations of AI-Hub's *CCTV traffic footage for solving
transport problems (highway)* dataset. The six team members shared one notebook and split the
models, resolutions and hyperparameters between them — 21 runs in total. Preprocessing, EDA,
splitting and training code, plus the full experiment log, live in [`training/`](training/README.md)
(Korean).

| | |
|---|---|
| Data | 25,589 images after cleaning · 49 CCTV cameras · `car` / `bus` / `truck` |
| Split | train 2,500 / val 1,000 / test 5,000. Split by camera and clip so that near-identical frames never land in both train and test; val and test keep the real distribution, while train was cut down with hard images (night, bad weather, tunnels …) prioritised |
| Final model | **YOLO26s-seg · imgsz 1280** — test mask mAP50-95 **0.645** · mAP50 **0.841** |

| Model | imgsz | test mask mAP50-95 | Params |
|---|---|---|---|
| YOLO26m-seg | 1280 | 0.6549 | 27.0M |
| YOLO11m-seg | 1280 | 0.6453 | 22.4M |
| **YOLO26s-seg** | **1280** | **0.6452** | **11.4M** |
| YOLO11s-seg | 1280 | 0.6376 | 10.1M |
| YOLO26n-seg | 1280 | 0.6177 | 3.1M |
| YOLO26n-seg | 640 | 0.4232 | 3.1M |

- **Resolution mattered most.** With every other setting unchanged, YOLO26n went from 0.423 to
  0.618 when imgsz was raised from 640 to 1280 — most vehicles in CCTV frames are small objects.
- **Why the s model** — the choice weighed mAP against compute (GFLOPs). It trails YOLO26m by 0.010
  with less than half the parameters.

---

## Screens

| Route | What it shows |
|---|---|
| `/overview` | Nationwide map with per-site occupancy coloured by congestion level. Live or 15-min / 1-hour / 24-hour averages, per-direction heading arrows, grouping by route or region, ranking list |
| `/` | Map console. KPIs, level-coloured markers, filters, **timeline playback** (scrub back through the last 1–24 hours), live segmentation for the selected camera |
| `/register` | Three-step camera registration. ITS auto-search → **mask editor** → details and destinations (heading computed automatically) |
| `/cameras` | Grid view (live MJPEG) and table view (bulk control). Detail pages add trend charts, captures, a **day-of-week × hour heatmap**, and full-video analysis |
| `/stats` | Averages by route, region, section or camera; sparklines, delay alerts, CSV export |
| `/apps` | Group reports with a **26 Han River bridges** preset. Daily reports, a **traffic redistribution policy scenario**, automatic insights, PDF output |
| `/system` | Model, ITS, threshold and worker status |

### Mask editor (registration step 2)

- **Auto-suggest road** — SAM 3 text prompt `"road"` produces the initial mask, with the footprint
  of vehicles touching the road filled in
- **SAM point** (click to include, Shift/right-click to exclude) · **SAM box** · **brush / eraser** · **polygon**
- **Split line** — draw along the median and the road pixels are divided between the two
  directions. The fastest way to separate a two-way road
- Wheel to zoom, Space+drag to pan, undo, hole filling, keyboard shortcuts
- Warns when the road covers too little of the frame, or when a direction has no pixels assigned

---

## Tech stack

| Area | |
|---|---|
| Models & inference | PyTorch · Ultralytics (YOLO-seg, SAM 3 / SAM 2.1) · OpenCV · NumPy · Shapely |
| Training & analysis | Ultralytics 8.4 · pandas · matplotlib · Pillow |
| Backend | FastAPI · Uvicorn · Pydantic v2 · SQLAlchemy 2.0 · psycopg 3 · httpx |
| Database | PostgreSQL — cameras, directions, occupancy time series, app groups, analysis jobs |
| Frontend | React 19 · TypeScript · Vite · React Router 7 |
| Map, charts, video | Leaflet / react-leaflet · Recharts · hls.js · Canvas mask editor |
| External | Korea ITS CCTV Open API · OpenStreetMap Nominatim · Hugging Face Hub |
| Tests | pytest (occupancy, ITS parsing, thresholds, rendering, road filling, route headings) · Puppeteer E2E |

---

## Engineering notes

Parts that were designed deliberately, for accuracy or throughput.

**ROI-cropped inference** — Only the bounding box of the road mask is fed to YOLO, and results are
mapped back to original coordinates. Instances that overlap the mask by less than half are dropped,
so vehicles on an unrelated road or car park that happen to be in frame never contaminate the
denominator. The smaller input also speeds up inference.

**Histogram-based area accumulation** — Building a separate mask per direction would scan the frame
more than twenty times. Instead, direction and vehicle class are packed into a single value
(`d × stride + c`) so one `calcHist` call counts every combination at once.

**Single-threaded GPU serialisation** — One worker thread per camera means several threads would
otherwise touch the GPU concurrently. Model loading, inference and tensor→numpy conversion all run
on one dedicated GPU thread. This exists because on macOS MPS, multiple threads touching Metal
command buffers crash the whole process.

**Frame-count-based HLS sampling** — ITS HLS delivers frames in bursts, one 2-second segment at a
time. Filtering by wall-clock time would yield only one inference per segment, so frames are picked
by count (`src_fps / INFER_FPS`) instead.

**Automatic ITS URL refresh** — Issued stream URLs are valid for 24 hours, so each worker re-fetches
its own at the 23-hour mark. When the CDN refuses a connection the retry interval backs off
exponentially, with jitter so that cameras do not all reconnect at the same instant.

**Adaptive stream mode** — Below a 60-second cadence the connection stays open and frames are simply
discarded between inferences; at 60 seconds or more the worker reconnects each cycle, grabs one
frame and disconnects. This is a compromise around the ITS CDN's limits on concurrent sessions and
reconnection rate.

**Road mask = the whole road surface** — The occupancy denominator has to include the surface that
vehicles are standing on. SAM 3's `"road"` only captures visible tarmac, so vehicles in the
registration snapshot leave holes, and the more congested the road, the smaller the denominator and
the more occupancy is inflated (Gwanggyo soundproof tunnel, jammed snapshot: road 32.1% → 47.9%
once vehicle footprints are included). One SAM 3 call therefore returns `road` and `car, truck, bus`
separately; only vehicles touching the road are merged into it, and small holes enclosed by road are
filled. Vehicles in car parks or on side roads, and large islands such as the median, are left out.

**Heading from destination signs** — Each direction is named after the destination on the road
sign visible in the frame (e.g. Seoul). CCTV coordinates on the same route are fetched from ITS and
chained into a route line, and the heading is the direction along that line towards the destination
(located via a built-in table, then OpenStreetMap Nominatim, cached). If only one destination
resolves, the other direction gets the opposite heading. Directions with no painted mask are shown
as **not measured** rather than 0%, and nothing is written to the database for them.

---

## Limitations

- **Perspective** inflates the pixel area of vehicles near the bottom of the frame. Comparisons
  across time and direction **at the same site** are therefore more trustworthy than absolute
  comparisons between sites.
- Distant (small) vehicles are easily missed, and accuracy drops sharply in **bad weather**
  (final model, test mask mAP50-95: 0.645 overall, 0.407 in bad weather).
- Vehicle class (`car` / `bus` / `truck`) accuracy is comparatively low, so the UI allows the class
  breakdown to be switched off. Occupancy itself does not depend on class.
- The denominator must be the **full driveable surface**, so masks are expected to be painted with
  the hard shoulder and median excluded.

---

## Repository layout

| | |
|---|---|
| `traffic_project/` | **The application.** Backend, frontend and ML pipeline. See [traffic_project/README.md](traffic_project/README.md) |
| `training/` | **Vehicle model training.** Data preprocessing, EDA and split scripts, the fine-tuning notebook, and the team's experiment log. See [training/README.md](training/README.md) (Korean) |
| `traffic-deploy/` | Install scripts (`install.sh`, `windows-setup.ps1`, `provision.sh` for cloud) |
| [`docs/발표자료.pdf`](docs/발표자료.pdf) | **Presentation deck**, 34 slides, final version (Korean). Slide script: [한국어](docs/presentation.ko.md) · [English](docs/presentation.en.md) |
| `setup/` | Windows power settings · WSL keepalive |
| [`DEPLOY.md`](DEPLOY.md) | **Laptop deployment notes** — install procedure, recovery commands, five problems actually hit along the way (Korean) |
| `키입력.example.txt` | Credentials template |

### Provenance of the vendored snapshots

`traffic_project/` and `traffic-deploy/` are point-in-time copies of separate repositories, with
their `.git` directories removed so that a single clone brings everything down.

| Directory | Upstream | Commit | Date |
|---|---|---|---|
| `traffic_project/` | [hantaeho123/traffic_project](https://github.com/hantaeho123/traffic_project) | `6acc9b7` sam3 alone | 2026-09-22 |
| `traffic-deploy/` | [Hanhws/traffic-deploy](https://github.com/Hanhws/traffic-deploy) | `484ad98` install kit | 2026-09-20 |

`traffic_project/` matches the demo code in the team's final submission.

These directories do not follow upstream automatically. To update, copy the newer source over them
and refresh the commit hashes in this table.

### Not in this repository

| | Why |
|---|---|
| `키입력.txt` | ITS API key, HF token, ngrok token. Copy `키입력.example.txt` and fill it in |
| `models/*.pt` | Trained on AI-Hub data, so not redistributed here |
| Training dataset | The AI-Hub source data and its preprocessed, split copy (4.4 GB) cannot be redistributed; rebuild it with the scripts in `training/` |
| `data/` `.venv/` `node_modules/` | Produced at runtime; `install.sh` creates them |

---

## Running it

Requires Python ≥ 3.10, Node ≥ 18, PostgreSQL, and optionally an NVIDIA GPU (CPU works but is
20–30× slower).

```bash
cd traffic_project
bash scripts/setup.sh          # .venv + packages + .env + database + frontend deps
# fill in ITS_API_KEY and DATABASE_URL in .env
bash scripts/run_prod.sh       # http://localhost:8000
```

Model weights go in `traffic_project/models/weights/`:

| | |
|---|---|
| Vehicle YOLO-seg | The fine-tuned checkpoint. The default filename is `yolov8s_seg_vehicle.pt`; point `YOLO_WEIGHTS` in `.env` at any other file. Use the final model (YOLO26s-seg, `yolo26_s_1280_best.pt`) together with `YOLO_IMGSZ=1280` |
| `sam3.pt` | `python scripts/download_sam3.py --token hf_xxx` — requires accepting the [facebook/sam3](https://huggingface.co/facebook/sam3) licence first |

For running this continuously on a Windows laptop (WSL2 + GPU), see [DEPLOY.md](DEPLOY.md).
