[한국어](presentation.ko.md) · [English]

# Presentation — Slide Script (English)

Original deck: [`발표자료.pdf`](발표자료.pdf) · 35 slides · Canva (Korean)

> This file carries the English text for every slide. To produce the English deck, duplicate the
> Canva design and swap the text in — the layout and figures stay as they are.

---

## 1. Title

**Measuring Traffic Congestion from the Vehicle-to-Road Area Ratio in CCTV Footage, Using Segmentation**

SNU KDT Cohort 13, Team 6 — Seojin Kwon, Juhee Lee, Seoyoung Jeon, Uijin Jeong, Wooseok Han, Taeho Han

## 2. Contents

| | |
|---|---|
| 01 | Topic and motivation |
| 02 | Road segmentation |
| 03 | Vehicle segmentation |
| 04 | Demo |
| 05 | Applications |
| 06 | Pain points |

---

# 01. Topic and Motivation

## 3. The problem

- Driven partly by the rise in single-person households, registered vehicles in Korea reached
  **26.3 million** (December 2024) and keep growing every year
- Congestion is expected to keep worsening
- Concrete, workable transport policy measures are needed to relieve it
- According to the Ministry of Land, Infrastructure and Transport and the Korea Transport
  Institute, congestion cost the country roughly **KRW 81.3 trillion in 2023**

> Source: *2025 National Transport Policy Evaluation Indicator Survey*, Vol. 3 — Congestion Cost (2023)

## 4. What it feels like

- Commute delays
- Holiday exodus and return traffic
- Specific roads backing up

## 5. Our idea

> Build a system that turns road CCTV footage into **numbers you can actually make transport
> decisions with**

## 6. Topic

**Measuring traffic congestion from the vehicle-to-road area ratio in CCTV footage, using segmentation**

## 7. Prior work

DAMI Lab, Department of Computer Science and AI, Dongguk University

## 8. Defining congestion

```
                   vehicle segmentation pixels
congestion  =  ──────────────────────────────────
                    road segmentation pixels
```

> Reference: *Emergency Information Extraction of Transport Congestion Based on Object-oriented
> Classification Using High Spatial Resolution Remote Sensing Image*

## 9. Processing flow

Image input → count road pixels → count vehicle pixels → compute congestion

## 10. Topic (restated)

Measuring traffic congestion from the vehicle-to-road area ratio in CCTV footage, using segmentation

---

# 02. Road Segmentation

## 11. Approach

- Road segmentation = **SAM 3 + manual refinement**
- Vehicle segmentation = **YOLO-seg**

## 12. Problems we had to solve

1. How to detect the road itself
2. How to determine the **direction** of travel
3. How to handle frames containing **multiple roads**

## 13. Attempt 1 — automatic road detection

Segment the road with Mask2Former or SAM 3, using lane markings

## 14. Attempt 2 — direction from motion

Use the median strip plus **eight consecutive frames** to infer direction

## 15. Conclusion

Neither worked reliably → but the road only has to be captured **once** → **SAM 3 + manual refinement**

> Because each camera is fixed, the expensive step can be pushed to registration time and done once.

---

# 03. Vehicle Segmentation

## 16. Dataset

AI-Hub — CCTV traffic footage for solving transport problems (highway)

## 17. Shrinking the data

```
550 GB  →  60 GB  →  13 GB  →  4 GB
       validation   convert    downsample
          only      to JPG
```

## 18. Cleaning

- Removed **524** images with no labels
- Removed **197** duplicate copies sharing a filename
- **26,310 → 25,589** images

## 19–20. (Figures)

## 21. Downsampling strategy

Problem: the dataset was too large, and performance was poor in rain and at night
→ downsample so that **hard images are over-represented**

| | train | val | test |
|---|---|---|---|
| Before | 20,000 | 2,500 | 2,500 |
| After | 2,500 | 1,000 | 5,000 |

## 22. easy / hard criteria

- **easy** — daytime *and* clear weather, both satisfied
- **hard** — either condition not met

## 23. Why YOLO

1. **Training cost**
2. **Realtime capability**

YOLO is CNN-based and therefore lightweight.

## 24. How we ran experiments

Each member trained a different YOLO variant with different hyperparameters, and results were
shared in a common tracking sheet.

## 25. Model selection

Judged on **mAP and GFLOPs**; **YOLO26s-seg** selected as the final model.

## 26. Segmentation output

YOLO26s-seg segmentation results

## 27. Training

## 28. Final performance

| Metric | Value |
|---|---|
| mAP50-95-seg (test) | **0.6452** |
| mAP50-seg (test) | **0.8414** |

---

# 04. Demo

## 29. The service

Demo: `https://traffic-project-lilac.vercel.app/`

Built on the Open API of the Intelligent Transport Systems (ITS) centre, Korea's public data
gateway for intelligent transport data.

---

# 05. Applications

# 30. Policy case

Congestion cost Korea roughly **KRW 81.3 trillion** in 2023.

- Transport infrastructure investment decisions
- Analysing where road expansion is needed
- Measuring the effect of transport policy

## 31. Commercial ideas

- A sharp rise in occupancy on a given stretch gives **early warning of an accident or sudden jam**
- Compare congestion across roads to **suggest detours**
- Works on **existing CCTV infrastructure** — no new hardware to install

## 32. Comparable services

| Rank | Service | Existing approach |
|---|---|---|
| 1 | Ministry of the Interior and Safety — AI CCTV traffic analysis | Existing CCTV plus AI to classify 12 vehicle types and count traffic per lane |
| 2 | Nota · RaonRoad | CCTV AI analysis for traffic volume and congestion, feeding operations and signal control |
| 3 | Korea Expressway Corporation VDS | Loop detectors measuring volume, mean speed, occupancy and other indicators |

---

# 06. Pain Points

## 33. Limitations

- Distant vehicles are often missed
- Overall performance drops in low light and similar conditions
- **Perspective correction** remains unsolved
- Only the segmented road area is processed, so anything outside it is ignored
- Incoming camera frames do not contain roads and vehicles alone

## 34–35. (Closing slides)
