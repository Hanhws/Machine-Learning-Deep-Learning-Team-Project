[한국어](presentation.ko.md) · [English]

# Presentation — Slide Script (English)

Original deck: [`발표자료.pdf`](발표자료.pdf) · 34 slides · Canva (Korean) · final version (2026-09-23)

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
| 06 | Future work |

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

## 11. Approach

- Vehicle segmentation = **YOLO-seg**
- Road segmentation = **SAM 3 + manual refinement**

---

# 02. Road Segmentation

## 12. Problems we had to solve

1. How to detect the road itself
2. How to determine the **direction** of travel
3. How to handle frames containing **multiple roads**

## 13. The road itself

Segment the road with Mask2Former or SAM 3, using lane markings

## 14. Direction of travel

Use the median strip plus **whether vehicles show their front or their rear** to infer direction

## 15. Multiple roads

Could not be solved automatically → but the road only has to be captured **once** →
**SAM 3 + manual refinement**

> Because each camera is fixed, the expensive step can be pushed to registration time and done once.

---

# 03. Vehicle Segmentation

## 16. Model — why YOLO

1. **Training cost**
2. **Realtime capability**

YOLO is CNN-based and therefore lightweight.

## 17. EDA — dataset

AI-Hub — CCTV traffic footage for solving transport problems (highway)

## 18. EDA — shrinking the data

```
550 GB  →  60 GB  →  13 GB  →  4 GB
       validation   convert    downsample
          only      to JPG
```

## 19. EDA — data cleansing

- Removed **524** images with no labels
- Removed **197** duplicate copies sharing a filename
- **26,310 → 25,589** images

## 20–21. EDA (figures)

## 22. EDA — dataset reduction

Problem: the dataset was too large, and performance was poor in rain and at night
→ reduce the dataset so that **hard images are over-represented**

| | train | val | test |
|---|---|---|---|
| Before | 20,000 | 2,500 | 2,500 |
| After | 2,500 | 1,000 | 5,000 |

## 23. EDA — easy / hard criteria

- **easy** — daytime *and* clear weather, both satisfied
- **hard** — either condition not met

## 24. Results — how we ran experiments

Each member trained a different YOLO variant with different hyperparameters, and results were
shared in a common tracking sheet.

> The spreadsheet linked on the slide is the team's experiment log.
> Copy in this repository: [`training/results/결과테이블.csv`](../training/results/결과테이블.csv)

## 25. Results — model selection

Judged on **mAP and GFLOPs**; **YOLO26s-seg** selected as the final model.

## 26. Results — segmentation output

YOLO26s-seg segmentation results

## 27. Results — training

## 28. Results — final performance

| Metric | Value |
|---|---|
| mAP50-95-seg (test) | **0.6339** |
| mAP50-seg (test) | **0.8414** |

> The experiment log records mAP50-95-seg (test) as **0.6452** for the same run
> (yolo26s-seg · imgsz 1280), which is also what the previous version of this slide (09-20) showed.

---

# 04. Demo

## 29. The service

Demo: `https://traffic-project-lilac.vercel.app/`

Built on the Open API of the Intelligent Transport Systems (ITS) centre, Korea's public data
gateway for intelligent transport data.

---

# 05. Applications

## 30. Commercial ideas

- A sharp rise in congestion on a given stretch gives **early warning of an accident or sudden jam**
- Compare congestion across roads to **suggest detours**
- Works on **existing CCTV infrastructure** — no new hardware to install

## 31. Policy use

- Transport infrastructure investment decisions
- Analysing where road expansion is needed
- Measuring the effect of transport policy

---

# 06. Future Work

## 32. Future work

- Distant vehicles are often missed
- Overall performance drops in low light and similar conditions
- **Perspective correction** remains unsolved
- Only the segmented road area is processed, so anything outside it is ignored
- Incoming camera frames do not contain roads and vehicles alone

## 33–34. (Closing slides)

Thank you · Q & A

---

## Changes from the previous version (2026-09-20, 35 slides)

- Section 03 order: "Why YOLO" moved ahead of the EDA slides (now slide 16)
- Slide 14: direction cue changed from "eight consecutive frames" to "front or rear view of vehicles"
- Slide 19 titled "data cleansing"; slide 22 "downsampling" → "dataset reduction"
- Slide 28: test mAP50-95-seg 0.6452 → 0.6339
- Section 05: commercial ideas now come first; "policy case" became "policy use" (KRW 81.3 trillion
  line dropped); the comparable-services slide was removed
- Section 06: "pain points" → "future work"
