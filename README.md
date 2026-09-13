# Dental-Pathology-Detection-SSL
Self-supervised learning for label-efficient dental pathology detection and multi-object tracking on panoramic X-rays using YOLOv12 and ByteTrack.


## Overview

This project investigates whether self-supervised learning (SSL) can improve label-efficient object detection for dental panoramic X-rays. The study uses unlabeled dental X-ray images to learn visual representations before fine-tuning a YOLOv12-m detector with only a fraction of the available bounding-box annotations.

Four self-supervised learning approaches are investigated: SimCLR, BYOL, I-JEPA, and DINOv3. Their learned representations are transferred to a YOLOv12-m detection backbone and compared against random initialization and COCO-pretrained initialization.

The project also evaluates the best-performing SSL-initialized detector for multi-object tracking using ByteTrack. The complete pipeline covers dataset analysis, self-supervised representation learning, label-efficient object detection, error analysis, and multi-object tracking.

## Key Features

* Dental panoramic X-ray dataset exploration and preprocessing
* Leakage-safe train/validation/test partitioning
* YOLOv10, YOLOv12, YOLOv26, and RF-DETR baseline comparison
* Self-supervised pretraining using:

  * SimCLR
  * BYOL
  * I-JEPA
  * DINOv3
* Label-efficient detection using only 20% of the available labeled images
* SSL-to-YOLOv12 backbone transfer
* Detection evaluation using mAP@50, mAP@50:95, precision, and recall
* Per-class performance and error analysis
* Representation-quality analysis using embedding similarity and t-SNE
* Multi-object tracking using ByteTrack
* Tracking evaluation using MOTA, MOTP, IDF1, identity switches, and fragmentation



## Dataset

The project uses a dental panoramic X-ray dataset containing:

| Property                  |                                         Value |
| ------------------------- | --------------------------------------------: |
| Images                    |                                        10,000 |
| Image resolution          |                                     640 × 640 |
| Classes                   |                                             4 |
| Annotation types          | Bounding boxes and polygon-format annotations |
| Train / Validation / Test |                               80% / 10% / 10% |

### Classes

* Cavities
* Damage
* Infection
* Wisdom

The dataset was analyzed for class imbalance, duplicate images, annotation consistency, object-size distribution, image characteristics, and potential data leakage before model training.


## Methodology

```text
                         Dental X-ray Images
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Part A: Data &       │
                    │  Detection Baselines  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Dataset EDA &          │
                    │ Preprocessing          │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
          YOLOv10            YOLOv12            YOLOv26
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                       YOLOv12-m Selected
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Part B: Data          │
                    │ Partitioning          │
                    └───────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │ 80% Unlabeled       │
                     │ SSL Pool            │
                     └──────────┬──────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
       SimCLR                  BYOL             I-JEPA / DINOv3
          │                     │                     │
          │                     │              ViT Representations
          │                     │                     │
          │                     │                     ▼
          │                     │              CNN Distillation
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                                ▼
                       YOLOv12-m Backbone
                                │
                                ▼
                     20% Labeled Data
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Dental Pathology      │
                    │ Object Detection      │
                    └───────────┬───────────┘
                                │
                                ▼
                       Detection Results
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Best SSL Detector     │
                    │ (DINOv3)              │
                    └───────────┬───────────┘
                                │
                                ▼
                           ByteTrack
                                │
                                ▼
                     Multi-Object Tracking
                                │
                                ▼
                       Tracking Analysis



## 5. Part A Results



## Part A — Detection Baselines

Four object detection approaches were evaluated as part of the initial benchmark.

| Model | Precision | Recall | mAP@50 | mAP@50:95 | F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv10-m | 0.621 | 0.579 | 0.477 | 0.221 | 0.599 | 52.2 |
| **YOLOv12-m** | **0.655** | 0.572 | **0.481** | **0.226** | **0.611** | 24.4 |
| YOLOv26-m | 0.621 | 0.550 | 0.461 | 0.213 | 0.583 | 39.7 |

YOLOv12-m achieved the highest mAP@50 and mAP@50:95 among the evaluated YOLO models and was therefore selected as the downstream detector for Part B.
:::

---

## 6. Part B — Self-Supervised Learning



| Method | Main idea | Transfer |
|---|---|---|
| SimCLR | Contrastive representation learning | CNN → YOLOv12 |
| BYOL | Bootstrap representation learning without negatives | CNN → YOLOv12 |
| I-JEPA | Masked representation prediction | ViT → CNN distillation → YOLOv12 |
| DINOv3 | Teacher-student self-distillation | ViT → CNN distillation → YOLOv12 |

---

## 7. Main Results


All downstream experiments use YOLOv12-m with a 20% labeled-data budget.

| Initialization | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|
| Random | 0.4448 | 0.5229 | 0.4448 | 0.1916 |
| I-JEPA | 0.4335 | 0.5274 | 0.4487 | 0.1941 |
| BYOL | 0.4604 | 0.5293 | 0.4641 | 0.2019 |
| SimCLR | 0.4847 | 0.5183 | 0.4699 | 0.2050 |
| **DINOv3** | 0.4704 | **0.5318** | 0.4679 | **0.2054** |
| COCO | **0.5045** | 0.5187 | **0.4796** | 0.2073 |

### Key Finding

DINOv3 achieved the strongest performance among the SSL approaches, reaching 0.2054 mAP@50:95 with only 20% labeled data. This was close to the COCO-initialized baseline at 0.2073 mAP@50:95, suggesting that domain-specific self-supervised pretraining can provide useful representations for label-efficient dental X-ray detection.
:::

---

## 8. Tracking Results

Add your ByteTrack results.

| Metric | Result |
|---|---:|
| MOTA | 0.047 |
| MOTP | 0.300 |
| IDF1 | 0.137 |
| ID Switches | 0 |
| Mostly Tracked | 5 / 45 |
| Mostly Lost | 39 / 45 |
| Fragmentations | 22 |



> The zero identity switches indicate that ByteTrack maintained correct identities when detections were available. However, the low MOTA and high number of mostly-lost objects were primarily caused by missed detections from the upstream detector.


---



## Limitations

- Experiments were conducted using a single random seed.
- I-JEPA training was constrained by available computational resources.
- The study uses a single dental X-ray dataset containing four pathology classes.
- Tracking evaluation was performed using a synthetic panning video.
- The label-efficiency ablation should be interpreted separately because some exploratory runs used different training configurations.


