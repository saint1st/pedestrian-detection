# Low-Light Pedestrian Detection (YOLOv8, Leak-Free Split, WBF Ensemble)

Single-class pedestrian detector for **low-light** images (LLVIP subset).  
This repo provides a **scene/near-duplicate safe split**, tuned **YOLOv8s/n** training recipes, **inference parameter sweeps**, and **Weighted Box Fusion (WBF)** ensembling to produce a robust final submission.

> Low-light data is prone to label noise, tiny objects, and validation leakage. To fix this we split the data first, then squeezes reliable gains with smart inference and ensembling.

---



---

## 🌱 Dataset & Paths

Link to Kaggle: https://www.kaggle.com/competitions/find-person-in-the-dark/overview

**Title**: Find Person in the Dark
**Description**: Pedestrian detection in low light conditions.
**Dataset**: The dataset used in this competition contains **15030 visible light images**(11782 images for training and 3248 for testing), it is a subset of LLVIP: A Visible-infrared Paired Dataset for Low-light Vision. 

> Leak-Free, Scene-Aware Split

Groups near-duplicate frames via perceptual hash (pHash) and splits by group to avoid train/val leakage.
```tools/split_scene_aware.py```

> Inference Tuning (per-model conf/IoU sweep)
```tools/tune_inference.py``` — finds best (conf, iou) for each model on your validation.

> Predict (export per-image JSON for ensembling)
```tools/predict_save.py```

> Ensembling (WBF) → YOLO-TXT & CSV
```tools/ensemble_wbf.py```

## Results

#  pedestrian-detection: YOLOv8 Experiment Log

This repository documents the training, experimentation, and results for a pedestrian detection task, primarily utilizing **YOLOv8** models. The focus of this work was to move beyond the baseline and explore the impact of augmentation, custom training loops, and data splitting strategies on final performance.

---

## 🚀 Final Results Summary

The most significant performance gain was achieved by combining a clean data split with careful inference parameter tuning.

| Model Name | Local mAP@0.5 | Local mAP@0.75 | Final Kaggle Score (Lower is Better) | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **YOLOv8s - Fine-tuned (Best Model)** | **0.9025** | **0.5348** | **0.28135** | Data hygiene + calibrated inference is key. |
| YOLOv8s - Heavy Augmentation | 0.8800 | 0.5071 | 0.30645 | High local AP, but overfitting/FP issues. |
| YOLOv8n - Baseline | 0.8468 | 0.4839 | 0.33465 | Solid starting point (YOLOv8n). |
| YOLOv8s - Clean Split + imgsz=960 | 0.8382 | 0.4677 | 0.34705 | Confirmed stability on new split. |
| YOLOv8s - Custom Trainer | 0.6791 | 0.2397 | 0.54060 | Custom pipeline flaws significantly hurt performance. |

---

## 📈 Training Process Illustrations

The following plots demonstrate the training and validation behavior of the **YOLOv8n Baseline model** over 50 epochs. These visualizations are critical for diagnosing convergence, loss trends, and potential overfitting/underfitting.


### Key Observations from Baseline Plots:

* **Loss Convergence:** All loss components (`box_loss`, `cls_loss`, `dfl_loss`) show smooth and consistent convergence across both training (`train/`) and validation (`val/`) sets.
* **Validation Loss:** Validation losses consistently track or stay slightly above the training losses, indicating good generalization without severe overfitting.
* **Metrics:** Metrics (Precision, Recall, mAP) show rapid increase in the first 10-20 epochs, plateauing afterwards.

---

## 🔬 Detailed Model Experimentation

This section provides the full configuration and results for each model variant tested.

### 1. YOLOv8n — Baseline (Random Split)
* **Setup:** `yolov8n.pt` weights, 50 epochs, `imgsz=640`, `batch=16`, using Ultralytics defaults.
* **Notes:** A solid starter model demonstrating the baseline performance on the original data split. Limited by the small backbone size for handling small objects and low-light conditions.

![Training results] (plots/baseline.png)

| Metric ID | Value |
| :--- | :--- |
| `AP@0.5` | 0.8468 |
| `AP@0.75` | 0.4839 |
| **Final Score** | **0.33465** |

### 2. YOLOv8s — CLAHE + Heavy Augmentation (Random Split)
* **Setup:** `yolov8s.pt`, `imgsz=800`, 100 epochs, **SGD optimizer**, `mosaic=1.0`, `hsv_v=0.5`, `scale=0.5`. Used **CLAHE** as a preprocessing step.
* **Notes:** Achieved high local AP, but the aggressive augmentations (CLAHE, high HSV values) and noisy data likely introduced artifacts, leading to an increase in False Positives and a slight degradation in the final score compared to the best model.

![Training results] (plots/model2.png)

| Metric ID | Value |
| :--- | :--- |
| `AP@0.5` | 0.8800 |
| `AP@0.75` | 0.5071 |
| **Final Score** | **0.30645** |

### 3. YOLOv8s — Custom Trainer + Albumentations (Random Split)
* **Setup:** Custom Dataset/collate function; heavy **Albumentations** pipeline (Flip, Rotate90, ColorJitter, Resize); `imgsz=800`.
* **Notes:** The custom pipeline resulted in the worst performance. Issues included a pipeline mismatch (e.g., handling bounding box coordinates during `Rotate90`), inconsistent scaling, and an initial error of dropping "negative" samples, causing significant divergence from the standard Ultralytics trainer behavior.

![Training results] (plots/model3.png)

| Metric ID | Value |
| :--- | :--- |
| `AP@0.5` | 0.6791 |
| `AP@0.75` | 0.2397 |
| **Final Score** | **0.54060** |

### 4. YOLOv8s — Clean Recipe on Leak-Free Split (New Split)
* **Setup:** `yolov8s.pt`, `imgsz≈960`, **AdamW** + cosine LR schedule, reduced `mosaic≈0.25`, no `rotate90`, negatives kept, and a single-class loss bias (increased box loss weight, reduced classification loss weight, added focal loss).
* **Notes:** Focused on data hygiene by using a **leak-free (scene-aware)** data split. This configuration served as a fast, stable backbone model used for ensembling and threshold tuning.

![Training results] (plots/model4.png)

| Metric ID | Value |
| :--- | :--- |
| `AP@0.5` | 0.8382 |
| `AP@0.75` | 0.4677 |
| **Final Score** | **0.34705** |

### 5. YOLOv8s — Fine-tuned (from Model #2) on Leak-Free Split + Inference Tuning
* **Setup:** Started from the best weights of Model #2, then fine-tuned on the **new leak-free split** (from Model #4). Inference parameters were swept and optimized, leading to the best result at `conf=0.10` and `iou=0.55`.
* **Notes:** This model proved that the combination of clean, high-quality data (new split) and meticulous calibration of post-processing (confidence and NMS thresholds) yielded the greatest gain, moving the final score to the top result.

![Training results] (plots/model5.png)

| Metric ID | Value |
| :--- | :--- |
| `AP@0.5` | 0.8586 |
| `AP@0.75` | 0.4549 |
| **Final Score** | **0.34325** |
