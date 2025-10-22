# Low-Light Pedestrian Detection (YOLOv8, Leak-Free Split, WBF Ensemble)

Single-class pedestrian detector for **low-light** images (LLVIP subset).  
This repo provides a **scene/near-duplicate safe split**, tuned **YOLOv8s/n** training recipes, **inference parameter sweeps**, and **Weighted Box Fusion (WBF)** ensembling to produce a robust final submission.

> Low-light data is prone to label noise, tiny objects, and validation leakage. To fix this we split the data first, then squeezes reliable gains with smart inference and ensembling.

---

## 📸 Illustrations (add these in `/docs/`)

- **Hero banner**: side-by-side **original vs detection** on a dark street  
  *Place at top of README:* `![Hero](docs/hero_detect.jpg)`
- **Leakage visualization**: montage of **train/val near-duplicates** before/after new split  
  *Place under Split section:* `![Leakage](docs/split_leakage.png)`
- **Quantitative chart**: **PR/F1 vs conf** curves on the new val set  
  *Place under Inference Tuning:* `![Thresholds](docs/threshold_sweep.png)`
- **Qualitative gallery**: grid of test predictions (small/occluded pedestrians)  
  *Place under Results:* `![Qualitative](docs/qual_gallery.jpg)`

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
