# Low-Light Pedestrian Detection (YOLOv8, Leak-Free Split, WBF Ensemble)

Single-class pedestrian detector for **low-light** images (LLVIP subset).  
This repo provides a **scene/near-duplicate safe split**, tuned **YOLOv8s/n** training recipes, **inference parameter sweeps**, and **Weighted Box Fusion (WBF)** ensembling to produce a robust final submission.

> 💡 **Why this repo?**  
> Low-light data is prone to label noise, tiny objects, and validation leakage. This template fixes the split first, then squeezes reliable gains with smart inference and ensembling.

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






Environment
pip install ultralytics ensemble-boxes numpy opencv-python tqdm

