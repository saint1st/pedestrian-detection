# final_infer.py

import os, json, glob, shutil, re, subprocess, sys, uuid
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import torch
from torchvision.ops import nms

MODEL_PREDS = [
    r'X:\Desktop\sergek\preds_raw\train',
    r'X:\Desktop\sergek\preds_raw\yolo_baseline',
    r'X:\Desktop\sergek\preds_raw\yolo_new_split'
   
]


MODEL_WEIGHTS = [1.2, 1.0, 1.0]  

OUT_DIR = Path(r'X:\Desktop\sergek\final_dets')
EVAL_ROOT = Path(r'X:\Desktop\sergek\evaluate\evaluate')
RUN_EVAL = True  # set False to skip evaluator


# ------------ Best-found fixed config --------------
WBF_IOU_THR        = 0.65
WBF_SKIP_BOX_THR   = 0.005
FINAL_SCORE_THR    = 0.20
FINAL_NMS_IOU      = 0.55
MIN_MODELS         = 1           
MAX_DETS           = 5          
SUPPORT_IOU        = 0.50        
# ---------------------------------------------------

def _parse_percent(s: str) -> float:
    s = s.strip().rstrip('%').replace(',', '.')
    return float(s) / 100.0

def load_index(folder: str):
    files = glob.glob(os.path.join(folder, '*.json'))
    return {os.path.basename(f): f for f in files}

def union_image_list(indices):
    names = set()
    for idx in indices:
        names.update(idx.keys())
    return sorted(names)

def xyxy_norm_to_xyxy_pixels(b, W, H):
    x1, y1, x2, y2 = b
    x1 *= W; y1 *= H; x2 *= W; y2 *= H
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(0, min(int(round(x2)), W - 1))
    y2 = max(0, min(int(round(y2)), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def xyxy_to_xywh(x1, y1, x2, y2):
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def build_preds_cache(model_pred_dirs: List[str]):
    idxs = [load_index(m) for m in model_pred_dirs]
    names = union_image_list(idxs)
    cache = []
    for mi, idx in enumerate(idxs):
        mcache = {}
        for n in names:
            if n in idx:
                data = json.load(open(idx[n], 'r'))
            else:
                data = {"width": None, "height": None, "boxes": [], "scores": [], "labels": []}
            mcache[n] = data
        cache.append(mcache)
    return names, cache

def run_final(names, cache, out_det_dir: Path):
  
    if len(MODEL_WEIGHTS) != len(cache):
        w = list(MODEL_WEIGHTS) + [1.0] * (len(cache) - len(MODEL_WEIGHTS))
        weights = [wi / sum(w) for wi in w]
    else:
        sw = sum(MODEL_WEIGHTS); weights = [w / sw for w in MODEL_WEIGHTS]

    if out_det_dir.exists():
        shutil.rmtree(out_det_dir)
    out_det_dir.mkdir(parents=True, exist_ok=True)

    for name in tqdm(names, desc="Final WBF + NMS"):
        boxes_list, scores_list, labels_list = [], [], []
        pix_boxes_per_model = []
        img_w = img_h = None

        for mcache in cache:
            data = mcache[name]
            if img_w is None and data["width"] is not None:
                img_w, img_h = data["width"], data["height"]
            boxes_list.append(data["boxes"])
            scores_list.append(data["scores"])
            labels_list.append(data["labels"])
    
            if data["width"] is not None:
                pix_boxes = [xyxy_norm_to_xyxy_pixels(b, data["width"], data["height"]) for b in data["boxes"]]
            else:
                pix_boxes = []
            pix_boxes_per_model.append(pix_boxes)

        out_txt = out_det_dir / name.replace('.json','').replace('.jpg','.txt').replace('.png','.txt')
        if sum(map(len, boxes_list)) == 0 or img_w is None:
            out_txt.write_text('')
            continue

        # WBF
        fb, fs, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_BOX_THR
        )

        # Score filter
        keep_idx_score = [i for i, sc in enumerate(fs) if sc >= FINAL_SCORE_THR]
        fb = [fb[i] for i in keep_idx_score]
        fs = [fs[i] for i in keep_idx_score]
        if not fb:
            out_txt.write_text('')
            continue

       
        xyxy_pix = [xyxy_norm_to_xyxy_pixels(b, img_w, img_h) for b in fb]
        supporters_list = [1] * len(xyxy_pix)  

  
        if MIN_MODELS > 1:
            consensus_keep = []
            for idx_box, B in enumerate(xyxy_pix):
                supporters = 0
                for model_boxes in pix_boxes_per_model:
                    if any(iou_xyxy(B, Mb) >= SUPPORT_IOU for Mb in model_boxes):
                        supporters += 1
                if supporters >= MIN_MODELS:
                    consensus_keep.append((idx_box, supporters))
            if not consensus_keep:
                out_txt.write_text('')
                continue
            idxs_keep = [i for i, s in consensus_keep]
            supporters_list = [s for i, s in consensus_keep]
            xyxy_pix = [xyxy_pix[i] for i in idxs_keep]
            fs = [fs[i] for i in idxs_keep]

        # Final NMS
        boxes_t = torch.tensor(xyxy_pix, dtype=torch.float32)
        scores_t = torch.tensor(fs, dtype=torch.float32)
        keep_idx = nms(boxes_t, scores_t, FINAL_NMS_IOU).tolist()
        boxes_t = boxes_t[keep_idx]
        scores_t = scores_t[keep_idx]
        supporters_list = [supporters_list[i] for i in keep_idx]

        # Cap detections
        if boxes_t.shape[0] > MAX_DETS:
            order = torch.argsort(scores_t, descending=True)[:MAX_DETS]
            boxes_t = boxes_t[order]
            scores_t = scores_t[order]
            supporters_list = [supporters_list[i.item()] for i in order]

     
        with open(out_txt, 'w') as f:
            for (x1, y1, x2, y2), sc in zip(boxes_t.tolist(), scores_t.tolist()):
                xmin, ymin, w, h = xyxy_to_xywh(int(x1), int(y1), int(x2), int(y2))
                f.write(f"person {sc:.6f} {xmin} {ymin} {w} {h}\n")

def run_evaluator(det_dir: Path) -> Tuple[float, float]:

    target = EVAL_ROOT / 'detections'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(det_dir, target)

    def call_thresh(thr: float) -> float:
       
        tag = f"results_{uuid.uuid4().hex[:8]}_{int(thr*100)}"
        save_rel = tag
        save_abs = EVAL_ROOT / save_rel
        save_abs.mkdir(parents=True, exist_ok=True)

        proc = subprocess.run(
            [
                sys.executable, "pascalvoc.py",
                "--threshold", str(thr),
                "--noplot",
                "--savepath", save_rel
            ],
            cwd=str(EVAL_ROOT),
            input="Y\n",  
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False
        )


        results_txt = save_abs / "results.txt"
        ap_val = None
        if results_txt.exists():
            with open(results_txt, "r", encoding="utf-8-sig", errors="ignore") as f:
                for raw in f:
                    line = raw.replace('\x00', '').strip()
                    m = re.match(r"^AP:\s*([0-9]+(?:[.,][0-9]+)?)\s*%", line, re.IGNORECASE)
                    if m:
                        try:
                            ap_val = _parse_percent(m.group(1))
                            break
                        except Exception:
                            pass
            if ap_val is None:
                with open(results_txt, "r", encoding="utf-8-sig", errors="ignore") as f:
                    for raw in f:
                        line = raw.replace('\x00', '').strip()
                        m = re.match(r"^mAP:\s*([0-9]+(?:[.,][0-9]+)?)\s*%", line, re.IGNORECASE)
                        if m:
                            try:
                                ap_val = _parse_percent(m.group(1))
                                break
                            except Exception:
                                pass

        if ap_val is None:
            m = re.search(r"AP:\s*([0-9]+(?:[.,][0-9]+)?)\s*%(?:\s*\(person\))?", proc.stdout, re.IGNORECASE)
            if m:
                ap_val = _parse_percent(m.group(1))

        if ap_val is None:
            print("----- pascalvoc.py output -----")
            print(proc.stdout)
            print("----- end output -----")
            raise RuntimeError(f"Could not parse AP at IoU={thr} from {results_txt}")

        return ap_val

    ap50 = call_thresh(0.5)
    ap75 = call_thresh(0.75)
    return ap50, ap75

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det_dir = OUT_DIR / 'detections'

    names, cache = build_preds_cache(MODEL_PREDS)

    run_final(names, cache, det_dir)
    if RUN_EVAL:
        ap50, ap75 = run_evaluator(det_dir)
        score = (1.0 - ap50 + 1.0 - ap75) / 2.0
        print("\n=== FINAL RESULTS (fixed best config) ===")
        print(f"AP50 = {ap50:.4f}")
        print(f"AP75 = {ap75:.4f}")
        print(f"Kaggle MAE-to-1 score = {score:.5f}")
    else:
        print(f"Detections written to: {det_dir}")

if __name__ == "__main__":
    main()
