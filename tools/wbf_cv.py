# sweep_wbf.py
# Grid-search WBF & post-filters, run official evaluator, pick best (lowest MAE-to-1 score).

import os, json, glob, csv, shutil, re, subprocess, sys
from pathlib import Path
from itertools import product
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import torch
from torchvision.ops import nms
import uuid

MODEL_PREDS = [
    r'X:\Desktop\sergek\preds_raw\train',
    r'X:\Desktop\sergek\preds_raw\yolo_baseline',
    r'X:\Desktop\sergek\preds_raw\yolo_new_split',
]


MODEL_WEIGHTS = [1.2, 1.0, 1.0]  

EVAL_ROOT = Path(r'X:\Desktop\sergek\evaluate\evaluate')
SWEEP_OUT = Path(r'X:\Desktop\sergek\wbf_sweep')

IOU_THR_LIST        = [0.60, 0.65, 0.68]
SKIP_BOX_THR_LIST   = [0.005, 0.01, 0.02]
FINAL_SCORE_THR_LST = [0.20, 0.22, 0.25]
FINAL_NMS_IOU_LIST  = [0.50, 0.55]
MIN_MODELS_LIST     = [1, 2]     
MAX_DETS_LIST       = [4, 5]       
SUPPORT_IOU         = 0.50         
# ---------------------------------------------------

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

def build_preds_cache():
    """Load all model JSONs once into memory (fast repeat runs)."""
    idxs = [load_index(m) for m in MODEL_PREDS]
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

def run_wbf_variant(names, cache, cfg, out_det_dir, debug_csv=None):
    """
    Build detections for a given (IOU_THR, SKIP_BOX_THR, FINAL_SCORE_THR, FINAL_NMS_IOU, MIN_MODELS, MAX_DETS),
    writing competition-format files under out_det_dir.
    """
    IOU_THR, SKIP_BOX_THR, FINAL_SCORE_THR, FINAL_NMS_IOU, MIN_MODELS, MAX_DETS = cfg
    # Normalize model weights
    sw = sum(MODEL_WEIGHTS); weights = [w / sw for w in MODEL_WEIGHTS]

    if out_det_dir.exists():
        shutil.rmtree(out_det_dir)
    out_det_dir.mkdir(parents=True, exist_ok=True)

    cw = None
    if debug_csv is not None:
        debug_csv.parent.mkdir(parents=True, exist_ok=True)
        csvf = open(debug_csv, 'w', newline='')
        cw = csv.writer(csvf)
        cw.writerow(['image_id','confidence','xmin','ymin','width','height','supporters'])

    for name in tqdm(names, desc=f"WBF {cfg}"):
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
            # pixel boxes for consensus
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
            weights=weights, iou_thr=IOU_THR, skip_box_thr=SKIP_BOX_THR
        )

        # Score filter
        keep_idx_score = [i for i, sc in enumerate(fs) if sc >= FINAL_SCORE_THR]
        fb = [fb[i] for i in keep_idx_score]
        fs = [fs[i] for i in keep_idx_score]
        if not fb:
            out_txt.write_text('')
            continue

        # To pixel xyxy
        xyxy_pix = [xyxy_norm_to_xyxy_pixels(b, img_w, img_h) for b in fb]

        # Consensus (if MIN_MODELS > 1)
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
        else:
            supporters_list = [1] * len(xyxy_pix)

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

        # Write competition format
        with open(out_txt, 'w') as f:
            for (x1, y1, x2, y2), sc, supp in zip(boxes_t.tolist(), scores_t.tolist(), supporters_list):
                xmin, ymin, w, h = xyxy_to_xywh(int(x1), int(y1), int(x2), int(y2))
                f.write(f"person {sc:.6f} {xmin} {ymin} {w} {h}\n")
                if cw: cw.writerow([name, f"{sc:.6f}", xmin, ymin, w, h, supp])

    if debug_csv is not None:
        csvf.close()

def run_evaluator(det_dir: Path):
   
    target = EVAL_ROOT / 'detections'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(det_dir, target)

    def _parse_percent(s: str):
        s = s.strip().rstrip('%').replace(',', '.')
        return float(s) / 100.0

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
    names, cache = build_preds_cache()
    SWEEP_OUT.mkdir(parents=True, exist_ok=True)
    results = [] 

    grid = list(product(IOU_THR_LIST, SKIP_BOX_THR_LIST, FINAL_SCORE_THR_LST,
                        FINAL_NMS_IOU_LIST, MIN_MODELS_LIST, MAX_DETS_LIST))
    print(f"Total variants: {len(grid)}")

    for cfg in grid:
        iou_thr, skip_thr, final_sc, nms_iou, min_models, max_dets = cfg
        tag = f"wbf_i{iou_thr}_s{skip_thr}_fs{final_sc}_n{nms_iou}_m{min_models}_k{max_dets}".replace('.', '')
        det_dir = SWEEP_OUT / tag / 'detections'
        debug_csv = SWEEP_OUT / tag / 'debug.csv'

        run_wbf_variant(names, cache, cfg, det_dir, debug_csv)
        ap50, ap75 = run_evaluator(det_dir)
        score = (1.0 - ap50 + 1.0 - ap75) / 2.0  

        results.append((score, ap50, ap75, cfg, det_dir))
        print(f"[{tag}] AP50={ap50:.4f} AP75={ap75:.4f} SCORE={score:.5f}")

    results.sort(key=lambda x: x[0])
    print("\n=== TOP 10 VARIANTS (lower score = better) ===")
    for i, (score, ap50, ap75, cfg, det_dir) in enumerate(results[:10], 1):
        print(f"{i:02d}) score={score:.5f}  AP50={ap50:.4f}  AP75={ap75:.4f}  cfg={cfg}  dir={det_dir}")

    best = results[0]
    print("\nBEST:")
    print(f"score={best[0]:.5f}, AP50={best[1]:.4f}, AP75={best[2]:.4f}, cfg={best[3]}")
    print(f"Detections in: {best[4]}")
    print("Evaluator detections folder now points to the last-evaluated variant; "
          "copy the best variant there if needed.")
    
if __name__ == "__main__":
    main()
