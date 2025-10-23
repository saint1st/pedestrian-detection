# ensemble_wbf_strict_competition.py
import json, glob, os, csv, shutil
from pathlib import Path
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import torch
from torchvision.ops import nms

MODEL_PREDS = [
    r'X:\Desktop\sergek\preds_raw\train',
    r'X:\Desktop\sergek\preds_raw\yolo_baseline',
    r'X:\Desktop\sergek\preds_raw\yolo_new_split',
]


OUT_COMP_TXT = Path(r'X:\Desktop\sergek\evaluate\evaluate\detections')
OUT_DEBUG_CSV = Path(r'X:\Desktop\sergek\wbf_preds\preds_final_debug.csv')


IOU_THR = 0.68         
SKIP_BOX_THR = 0.03   
WEIGHTS = [1.0, 1.0, 1.2]


FINAL_SCORE_THR = 0.25     
FINAL_NMS_IOU = 0.50      
SUPPORT_IOU = 0.50        
MIN_MODELS = 2            
MAX_DETS = 4               

def load_index(folder: str):
    """Return {filename.json: full_path} for all per-image JSONs in folder."""
    files = glob.glob(os.path.join(folder, '*.json'))
    return {os.path.basename(f): f for f in files}

def union_image_list(indices):
    """Union of all filenames across models (so we don't miss images)."""
    names = set()
    for idx in indices:
        names.update(idx.keys())
    return sorted(names)

def xyxy_norm_to_xyxy_pixels(b, W, H):
    """Normalized [x1,y1,x2,y2] -> clamped integer [x1,y1,x2,y2] in pixels."""
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

if __name__ == '__main__':
   
    if OUT_COMP_TXT.exists():
        shutil.rmtree(OUT_COMP_TXT)
    OUT_COMP_TXT.mkdir(parents=True, exist_ok=True)

    OUT_DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    csvf = open(OUT_DEBUG_CSV, 'w', newline='')
    cw = csv.writer(csvf)
    cw.writerow(['image_id', 'confidence', 'xmin', 'ymin', 'width', 'height', 'support_models'])


    idxs = [load_index(p) for p in MODEL_PREDS]
    names = union_image_list(idxs)


    s = sum(WEIGHTS)
    weights = [w / s for w in WEIGHTS] if s > 0 else [1.0 / len(MODEL_PREDS)] * len(MODEL_PREDS)

    for name in tqdm(names, desc='Strict ensemble'):
        boxes_list, scores_list, labels_list = [], [], []
        pix_boxes_per_model = []  
        img_w = img_h = None

        for idx in idxs:
            if name not in idx:
                boxes_list.append([]); scores_list.append([]); labels_list.append([]); pix_boxes_per_model.append([])
                continue
            data = json.load(open(idx[name], 'r'))
            if img_w is None:
                img_w, img_h = data['width'], data['height']
            boxes_list.append(data['boxes'])      
            scores_list.append(data['scores'])
            labels_list.append(data['labels'])
        
            pix_boxes = [xyxy_norm_to_xyxy_pixels(b, img_w, img_h) for b in data['boxes']]
            pix_boxes_per_model.append(pix_boxes)

        out_txt = OUT_COMP_TXT / name.replace('.json', '').replace('.jpg', '.txt').replace('.png', '.txt')
        if sum(map(len, boxes_list)) == 0 or img_w is None:
            out_txt.write_text('')
            continue

        # ---------- WBF ----------
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

   
        xyxy_pix = [xyxy_norm_to_xyxy_pixels(b, img_w, img_h) for b in fb]

        # ---------- Consensus filter ----------
    
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

        # ---------- Final NMS ----------
        boxes_t = torch.tensor(xyxy_pix, dtype=torch.float32)
        scores_t = torch.tensor(fs, dtype=torch.float32)
        keep_idx = nms(boxes_t, scores_t, FINAL_NMS_IOU).tolist()

        boxes_t = boxes_t[keep_idx]
        scores_t = scores_t[keep_idx]
        kept_support = [supporters_list[i] for i in keep_idx]  

       
        if boxes_t.shape[0] > MAX_DETS:
            order = torch.argsort(scores_t, descending=True)[:MAX_DETS]
            boxes_t = boxes_t[order]
            scores_t = scores_t[order]
            kept_support = [kept_support[i.item()] for i in order]

    
        with open(out_txt, 'w') as f:
            for (x1, y1, x2, y2), sc, supp in zip(boxes_t.tolist(), scores_t.tolist(), kept_support):
                xmin, ymin, w, h = xyxy_to_xywh(int(x1), int(y1), int(x2), int(y2))
                f.write(f"person {sc:.6f} {xmin} {ymin} {w} {h}\n")
                cw.writerow([name, f"{sc:.6f}", xmin, ymin, w, h, supp])

    csvf.close()
    print(f"\nSaved strict competition TXT → {OUT_COMP_TXT}")
    print("Tune quickly if needed:")
    print("  IOU_THR 0.65–0.70 | SKIP_BOX_THR 0.02–0.05 | FINAL_SCORE_THR 0.20–0.30 | FINAL_NMS_IOU 0.45–0.55 | MIN_MODELS 2–3 | MAX_DETS 3–4")
