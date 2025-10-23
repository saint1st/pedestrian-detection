# ensemble_wbf_competition.py

import json, glob, os, csv, shutil
from pathlib import Path
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

MODEL_PREDS = [
    r'X:\Desktop\sergek\preds_raw\train',
    r'X:\Desktop\sergek\preds_raw\yolo_baseline',
    r'X:\Desktop\sergek\preds_raw\yolo_new_split',
]
# --------------------------------------------------------------------------------


OUT_COMP_TXT = Path(r'X:\Desktop\sergek\evaluate\evaluate\detections')

OUT_DEBUG_CSV = Path(r'X:\Desktop\sergek\wbf_preds\preds_final_debug.csv')


IOU_THR = 0.60
SKIP_BOX_THR = 0.01
WEIGHTS = [1.0] * len(MODEL_PREDS)

def load_index(folder: str):
    files = glob.glob(os.path.join(folder, '*.json'))
    return {os.path.basename(f): f for f in files}

def union_image_list(indices):
    names = set()
    for idx in indices:
        names.update(idx.keys())
    return sorted(names)

def xyxy_norm_to_xywh_pixels(b, W, H):
    x1, y1, x2, y2 = b
    xmin = int(round(x1 * W))
    ymin = int(round(y1 * H))
    wpx  = int(round((x2 - x1) * W))
    hpx  = int(round((y2 - y1) * H))
    xmin = max(0, min(xmin, W - 1))
    ymin = max(0, min(ymin, H - 1))
    wpx  = max(0, min(wpx, W - xmin))
    hpx  = max(0, min(hpx, H - ymin))
    return xmin, ymin, wpx, hpx

def main():
    
    if OUT_COMP_TXT.exists():
        shutil.rmtree(OUT_COMP_TXT)
    OUT_COMP_TXT.mkdir(parents=True, exist_ok=True)

    
    model_indices = [load_index(p) for p in MODEL_PREDS]
    s = sum(WEIGHTS)
    weights = [w / s for w in WEIGHTS] if s > 0 else [1.0 / len(MODEL_PREDS)] * len(MODEL_PREDS)

    
    all_names = union_image_list(model_indices)

 
    OUT_DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    csvf = open(OUT_DEBUG_CSV, 'w', newline='')
    cw = csv.writer(csvf)
    cw.writerow(['image_id', 'confidence', 'xmin', 'ymin', 'width', 'height'])

    for name in tqdm(all_names, desc='WBF ensemble'):
        boxes_list, scores_list, labels_list = [], [], []
        img_w = img_h = None

        for idx in model_indices:
            if name not in idx:
                boxes_list.append([]); scores_list.append([]); labels_list.append([])
                continue
            data = json.load(open(idx[name], 'r'))
            if img_w is None:
                img_w, img_h = data['width'], data['height']
            boxes_list.append(data['boxes'])
            scores_list.append(data['scores'])
            labels_list.append(data['labels'])

        
        out_txt = OUT_COMP_TXT / name.replace('.json', '').replace('.jpg', '.txt').replace('.png', '.txt')
        if sum(map(len, boxes_list)) == 0 or img_w is None:
            out_txt.write_text('')
            continue

        fb, fs, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=IOU_THR, skip_box_thr=SKIP_BOX_THR
        )

        with open(out_txt, 'w') as f:
            for b, s in zip(fb, fs):
                xmin, ymin, wpx, hpx = xyxy_norm_to_xywh_pixels(b, img_w, img_h)
                f.write(f"person {s:.6f} {xmin} {ymin} {wpx} {hpx}\n")
                cw.writerow([name, f"{s:.6f}", xmin, ymin, wpx, hpx])

    csvf.close()
    print(f"\nWrote competition TXT files to:\n  {OUT_COMP_TXT}")
    print("Run the evaluator from its folder, e.g.:")
    print("  python pascalvoc.py --threshold 0.5")
    print("  python pascalvoc.py --threshold 0.75")

if __name__ == '__main__':
    main()
