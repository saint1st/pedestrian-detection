from ultralytics import YOLO
from pathlib import Path
import json, os
from tqdm import tqdm

DATA = 'X:/Desktop/sergek/data.yaml'
TEST_DIR = 'X:/Desktop/sergek/test/test/test_images'  
IMG_SIZE = 960
TUNING_JSON = 'X:/Desktop/sergek/tuning/best_inference.json'

OUT_ROOT = Path('X:/Desktop/sergek/preds_raw')  

def results_to_json(res):
    """
    Convert a single Ultralytics result to:
    {
      "image_id": "xxx.jpg",
      "width": W, "height": H,
      "boxes": [[x1,y1,x2,y2],... in 0-1],
      "scores": [...],
      "labels": [...]
    }
    """
    im_path = res.path
    h, w = res.orig_shape
    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
    labels = res.boxes.cls.cpu().numpy().astype(int).tolist() if res.boxes is not None else []

    boxes_norm = []
    for x1,y1,x2,y2 in boxes_xyxy:
        boxes_norm.append([x1/w, y1/h, x2/w, y2/h])

    return {
        "image_id": os.path.basename(im_path),
        "width": int(w), "height": int(h),
        "boxes": boxes_norm,
        "scores": scores,
        "labels": labels
    }

def predict_for_model(weights, conf, iou):
    model = YOLO(weights)
    out_dir = OUT_ROOT / Path(weights).parent.parent.name 
    out_dir.mkdir(parents=True, exist_ok=True)


    results = model.predict(
        source=TEST_DIR, imgsz=IMG_SIZE, conf=conf, iou=iou,
        augment=True,                
        agnostic_nms=True,           
        stream=True,                 
        save=False, save_txt=False, save_conf=False
    )

    for r in tqdm(results, desc=f"Predict {out_dir.name}"):
        js = results_to_json(r)
        with open(out_dir / (js["image_id"] + '.json'), 'w') as f:
            json.dump(js, f)

def main():
    with open(TUNING_JSON, 'r') as f:
        best = json.load(f)
    for w, cfg in best.items():
        predict_for_model(w, cfg['conf'], cfg['iou'])

if __name__ == "__main__":
    main()
