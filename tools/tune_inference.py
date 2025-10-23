# tune_inference.py
from ultralytics import YOLO
from pathlib import Path
import json

DATA = 'X:/Desktop/sergek/data.yaml'
MODELS = [
    
    'X:/Desktop/sergek/runs/detect/previous/yolo_baseline/weights/best.pt',
    'X:/Desktop/sergek/runs/detect/previous/train/weights/best.pt',
    'X:/Desktop/sergek/runs/detect/previous/yolov8s_custom_albu33/weights/best.pt',
    'X:/Desktop/sergek/runs/detect/previous/yolo_new_split/weights/best.pt',
]

IMG_SIZE = 960
CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
IOU_GRID  = [0.45, 0.50, 0.55, 0.60]

def metric_combo(m):
    return 0.7 * m.box.map50 + 0.3 * m.box.map75

def main():
    results = {}
    for w in MODELS:
        print(f"\nTuning: {w}")
        model = YOLO(w)
        best = None
        for conf in CONF_GRID:
            for iou in IOU_GRID:
                m = model.val(data=DATA, imgsz=IMG_SIZE, conf=conf, iou=iou, device=0,
                               rect=True, batch=8, plots=False, save_json=False)
                score = metric_combo(m)
                if not best or score > best['score']:
                    best = {'score': float(score), 'conf': conf, 'iou': iou}
        print("Best:", best)
        results[w] = best

    Path('X:/Desktop/sergek/tuning').mkdir(exist_ok=True)
    with open('X:/Desktop/sergek/tuning/best_inference.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved → X:/Desktop/sergek/tuning/best_inference.json")

if __name__ == "__main__":
    main()
