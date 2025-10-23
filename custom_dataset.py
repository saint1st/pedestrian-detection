import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, imgsz=640, transform=None,
                 debug=False, log_invalid=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.imgsz = imgsz
        self.debug = debug
        self.log_invalid = log_invalid

        os.makedirs("logs", exist_ok=True)
        self.log_path = f"logs/invalid_boxes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        if transform is None:
            self.transform = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(height=self.imgsz, width=self.imgsz),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        h0, w0 = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.label_dir, self.label_files[idx])
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])
                    labels.append(int(cls))

        try:
            if self.transform:
                transformed = self.transform(image=image, bboxes=boxes, labels=labels)
                image = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["labels"]
        except Exception as e:
            self.log_problem(f"{img_path} | Transform failed: {e}")
            return None

        if any(v < 0 or v > 1 for box in boxes for v in box):
            self.log_problem(f"{img_path} | Invalid box after transform: {boxes}")
            return None

        ratio_pad = (1.0, (0.0, 0.0))

        return {
            "image": image,
            "bboxes": boxes,
            "labels": labels,
            "ori_shape": (h0, w0),
            "ratio_pad": ratio_pad,
            "im_file": img_path
        }

    def log_problem(self, message: str):
        """Append issues to file silently."""
        if self.log_invalid:
            with open(self.log_path, "a") as f:
                f.write(message + "\n")
        if self.debug:
            print(message)
