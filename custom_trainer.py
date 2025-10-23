from ultralytics.models.yolo.detect.train import DetectionTrainer
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
import torch


def yolo_collate_fn(batch):
    """Collate function compatible with Ultralytics validator."""
    images, bboxes, classes, batch_idx = [], [], [], []
    ori_shapes, ratio_pads, im_files = [], [], []

    for i, sample in enumerate(batch):
        img = sample["image"]
        boxes = sample["bboxes"]
        labels = sample["labels"]

        images.append(img)
        ori_shapes.append(sample["ori_shape"])
        ratio_pads.append(sample["ratio_pad"])
        im_files.append(sample["im_file"])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            bboxes.append(boxes)
            classes.append(labels)
            batch_idx.append(torch.full((len(boxes), 1), i, dtype=torch.float32))

    
    if len(bboxes) == 0:
        return None

    images = torch.stack(images, 0)
    bboxes = torch.cat(bboxes, 0)
    classes = torch.cat(classes, 0)
    batch_idx = torch.cat(batch_idx, 0).squeeze(1) 

    return {
        "img": images,
        "bboxes": bboxes,
        "cls": classes,
        "batch_idx": batch_idx,
        "ori_shape": ori_shapes,
        "ratio_pad": ratio_pads,
        "im_file": im_files
    }


def safe_collate_fn(batch):
    """Wrapper to drop None samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return yolo_collate_fn(batch)


class CustomTrainer(DetectionTrainer):
    """Custom YOLOv8 trainer using Albumentations-augmented dataset."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "train":
            dataset = CustomDataset(
                image_dir='X:/Desktop/sergek/yolo/images/train',
                label_dir='X:/Desktop/sergek/yolo/labels/train',
                debug=False
            )
            shuffle = True
        else:
            dataset = CustomDataset(
                image_dir='X:/Desktop/sergek/yolo/images/val',
                label_dir='X:/Desktop/sergek/yolo/labels/val',
                debug=False
            )
            shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,   # Windows safe
            pin_memory=True,
            collate_fn=safe_collate_fn
        )

        
        loader.reset = lambda: None

        return loader

        # return DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=shuffle,
        #     num_workers=0,   # Windows safe
        #     pin_memory=True,
        #     collate_fn=safe_collate_fn
        # )
