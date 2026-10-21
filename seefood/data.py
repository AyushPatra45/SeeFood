"""Shared transforms and a manifest-based dataset: explicit labels, no folder-order assumptions."""
import json
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transform(training=False):
    spatial = [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
    ] if training else [transforms.Resize(232), transforms.CenterCrop(224)]
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def open_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class ImageManifest(Dataset):
    def __init__(self, root, manifest, split, training=False):
        self.root = Path(root)
        rows = json.loads(Path(manifest).read_text())
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No images in {split} split")
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(open_rgb(self.root / row["path"])), row["label"]
