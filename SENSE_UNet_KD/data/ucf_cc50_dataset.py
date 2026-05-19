import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _read_points(mat_path: Path) -> np.ndarray:
    mat_data = loadmat(str(mat_path))
    for value in mat_data.values():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 2:
            return value.astype(np.float32)
    raise ValueError(f"Could not find Nx2 points in {mat_path}")


def points_to_density_map(points: np.ndarray, image_h: int, image_w: int) -> torch.Tensor:
    density = np.zeros((image_h, image_w), dtype=np.float32)
    if points.size == 0:
        return torch.from_numpy(density).unsqueeze(0)
    xs = np.clip(points[:, 0].round().astype(np.int64), 0, image_w - 1)
    ys = np.clip(points[:, 1].round().astype(np.int64), 0, image_h - 1)
    for x, y in zip(xs, ys):
        density[y, x] += 1.0
    return torch.from_numpy(density).unsqueeze(0)


class UCFCC50DistillDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: Tuple[int, int] = (768, 1024),
        train_scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train_scale_range = train_scale_range

        image_ids = sorted([int(p.stem) for p in self.root.glob("*.jpg")])
        if len(image_ids) < 10:
            raise ValueError(f"Expected UCF_CC_50 images in {root}, found {len(image_ids)}")

        val_ids = set(image_ids[-10:])
        train_ids = [idx for idx in image_ids if idx not in val_ids]
        selected = train_ids if split == "train" else sorted(list(val_ids))

        self.samples: List[Dict[str, Path]] = []
        for sample_id in selected:
            img_path = self.root / f"{sample_id}.jpg"
            ann_path = self.root / f"{sample_id}_ann.mat"
            if img_path.exists() and ann_path.exists():
                self.samples.append({"img": img_path, "ann": ann_path})
        if not self.samples:
            raise ValueError(f"No samples found for split={split} at {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _random_scale(self, image: Image.Image, points: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        scale = random.uniform(self.train_scale_range[0], self.train_scale_range[1])
        new_w = max(64, int(round(image.width * scale)))
        new_h = max(64, int(round(image.height * scale)))
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        if points.size > 0:
            points = points.copy()
            points[:, 0] *= scale
            points[:, 1] *= scale
        return image, points

    def _random_crop(self, image: Image.Image, points: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        target_h, target_w = self.image_size
        if image.height < target_h or image.width < target_w:
            pad_h = max(0, target_h - image.height)
            pad_w = max(0, target_w - image.width)
            image = TF.pad(image, padding=[0, 0, pad_w, pad_h], fill=0)
        top = 0 if image.height == target_h else random.randint(0, image.height - target_h)
        left = 0 if image.width == target_w else random.randint(0, image.width - target_w)
        image = TF.crop(image, top=top, left=left, height=target_h, width=target_w)
        if points.size > 0:
            points = points.copy()
            points[:, 0] -= left
            points[:, 1] -= top
            valid_x = np.logical_and(points[:, 0] >= 0, points[:, 0] < target_w)
            valid_y = np.logical_and(points[:, 1] >= 0, points[:, 1] < target_h)
            points = points[np.logical_and(valid_x, valid_y)]
        return image, points

    def _random_flip(self, image: Image.Image, points: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            if points.size > 0:
                points = points.copy()
                points[:, 0] = (image.width - 1) - points[:, 0]
        return image, points

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["img"]).convert("RGB")
        points = _read_points(sample["ann"])
        target_h, target_w = self.image_size
        orig_w, orig_h = image.width, image.height

        if self.split == "train":
            image, points = self._random_scale(image, points)
            image, points = self._random_crop(image, points)
            image, points = self._random_flip(image, points)
        else:
            image = image.resize((target_w, target_h), resample=Image.BILINEAR)
            if points.size > 0:
                scale_x = target_w / max(1, orig_w)
                scale_y = target_h / max(1, orig_h)
                points = points.copy()
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        gt_density = points_to_density_map(points, target_h, target_w)
        gt_count = torch.tensor([float(points.shape[0])], dtype=torch.float32)
        return {"image": image_tensor, "gt_density": gt_density, "gt_count": gt_count}
