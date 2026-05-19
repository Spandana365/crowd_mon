import math
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


def points_to_density_map(
    points: np.ndarray, image_h: int, image_w: int, downsample: int = 8
) -> torch.Tensor:
    out_h = max(1, math.ceil(image_h / downsample))
    out_w = max(1, math.ceil(image_w / downsample))
    density = np.zeros((out_h, out_w), dtype=np.float32)
    if points.size == 0:
        return torch.from_numpy(density).unsqueeze(0)

    xs = np.clip((points[:, 0] / downsample).astype(np.int64), 0, out_w - 1)
    ys = np.clip((points[:, 1] / downsample).astype(np.int64), 0, out_h - 1)
    for x, y in zip(xs, ys):
        density[y, x] += 1.0

    return torch.from_numpy(density).unsqueeze(0)


class UCFCC50Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: Tuple[int, int] = (768, 1024),
        downsample: int = 8,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.downsample = downsample

        image_ids = sorted([int(p.stem) for p in self.root.glob("*.jpg")])
        if len(image_ids) < 10:
            raise ValueError(f"Expected UCF_CC_50 images in {root}, found {len(image_ids)}")

        val_ids = set(image_ids[-10:])
        train_ids = [idx for idx in image_ids if idx not in val_ids]
        selected_ids = train_ids if split == "train" else sorted(list(val_ids))

        self.samples: List[Dict[str, Path]] = []
        for sample_id in selected_ids:
            img_path = self.root / f"{sample_id}.jpg"
            ann_path = self.root / f"{sample_id}_ann.mat"
            if img_path.exists() and ann_path.exists():
                self.samples.append({"img": img_path, "ann": ann_path})

        if not self.samples:
            raise ValueError(f"No samples found for split={split} at {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["img"]).convert("RGB")
        orig_w, orig_h = image.size
        points = _read_points(sample["ann"])

        target_h, target_w = self.image_size
        image = image.resize((target_w, target_h), resample=Image.BILINEAR)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        scale_x = target_w / max(1, orig_w)
        scale_y = target_h / max(1, orig_h)
        if points.size > 0:
            points[:, 0] = points[:, 0] * scale_x
            points[:, 1] = points[:, 1] * scale_y

        gt_density = points_to_density_map(points, target_h, target_w, self.downsample)
        gt_count = torch.tensor([points.shape[0]], dtype=torch.float32)

        return {
            "image": image_tensor,
            "gt_density": gt_density,
            "gt_count": gt_count,
        }
