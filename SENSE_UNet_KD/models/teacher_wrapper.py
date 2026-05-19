import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn


def _add_oman_to_path(repo_root: Path) -> None:
    oman_root = repo_root / "OMAN"
    if not oman_root.exists():
        oman_root = repo_root
    if str(oman_root) not in sys.path:
        sys.path.insert(0, str(oman_root))


def _prefer_oman_models_module() -> None:
    existing = sys.modules.get("models")
    if existing is None:
        return
    module_file = str(getattr(existing, "__file__", ""))
    if "SENSE_UNet_KD" in module_file.replace("\\", "/"):
        del sys.modules["models"]


def _enable_oman_cpu_compat_if_needed() -> torch.device:
    try:
        from cpu_compat import enable_cpu_compat_if_needed

        return enable_cpu_compat_if_needed()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def points_to_density_map(points: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    if points.ndim == 1:
        if points.numel() == 0:
            points = points.reshape(0, 2)
        elif points.numel() % 2 == 0:
            points = points.reshape(-1, 2)
        else:
            points = points.new_zeros((0, 2))
    elif points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if points.shape[-1] != 2:
        points = points.new_zeros((0, 2))

    density = torch.zeros((1, image_h, image_w), dtype=torch.float32, device=points.device)
    if points.numel() == 0:
        return density
    xs = points[:, 0].round().long().clamp_(0, image_w - 1)
    ys = points[:, 1].round().long().clamp_(0, image_h - 1)
    density[0, ys, xs] += 1.0
    return density


class SenseTeacher(nn.Module):
    def __init__(self, repo_root: Path, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        _add_oman_to_path(repo_root)
        _prefer_oman_models_module()
        _enable_oman_cpu_compat_if_needed()

        from models import build_model
        from test import get_args_parser
        from util.misc import nested_tensor_from_tensor_list

        args = get_args_parser().parse_args([])
        args.device = str(device)
        args.dataset_file = "SENSE"
        args.gpu = "0"

        self._nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
        self.model, _ = build_model(args)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # OMAN test-time query generation can fail on batched inputs.
        # Run teacher inference per image for stable behavior on Windows and small custom batches.
        batch_features = []
        batch_density = []
        for i in range(image.shape[0]):
            single = image[i : i + 1]
            nested = self._nested_tensor_from_tensor_list(single)
            points_list, features = self.model(nested, [], [], test=True)
            teacher_feature = features["4x"].tensors
            if isinstance(points_list, (list, tuple)):
                raw_points = points_list[0] if len(points_list) > 0 else []
            else:
                raw_points = points_list
            points = torch.as_tensor(raw_points, dtype=torch.float32, device=image.device)
            density = points_to_density_map(points=points, image_h=image.shape[-2], image_w=image.shape[-1])
            batch_features.append(teacher_feature)
            batch_density.append(density)

        teacher_feature = torch.cat(batch_features, dim=0)
        teacher_density = torch.stack(batch_density, dim=0)
        return {"density_map": teacher_density, "feature_map": teacher_feature}
