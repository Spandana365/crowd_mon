import argparse
import sys
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from cpu_compat import enable_cpu_compat_if_needed
from models import build_model
from test import get_args_parser
from util.misc import nested_tensor_from_tensor_list


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _add_kd_path(repo_root: Path) -> None:
    kd_root = repo_root / "SENSE_MobileNet_KD"
    if str(kd_root) not in sys.path:
        sys.path.insert(0, str(kd_root))


class SenseTeacherWrapper(nn.Module):
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        args = get_args_parser().parse_args([])
        args.device = str(device)
        args.dataset_file = "SENSE"
        args.gpu = "0"
        self.model, _ = build_model(args)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        nested = nested_tensor_from_tensor_list(x)
        _, features = self.model(nested, [], [], test=True)
        return {"feature_map": features["4x"].tensors}


def visualize_comparison(
    model_path: Path,
    teacher_ckpt: Path,
    image_path: Path,
    output_path: Path,
    device: str = "cpu",
) -> None:
    repo_root = _repo_root()
    _add_kd_path(repo_root)
    from student_model import MobileNetV3DensityStudent  # pylint: disable=import-outside-toplevel

    patched_device = enable_cpu_compat_if_needed()
    if device == "cuda" and patched_device.type != "cuda":
        print("CUDA requested but unavailable; using CPU compatibility mode.")
    effective_device = patched_device if device == "cpu" else torch.device(
        "cuda" if torch.cuda.is_available() else patched_device.type
    )

    student = MobileNetV3DensityStudent(pretrained=False).to(effective_device)
    checkpoint = torch.load(str(model_path), map_location=effective_device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    student.load_state_dict(state_dict, strict=True)
    student.eval()

    teacher = SenseTeacherWrapper(checkpoint_path=teacher_ckpt, device=effective_device)

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    raw_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw_img).unsqueeze(0).to(effective_device)

    with torch.no_grad():
        t_out = teacher(input_tensor)
        s_out = student(input_tensor)

    student_map = s_out["density_map"].squeeze().detach().cpu().numpy()
    teacher_vis = t_out["feature_map"].squeeze().mean(0).detach().cpu().numpy()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.resize(image_bgr[:, :, ::-1], (512, 512))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image (Resized)")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Student Prediction\nSum: {student_map.sum():.2f}")
    plt.imshow(student_map, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Teacher Feature Focus")
    plt.imshow(teacher_vis, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Saved visualization: {output_path}")


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser("Visualize teacher/student comparison on one frame")
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default=str(repo_root / "OMAN" / "checkpoints" / "student_best.pth"),
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=str(repo_root / "OMAN" / "pretrained" / "SENSE.pth"),
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "chaozhuang"
            / "mall-dataset"
            / "versions"
            / "3"
            / "frames"
            / "frames"
            / "seq_000001.jpg"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "OMAN" / "outputs" / "teacher_student_viz.png"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    student_ckpt = Path(args.student_ckpt).resolve()
    teacher_ckpt = Path(args.teacher_ckpt).resolve()
    image_path = Path(args.image).resolve()
    output = Path(args.output).resolve()

    if not student_ckpt.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt}")
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    visualize_comparison(
        model_path=student_ckpt,
        teacher_ckpt=teacher_ckpt,
        image_path=image_path,
        output_path=output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
