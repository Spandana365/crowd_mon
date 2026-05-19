import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure these files exist in your VS Code workspace
from kd_dataset import UCFCC50Dataset, points_to_density_map
from student_model import MobileNetV3DensityStudent, feature_distill_loss


def _resize_density_preserve_count(density: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    in_h, in_w = density.shape[-2:]
    out_h, out_w = size
    resized = F.interpolate(density, size=size, mode="bilinear", align_corners=False)
    scale = (in_h * in_w) / max(1.0, float(out_h * out_w))
    return resized * scale


def _default_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return {
        "repo_root": repo_root,
        "teacher_ckpt": repo_root / "OMAN" / "pretrained" / "SENSE.pth",
        "ucf_root": repo_root / "dataset" / "UCF_CC_50",
        "save_dir": repo_root / "OMAN" / "checkpoints",
    }


def _add_oman_to_path(repo_root: Path) -> None:
    oman_root = repo_root / "OMAN"
    if not oman_root.exists():
        oman_root = repo_root
    if str(oman_root) not in sys.path:
        sys.path.insert(0, str(oman_root))


def _enable_oman_cpu_compat_if_needed() -> torch.device:
    try:
        from cpu_compat import enable_cpu_compat_if_needed
        return enable_cpu_compat_if_needed()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SenseTeacherWrapper(nn.Module):
    def __init__(self, repo_root: Path, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        _add_oman_to_path(repo_root)
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
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        nested = self._nested_tensor_from_tensor_list(image)
        points, features = self.model(nested, [], [], test=True)
        teacher_feature = features["4x"].tensors
        return {"points": points, "feature_map": teacher_feature}


def run_epoch(
    teacher: SenseTeacherWrapper,
    student: MobileNetV3DensityStudent,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW,
    alpha: float,
    beta: float,
    empty_sample_prob: float,
) -> float:
    student.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        gt_density = batch["gt_density"].to(device)
        gt_count = batch["gt_count"].to(device)

        with torch.no_grad():
            teacher_out = teacher(image)
            teacher_points = teacher_out["points"]
            teacher_feature = teacher_out["feature_map"]
            
            # Density alignment
            teacher_density = points_to_density_map(
                teacher_points,
                image_h=image.shape[-2],
                image_w=image.shape[-1],
                downsample=1,
            ).unsqueeze(0).to(device)

        student_out = student(image)
        student_density = student_out["density_map"]
        student_feature = student_out["feature_map"]
        
        density_size = student_density.shape[-2:]
        gt_density = _resize_density_preserve_count(gt_density, density_size)
        teacher_density = _resize_density_preserve_count(teacher_density, density_size)

        # Optional background regularization by occasionally forcing empty supervision.
        if empty_sample_prob > 0.0 and torch.rand(1).item() < empty_sample_prob:
            gt_density = torch.zeros_like(gt_density)
            gt_count = torch.zeros_like(gt_count)

        # Counting and losses (head-focus strategy)
        student_count = student_density.flatten(1).sum(dim=1, keepdim=True)
        gt_count_loss = F.smooth_l1_loss(student_count, gt_count)
        distill_density = F.mse_loss(student_density, teacher_density) * 10.0
        loss = (alpha * gt_count_loss) + (beta * distill_density)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # --- STABILITY FIX: Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    student: MobileNetV3DensityStudent,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    student.eval()
    mae = 0.0
    mse = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        gt_count = batch["gt_count"].to(device)
        pred_density = student(image)["density_map"]
        pred_count = pred_density.flatten(1).sum(dim=1, keepdim=True)
        err = pred_count - gt_count
        mae += torch.abs(err).mean().item()
        mse += (err**2).mean().item()
    
    mae /= max(1, len(loader))
    mse = (mse / max(1, len(loader))) ** 0.5
    return {"mae": mae, "rmse": mse}


def plot_distillation_progress(train_losses: list[float], val_maes: list[float], save_path: Path) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Total Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_maes, label="MAE", color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Headcount Accuracy (MAE)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser("SENSE teacher -> MobileNetV3 student distillation")
    parser.add_argument("--repo_root", default=str(defaults["repo_root"]), type=str)
    parser.add_argument("--teacher_ckpt", default=str(defaults["teacher_ckpt"]), type=str)
    parser.add_argument("--ucf_root", default=str(defaults["ucf_root"]), type=str)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--beta", default=0.05, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--alpha", default=2.0, type=float)
    parser.add_argument(
        "--empty_sample_prob",
        default=0.0,
        type=float,
        help="Probability of using empty supervision for a batch to reduce false positives.",
    )
    parser.add_argument(
        "--resume_student",
        default="",
        type=str,
        help="Path to student checkpoint for warm fine-tuning (student_best.pth supported).",
    )
    parser.add_argument("--save_dir", default=str(defaults["save_dir"]), type=str)
    args = parser.parse_args()

    # Environment Setup
    repo_root = Path(args.repo_root).resolve()
    teacher_ckpt = Path(args.teacher_ckpt).resolve()
    ucf_root = Path(args.ucf_root).resolve()
    os.makedirs(args.save_dir, exist_ok=True)
    
    _add_oman_to_path(repo_root)
    device = _enable_oman_cpu_compat_if_needed()
    
    # Init Models
    teacher = SenseTeacherWrapper(repo_root=repo_root, checkpoint_path=teacher_ckpt, device=device)
    use_pretrained = not bool(args.resume_student)
    student = MobileNetV3DensityStudent(pretrained=use_pretrained).to(device)
    if args.resume_student:
        resume_path = Path(args.resume_student).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(str(resume_path), map_location="cpu")
        resume_state = resume_ckpt["model"] if isinstance(resume_ckpt, dict) and "model" in resume_ckpt else resume_ckpt
        student.load_state_dict(resume_state, strict=True)
        print(f"Loaded warm-start student checkpoint: {resume_path}")

    # Data
    train_set = UCFCC50Dataset(str(ucf_root), split="train")
    val_set = UCFCC50Dataset(str(ucf_root), split="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_losses, val_maes, val_rmses = [], [], []
    best_mae = float("inf")

    print(f"Starting Distillation on {device}...")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            teacher=teacher, student=student, loader=train_loader,
            device=device, optimizer=optimizer, alpha=args.alpha, beta=args.beta,
            empty_sample_prob=args.empty_sample_prob
        )
        metrics = validate(student=student, loader=val_loader, device=device)
        
        train_losses.append(train_loss)
        val_maes.append(metrics["mae"])
        val_rmses.append(metrics["rmse"])

        print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | MAE: {metrics['mae']:.2f}")

        # Save Best Model
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save(student.state_dict(), Path(args.save_dir) / "student_best.pth")

    # Plot and Log History
    save_dir = Path(args.save_dir)
    plot_distillation_progress(train_losses, val_maes, save_dir / "distillation_progress.png")

    history_csv = save_dir / "training_history.csv"
    with history_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mae", "val_rmse"])
        for i, (l, m, r) in enumerate(zip(train_losses, val_maes, val_rmses), 1):
            writer.writerow([i, f"{l:.6f}", f"{m:.2f}", f"{r:.2f}"])

    print(f"Done! Best MAE: {best_mae:.2f}. Graph and CSV saved to {save_dir}")

if __name__ == "__main__":
    main()