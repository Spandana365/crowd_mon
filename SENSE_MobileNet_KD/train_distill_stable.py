import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from kd_dataset import UCFCC50Dataset, points_to_density_map
from lite_student import MobileNetDensityStudent


def _default_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return {
        "repo_root": repo_root,
        "teacher_ckpt": repo_root / "OMAN" / "pretrained" / "SENSE.pth",
        "ucf_root": repo_root / "dataset" / "UCF_CC_50",
        "save_dir": repo_root / "OMAN" / "checkpoints" / "mobile_kd_stable",
    }


def _add_oman_to_path(repo_root: Path) -> None:
    oman_root = repo_root / "OMAN"
    if str(oman_root) not in sys.path:
        sys.path.insert(0, str(oman_root))


def _prefer_oman_models_module() -> None:
    existing = sys.modules.get("models")
    if existing is None:
        return
    module_file = str(getattr(existing, "__file__", ""))
    if "SENSE_MobileNet_KD" in module_file.replace("\\", "/"):
        del sys.modules["models"]


def _enable_oman_cpu_compat_if_needed() -> torch.device:
    try:
        from cpu_compat import enable_cpu_compat_if_needed

        return enable_cpu_compat_if_needed()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resize_density_preserve_count(density: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    in_h, in_w = density.shape[-2:]
    out_h, out_w = size
    resized = F.interpolate(density, size=size, mode="bilinear", align_corners=False)
    scale = (in_h * in_w) / max(1.0, float(out_h * out_w))
    return resized * scale


def _set_warmup_lr(optimizer: Adam, epoch: int, warmup_epochs: int, start_lr: float, base_lr: float) -> float:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return optimizer.param_groups[0]["lr"]
    progress = 0.0 if warmup_epochs == 1 else float(epoch - 1) / float(warmup_epochs - 1)
    lr = start_lr + progress * (base_lr - start_lr)
    for group in optimizer.param_groups:
        group["lr"] = min(lr, group.get("max_lr", base_lr))
    return lr


def _points_to_tensor(points_like, device: torch.device) -> torch.Tensor:
    points = torch.as_tensor(points_like, dtype=torch.float32, device=device)
    if points.ndim == 1:
        if points.numel() == 0:
            return points.reshape(0, 2)
        if points.numel() % 2 == 0:
            return points.reshape(-1, 2)
        return points.new_zeros((0, 2))
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if points.shape[-1] != 2:
        return points.new_zeros((0, 2))
    return points


class FeatureAligner(nn.Module):
    def __init__(self, student_channels: int) -> None:
        super().__init__()
        self.student_channels = student_channels
        self.teacher_adapter = nn.LazyConv2d(student_channels, kernel_size=1, bias=False)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        teacher_feat = F.interpolate(
            teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        if teacher_feat.shape[1] != self.student_channels:
            teacher_feat = self.teacher_adapter(teacher_feat)
        return F.mse_loss(student_feat, teacher_feat)


class SenseTeacherWrapper(nn.Module):
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
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        density_batch = []
        feat_batch = []
        for i in range(image.shape[0]):
            single = image[i : i + 1]
            nested = self._nested_tensor_from_tensor_list(single)
            points_list, features = self.model(nested, [], [], test=True)
            feat_batch.append(features["4x"].tensors)
            if isinstance(points_list, (list, tuple)):
                raw_points = points_list[0] if len(points_list) > 0 else []
            else:
                raw_points = points_list
            points = _points_to_tensor(raw_points, image.device)
            density = points_to_density_map(
                points.detach().cpu().numpy(), image_h=image.shape[-2], image_w=image.shape[-1], downsample=1
            ).to(image.device)
            density_batch.append(density)
        return {"density_map": torch.stack(density_batch, dim=0), "feature_map": torch.cat(feat_batch, dim=0)}


def run_epoch(
    teacher: SenseTeacherWrapper,
    student: MobileNetDensityStudent,
    feat_aligner: FeatureAligner,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam,
    max_train_steps: int = -1,
) -> Dict[str, float]:
    student.train()
    feat_aligner.train()
    total_loss = 0.0
    total_ratio = 0.0
    total_count_l1 = 0.0
    total_batches = 0

    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        if max_train_steps > 0 and step >= max_train_steps:
            break
        image = batch["image"].to(device)
        gt_density = batch["gt_density"].to(device)
        gt_count = gt_density.flatten(1).sum(dim=1)

        with torch.no_grad():
            teacher_out = teacher(image)
            teacher_density = torch.relu(teacher_out["density_map"])
            teacher_feature = teacher_out["feature_map"]

        student_out = student(image)
        student_density = torch.relu(student_out["density_map"])
        student_feature = student_out["feature_map"]

        density_size = student_density.shape[-2:]
        gt_density = _resize_density_preserve_count(gt_density, density_size)
        teacher_density = _resize_density_preserve_count(teacher_density, density_size)

        # Align teacher count scale to GT to avoid distilling a wrong count prior.
        teacher_count = teacher_density.flatten(1).sum(dim=1, keepdim=True)
        gt_count_keep = gt_density.flatten(1).sum(dim=1, keepdim=True)
        teacher_density = teacher_density * (gt_count_keep / (teacher_count + 1e-6)).view(-1, 1, 1, 1)

        pred_count = student_density.flatten(1).sum(dim=1)
        l_gt = F.mse_loss(student_density, gt_density) + 0.5 * F.l1_loss(student_density, gt_density)
        l_teacher = F.mse_loss(student_density, teacher_density)
        l_feat = feat_aligner(student_feature, teacher_feature)
        l_count = torch.mean(torch.abs(pred_count - gt_count))
        loss = l_gt + (0.3 * l_teacher) + (0.1 * l_feat) + (0.2 * l_count)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(feat_aligner.parameters()), max_norm=1.0)
        optimizer.step()

        batch_ratio = ((pred_count + 1e-6) / (gt_count + 1e-6)).mean().item()
        batch_count_l1 = torch.mean(torch.abs(pred_count - gt_count)).item()
        total_loss += float(loss.item())
        total_ratio += float(batch_ratio)
        total_count_l1 += float(batch_count_l1)
        total_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            l_gt=f"{l_gt.item():.3f}",
            l_t=f"{l_teacher.item():.3f}",
            l_f=f"{l_feat.item():.3f}",
            l_c=f"{l_count.item():.3f}",
            ratio=f"{batch_ratio:.2f}",
            pred=f"{pred_count.mean().item():.1f}",
            gt=f"{gt_count.mean().item():.1f}",
        )

    denom = max(1, total_batches)
    return {
        "loss": total_loss / denom,
        "count_ratio": total_ratio / denom,
        "count_l1": total_count_l1 / denom,
    }


@torch.no_grad()
def validate(student: MobileNetDensityStudent, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    student.eval()
    mae = 0.0
    mse = 0.0
    ratio = 0.0
    n = 0
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        gt_count = batch["gt_density"].to(device).flatten(1).sum(dim=1)
        pred_density = torch.relu(student(image)["density_map"])
        pred_count = pred_density.flatten(1).sum(dim=1)
        err = pred_count - gt_count
        mae += torch.abs(err).mean().item()
        mse += (err**2).mean().item()
        ratio += ((pred_count + 1e-6) / (gt_count + 1e-6)).mean().item()
        n += 1
    n = max(1, n)
    return {"mae": mae / n, "rmse": (mse / n) ** 0.5, "ratio": ratio / n}


def benchmark_latency(
    student: MobileNetDensityStudent, loader: DataLoader, device: torch.device, num_iters: int = 20
) -> float:
    student.eval()
    sample = next(iter(loader))["image"][:1].to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = student(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if device.type == "cuda":
        start.record()
    else:
        import time

        t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = student(sample)
    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        import time

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / num_iters


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser("Stable KD: SENSE teacher -> MobileNet student")
    parser.add_argument("--repo_root", default=str(defaults["repo_root"]), type=str)
    parser.add_argument("--teacher_ckpt", default=str(defaults["teacher_ckpt"]), type=str)
    parser.add_argument("--ucf_root", default=str(defaults["ucf_root"]), type=str)
    parser.add_argument("--save_dir", default=str(defaults["save_dir"]), type=str)
    parser.add_argument("--variant", default="small", choices=["small", "large"])
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0 if os.name == "nt" else 2, type=int)
    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_start_lr", default=1e-5, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--save_every", default=5, type=int)
    parser.add_argument("--max_train_steps", default=-1, type=int)
    parser.add_argument("--resume_ckpt", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    teacher_ckpt = Path(args.teacher_ckpt).resolve()
    ucf_root = Path(args.ucf_root).resolve()
    save_dir = Path(args.save_dir).resolve()
    os.makedirs(save_dir, exist_ok=True)

    _add_oman_to_path(repo_root)
    device = _enable_oman_cpu_compat_if_needed()

    teacher = SenseTeacherWrapper(repo_root=repo_root, checkpoint_path=teacher_ckpt, device=device)
    student = MobileNetDensityStudent(variant=args.variant, pretrained=True).to(device)
    feat_aligner = FeatureAligner(student_channels=128).to(device)

    train_set = UCFCC50Dataset(str(ucf_root), split="train")
    val_set = UCFCC50Dataset(str(ucf_root), split="val")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    optimizer = Adam(
        [
            {"params": student.features.parameters(), "lr": 5e-5, "max_lr": 5e-5},
            {
                "params": list(student.feature_proj.parameters())
                + list(student.head.parameters())
                + list(feat_aligner.parameters()),
                "lr": args.base_lr,
                "max_lr": args.base_lr,
            },
        ],
        lr=args.base_lr,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6)

    history: List[Dict[str, float]] = []
    best_mae = float("inf")
    start_epoch = 1

    resume_path = None
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt).resolve()
    elif args.auto_resume and (save_dir / "latest.pth").exists():
        resume_path = (save_dir / "latest.pth").resolve()

    if resume_path is not None:
        ckpt = torch.load(str(resume_path), map_location="cpu")
        student.load_state_dict(ckpt["student"], strict=True)
        feat_aligner.load_state_dict(ckpt["feature_aligner"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            cosine.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", [])
        best_mae = ckpt.get("best_mae", float("inf"))
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {resume_path} at epoch {ckpt['epoch']}.")

    latency_ms = benchmark_latency(student, train_loader, device, num_iters=20)
    print(f"Student variant={args.variant} | approx latency per image: {latency_ms:.2f} ms")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            lr_now = _set_warmup_lr(
                optimizer=optimizer,
                epoch=epoch,
                warmup_epochs=args.warmup_epochs,
                start_lr=args.warmup_start_lr,
                base_lr=args.base_lr,
            )
        else:
            lr_now = optimizer.param_groups[0]["lr"]

        train_stats = run_epoch(
            teacher=teacher,
            student=student,
            feat_aligner=feat_aligner,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_train_steps=args.max_train_steps,
        )
        val_stats = validate(student=student, loader=val_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_stats["loss"],
                "train_ratio": train_stats["count_ratio"],
                "train_count_l1": train_stats["count_l1"],
                "val_mae": val_stats["mae"],
                "val_rmse": val_stats["rmse"],
                "val_ratio": val_stats["ratio"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | LR {lr_now:.2e} | "
            f"TrainLoss {train_stats['loss']:.3f} | TrainRatio {train_stats['count_ratio']:.2f} | "
            f"Val MAE {val_stats['mae']:.2f} | Val RMSE {val_stats['rmse']:.2f} | ValRatio {val_stats['ratio']:.2f}"
        )

        ckpt = {
            "epoch": epoch,
            "student": student.state_dict(),
            "feature_aligner": feat_aligner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": cosine.state_dict(),
            "history": history,
            "best_mae": best_mae,
            "variant": args.variant,
        }
        torch.save(ckpt, save_dir / "latest.pth")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pth")
        if val_stats["mae"] < best_mae:
            best_mae = val_stats["mae"]
            ckpt["best_mae"] = best_mae
            torch.save(ckpt, save_dir / "best.pth")
        if epoch > args.warmup_epochs:
            cosine.step()

    with (save_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_ratio",
                "train_count_l1",
                "val_mae",
                "val_rmse",
                "val_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"Done. Best MAE: {best_mae:.2f}. Artifacts: {save_dir}")


if __name__ == "__main__":
    main()
