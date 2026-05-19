import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SENSE_UNet_KD.data.ucf_cc50_dataset import UCFCC50DistillDataset
from SENSE_UNet_KD.losses.distill_losses import FeatureAligner, compute_losses
from SENSE_UNet_KD.models.teacher_wrapper import SenseTeacher, _enable_oman_cpu_compat_if_needed
from SENSE_UNet_KD.models.unet_student import UNetStudent


def _default_paths() -> Dict[str, Path]:
    repo_root = REPO_ROOT
    return {
        "repo_root": repo_root,
        "teacher_ckpt": repo_root / "OMAN" / "pretrained" / "SENSE.pth",
        "ucf_root": repo_root / "dataset" / "UCF_CC_50",
        "save_dir": repo_root / "OMAN" / "checkpoints" / "unet_kd",
    }


def _set_warmup_lr(optimizer: Adam, epoch: int, warmup_epochs: int, start_lr: float, base_lr: float) -> float:
    if warmup_epochs <= 0:
        return optimizer.param_groups[0]["lr"]
    if epoch > warmup_epochs:
        return optimizer.param_groups[0]["lr"]
    progress = 0.0 if warmup_epochs == 1 else float(epoch - 1) / float(warmup_epochs - 1)
    lr = start_lr + progress * (base_lr - start_lr)
    for group in optimizer.param_groups:
        base = group.get("base_lr", base_lr)
        group["lr"] = lr if base == base_lr else min(lr, base)
    return lr


def _prepare_optimizer(
    student: UNetStudent, feat_aligner: FeatureAligner, use_split_lr: bool, base_lr: float
) -> Adam:
    if use_split_lr:
        param_groups = [
            {"params": student.encoder.parameters(), "lr": 5e-5, "base_lr": 5e-5},
            {
                "params": list(student.bottleneck.parameters())
                + list(student.decoder.parameters())
                + list(feat_aligner.parameters()),
                "lr": base_lr,
                "base_lr": base_lr,
            },
        ]
    else:
        param_groups = [
            {"params": list(student.parameters()) + list(feat_aligner.parameters()), "lr": base_lr, "base_lr": base_lr}
        ]
    return Adam(param_groups, lr=base_lr)


def train_one_epoch(
    teacher: SenseTeacher,
    student: UNetStudent,
    feat_aligner: FeatureAligner,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    use_ssim: bool,
    ssim_weight: float,
) -> float:
    student.train()
    feat_aligner.train()
    total_loss = 0.0

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        image = batch["image"].to(device)
        gt_density = batch["gt_density"].to(device)
        gt_count = gt_density.sum(dim=[1, 2, 3], keepdim=True)

        with torch.no_grad():
            teacher_out = teacher(image)
            teacher_density = torch.relu(teacher_out["density_map"])
            teacher_feature = teacher_out["feature_map"]
            teacher_count = teacher_density.sum(dim=[1, 2, 3], keepdim=True)
            teacher_scale = gt_count / (teacher_count + 1e-6)
            teacher_density = teacher_density * teacher_scale

        student_out = student(image)
        student_density = torch.relu(student_out["density_map"])
        student_bottleneck = student_out["bottleneck"]
        pred_count_raw = student_density.sum(dim=[1, 2, 3], keepdim=True)
        l_count = torch.mean(torch.abs(pred_count_raw.squeeze(-1).squeeze(-1) - gt_count.squeeze(-1).squeeze(-1)))

        scale_factor = gt_count / (pred_count_raw + 1e-6)
        student_density = student_density * scale_factor

        l_feat = feat_aligner(student_bottleneck, teacher_feature)
        losses = compute_losses(
            student_density=student_density,
            gt_density=gt_density,
            teacher_density=teacher_density,
            l_feat=l_feat,
            l_count=l_count,
            use_ssim=use_ssim,
            ssim_weight=ssim_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(feat_aligner.parameters()), max_norm=1.0
        )
        optimizer.step()

        pred_count = student_density.sum(dim=[1, 2, 3], keepdim=True).detach().mean().item()
        gt_count_val = gt_count.detach().mean().item()
        teacher_count_val = teacher_density.sum(dim=[1, 2, 3], keepdim=True).detach().mean().item()
        progress.set_postfix(
            L_gt=f"{losses['l_gt'].item():.4f}",
            L_teacher=f"{losses['l_teacher'].item():.4f}",
            L_feat=f"{losses['l_feat'].item():.4f}",
            L_count=f"{losses['l_count'].item():.4f}",
            L_total=f"{losses['total'].item():.4f}",
            pred_cnt=f"{pred_count:.2f}",
            gt_cnt=f"{gt_count_val:.2f}",
            teacher_cnt=f"{teacher_count_val:.2f}",
        )
        total_loss += losses["total"].item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(student: UNetStudent, loader: DataLoader, device: torch.device) -> Dict[str, float]:
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
    rmse = (mse / max(1, len(loader))) ** 0.5
    return {"mae": mae, "rmse": rmse}


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser("U-Net student distillation with teacher guidance")
    parser.add_argument("--repo_root", default=str(defaults["repo_root"]), type=str)
    parser.add_argument("--teacher_ckpt", default=str(defaults["teacher_ckpt"]), type=str)
    parser.add_argument("--ucf_root", default=str(defaults["ucf_root"]), type=str)
    parser.add_argument("--save_dir", default=str(defaults["save_dir"]), type=str)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--image_h", default=768, type=int)
    parser.add_argument("--image_w", default=1024, type=int)
    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_start_lr", default=1e-5, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--use_split_lr", action="store_true")
    parser.add_argument("--use_ssim", action="store_true")
    parser.add_argument("--ssim_weight", default=0.05, type=float)
    parser.add_argument(
        "--num_workers",
        default=0 if os.name == "nt" else 2,
        type=int,
        help="DataLoader workers. Keep 0 on Windows for stable spawn behavior.",
    )
    parser.add_argument("--save_every", default=5, type=int)
    parser.add_argument(
        "--resume_ckpt",
        default="",
        type=str,
        help="Path to checkpoint (.pth). If empty, tries save_dir/latest.pth when --auto_resume is enabled.",
    )
    parser.add_argument("--auto_resume", action="store_true")
    args = parser.parse_args()

    device = _enable_oman_cpu_compat_if_needed()
    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    teacher = SenseTeacher(
        repo_root=Path(args.repo_root).resolve(),
        checkpoint_path=Path(args.teacher_ckpt).resolve(),
        device=device,
    )
    student = UNetStudent().to(device)
    feat_aligner = FeatureAligner(student_channels=student.bottleneck_channels).to(device)

    train_set = UCFCC50DistillDataset(
        root=args.ucf_root, split="train", image_size=(args.image_h, args.image_w)
    )
    val_set = UCFCC50DistillDataset(
        root=args.ucf_root, split="val", image_size=(args.image_h, args.image_w)
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    optimizer = _prepare_optimizer(
        student=student, feat_aligner=feat_aligner, use_split_lr=args.use_split_lr, base_lr=args.base_lr
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - max(0, args.warmup_epochs)), eta_min=1e-6
    )

    history: List[Dict[str, float]] = []
    best_mae = float("inf")
    start_epoch = 1

    resume_path = None
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt).resolve()
    elif args.auto_resume:
        candidate = save_dir / "latest.pth"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location="cpu")
        student.load_state_dict(ckpt["student"], strict=True)
        feat_aligner.load_state_dict(ckpt["feature_aligner"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            cosine.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", [])
        best_mae = ckpt.get("best_mae", float("inf"))
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {resume_path} at epoch {ckpt['epoch']}. Continuing from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            lr_now = _set_warmup_lr(
                optimizer,
                epoch=epoch,
                warmup_epochs=args.warmup_epochs,
                start_lr=args.warmup_start_lr,
                base_lr=args.base_lr,
            )
        else:
            lr_now = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            teacher=teacher,
            student=student,
            feat_aligner=feat_aligner,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_ssim=args.use_ssim,
            ssim_weight=args.ssim_weight,
        )
        metrics = validate(student=student, loader=val_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | LR: {lr_now:.6e} | "
            f"TrainLoss: {train_loss:.4f} | MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f}"
        )

        ckpt = {
            "epoch": epoch,
            "student": student.state_dict(),
            "feature_aligner": feat_aligner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": cosine.state_dict(),
            "metrics": metrics,
            "history": history,
            "best_mae": best_mae,
        }
        torch.save(ckpt, save_dir / "latest.pth")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pth")
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            ckpt["best_mae"] = best_mae
            torch.save(ckpt, save_dir / "best.pth")
        if epoch > args.warmup_epochs:
            cosine.step()

    with (save_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "val_mae", "val_rmse"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    print(f"Training complete. Best MAE: {best_mae:.3f}. Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
