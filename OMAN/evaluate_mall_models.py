import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torchvision.transforms import functional as TF

from cpu_compat import enable_cpu_compat_if_needed
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from video_inference import InferenceConfig, _build_args


@dataclass
class EvalConfig:
    dataset_root: str
    sample_interval: int
    resize_width: int
    max_sampled_frames: int
    gpu: str
    parent_weight: str
    backbone_weight: str
    student_weight: str
    output_dir: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _maybe_download_kaggle_dataset(dataset_root: str) -> Path:
    root = Path(dataset_root).resolve()
    if root.exists():
        return root
    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. Install kagglehub or provide --dataset_root."
        ) from exc

    downloaded = Path(kagglehub.dataset_download("chaozhuang/mall-dataset")).resolve()
    return downloaded


def _find_first_file(root: Path, patterns: List[str]) -> Path:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {patterns} under {root}")


def _find_frames_dir(root: Path) -> Path:
    candidates = [
        root / "frames",
        root / "frames" / "frames",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            jpgs = list(c.glob("*.jpg"))
            if jpgs:
                return c
    for c in root.rglob("*"):
        if c.is_dir() and list(c.glob("*.jpg")):
            return c
    raise FileNotFoundError(f"Could not find frame directory with JPG files under {root}")


def _normalize_parent(frame_bgr: np.ndarray, resize_width: int) -> torch.Tensor:
    if resize_width > 0 and frame_bgr.shape[1] > resize_width:
        scale = resize_width / frame_bgr.shape[1]
        new_h = max(1, int(frame_bgr.shape[0] * scale))
        frame_bgr = cv2.resize(frame_bgr, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame_rgb = (frame_rgb - mean) / std
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()


def _normalize_student(frame_bgr: np.ndarray, resize_width: int) -> torch.Tensor:
    if resize_width > 0 and frame_bgr.shape[1] > resize_width:
        scale = resize_width / frame_bgr.shape[1]
        new_h = max(1, int(frame_bgr.shape[0] * scale))
        frame_bgr = cv2.resize(frame_bgr, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = TF.to_tensor(frame_rgb)
    return TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _load_parent_model(cfg: EvalConfig) -> Tuple[torch.nn.Module, torch.device]:
    inf_cfg = InferenceConfig(
        sample_interval=cfg.sample_interval,
        resize_width=cfg.resize_width,
        max_sampled_frames=cfg.max_sampled_frames,
        gpu=cfg.gpu,
        model_weight=cfg.parent_weight,
        backbone_weight=cfg.backbone_weight,
    )
    args = _build_args(inf_cfg)
    effective_device = enable_cpu_compat_if_needed()
    if effective_device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = "cuda"
    else:
        args.device = "cpu"

    if not os.path.exists(args.backbone_weight):
        raise FileNotFoundError(f"Backbone weight not found: {args.backbone_weight}")
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Parent model weight not found: {args.resume}")

    model, _ = build_model(args)
    model = model.cuda().eval() if args.device == "cuda" else model.eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, torch.device(args.device)


def _load_student_model(cfg: EvalConfig, device: torch.device) -> torch.nn.Module:
    kd_dir = _repo_root() / "SENSE_MobileNet_KD"
    if str(kd_dir) not in sys.path:
        sys.path.insert(0, str(kd_dir))
    from student_model import MobileNetV3DensityStudent  # pylint: disable=import-outside-toplevel
    from lite_student import MobileNetDensityStudent  # pylint: disable=import-outside-toplevel

    if not os.path.exists(cfg.student_weight):
        raise FileNotFoundError(f"Student model weight not found: {cfg.student_weight}")

    checkpoint = torch.load(cfg.student_weight, map_location="cpu")
    if isinstance(checkpoint, dict) and "student" in checkpoint:
        variant = checkpoint.get("variant", "small")
        student = MobileNetDensityStudent(variant=variant, pretrained=False).to(device).eval()
        student.load_state_dict(checkpoint["student"], strict=True)
    else:
        student = MobileNetV3DensityStudent(pretrained=False).to(device).eval()
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        student.load_state_dict(state, strict=True)
    return student


def _extract_gt_counts(mat_obj: Any, expected_len: int) -> np.ndarray:
    if isinstance(mat_obj, np.ndarray):
        if mat_obj.dtype == np.object_:
            for item in mat_obj.flat:
                result = _extract_gt_counts(item, expected_len)
                if result.size == expected_len:
                    return result
        else:
            arr = np.asarray(mat_obj).squeeze()
            if arr.ndim == 1 and arr.size == expected_len:
                return arr.astype(np.float32)
            if arr.ndim == 2:
                if arr.shape[0] == expected_len and arr.shape[1] == 1:
                    return arr[:, 0].astype(np.float32)
                if arr.shape[1] == expected_len and arr.shape[0] == 1:
                    return arr[0, :].astype(np.float32)
    if isinstance(mat_obj, dict):
        for value in mat_obj.values():
            result = _extract_gt_counts(value, expected_len)
            if result.size == expected_len:
                return result
    return np.array([], dtype=np.float32)


def _load_gt_counts(mat_path: Path, expected_len: int) -> np.ndarray:
    mat = loadmat(str(mat_path))
    # Common keys first for Mall dataset.
    for key in ["count", "counts", "frameCount", "gt_count"]:
        if key in mat:
            arr = _extract_gt_counts(mat[key], expected_len)
            if arr.size == expected_len:
                return arr
    # Fallback: recursive search through all keys.
    arr = _extract_gt_counts(mat, expected_len)
    if arr.size == expected_len:
        return arr
    raise ValueError(f"Could not extract {expected_len} frame-level GT counts from {mat_path}")


@torch.no_grad()
def _predict_parent_count(model: torch.nn.Module, image_t: torch.Tensor, device: torch.device) -> int:
    img = image_t.to(device)
    nested = nested_tensor_from_tensor_list([img])
    points, _ = model(nested, [], [], test=True)
    return int(len(points))


@torch.no_grad()
def _predict_student_count(student: torch.nn.Module, image_t: torch.Tensor, device: torch.device) -> float:
    img = image_t.unsqueeze(0).to(device)
    density = student(img)["density_map"]
    return float(max(0.0, density.sum().item()))


def _metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mape = float(np.mean(np.abs(err) / np.maximum(gt, 1.0)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape}


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    dataset_root = _maybe_download_kaggle_dataset(cfg.dataset_root)
    frames_dir = _find_frames_dir(dataset_root)
    gt_path = _find_first_file(dataset_root, ["*gt*.mat", "*GT*.mat", "*.mat"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No JPG frames found in {frames_dir}")
    total_frames = len(frame_paths)
    fps = 25.0
    gt_counts = _load_gt_counts(gt_path, total_frames)

    parent_model, device = _load_parent_model(cfg)
    student_model = _load_student_model(cfg, device)

    sampled_indices: List[int] = []
    parent_pred: List[float] = []
    student_pred: List[float] = []
    gt_sampled: List[float] = []

    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx % cfg.sample_interval == 0:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            parent_t = _normalize_parent(frame, cfg.resize_width)
            student_t = _normalize_student(frame, cfg.resize_width)
            p = _predict_parent_count(parent_model, parent_t, device)
            s = _predict_student_count(student_model, student_t, device)
            sampled_indices.append(frame_idx)
            parent_pred.append(float(p))
            student_pred.append(float(s))
            gt_sampled.append(float(gt_counts[frame_idx]))
            if len(sampled_indices) >= cfg.max_sampled_frames:
                break

    if not sampled_indices:
        raise RuntimeError("No frames sampled for evaluation.")

    gt_np = np.array(gt_sampled, dtype=np.float32)
    parent_np = np.array(parent_pred, dtype=np.float32)
    student_np = np.array(student_pred, dtype=np.float32)

    parent_metrics = _metrics(parent_np, gt_np)
    student_metrics = _metrics(student_np, gt_np)

    rows = []
    for idx, gt, p, s in zip(sampled_indices, gt_sampled, parent_pred, student_pred):
        rows.append(
            {
                "frame_idx": int(idx),
                "gt_count": round(float(gt), 4),
                "parent_pred": round(float(p), 4),
                "student_pred": round(float(s), 4),
                "parent_abs_err": round(abs(float(p - gt)), 4),
                "student_abs_err": round(abs(float(s - gt)), 4),
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "frames_dir": str(frames_dir),
        "gt_path": str(gt_path),
        "fps": fps,
        "total_frames": total_frames,
        "sample_interval": cfg.sample_interval,
        "resize_width": cfg.resize_width,
        "max_sampled_frames": cfg.max_sampled_frames,
        "num_sampled_frames": len(sampled_indices),
        "parent_metrics": parent_metrics,
        "student_metrics": student_metrics,
        "winner_by_mae": "parent" if parent_metrics["mae"] <= student_metrics["mae"] else "student",
    }
    return {"summary": summary, "rows": rows}


def _write_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mall_eval_results.json"
    csv_path = output_dir / "mall_eval_per_frame.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "gt_count",
                "parent_pred",
                "student_pred",
                "parent_abs_err",
                "student_abs_err",
            ],
        )
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(row)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def parse_args() -> EvalConfig:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser("Evaluate OMAN parent and MobileNetV3 student on Mall dataset")
    parser.add_argument("--dataset_root", type=str, default=str(repo_root / "dataset" / "mall-dataset"))
    parser.add_argument("--sample_interval", type=int, default=15)
    parser.add_argument("--resize_width", type=int, default=640)
    parser.add_argument("--max_sampled_frames", type=int, default=120)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--parent_weight", type=str, default=str(repo_root / "OMAN" / "pretrained" / "SENSE.pth"))
    parser.add_argument(
        "--backbone_weight",
        type=str,
        default=str(repo_root / "OMAN" / "pretrained" / "convnext_small_384_in22ft1k.pth"),
    )
    parser.add_argument(
        "--student_weight", type=str, default=str(repo_root / "OMAN" / "checkpoints" / "student_best.pth")
    )
    parser.add_argument("--output_dir", type=str, default=str(repo_root / "OMAN" / "outputs" / "mall_eval"))
    args = parser.parse_args()
    return EvalConfig(
        dataset_root=args.dataset_root,
        sample_interval=args.sample_interval,
        resize_width=args.resize_width,
        max_sampled_frames=args.max_sampled_frames,
        gpu=args.gpu,
        parent_weight=args.parent_weight,
        backbone_weight=args.backbone_weight,
        student_weight=args.student_weight,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    result = evaluate(cfg)
    _write_outputs(result, Path(cfg.output_dir))

    summary = result["summary"]
    print("\n=== Mall Evaluation Summary ===")
    print(f"Sampled frames: {summary['num_sampled_frames']} / total {summary['total_frames']}")
    print(
        f"Parent   -> MAE: {summary['parent_metrics']['mae']:.3f}, "
        f"RMSE: {summary['parent_metrics']['rmse']:.3f}, "
        f"MAPE: {summary['parent_metrics']['mape_percent']:.2f}%"
    )
    print(
        f"Student  -> MAE: {summary['student_metrics']['mae']:.3f}, "
        f"RMSE: {summary['student_metrics']['rmse']:.3f}, "
        f"MAPE: {summary['student_metrics']['mape_percent']:.2f}%"
    )
    print(f"Winner by MAE: {summary['winner_by_mae']}")


if __name__ == "__main__":
    main()
