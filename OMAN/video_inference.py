import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cpu_compat import enable_cpu_compat_if_needed
from models import build_model
from util.misc import nested_tensor_from_tensor_list


@dataclass
class InferenceConfig:
    device: str = "cpu"
    gpu: str = "0"
    sample_interval: int = 15
    resize_width: int = 640
    max_sampled_frames: int = 80
    backbone_weight: str = "pretrained/convnext_small_384_in22ft1k.pth"
    model_weight: str = "pretrained/SENSE.pth"


def _build_args(config: InferenceConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser("OMAN video inference")
    parser.add_argument("--backbone", default="convnext", type=str)
    parser.add_argument("--position_embedding", default="sine", type=str)
    parser.add_argument("--dec_layers", default=2, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--set_cost_class", default=1, type=float)
    parser.add_argument("--set_cost_point", default=0.05, type=float)
    parser.add_argument("--ce_loss_coef", default=1.0, type=float)
    parser.add_argument("--point_loss_coef", default=5.0, type=float)
    parser.add_argument("--eos_coef", default=0.5, type=float)
    parser.add_argument("--dataset_file", default="SENSE")
    parser.add_argument("--test_root", default="")
    parser.add_argument("--ann_dir", default="")
    parser.add_argument("--max_len", default=3000)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--gpu", default=config.gpu)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default=config.model_weight)
    parser.add_argument("--vis_dir", default="./outputs/SENSE/img_VIC")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--sample_interval", default=config.sample_interval, type=int)
    parser.add_argument("--resize_width", default=config.resize_width, type=int)
    parser.add_argument("--max_sampled_frames", default=config.max_sampled_frames, type=int)
    parser.add_argument("--backbone_weight", default=config.backbone_weight)
    return parser.parse_args([])


def _normalize_frame(frame_bgr: np.ndarray, resize_width: int = 0) -> torch.Tensor:
    if resize_width and resize_width > 0 and frame_bgr.shape[1] > resize_width:
        scale = resize_width / frame_bgr.shape[1]
        new_h = max(1, int(frame_bgr.shape[0] * scale))
        frame_bgr = cv2.resize(frame_bgr, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame_rgb = (frame_rgb - mean) / std
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
    return tensor


def _read_pts(model: torch.nn.Module, img: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
    samples = nested_tensor_from_tensor_list([img.cuda()])
    points, features = model(samples, [], [], test=True)
    return points, features["4x"].tensors


def _load_model(args: argparse.Namespace) -> torch.nn.Module:
    effective_device = enable_cpu_compat_if_needed()
    if effective_device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = "cuda"
    else:
        args.device = "cpu"
    os.makedirs("pretrained", exist_ok=True)
    if not os.path.exists(args.backbone_weight):
        raise FileNotFoundError(f"Backbone weight not found: {args.backbone_weight}")
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Model weight not found: {args.resume}")

    model, _ = build_model(args)
    model = model.cuda().eval()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def infer_video(video_path: str, config: InferenceConfig) -> Dict:
    args = _build_args(config)
    model = _load_model(args)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_frames: List[torch.Tensor] = []
    sampled_idx: List[int] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.sample_interval == 0:
            sampled_frames.append(_normalize_frame(frame, resize_width=args.resize_width))
            sampled_idx.append(i)
            if len(sampled_frames) >= args.max_sampled_frames:
                break
        i += 1
    cap.release()
    if not sampled_frames:
        raise RuntimeError("No frames sampled from video.")

    with torch.no_grad():
        pos0, feature0 = _read_pts(model, sampled_frames[0])
        pre_z = None
        if len(pos0) > 0:
            z0 = model.forward_single_image(sampled_frames[0].cuda().unsqueeze(0), [pos0], feature0, True)
            pre_z = z0
        pre_pos = pos0

        first_frame_num = len(pos0)
        cum_cnt = first_frame_num
        cnt_list = [first_frame_num]
        outflow_cnt_list = [0]
        pos_lists = [pos0.tolist()]
        inflow_lists = [[1 for _ in range(len(pos0))]]
        outflow_lists: List[List[int]] = []

        for f_idx in range(1, len(sampled_frames)):
            pos, feature1 = _read_pts(model, sampled_frames[f_idx])

            # Guard: empty current detections can make downstream matching paths
            # call torch.cat on an empty tensor list inside model internals.
            if len(pos) == 0:
                outflow_cnt = int(len(pre_pos))
                pos_lists.append([])
                inflow_lists.append([])
                outflow_lists.append([1 for _ in range(len(pre_pos))])
                cnt_list.append(0)
                outflow_cnt_list.append(outflow_cnt)
                pre_pos = pos
                pre_z = None
                continue

            # If previous frame had no valid memory, bootstrap from current frame.
            if pre_z is None or len(pre_pos) == 0:
                pre_z = model.forward_single_image(
                    sampled_frames[f_idx].cuda().unsqueeze(0), [pos], feature1, True
                )
                pre_pos = pos
                pos_lists.append(pos.tolist())
                inflow_lists.append([1 for _ in range(len(pos))])
                outflow_lists.append([])
                inflow_cnt = int(len(pos))
                cnt_list.append(inflow_cnt)
                outflow_cnt_list.append(0)
                cum_cnt += inflow_cnt
                continue

            pre_pre_z = pre_z
            z1, z2, pre_z = model.forward_single_image(
                sampled_frames[f_idx].cuda().unsqueeze(0), [pos], feature1, True, pre_z
            )
            z1 = F.normalize(z1, dim=-1).transpose(0, 1)
            z2 = F.normalize(z2, dim=-1).transpose(0, 1)
            sim_feats = torch.einsum("bnc,bmc->bnmc", z2, z1).view(1, -1, z1.shape[-1])
            pred_logits = model.vic.regression(sim_feats.squeeze(0))
            pred_probs = F.softmax(pred_logits, dim=1)
            _, pred_classes = pred_probs.max(dim=1)

            pedestrian_idx = torch.nonzero(pred_classes == 0).squeeze(1).cpu().numpy()
            pedestrian_list = pedestrian_idx // z1.shape[1]
            pre_pedestrian_list = pedestrian_idx % z1.shape[1]

            inflow_idx_list = [j for j in range(len(pos)) if j not in pedestrian_list]
            outflow_idx_list = [j for j in range(len(pre_pos)) if j not in pre_pedestrian_list]

            inflow_list = [1 if j in inflow_idx_list else 0 for j in range(len(pos))]
            outflow_list = [1 if j in outflow_idx_list else 0 for j in range(len(pre_pos))]

            pos_lists.append(pos.tolist())
            inflow_lists.append(inflow_list)
            outflow_lists.append(outflow_list)

            inflow_cnt = len(inflow_idx_list)
            outflow_cnt = len(outflow_idx_list)
            cnt_list.append(inflow_cnt)
            outflow_cnt_list.append(outflow_cnt)
            cum_cnt += inflow_cnt

            z_mask = np.array(outflow_list, dtype=bool)
            mem = pre_pre_z[0][: len(pre_pos)][z_mask]
            pre_z = [torch.cat((pre_z[0], mem), dim=0)]
            pre_pos = pos

    effective_minutes = (total_frames / fps) / 60.0 if fps > 0 else 0.0
    overall_flow_per_min = (cum_cnt / effective_minutes) if effective_minutes > 0 else 0.0

    return {
        "video_name": os.path.basename(video_path),
        "video_num": int(cum_cnt),
        "first_frame_num": int(first_frame_num),
        "cnt_list": [int(x) for x in cnt_list],
        "outflow_cnt_list": [int(x) for x in outflow_cnt_list],
        "frame_num": int(total_frames),
        "fps": fps,
        "sample_interval": int(args.sample_interval),
        "resize_width": int(args.resize_width),
        "max_sampled_frames": int(args.max_sampled_frames),
        "sampled_frame_indices": sampled_idx,
        "overall_flow_per_min": float(overall_flow_per_min),
        "pos_lists": pos_lists,
        "inflow_lists": inflow_lists,
        "outflow_lists": outflow_lists,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Run OMAN on one video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output_json", default="outputs/json/video_result_single.json")
    parser.add_argument("--sample_interval", default=15, type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model_weight", default="pretrained/SENSE.pth")
    parser.add_argument("--backbone_weight", default="pretrained/convnext_small_384_in22ft1k.pth")
    args = parser.parse_args()

    cfg = InferenceConfig(
        sample_interval=args.sample_interval,
        gpu=args.gpu,
        model_weight=args.model_weight,
        backbone_weight=args.backbone_weight,
    )
    result = infer_video(args.video, cfg)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.output_json}")
    print(
        f"video_num={result['video_num']}, first_frame_num={result['first_frame_num']}, "
        f"overall_flow_per_min={result['overall_flow_per_min']:.2f}"
    )


if __name__ == "__main__":
    main()
