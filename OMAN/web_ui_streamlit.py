import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.transforms import functional as TF

from video_inference import InferenceConfig, infer_video


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_oman_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    base = Path(__file__).resolve().parent
    return str((base / path).resolve())


def _student_normalize_frame(frame_bgr: np.ndarray, resize_width: int = 0) -> torch.Tensor:
    if resize_width and resize_width > 0 and frame_bgr.shape[1] > resize_width:
        scale = resize_width / frame_bgr.shape[1]
        new_h = max(1, int(frame_bgr.shape[0] * scale))
        frame_bgr = cv2.resize(frame_bgr, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(frame_rgb)
    return TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def infer_video_student(video_path: str, student_weight: str, sample_interval: int, resize_width: int, max_sampled_frames: int) -> Dict:
    repo_root = _repo_root()
    kd_dir = repo_root / "SENSE_MobileNet_KD"
    if str(kd_dir) not in sys.path:
        sys.path.insert(0, str(kd_dir))
    from student_model import MobileNetV3DensityStudent  # pylint: disable=import-outside-toplevel
    from lite_student import MobileNetDensityStudent  # pylint: disable=import-outside-toplevel

    student_weight = _resolve_oman_path(student_weight)
    if not os.path.exists(student_weight):
        raise FileNotFoundError(f"Student weight not found: {student_weight}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(student_weight, map_location="cpu")
    if isinstance(checkpoint, dict) and "student" in checkpoint:
        variant = checkpoint.get("variant", "small")
        student = MobileNetDensityStudent(variant=variant, pretrained=False).to(device).eval()
        student.load_state_dict(checkpoint["student"], strict=True)
    else:
        student = MobileNetV3DensityStudent(pretrained=False).to(device).eval()
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        student.load_state_dict(state, strict=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_idx: List[int] = []
    frame_counts: List[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_interval == 0:
            img = _student_normalize_frame(frame, resize_width=resize_width).unsqueeze(0).to(device)
            pred_density = student(img)["density_map"]
            pred_count = float(pred_density.sum().item())
            frame_counts.append(max(0.0, pred_count))
            sampled_idx.append(i)
            if len(frame_counts) >= max_sampled_frames:
                break
        i += 1
    cap.release()
    if not frame_counts:
        raise RuntimeError("No frames sampled from video.")

    avg_count = float(np.mean(frame_counts))
    max_count = float(np.max(frame_counts))
    min_count = float(np.min(frame_counts))
    return {
        "video_name": os.path.basename(video_path),
        "model_type": "mobilenetv3_student",
        "video_num": int(round(avg_count)),
        "first_frame_num": int(round(frame_counts[0])),
        "avg_count": avg_count,
        "max_count": max_count,
        "min_count": min_count,
        "frame_counts": [float(x) for x in frame_counts],
        "frame_num": int(total_frames),
        "fps": fps,
        "sample_interval": int(sample_interval),
        "resize_width": int(resize_width),
        "max_sampled_frames": int(max_sampled_frames),
        "sampled_frame_indices": sampled_idx,
    }


def _build_rows(result: Dict) -> List[Dict]:
    rows = []
    sampled = result.get("sampled_frame_indices", [])
    inflows = result.get("cnt_list", [])
    outflows = result.get("outflow_cnt_list", [0] * len(inflows))
    cumulative = 0
    fps = float(result.get("fps", 25.0))
    for i, frame_idx in enumerate(sampled):
        inflow = int(inflows[i]) if i < len(inflows) else 0
        outflow = int(outflows[i]) if i < len(outflows) else 0
        cumulative += inflow if i == 0 else inflow - outflow
        elapsed_sec = frame_idx / fps if fps > 0 else 0.0
        flow_per_min = (cumulative / elapsed_sec * 60.0) if elapsed_sec > 0 else 0.0
        rows.append(
            {
                "step": i,
                "frame_idx": frame_idx,
                "inflow": inflow,
                "outflow": outflow,
                "net_cumulative": cumulative,
                "flow_per_min": round(flow_per_min, 3),
            }
        )
    return rows


st.set_page_config(page_title="OMAN Video Counting", layout="wide")
st.title("OMAN Video Inference UI")
st.caption("Upload a crowd video, run pretrained OMAN, and inspect count + flow rate.")
if not torch.cuda.is_available():
    st.warning("CUDA GPU not detected. Running in CPU fallback mode (slower).")

with st.sidebar:
    st.header("Model Settings")
    model_type = st.selectbox(
        "Model",
        [
            "OMAN (SENSE teacher)",
            "MobileNet Student (legacy)",
            "MobileNet KD Stable (epoch 10)",
        ],
    )
    fast_mode = st.checkbox("Fast mode (recommended for CPU)", value=True)
    sample_interval_default = 45 if fast_mode else 15
    resize_width_default = 512 if fast_mode else 640
    max_sampled_default = 40 if fast_mode else 80

    sample_interval = st.number_input(
        "Sample interval (frames)",
        min_value=1,
        value=sample_interval_default,
        step=1,
        help="Process every Nth frame. Larger N is faster but less detailed.",
    )
    resize_width = st.number_input(
        "Resize width (px)",
        min_value=256,
        value=resize_width_default,
        step=32,
        help="Frames wider than this are downscaled before inference.",
    )
    max_sampled_frames = st.number_input(
        "Max sampled frames",
        min_value=8,
        value=max_sampled_default,
        step=4,
        help="Hard cap on processed sampled frames to bound runtime.",
    )
    gpu = st.text_input("CUDA device id", value="0")
    model_weight = st.text_input("SENSE weight path", value="pretrained/SENSE.pth")
    backbone_weight = st.text_input("Backbone weight path", value="pretrained/convnext_small_384_in22ft1k.pth")
    student_weight_default = (
        "checkpoints/mobile_kd_stable/epoch_010.pth"
        if model_type == "MobileNet KD Stable (epoch 10)"
        else "checkpoints/student_best.pth"
    )
    student_weight = st.text_input("Student weight path", value=student_weight_default)
    save_json = st.checkbox("Save result JSON", value=True)

uploaded = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv", "webm"])

if uploaded is not None:
    st.video(uploaded)
    if st.button("Run Inference", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_video_path = tmp.name

        try:
            if model_type == "OMAN (SENSE teacher)":
                resolved_model_weight = _resolve_oman_path(model_weight)
                resolved_backbone_weight = _resolve_oman_path(backbone_weight)
                config = InferenceConfig(
                    sample_interval=int(sample_interval),
                    resize_width=int(resize_width),
                    max_sampled_frames=int(max_sampled_frames),
                    gpu=gpu,
                    model_weight=resolved_model_weight,
                    backbone_weight=resolved_backbone_weight,
                )
                with st.spinner("Running OMAN inference... this can take time on long videos."):
                    result = infer_video(tmp_video_path, config)

                c1, c2, c3 = st.columns(3)
                c1.metric("Final Count (video_num)", int(result["video_num"]))
                c2.metric("First Frame Count", int(result["first_frame_num"]))
                c3.metric("Overall Flow (persons/min)", f"{result['overall_flow_per_min']:.2f}")

                st.subheader("Per-step Flow")
                rows = _build_rows(result)
                st.dataframe(rows, use_container_width=True)
                st.line_chart(
                    {
                        "flow_per_min": [r["flow_per_min"] for r in rows],
                        "net_cumulative": [r["net_cumulative"] for r in rows],
                    }
                )
            else:
                with st.spinner("Running MobileNet student inference..."):
                    result = infer_video_student(
                        video_path=tmp_video_path,
                        student_weight=student_weight,
                        sample_interval=int(sample_interval),
                        resize_width=int(resize_width),
                        max_sampled_frames=int(max_sampled_frames),
                    )
                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated Count (avg/frame)", int(result["video_num"]))
                c2.metric("First Frame Count", int(result["first_frame_num"]))
                c3.metric("Max Sampled Frame Count", int(round(result["max_count"])))
                st.subheader("Per-sampled-frame Count")
                st.line_chart({"count": result["frame_counts"]})

            st.subheader("Raw Result")
            st.json(result)

            if save_json:
                os.makedirs("outputs/json", exist_ok=True)
                out_path = os.path.join("outputs", "json", "video_result_uploaded.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                st.success(f"Saved result to {out_path}")
        except Exception as exc:
            st.error(str(exc))
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
