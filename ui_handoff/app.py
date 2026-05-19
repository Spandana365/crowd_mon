import os
import json
import random
import sqlite3
import sys
import tempfile
from datetime import datetime
from urllib.request import Request, urlopen
from pathlib import Path

import joblib
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS

# Load homography module
try:
    from homography import HomographyTransformer, ZoneChecker, CrowdHeatmapGenerator, create_transformer_from_config, create_zone_checker_from_config
    HOMOGRAPHY_AVAILABLE = True
except ImportError:
    HOMOGRAPHY_AVAILABLE = False

# Load local environment variables from .env (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except (ImportError, KeyboardInterrupt, OSError):
    pd = None
    PANDAS_AVAILABLE = False

if not PANDAS_AVAILABLE:
    import numpy as np

from insta_facebook_reddit_ocr import analyze_and_export

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["TIMEOUT"] = 300
CORS(app, origins=[
    "http://localhost:5001", "http://127.0.0.1:5001",
    "http://localhost:5002", "http://127.0.0.1:5002",
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
])

# ---------------------------------------------------------------------------
# Live simulation state — shared across requests
# ---------------------------------------------------------------------------
LIVE_TRACKING_CACHE = {
    "zones": [],
    "heatmap_points": [],
    "animation_frames": [],
}

SIMULATED_STREAM_STATE = {
    "video_path": None,       # Set on first analyze call
    "current_frame_pointer": 0,
    "fps": 25,
}
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = Path(BASE_DIR).resolve().parent
OMAN_DIR = REPO_ROOT / "OMAN"
DB_PATH = os.path.join(BASE_DIR, "data.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crowd_prediction_model.joblib")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.joblib")
MAPPING_CONFIG_PATH = os.path.join(BASE_DIR, "mapping_config.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    raise FileNotFoundError("Model or encoder files not found in the models/ folder.")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

FEATURE_COLUMNS = ["Day", "Popularity", "Type", "Location", "Weather", "Venue Capacity"]


def _load_mapping_config() -> dict:
    if not os.path.exists(MAPPING_CONFIG_PATH):
        raise FileNotFoundError(f"Missing mapping config at {MAPPING_CONFIG_PATH}")
    with open(MAPPING_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sources = raw.get("video_sources", {})
    if not isinstance(sources, dict):
        raise ValueError("mapping_config.json must contain object field 'video_sources'.")

    normalized_sources = {}
    for source_id, meta in sources.items():
        if not isinstance(meta, dict):
            continue
        centroid = meta.get("centroid", [])
        if not isinstance(centroid, list) or len(centroid) != 2:
            continue
        try:
            lat = float(centroid[0])
            lon = float(centroid[1])
        except Exception:
            continue
        max_capacity = int(meta.get("max_capacity", 100))
        if max_capacity <= 0:
            max_capacity = 100
        normalized_sources[str(source_id)] = {
            "display_name": str(meta.get("display_name", source_id)),
            "centroid": [lat, lon],
            "polygon_bounds": meta.get("polygon_bounds"),
            "max_capacity": max_capacity,
        }

    if not normalized_sources:
        raise ValueError("No valid video_sources found in mapping_config.json")

    return {"video_sources": normalized_sources}


MAPPING_CONFIG = _load_mapping_config()

# Load homography configuration
def _load_homography_config() -> dict:
    config_path = os.path.join(BASE_DIR, "homography_config.json")
    if not os.path.exists(config_path):
        return {"homography_zones": {}, "global_settings": {}}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load homography config: {e}")
        return {"homography_zones": {}, "global_settings": {}}

HOMOGRAPHY_CONFIG = _load_homography_config()


def _resolve_oman_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((OMAN_DIR / path).resolve())


def _point_in_polygon(point, ring):
    if not ring or len(ring) < 3:
        return False
    x, y = point
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _geo_bounds_from_feature(feature):
    """Extract lat/lng bounding box from a single GeoJSON feature."""
    all_lats, all_lngs = [], []
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or []
    gtype = geom.get("type", "")
    rings = []
    if gtype == "Polygon":
        rings = coords
    elif gtype == "MultiPolygon":
        rings = [r for poly in coords for r in poly]
    for ring in rings:
        for pt in ring:
            all_lngs.append(float(pt[0]))
            all_lats.append(float(pt[1]))
    if all_lats and all_lngs:
        return (min(all_lats), max(all_lats), min(all_lngs), max(all_lngs))
    return None


def _geo_bounds_from_geojson(zones_geojson):
    """Extract lat/lng bounding box from all zone GeoJSON features combined."""
    all_lats, all_lngs = [], []
    if not zones_geojson:
        return None
    for feature in (zones_geojson.get("features") or []):
        b = _geo_bounds_from_feature(feature)
        if b:
            all_lats += [b[0], b[1]]
            all_lngs += [b[2], b[3]]
    if all_lats and all_lngs:
        return (min(all_lats), max(all_lats), min(all_lngs), max(all_lngs))
    return None


def _pixel_to_geo(px, py, frame_w, frame_h, geo_bounds):
    """Linearly map pixel (px,py) into geographic bounding box."""
    min_lat, max_lat, min_lng, max_lng = geo_bounds
    lat = max_lat - (py / frame_h) * (max_lat - min_lat)
    lng = min_lng + (px / frame_w) * (max_lng - min_lng)
    return lat, lng


def _extract_zone_counts_and_heatmap(zones_geojson, result_payload, zone_feature=None):
    zone_counts = []
    sampled_positions = result_payload.get("pos_lists") or []
    if not sampled_positions:
        if zones_geojson and zones_geojson.get("features"):
            for idx, feature in enumerate(zones_geojson["features"], start=1):
                props = feature.get("properties") or {}
                zone_counts.append({"zone_name": props.get("name") or f"Zone {idx}", "people_count": 0})
        return zone_counts, [], []

    if zone_feature:
        geo_bounds = _geo_bounds_from_feature(zone_feature)
        geom = zone_feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        zone_ring = coords[0] if geom.get("type") == "Polygon" and coords else (
            coords[0][0] if geom.get("type") == "MultiPolygon" and coords and coords[0] else [])
    else:
        geo_bounds = _geo_bounds_from_geojson(zones_geojson)
        zone_ring = None

    all_px = [float(p[0]) for frame in sampled_positions for p in frame if len(p) >= 2]
    all_py = [float(p[1]) for frame in sampled_positions for p in frame if len(p) >= 2]
    frame_w = max(all_px) if all_px else 1920.0
    frame_h = max(all_py) if all_py else 1080.0
    frame_w = max(frame_w, 1.0)
    frame_h = max(frame_h, 1.0)

    # Build per-frame geo points for animation AND flat list for static view
    animation_frames = []
    all_heat_points = []
    latest_positions = sampled_positions[-1]

    for frame_pos in sampled_positions:
        frame_geo = []
        for p in frame_pos:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            if not geo_bounds:
                continue
            lat, lng = _pixel_to_geo(float(p[0]), float(p[1]), frame_w, frame_h, geo_bounds)
            intensity = float(p[2]) if len(p) > 2 else 0.5
            if zone_ring and not _point_in_polygon((lng, lat), zone_ring):
                continue
            pt = {"lat": lat, "lng": lng, "intensity": round(intensity, 3)}
            frame_geo.append(pt)
            all_heat_points.append(pt)
        animation_frames.append(frame_geo)

    if zones_geojson and zones_geojson.get("features"):
        for idx, feature in enumerate(zones_geojson["features"], start=1):
            props = feature.get("properties") or {}
            zone_name = props.get("name") or f"Zone {idx}"
            geom = feature.get("geometry") or {}
            coords = geom.get("coordinates") or []
            ring = coords[0] if geom.get("type") == "Polygon" and coords else (
                coords[0][0] if geom.get("type") == "MultiPolygon" and coords and coords[0] else [])
            count = sum(1 for p in latest_positions
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                        and ring and _point_in_polygon((p[0], p[1]), ring))
            zone_counts.append({"zone_name": zone_name, "people_count": int(count)})

    return zone_counts, all_heat_points, animation_frames


def _run_oman_inference(video_path, model_type, sample_interval, resize_width, max_sampled_frames, gpu, model_weight, backbone_weight, student_weight):
    if str(OMAN_DIR) not in sys.path:
        sys.path.insert(0, str(OMAN_DIR))

    import cv2
    import numpy as np
    import torch
    from torchvision.transforms import functional as TF
    try:
        from video_inference import InferenceConfig, infer_video
    except ModuleNotFoundError as exc:
        if exc.name == "timm":
            raise RuntimeError(
                "Missing dependency 'timm'. Install it in ui_handoff venv with: python -m pip install timm"
            ) from exc
        raise

    @torch.no_grad()
    def infer_video_student(local_video_path: str, local_student_weight: str):
        kd_dir = REPO_ROOT / "SENSE_MobileNet_KD"
        if str(kd_dir) not in sys.path:
            sys.path.insert(0, str(kd_dir))
        from student_model import MobileNetV3DensityStudent
        from lite_student import MobileNetDensityStudent

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(local_student_weight, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "student" in checkpoint:
            variant = checkpoint.get("variant", "small")
            student = MobileNetDensityStudent(variant=variant, pretrained=False).to(device).eval()
            student.load_state_dict(checkpoint["student"], strict=True)
        else:
            student = MobileNetV3DensityStudent(pretrained=False).to(device).eval()
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            student.load_state_dict(state, strict=True)

        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {local_video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_idx = []
        frame_counts = []
        pos_lists = []
        orig_w, orig_h = 0, 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % sample_interval == 0:
                if orig_w == 0:
                    orig_h, orig_w = frame.shape[:2]
                if resize_width and resize_width > 0 and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    new_h = max(1, int(frame.shape[0] * scale))
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = TF.to_tensor(frame_rgb)
                img = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
                pred_density = student(img)["density_map"]
                pred_count = float(pred_density.sum().item())
                frame_counts.append(max(0.0, pred_count))
                sampled_idx.append(i)

                # Extract TOP PEAKS from density map (not flood-fill sampling)
                dmap = pred_density.squeeze().cpu().numpy()  # (H, W)
                dmap_h, dmap_w = dmap.shape
                scale_x = orig_w / dmap_w
                scale_y = orig_h / dmap_h

                frame_positions = []
                if pred_count > 0:
                    # Find top-N peaks proportional to count, max 30 per frame
                    n_peaks = min(30, max(1, int(round(pred_count))))
                    # Use a minimum threshold: only cells above 5% of max density
                    threshold = dmap.max() * 0.05
                    candidates = np.argwhere(dmap > threshold)  # (row, col) pairs
                    if len(candidates) > 0:
                        # Weight candidates by their density value
                        weights = dmap[candidates[:, 0], candidates[:, 1]]
                        weights = weights / weights.sum()
                        chosen = np.random.choice(len(candidates), size=min(n_peaks, len(candidates)), replace=False, p=weights)
                        for ci in chosen:
                            gy, gx = candidates[ci]
                            px = float(gx * scale_x + scale_x / 2)
                            py = float(gy * scale_y + scale_y / 2)
                            # Store intensity proportional to local density
                            intensity = float(dmap[gy, gx] / dmap.max())
                            frame_positions.append([px, py, intensity])
                pos_lists.append(frame_positions)

                if len(frame_counts) >= max_sampled_frames:
                    break
            i += 1
        cap.release()
        if not frame_counts:
            raise RuntimeError("No frames sampled from video.")
        avg_count = float(np.mean(frame_counts))
        return {
            "video_name": os.path.basename(local_video_path),
            "model_type": "mobilenet_student",
            "video_num": int(round(avg_count)),
            "first_frame_num": int(round(frame_counts[0])),
            "avg_count": avg_count,
            "max_count": float(np.max(frame_counts)),
            "min_count": float(np.min(frame_counts)),
            "frame_counts": [float(x) for x in frame_counts],
            "frame_num": int(total_frames),
            "fps": fps,
            "sample_interval": int(sample_interval),
            "resize_width": int(resize_width),
            "max_sampled_frames": int(max_sampled_frames),
            "sampled_frame_indices": sampled_idx,
            "pos_lists": pos_lists,
        }

    if model_type == "OMAN (SENSE teacher)":
        cfg = InferenceConfig(
            sample_interval=int(sample_interval),
            resize_width=int(resize_width),
            max_sampled_frames=int(max_sampled_frames),
            gpu=gpu,
            model_weight=_resolve_oman_path(model_weight),
            backbone_weight=_resolve_oman_path(backbone_weight),
        )
        return infer_video(video_path, cfg)

    resolved_student = _resolve_oman_path(student_weight)
    if not os.path.exists(resolved_student):
        raise FileNotFoundError(f"Student weight not found: {resolved_student}")
    return infer_video_student(video_path, resolved_student)

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                event_type TEXT,
                location TEXT,
                weather TEXT,
                popularity TEXT,
                venue_capacity INTEGER,
                baseline_prediction REAL,
                average_sentiment REAL,
                sentiment_adjustment REAL,
                final_crowd INTEGER,
                congestion_ratio REAL,
                risk_level TEXT,
                risk_color TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS layouts (
                event_id INTEGER PRIMARY KEY,
                boundary_json TEXT,
                zones_json TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
            );
            """
        )


init_db()


def encode_feature(name, value):
    encoder = encoders.get(name)
    if encoder is None:
        return value
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return -1


def sentiment_adjustment(score):
    if score >= 0.90:
        return 0.20
    if score >= 0.75:
        return 0.12
    if score >= 0.50:
        return 0.08
    if score <= -0.6:
        return -0.25
    if score <= -0.2:
        return -0.12
    return 0.0


def risk_level(final_crowd, capacity):
    if capacity <= 0:
        return "Unknown", "gray"
    ratio = final_crowd / capacity
    if ratio < 1.0:
        return "Safe", "green"
    if ratio < 1.5:
        return "Moderate", "gold"
    return "High Risk", "orange"


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/organizer", methods=["GET"])
def organizer_dashboard():
    return render_template("organizer.html")


@app.route("/public", methods=["GET"])
def public_dashboard():
    return render_template("public.html")


@app.route("/organizer-monitoring", methods=["GET"])
def organizer_monitoring():
    return render_template("organizer_monitoring.html")


@app.route("/heatmap", methods=["GET"])
def heatmap_dashboard():
    return render_template("heatmap_dashboard.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    error = None
    backend_status = None
    saved_event_id = None
    sentiment_warning = None

    if request.method == "POST":
        try:
            event_name = request.form.get("event_name", "").strip()
            start_date = request.form.get("start_date", "").strip()
            end_date = request.form.get("end_date", "").strip()
            event_type = request.form.get("event_type", "").strip()
            location = request.form.get("location", "").strip()
            weather = request.form.get("weather", "").strip()
            popularity = request.form.get("popularity", "").strip()
            venue_capacity = int(request.form.get("capacity", "0") or 0)

            backend_status = "Backend received the form submission and is processing it."

            if not event_name or not start_date or not end_date or venue_capacity <= 0:
                raise ValueError("Please enter a valid event name, start date, end date, and venue capacity.")

            try:
                event_day = datetime.strptime(end_date, "%Y-%m-%d").strftime("%A")
            except ValueError:
                raise ValueError("Enter the dates in YYYY-MM-DD format.")

            # Scrape and analyze live social media sentiment
            sentiment_summary = analyze_and_export(event_name, start_date, end_date, timeout_seconds=90)
            print("analyze_and_export returned:", sentiment_summary)

            # Check if scraping failed
            if "error" in sentiment_summary:
                raise ValueError(sentiment_summary["error"])

            avg_sentiment = sentiment_summary.get("average_sentiment_overall", 0.0)

            input_vector = [
                encode_feature("Day", event_day),
                encode_feature("Popularity", popularity),
                encode_feature("Type", event_type),
                encode_feature("Location", location),
                encode_feature("Weather", weather),
                venue_capacity,
            ]

            if PANDAS_AVAILABLE:
                input_df = pd.DataFrame([input_vector], columns=FEATURE_COLUMNS)
                baseline_prediction = float(model.predict(input_df)[0])
            else:
                input_array = np.array([input_vector])
                baseline_prediction = float(model.predict(input_array)[0])

            baseline_prediction = max(0.0, baseline_prediction)
            adjustment = sentiment_adjustment(avg_sentiment)
            final_crowd = round(baseline_prediction * (1 + adjustment))
            ratio = final_crowd / venue_capacity if venue_capacity else 0.0
            level, color = risk_level(final_crowd, venue_capacity)

            result = {
                "event_name": event_name,
                "start_date": start_date,
                "end_date": end_date,
                "event_type": event_type,
                "location": location,
                "weather": weather,
                "venue_capacity": venue_capacity,
                "baseline_prediction": round(baseline_prediction),
                "average_sentiment": round(avg_sentiment, 3),
                "sentiment_adjustment": adjustment,
                "final_crowd": final_crowd,
                "congestion_ratio": round(ratio, 3),
                "risk_level": level,
                "risk_color": color,
                "summary": sentiment_summary,
                "sentiment_warning": sentiment_warning,
            }

            # Persist for public monitoring (upsert by (name,start,end))
            now = datetime.utcnow().isoformat()
            with _db() as conn:
                existing = conn.execute(
                    "SELECT id FROM events WHERE event_name=? AND start_date=? AND end_date=? ORDER BY id DESC LIMIT 1",
                    (event_name, start_date, end_date),
                ).fetchone()
                if existing:
                    saved_event_id = int(existing["id"])
                    conn.execute(
                        """
                        UPDATE events SET
                            event_type=?, location=?, weather=?, popularity=?, venue_capacity=?,
                            baseline_prediction=?, average_sentiment=?, sentiment_adjustment=?,
                            final_crowd=?, congestion_ratio=?, risk_level=?, risk_color=?
                        WHERE id=?
                        """,
                        (
                            event_type,
                            location,
                            weather,
                            popularity,
                            venue_capacity,
                            float(baseline_prediction),
                            float(avg_sentiment),
                            float(adjustment),
                            int(final_crowd),
                            float(ratio),
                            level,
                            color,
                            saved_event_id,
                        ),
                    )
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO events (
                            event_name,start_date,end_date,event_type,location,weather,popularity,venue_capacity,
                            baseline_prediction,average_sentiment,sentiment_adjustment,final_crowd,congestion_ratio,
                            risk_level,risk_color,created_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            event_name,
                            start_date,
                            end_date,
                            event_type,
                            location,
                            weather,
                            popularity,
                            venue_capacity,
                            float(baseline_prediction),
                            float(avg_sentiment),
                            float(adjustment),
                            int(final_crowd),
                            float(ratio),
                            level,
                            color,
                            now,
                        ),
                    )
                    saved_event_id = int(cur.lastrowid)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "prediction.html",
        result=result,
        error=error,
        backend_status=backend_status,
        saved_event_id=saved_event_id,
    )


@app.route("/layout", methods=["GET"])
def layout():
    event_id = request.args.get("event_id", "").strip()
    return render_template("layout.html", event_id=event_id)


@app.route("/monitoring/<int:event_id>", methods=["GET"])
def monitoring(event_id: int):
    with _db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        layout_row = conn.execute("SELECT * FROM layouts WHERE event_id=?", (event_id,)).fetchone()

    if not event:
        return redirect(url_for("public_dashboard"))

    boundary = None
    zones = None
    if layout_row:
        boundary = json.loads(layout_row["boundary_json"]) if layout_row["boundary_json"] else None
        zones = json.loads(layout_row["zones_json"]) if layout_row["zones_json"] else None

    return render_template(
        "monitoring_view.html",
        event=dict(event),
        boundary=boundary,
        zones=zones,
    )


@app.route("/monitoring-organizer/<int:event_id>", methods=["GET"])
def monitoring_organizer(event_id: int):
    with _db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        layout_row = conn.execute("SELECT * FROM layouts WHERE event_id=?", (event_id,)).fetchone()

    if not event:
        return redirect(url_for("organizer_monitoring"))

    boundary = None
    zones = None
    if layout_row:
        boundary = json.loads(layout_row["boundary_json"]) if layout_row["boundary_json"] else None
        zones = json.loads(layout_row["zones_json"]) if layout_row["zones_json"] else None

    return render_template(
        "monitoring_organizer_view.html",
        event=dict(event),
        boundary=boundary,
        zones=zones,
    )


@app.route("/monitoring-public/<int:event_id>", methods=["GET"])
def monitoring_public(event_id: int):
    with _db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        layout_row = conn.execute("SELECT * FROM layouts WHERE event_id=?", (event_id,)).fetchone()

    if not event:
        return redirect(url_for("public_dashboard"))

    boundary = None
    zones = None
    if layout_row:
        boundary = json.loads(layout_row["boundary_json"]) if layout_row["boundary_json"] else None
        zones = json.loads(layout_row["zones_json"]) if layout_row["zones_json"] else None

    return render_template(
        "monitoring_public_view.html",
        event=dict(event),
        boundary=boundary,
        zones=zones,
    )


@app.route("/api/events", methods=["GET"])
def api_events():
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, event_name, start_date, end_date, final_crowd, risk_level, created_at FROM events ORDER BY id DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/event/<int:event_id>", methods=["GET"])
def api_event(event_id: int):
    with _db() as conn:
        row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(row))


@app.route("/api/layout-data/<int:event_id>", methods=["GET"])
def api_layout_data(event_id: int):
    with _db() as conn:
        layout_row = conn.execute("SELECT * FROM layouts WHERE event_id=?", (event_id,)).fetchone()
    if not layout_row:
        return jsonify({"boundary": None, "zones": None})
    boundary = json.loads(layout_row["boundary_json"]) if layout_row["boundary_json"] else None
    zones = json.loads(layout_row["zones_json"]) if layout_row["zones_json"] else None
    return jsonify({"boundary": boundary, "zones": zones})


@app.route("/api/v1/mapping-config", methods=["GET"])
def api_v1_mapping_config():
    return jsonify(MAPPING_CONFIG)


@app.route("/api/v1/update-heatmap", methods=["GET", "POST"])
def api_v1_update_heatmap():
    payload = request.get_json(silent=True) or {}
    incoming_counts = payload.get("counts") if isinstance(payload, dict) else None
    if incoming_counts is None:
        incoming_counts = {}
    if not isinstance(incoming_counts, dict):
        return jsonify({"error": "counts must be an object keyed by video_source_id"}), 400

    sources = MAPPING_CONFIG.get("video_sources", {})
    heatmap_points = []
    warnings = []
    zone_metrics = []

    for source_id in incoming_counts.keys():
        if source_id not in sources:
            warnings.append(
                {
                    "video_source_id": source_id,
                    "message": "video_source_id not found in mapping_config; ignored.",
                }
            )

    for source_id, meta in sources.items():
        raw_count = incoming_counts.get(source_id, None)
        if raw_count is None:
            count = random.randint(0, int(meta["max_capacity"]))
        else:
            try:
                count = max(0, int(raw_count))
            except Exception:
                count = 0
                warnings.append(
                    {
                        "video_source_id": source_id,
                        "message": f"Invalid count value '{raw_count}', coerced to 0.",
                    }
                )

        max_capacity = max(1, int(meta.get("max_capacity", 100)))
        weight = min(1.0, max(0.0, float(count) / float(max_capacity)))
        lat, lon = meta["centroid"]

        point_obj = {
            "video_source_id": source_id,
            "display_name": meta.get("display_name", source_id),
            "centroid": [lat, lon],
            "weight": round(weight, 4),
            "count": int(count),
            "max_capacity": max_capacity,
        }
        zone_metrics.append(point_obj)
        heatmap_points.append({"lat": lat, "lon": lon, "weight": round(weight, 4)})

    return jsonify(
        {
            "heatmap_points": heatmap_points,
            "zones": zone_metrics,
            "warnings": warnings,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.route("/api/layout/<int:event_id>", methods=["POST"])
def api_save_layout(event_id: int):
    payload = request.get_json(silent=True) or {}
    boundary = payload.get("boundary")
    zones = payload.get("zones")

    with _db() as conn:
        exists = conn.execute("SELECT id FROM events WHERE id=?", (event_id,)).fetchone()
        if not exists:
            return jsonify({"error": "Unknown event_id"}), 404

        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT INTO layouts (event_id, boundary_json, zones_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                boundary_json=excluded.boundary_json,
                zones_json=excluded.zones_json,
                updated_at=excluded.updated_at
            """,
            (
                event_id,
                json.dumps(boundary) if boundary else None,
                json.dumps(zones) if zones else None,
                now,
            ),
        )
    return jsonify({"ok": True})


@app.route("/api/monitoring/<int:event_id>/analyze", methods=["POST"])
def api_monitoring_analyze(event_id: int):
    global LIVE_TRACKING_CACHE, SIMULATED_STREAM_STATE

    with _db() as conn:
        event = conn.execute("SELECT id FROM events WHERE id=?", (event_id,)).fetchone()
        layout_row = conn.execute("SELECT zones_json FROM layouts WHERE event_id=?", (event_id,)).fetchone()
    if not event:
        return jsonify({"error": "Unknown event_id"}), 404

    zones = None
    if layout_row and layout_row["zones_json"]:
        zones = json.loads(layout_row["zones_json"])

    model_type = request.form.get("model_type", "OMAN (SENSE teacher)")
    sample_interval = int(request.form.get("sample_interval", 45))
    resize_width = int(request.form.get("resize_width", 512))
    # 5-second window: fps(25) * 5 = 125 frames → sample every 45 frames → ~3 samples
    max_sampled_frames = int(request.form.get("max_sampled_frames", 8))
    gpu = request.form.get("gpu", "0")
    model_weight = request.form.get("model_weight", "pretrained/SENSE.pth")
    backbone_weight = request.form.get("backbone_weight", "pretrained/convnext_small_384_in22ft1k.pth")
    default_student = (
        "checkpoints/mobile_kd_stable/epoch_010.pth"
        if model_type == "MobileNet KD Stable (epoch 10)"
        else "checkpoints/student_best.pth"
    )
    student_weight = request.form.get("student_weight", default_student)

    zone_features = (zones or {}).get("features", []) if zones else []

    def _materialize_source(upload_obj, url_value):
        if upload_obj:
            suffix = os.path.splitext(upload_obj.filename or "uploaded.mp4")[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                upload_obj.save(tmp)
                return tmp.name
        if url_value:
            req = Request(url_value, headers={"User-Agent": "idp-monitoring-ui/1.0"})
            with urlopen(req, timeout=20) as response:
                payload = response.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(payload)
                return tmp.name
        return None

    global_upload = request.files.get("video_file")
    global_video_url = request.form.get("video_url", "").strip()

    zone_inputs = []
    for idx, feature in enumerate(zone_features):
        props = feature.get("properties") or {}
        zone_name = props.get("name") or f"Zone {idx + 1}"
        z_upload = request.files.get(f"zone_video_{idx}")
        z_url = request.form.get(f"zone_url_{idx}", "").strip()
        if z_upload or z_url:
            zone_inputs.append({"zone_idx": idx, "zone_name": zone_name, "upload": z_upload, "url": z_url})

    if not zone_inputs and not global_upload and not global_video_url:
        # No new upload — only allowed if we already have a cached video path
        if not SIMULATED_STREAM_STATE["video_path"]:
            return jsonify({"error": "Provide a video for each zone, or at least one global video input."}), 400

    tmp_paths = []
    try:
        zone_counts = []
        heatmap_points = []
        animation_frames_all = []
        zone_results = []
        result = None

        # ----------------------------------------------------------------
        # Sliding-window: materialise video path once, then reuse pointer
        # ----------------------------------------------------------------
        def _run_with_sliding_window(video_path):
            """Run inference starting from the current stream pointer."""
            import cv2

            # Store path on first call
            if SIMULATED_STREAM_STATE["video_path"] is None:
                SIMULATED_STREAM_STATE["video_path"] = video_path

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            SIMULATED_STREAM_STATE["fps"] = fps

            # Jump to current pointer
            ptr = SIMULATED_STREAM_STATE["current_frame_pointer"]
            if ptr >= total_frames:
                ptr = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, ptr)
            cap.release()

            # Run OMAN inference from that pointer position
            infer_result = _run_oman_inference(
                video_path=video_path,
                model_type=model_type,
                sample_interval=sample_interval,
                resize_width=resize_width,
                max_sampled_frames=max_sampled_frames,
                gpu=gpu,
                model_weight=model_weight,
                backbone_weight=backbone_weight,
                student_weight=student_weight,
            )

            # Advance pointer by 4500 frames (~3 min at 25 fps); wrap at end
            SIMULATED_STREAM_STATE["current_frame_pointer"] = (
                ptr + 4500
            ) % max(total_frames, 1)

            return infer_result

        if zone_inputs:
            for z in zone_inputs:
                tmp_video_path = _materialize_source(z["upload"], z["url"])
                if not tmp_video_path:
                    continue
                tmp_paths.append(tmp_video_path)
                z_result = _run_with_sliding_window(tmp_video_path)
                zone_results.append({"zone_name": z["zone_name"], "result": z_result})
                zone_counts.append({"zone_name": z["zone_name"], "people_count": int(z_result.get("video_num", 0))})
                zone_feature = zone_features[z["zone_idx"]] if z["zone_idx"] < len(zone_features) else None
                _, zone_heat, zone_anim = _extract_zone_counts_and_heatmap(zones, z_result, zone_feature=zone_feature)
                heatmap_points.extend(zone_heat)
                animation_frames_all.append(zone_anim)
            if zone_results:
                result = zone_results[0]["result"]
        else:
            # Use cached path if no new upload provided
            video_path_to_use = SIMULATED_STREAM_STATE["video_path"]
            if global_upload or global_video_url:
                video_path_to_use = _materialize_source(global_upload, global_video_url)
                if video_path_to_use:
                    tmp_paths.append(video_path_to_use)

            if not video_path_to_use:
                return jsonify({"error": "Unable to read video input."}), 400

            result = _run_with_sliding_window(video_path_to_use)
            zone_counts, heatmap_points, animation_frames_single = _extract_zone_counts_and_heatmap(zones, result)
            animation_frames_all = animation_frames_single

        if result is None:
            return jsonify({"error": "No valid zone video input provided."}), 400

        # Apply homography transformation if available
        homography_zone_counts = []
        homography_heatmap_points = []
        if HOMOGRAPHY_AVAILABLE and HOMOGRAPHY_CONFIG.get("homography_zones"):
            try:
                sampled_positions = result.get("pos_lists") or []
                if sampled_positions:
                    latest_positions = sampled_positions[-1]
                    zones_input = {}
                    for zone_id, zone_config in HOMOGRAPHY_CONFIG["homography_zones"].items():
                        if zone_config.get("enabled", True):
                            zones_input[zone_id] = latest_positions

                    if zones_input:
                        from homography import create_transformer_from_config, create_zone_checker_from_config, CrowdHeatmapGenerator
                        batch_results = {}
                        for zone_id, detections in zones_input.items():
                            zone_config = HOMOGRAPHY_CONFIG["homography_zones"][zone_id]
                            transformer = create_transformer_from_config(zone_config)
                            zone_checker = create_zone_checker_from_config(zone_config)
                            if transformer is None or zone_checker is None:
                                batch_results[zone_id] = {"error": "Zone coordinates not configured (contains placeholders)"}
                                continue
                            if not detections:
                                batch_results[zone_id] = {"transformed_coords": [], "points_in_zone": 0, "heatmap_data": []}
                                continue
                            try:
                                transformed_coords = transformer.batch_transform(detections)
                                points_in_zone = zone_checker.filter_points(transformed_coords)
                                heatmap_data = CrowdHeatmapGenerator.generate_heatmap_points(points_in_zone)
                                batch_results[zone_id] = {"transformed_coords": transformed_coords, "points_in_zone": len(points_in_zone), "heatmap_data": heatmap_data}
                            except Exception as exc:
                                batch_results[zone_id] = {"error": str(exc)}

                        for zone_id, zone_result in batch_results.items():
                            if "error" in zone_result:
                                continue
                            zone_config = HOMOGRAPHY_CONFIG["homography_zones"][zone_id]
                            zone_name = zone_config.get("name", zone_id)
                            people_count = zone_result["points_in_zone"]
                            max_capacity = zone_config.get("max_capacity", 0)
                            utilization = people_count / max_capacity if max_capacity > 0 else 0
                            homography_zone_counts.append({
                                "zone_id": zone_id,
                                "zone_name": zone_name,
                                "people_count": people_count,
                                "capacity": max_capacity,
                                "utilization": round(utilization, 3),
                            })
                            homography_heatmap_points.extend(zone_result["heatmap_data"])
            except Exception as exc:
                print(f"Homography processing failed: {exc}")

        max_frames = max((len(f) for f in animation_frames_all), default=0) if isinstance(animation_frames_all, list) and animation_frames_all and isinstance(animation_frames_all[0], list) and animation_frames_all[0] and isinstance(animation_frames_all[0][0], list) else 0
        if max_frames == 0:
            merged_frames = animation_frames_all if isinstance(animation_frames_all, list) else []
        else:
            merged_frames = []
            for fi in range(max_frames):
                frame_pts = []
                for zone_frames in animation_frames_all:
                    if fi < len(zone_frames):
                        frame_pts.extend(zone_frames[fi])
                merged_frames.append(frame_pts)

        final_zone_counts = homography_zone_counts if homography_zone_counts else zone_counts
        final_heatmap_points = homography_heatmap_points if homography_heatmap_points else heatmap_points

        # Update shared live cache so /api/public/live-status always has fresh data
        LIVE_TRACKING_CACHE["zones"] = final_zone_counts
        LIVE_TRACKING_CACHE["heatmap_points"] = final_heatmap_points
        LIVE_TRACKING_CACHE["animation_frames"] = merged_frames

        return jsonify({
            "ok": True,
            "model_type": model_type,
            "result": result,
            "zone_counts": final_zone_counts,
            "heatmap_points": final_heatmap_points,
            "animation_frames": merged_frames,
            "zone_results": zone_results,
            "homography_used": bool(homography_zone_counts),
            "stream_pointer": SIMULATED_STREAM_STATE["current_frame_pointer"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        for p in tmp_paths:
            if p and os.path.exists(p):
                os.remove(p)


@app.route("/proxy", methods=["GET"])
def proxy():
    target = request.args.get("url", "")
    if not target:
        return jsonify({"error": "Missing url parameter"}), 400

    if not (
        target.startswith("https://nominatim.openstreetmap.org/")
        or target.startswith("https://photon.komoot.io/")
    ):
        return jsonify({"error": "Proxy only supports approved geocoding endpoints"}), 403

    try:
        req = Request(
            target,
            headers={
                "User-Agent": "idp-layout-builder/1.0 (contact: example@example.com)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": request.host_url,
            },
        )
        with urlopen(req, timeout=10) as response:
            payload = response.read()
        return app.response_class(payload, mimetype="application/json")
    except Exception as exc:
        return jsonify({"error": f"Proxy request failed: {exc}"}), 502


# ============================================================================
# Homography & Heatmap API Routes
# ============================================================================

@app.route("/api/v1/homography-config", methods=["GET"])
def api_v1_homography_config():
    """Get homography configuration for all zones."""
    return jsonify(HOMOGRAPHY_CONFIG)


@app.route("/api/v1/homography/transform-pixels", methods=["POST"])
def api_homography_transform_pixels():
    """
    Transform pixel coordinates to geographic coordinates using homography.
    
    Expected JSON:
    {
        "zone_id": "zone_1",
        "detections": [[x1, y1], [x2, y2], ...]
    }
    
    Returns:
    {
        "zone_id": "zone_1",
        "transformed_coords": [[lat1, lng1], [lat2, lng2], ...],
        "points_in_zone": 3,
        "heatmap_data": [{"lat": ..., "lng": ..., "intensity": ...}, ...]
    }
    """
    if not HOMOGRAPHY_AVAILABLE:
        return jsonify({"error": "Homography module not available"}), 503
    
    payload = request.get_json(silent=True) or {}
    zone_id = payload.get("zone_id", "")
    detections = payload.get("detections", [])
    
    if not zone_id or zone_id not in HOMOGRAPHY_CONFIG.get("homography_zones", {}):
        return jsonify({"error": f"Unknown zone_id: {zone_id}"}), 400
    
    if not detections:
        return jsonify({
            "zone_id": zone_id,
            "transformed_coords": [],
            "points_in_zone": 0,
            "heatmap_data": []
        })
    
    try:
        zone_config = HOMOGRAPHY_CONFIG["homography_zones"][zone_id]
        
        # Create transformer
        transformer = create_transformer_from_config(zone_config)
        
        # Transform pixels to coordinates
        transformed_coords = transformer.batch_transform(detections)
        
        # Check which points are inside the zone
        zone_checker = create_zone_checker_from_config(zone_config)
        points_in_zone = zone_checker.filter_points(transformed_coords)
        
        # Generate heatmap data
        heatmap_data = CrowdHeatmapGenerator.generate_heatmap_points(points_in_zone)
        
        return jsonify({
            "zone_id": zone_id,
            "transformed_coords": transformed_coords,
            "points_in_zone": len(points_in_zone),
            "heatmap_data": heatmap_data,
            "total_detections": len(detections)
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/v1/homography/batch-transform", methods=["POST"])
def api_homography_batch_transform():
    """
    Transform detections for multiple zones in one request.
    
    Expected JSON:
    {
        "zones": {
            "zone_1": [[x1, y1], [x2, y2], ...],
            "zone_2": [[x3, y3], ...],
            ...
        }
    }
    
    Returns:
    {
        "results": {
            "zone_1": {
                "transformed_coords": [...],
                "points_in_zone": 3,
                "heatmap_data": [...]
            },
            ...
        }
    }
    """
    if not HOMOGRAPHY_AVAILABLE:
        return jsonify({"error": "Homography module not available"}), 503
    
    payload = request.get_json(silent=True) or {}
    zones_input = payload.get("zones", {})
    
    if not isinstance(zones_input, dict):
        return jsonify({"error": "zones must be an object"}), 400
    
    results = {}
    for zone_id, detections in zones_input.items():
        if zone_id not in HOMOGRAPHY_CONFIG.get("homography_zones", {}):
            results[zone_id] = {"error": f"Unknown zone_id: {zone_id}"}
            continue
        
        if not detections:
            results[zone_id] = {
                "transformed_coords": [],
                "points_in_zone": 0,
                "heatmap_data": []
            }
            continue
        
        try:
            zone_config = HOMOGRAPHY_CONFIG["homography_zones"][zone_id]
            transformer = create_transformer_from_config(zone_config)
            zone_checker = create_zone_checker_from_config(zone_config)
            
            if transformer is None or zone_checker is None:
                results[zone_id] = {"error": "Zone coordinates not configured (contains placeholders)"}
                continue
            
            transformed_coords = transformer.batch_transform(detections)
            
            points_in_zone = zone_checker.filter_points(transformed_coords)
            
            heatmap_data = CrowdHeatmapGenerator.generate_heatmap_points(points_in_zone)
            
            results[zone_id] = {
                "transformed_coords": transformed_coords,
                "points_in_zone": len(points_in_zone),
                "heatmap_data": heatmap_data,
                "total_detections": len(detections)
            }
        except Exception as exc:
            results[zone_id] = {"error": str(exc)}
    
    return jsonify({"results": results})


@app.route("/api/v1/homography/zones-summary", methods=["POST"])
def api_homography_zones_summary():
    """
    Get a summary of people counts in each zone.
    
    Expected JSON:
    {
        "zones": {
            "zone_1": [[x1, y1], ...],
            ...
        }
    }
    
    Returns:
    {
        "zones_summary": [
            {"zone_id": "zone_1", "zone_name": "...", "people_count": 5, "capacity": 120, "utilization": 0.04},
            ...
        ],
        "total_people": 10,
        "overall_utilization": 0.03
    }
    """
    if not HOMOGRAPHY_AVAILABLE:
        return jsonify({"error": "Homography module not available"}), 503
    
    payload = request.get_json(silent=True) or {}
    zones_input = payload.get("zones", {})
    
    zones_summary = []
    total_people = 0
    total_capacity = 0
    
    for zone_id, detections in zones_input.items():
        if zone_id not in HOMOGRAPHY_CONFIG.get("homography_zones", {}):
            continue
        
        zone_config = HOMOGRAPHY_CONFIG["homography_zones"][zone_id]
        zone_name = zone_config.get("name", zone_id)
        max_capacity = zone_config.get("max_capacity", 100)
        
        try:
            if detections:
                transformer = create_transformer_from_config(zone_config)
                transformed_coords = transformer.batch_transform(detections)
                
                zone_checker = create_zone_checker_from_config(zone_config)
                points_in_zone = zone_checker.filter_points(transformed_coords)
                people_count = len(points_in_zone)
            else:
                people_count = 0
            
            utilization = people_count / max_capacity if max_capacity > 0 else 0
            total_people += people_count
            total_capacity += max_capacity
            
            zones_summary.append({
                "zone_id": zone_id,
                "zone_name": zone_name,
                "people_count": people_count,
                "capacity": max_capacity,
                "utilization": round(utilization, 4)
            })
        except Exception as exc:
            zones_summary.append({
                "zone_id": zone_id,
                "zone_name": zone_name,
                "error": str(exc)
            })
    
    overall_utilization = total_people / total_capacity if total_capacity > 0 else 0
    
    return jsonify({
        "zones_summary": zones_summary,
        "total_people": total_people,
        "total_capacity": total_capacity,
        "overall_utilization": round(overall_utilization, 4)
    })


@app.route("/api/public/live-status", methods=["GET"])
def get_public_live_status():
    global LIVE_TRACKING_CACHE
    return jsonify({"success": True, "data": LIVE_TRACKING_CACHE})


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/.well-known/appspecific/com.chrome.devtools.json", methods=["GET"])
def chrome_devtools_json():
    return jsonify([])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)