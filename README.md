# Crowd Intelligence — Real-Time Event Crowd Monitoring System

A full-stack AI-powered crowd monitoring platform for large-scale events. It combines social media sentiment analysis, ML-based crowd prediction, live video inference using the OMAN crowd counting model, and a real-time geospatial heatmap dashboard — split into separate Organiser and Public user interfaces.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Model Weights](#model-weights)
- [API Reference](#api-reference)

---

## Overview

This system is designed for event organisers and public attendees of large gatherings (festivals, concerts, religious events, etc.). It provides:

- Pre-event crowd prediction using historical data + live social media sentiment
- Real-time crowd density monitoring via CCTV video inference
- Live heatmap and zone-wise people count visible to both organisers and the public
- GPS-based location awareness for public users inside the venue

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Flask Backend                      │
│              ui_handoff/app.py  :5000                │
│                                                      │
│  • Crowd prediction (ML + sentiment)                 │
│  • OMAN video inference (sliding window)             │
│  • SQLite DB (events + layouts)                      │
│  • LIVE_TRACKING_CACHE (shared state)                │
│  • REST API for both frontends                       │
└────────────────┬────────────────┬────────────────────┘
                 │                │
    ┌────────────▼───┐    ┌───────▼────────────┐
    │  Organiser UI  │    │    Public UI        │
    │  Flask :5001   │    │    Flask :5002       │
    │                │    │                     │
    │ • Prediction   │    │ • Event selection   │
    │ • Layout design│    │ • Live heatmap      │
    │ • Video upload │    │ • Zone counts       │
    │ • Monitoring   │    │ • GPS location      │
    └────────────────┘    └─────────────────────┘
```

---

## Features

### Organiser Frontend (port 5001)
- **Crowd Prediction** — Enter event details; system scrapes Instagram + Reddit sentiment via Apify, combines with an ML model (Random Forest) to predict expected crowd size and risk level (Safe / Moderate / High Risk)
- **Layout Designer** — Draw venue boundary and zones on an interactive Leaflet map with freehand drawing; save layout linked to an event
- **Video Inference** — Upload CCTV video per zone; runs OMAN (SENSE teacher) or MobileNet KD student model to count people and generate heatmaps
- **Sliding Window Simulation** — Subsequent inference calls advance a frame pointer through the video, simulating a live CCTV feed without re-uploading

### Public Frontend (port 5002)
- **Live Zone Counts** — Polls `/api/public/live-status` every 8 seconds for real inference results pushed by the organiser
- **Animated Heatmap** — Cycles through per-frame heatmap data between polls (0.9s frame interval) for a live feel
- **GPS Location** — Uses browser `watchPosition` with `enableHighAccuracy: true` to place a blue pulsing marker at the user's exact location
- **Zone Detection** — Ray-casting point-in-polygon check tells each user which zone they are currently inside
- **Accuracy Circle** — Shows GPS accuracy radius around the user's position

### Backend
- **`/api/public/live-status`** — Lightweight GET endpoint serving the latest `LIVE_TRACKING_CACHE` (zones + heatmap_points + animation_frames)
- **Homography support** — Optional pixel-to-geographic coordinate transformation for accurate geo-mapping of detected persons
- **SQLite persistence** — Events and layouts stored in `data.db`; survives server restarts

---

## Project Structure

```
idp_rtm/
├── ui_handoff/                  # Flask backend (port 5000)
│   ├── app.py                   # Main application — all routes and API
│   ├── homography.py            # Pixel-to-geo coordinate transformer
│   ├── insta_facebook_reddit_ocr.py  # Sentiment scraping (Apify + Reddit)
│   ├── mapping_config.json      # Video source zones and capacities
│   ├── homography_config.json   # Homography zone configuration
│   ├── data.db                  # SQLite database (events + layouts)
│   ├── models/
│   │   ├── crowd_prediction_model.joblib   # Trained Random Forest model
│   │   └── label_encoders.joblib           # Feature label encoders
│   ├── templates/               # Jinja2 HTML templates (legacy single-app)
│   ├── .env.example             # Environment variable template
│   └── requirements.txt
│
├── frontend_organizer/          # Organiser Flask frontend (port 5001)
│   ├── app.py
│   └── templates/
│       ├── organizer.html
│       ├── layout.html
│       ├── organizer_monitoring.html
│       └── monitoring_organizer_view.html
│
├── frontend_public/             # Public Flask frontend (port 5002)
│   ├── app.py
│   └── templates/
│       ├── public.html
│       └── monitoring_public_view.html
│
├── OMAN/                        # OMAN crowd counting model (SENSE teacher)
│   ├── video_inference.py       # Core inference engine
│   ├── models/                  # Model architecture
│   ├── checkpoints/
│   │   ├── student_best.pth     # MobileNet KD student weights (13 MB)
│   │   └── mobile_kd_stable/
│   │       └── epoch_010.pth    # Stable KD checkpoint (13 MB)
│   └── pretrained/              # SENSE.pth + convnext (download separately)
│
├── SENSE_MobileNet_KD/          # MobileNet knowledge distillation training
├── SENSE_UNet_KD/               # UNet knowledge distillation training
├── start_all.bat                # Starts all 3 servers (Windows)
└── .gitignore
```

---

## Prerequisites

- Python 3.10 or 3.11 (3.13 also works)
- pip
- Git
- An [Apify](https://apify.com) account with API token (for sentiment scraping)
- Windows (for `start_all.bat`) or any OS with 3 terminals

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Spandana365/crowd_mon.git
cd crowd_mon
```

### 2. Create and activate a virtual environment

```bash
cd ui_handoff
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

Both frontends only need Flask:

```bash
cd ../frontend_organizer
pip install -r requirements.txt

cd ../frontend_public
pip install -r requirements.txt
```

---

## Configuration

### Environment variables

Copy `.env.example` to `.env` inside `ui_handoff/` and fill in your Apify token:

```bash
cp ui_handoff/.env.example ui_handoff/.env
```

```env
APIFY_TOKEN=apify_api_your_token_here
```

Get your token from [console.apify.com/account/integrations](https://console.apify.com/account/integrations).

### Model weights (large files — download separately)

The two large pretrained weights are not in the repository. Download and place them at:

| File | Path | Size |
|------|------|------|
| `SENSE.pth` | `OMAN/pretrained/SENSE.pth` | ~875 MB |
| `convnext_small_384_in22ft1k.pth` | `OMAN/pretrained/convnext_small_384_in22ft1k.pth` | ~191 MB |

These are only needed if you use the **OMAN (SENSE teacher)** model option. The **MobileNet KD** student models (`student_best.pth`, `epoch_010.pth`) are already included in the repo and work without them.

### Video source zones

Edit `ui_handoff/mapping_config.json` to define your venue's camera zones, centroids, and capacities:

```json
{
  "video_sources": {
    "cam_gate_north": {
      "display_name": "North Gate",
      "centroid": [lat, lng],
      "max_capacity": 120
    }
  }
}
```

---

## Running the Application

### Option A — One command (Windows)

```bash
start_all.bat
```

This opens 3 terminal windows automatically.

### Option B — Three separate terminals

**Terminal 1 — Backend:**
```bash
cd ui_handoff
.venv\Scripts\activate
python app.py
```

**Terminal 2 — Organiser Frontend:**
```bash
cd frontend_organizer
python app.py
```

**Terminal 3 — Public Frontend:**
```bash
cd frontend_public
python app.py
```

### Access the application

| Interface | URL |
|-----------|-----|
| Backend API | http://localhost:5000 |
| Organiser Dashboard | http://localhost:5001 |
| Public Dashboard | http://localhost:5002 |

---

## How It Works

### Crowd Prediction Flow
1. Organiser enters event details (name, dates, type, location, weather, capacity)
2. Backend scrapes Instagram hashtags and Reddit posts via Apify
3. VADER sentiment analysis scores each post; EasyOCR extracts text from images
4. Average sentiment score adjusts the ML model's baseline prediction by ±8–25%
5. Final crowd count and risk level (Safe / Moderate / High Risk) saved to `data.db`

### Live Monitoring Flow
1. Organiser uploads a video (or provides URL) per zone in the monitoring dashboard
2. Backend runs OMAN/MobileNet inference on a **sliding 5-second window** of frames
3. Results (zone counts + heatmap points + animation frames) stored in `LIVE_TRACKING_CACHE`
4. On next inference call, the frame pointer advances 4500 frames (~3 min) simulating a live feed
5. Public frontend polls `/api/public/live-status` every **8 seconds** and animates through frames at 0.9s intervals

### GPS Location (Public Users)
- Browser requests GPS via `navigator.geolocation.watchPosition`
- A blue pulsing dot is placed at the user's exact coordinates
- Ray-casting algorithm checks which saved zone polygon the user is inside
- Each user's browser runs this independently — different users see their own location

---

## Model Weights

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Random Forest | `ui_handoff/models/crowd_prediction_model.joblib` | 220 KB | Pre-event crowd prediction |
| Label Encoders | `ui_handoff/models/label_encoders.joblib` | 1.6 KB | Feature encoding for RF model |
| MobileNet KD Student | `OMAN/checkpoints/student_best.pth` | 13.6 MB | Lightweight real-time inference |
| MobileNet KD Stable | `OMAN/checkpoints/mobile_kd_stable/epoch_010.pth` | 12.9 MB | Stable KD checkpoint |
| SENSE Teacher | `OMAN/pretrained/SENSE.pth` | 875 MB | High-accuracy inference (optional) |
| ConvNeXt Backbone | `OMAN/pretrained/convnext_small_384_in22ft1k.pth` | 191 MB | Backbone for SENSE teacher (optional) |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/events` | List all saved events |
| GET | `/api/event/<id>` | Get single event details |
| GET | `/api/layout-data/<id>` | Get boundary + zones for an event |
| POST | `/api/layout/<id>` | Save layout for an event |
| POST | `/api/monitoring/<id>/analyze` | Run video inference (organiser) |
| GET | `/api/public/live-status` | Latest zone counts + heatmap (public) |
| GET | `/api/v1/mapping-config` | Video source zone configuration |
| POST | `/api/v1/update-heatmap` | Update heatmap with manual counts |
| GET | `/proxy?url=` | Geocoding proxy (Nominatim / Photon) |
