# UI Handoff Package

This folder is a clean package containing only files required to run the complete UI handoff.

## Included modules
- `Crowd Prediction` dashboard card
  - Prediction UI (`/prediction`) with existing prediction logic
  - Layout design UI (`/layout`) with existing zone/boundary drawing logic
- `Crowd Monitoring` dashboard card
  - Empty placeholder page (`/monitoring`) for teammate implementation

## Folder structure
- `app.py` - Flask app with all routes and prediction logic
- `templates/` - Home, prediction, layout, and monitoring pages
- `models/` - required model binaries
- `insta_facebook_reddit_ocr.py` - sentiment + scraping pipeline used by prediction flow
- `requirements.txt` - Python dependencies

## Run locally
```bash
cd ui_handoff
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Environment variables
Copy `.env.example` to `.env` and set:
- `APIFY_TOKEN` (required for Instagram scraping)
