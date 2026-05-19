# Synthetic Homography Pipeline for Real-Time Crowd Monitoring

## Overview

This implementation provides a complete "Synthetic Homography" pipeline for mapping random crowd video pixels to an irregular polygon zone on an interactive map. The system enables real-time crowd monitoring with heatmap visualization, separating processing (organizer-only) from display (public-visible).

## Architecture

### Backend Components

#### 1. **Homography Module** (`homography.py`)
A Python module providing perspective transformation utilities:

- **`HomographyTransformer`**: Handles pixel-to-coordinate transformation
  - `get_homography_matrix(video_points, map_points)`: Creates perspective transform matrix using `cv2.getPerspectiveTransform()`
  - `pixel_to_coords(pixel_x, pixel_y)`: Transforms video pixels to (lat, lng) coordinates
  - `coords_to_pixel(lat, lng)`: Reverse transformation
  - `batch_transform(detections)`: Processes multiple detections efficiently

- **`ZoneChecker`**: Validates if points are within irregular polygon zones
  - Uses Shapely geometry for robust polygon containment checking
  - `is_inside(lat, lng)`: Check if point is inside zone
  - `is_inside_or_boundary(lat, lng)`: Include boundary points
  - `filter_points(points)`: Filter list to only valid zone points

- **`CrowdHeatmapGenerator`**: Converts detection data to heatmap visualization
  - `generate_heatmap_points(detections, intensity_range)`: Creates intensity-weighted points
  - `aggregate_detections(detection_frames, time_decay)`: Temporal aggregation with optional decay

#### 2. **Flask Routes** (`app.py`)

New API endpoints for homography processing:

```
POST /api/v1/homography/transform-pixels
  Input: {"zone_id": "zone_1", "detections": [[x1, y1], ...]}
  Output: {"transformed_coords": [...], "points_in_zone": N, "heatmap_data": [...]}

POST /api/v1/homography/batch-transform
  Input: {"zones": {"zone_1": [...], "zone_2": [...]}}
  Output: {"results": {...}}

POST /api/v1/homography/zones-summary
  Input: {"zones": {"zone_1": [...], ...}}
  Output: {"zones_summary": [...], "total_people": N, "overall_utilization": 0.XX}

GET /api/v1/homography-config
  Output: Full homography configuration
```

### Frontend Components

#### 1. **Organizer Dashboard** (`monitoring_organizer_view.html`)
- **Video Inference Panel**: Upload/URL input for each zone with model selection
- **Zone Counts**: Real-time display of people per zone
- **Heatmap Controls**: Toggle heatmap overlay, switch between zone/heatmap views
- **Dual Layer Visualization**:
  - Circle markers for individual detections
  - Leaflet.heat density map for aggregate visualization
- **Inference Metrics**: Final count, first frame count, max count, heatmap points generated

#### 2. **Public Dashboard** (`monitoring_public_view.html`)
- **Read-only Zones Display**: Shows current count per zone
- **Live Heatmap**: Auto-updates every 30 seconds from backend
- **Visual Indicators**: Risk level pill, capacity ratio, live update pulse
- **No Processing Controls**: Public users only see visualization

### Configuration

#### `homography_config.json`
Stores anchor points and zone definitions:

```json
{
  "homography_zones": {
    "zone_1": {
      "name": "North Gate Area",
      "video_anchor_points": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
      "map_anchor_points": [[23.5862, 58.4053], [23.5862, 58.4064], ...],
      "border_points": [[23.5862, 58.4053], [23.5862, 58.4064], ...],
      "max_capacity": 120,
      "enabled": true
    }
  },
  "global_settings": {
    "heatmap_intensity_range": [0.3, 1.0],
    "use_time_decay": true,
    "min_confidence_threshold": 0.5
  }
}
```

## Workflow

### Setup Steps

1. **Install Dependencies**:
   ```bash
   cd ui_handoff
   pip install -r requirements.txt
   ```
   This installs:
   - `opencv-python` (already present)
   - `shapely` (new - for polygon operations)
   - `flask-socketio` & `python-socketio` (for optional real-time updates)

2. **Configure Zones**:
   Edit `homography_config.json` with your specific anchor points:
   - **video_anchor_points**: 4 corners of your video frame
   - **map_anchor_points**: corresponding 4 points on the geographic map
   - **border_points**: irregular polygon boundary for your zone

3. **Run Application**:
   ```bash
   python app.py
   ```
   Access at `http://localhost:5000`

### Organizer Workflow

1. Navigate to **Organizer Dashboard** → Select/Create Event
2. Click **Monitoring-Organizer** link for your event
3. Ensure **Layout Design** has zones defined (appears as blue polygons)
4. In **Video Inference** panel:
   - Select model (OMAN or student variants)
   - Upload video for each zone (or single fallback video)
   - Click "Run Inference"
5. Results appear in real-time:
   - Zone counts update
   - Heatmap circles appear for each detection
   - Optional: Toggle to Leaflet.heat for dense visualization

### Public Workflow

1. Navigate to **Public Dashboard**
2. Select event to monitor
3. View:
   - Total predicted crowd count
   - Zone-wise current counts (auto-updates every 30 seconds)
   - Heatmap overlay showing density
4. No processing or configuration available (read-only)

## Technical Details

### Homography Mathematics

The transformation uses **4-point perspective transformation** (homography):

```
video_point = [pixel_x, pixel_y]
map_point = [lat, lng]

Matrix H = cv2.getPerspectiveTransform(video_points, map_points)
transformed = cv2.perspectiveTransform(video_point, H)
```

This projects video pixels onto geographic coordinates accounting for camera angle and view distortion.

### Zone Validation

Uses **Shapely polygon containment**:

```python
from shapely.geometry import Point, Polygon

polygon = Polygon(border_points)
point = Point(lat, lng)
if polygon.contains(point):
    # Point is inside zone
```

This handles irregular/complex polygon boundaries automatically.

### Heatmap Intensity

Points weighted by:
1. **Spatial density**: Multiple detections at same location = higher intensity
2. **Temporal decay**: Older frames weighted less if `use_time_decay=true`
3. **Normalization**: Intensity range 0.2-1.0 for visualization

```
intensity = (detection_weight / max_weight) * (max_intensity - min_intensity) + min_intensity
```

## Customization

### Add New Zone

1. Edit `homography_config.json`:
   ```json
   "zone_3": {
     "name": "Food Court",
     "video_anchor_points": [[...], [...], [...], [...]],
     "map_anchor_points": [[...], [...], [...], [...]],
     "border_points": [[[...], [...], ...]],
     "max_capacity": 90
   }
   ```

2. Use calibration tool to get accurate anchor points:
   - Mark 4 corners in video frame
   - Mark corresponding points on map
   - System calculates homography automatically

### Adjust Heatmap Appearance

In `monitoring_organizer_view.html`:
```javascript
heatmapLayer = L.heatLayer(heatPoints, {
    radius: 25,           // Increase for larger blur
    blur: 15,            // Increase for smoother
    minOpacity: 0.2,     // Minimum transparency
    gradient: {          // Color scale
        0.0: '#0000ff',  // Blue (cold)
        1.0: '#ff0000'   // Red (hot)
    }
});
```

### Enable Real-Time Updates

Modify `updateHeatmapData()` interval in `monitoring_public_view.html`:
```javascript
// Update every 10 seconds instead of 30
setInterval(updateHeatmapData, 10000);
```

## Data Flow

```
Video Frame
    ↓ [Detection Model - OMAN/Student]
Pixel Detections [[x1, y1], [x2, y2], ...]
    ↓ [Homography Transformer]
Geographic Coordinates [[lat1, lng1], [lat2, lng2], ...]
    ↓ [Zone Checker]
Filtered Points (inside zone boundary)
    ↓ [Heatmap Generator]
Intensity-weighted Points [[lat, lng, intensity], ...]
    ↓ [Leaflet.heat Visualization]
Heatmap Layer on Map
    ↓
Organizer Dashboard (Full controls)
Public Dashboard (Read-only, auto-refreshing)
```

## Performance Considerations

1. **Video Processing**: Runs server-side (slower but centralized)
   - Sample interval: Skip N frames for efficiency
   - Resize width: Smaller = faster, less detail

2. **Heatmap Rendering**: Client-side (browser)
   - Leaflet.heat optimized for ~1000+ points
   - Circle markers faster but less visually appealing
   - Toggle based on data volume

3. **Database**: SQLite stores
   - Event predictions
   - Layout definitions
   - Zone geometries

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'shapely'` | `pip install shapely` in venv |
| Heatmap not appearing | Check `enableHeatmap` checkbox; ensure points exist |
| Zones showing as empty | Verify `homography_config.json` `border_points` are valid |
| Transformation looks wrong | Re-calibrate video_anchor_points; ensure 4 distinct points |
| Slow inference | Increase `sample_interval` or decrease `resize_width` |

## API Examples

### Transform single detection:
```bash
curl -X POST http://localhost:5000/api/v1/homography/transform-pixels \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "zone_1",
    "detections": [[960, 540], [100, 200]]
  }'
```

### Batch transform multiple zones:
```bash
curl -X POST http://localhost:5000/api/v1/homography/batch-transform \
  -H "Content-Type: application/json" \
  -d '{
    "zones": {
      "zone_1": [[960, 540]],
      "zone_2": [[100, 200], [200, 300]]
    }
  }'
```

## Libraries Used

- **opencv-python**: Perspective transformation (`getPerspectiveTransform`, `perspectiveTransform`)
- **shapely**: Polygon geometry and containment checking
- **leaflet.js**: Interactive map rendering
- **leaflet-heat**: Heatmap visualization layer
- **flask**: Backend web framework

## Future Enhancements

1. **WebSocket Real-time Updates**: Replace polling with live socket updates
2. **Multi-model Ensemble**: Combine predictions from multiple detection models
3. **Temporal Tracking**: Track individual crowd movements across frames
4. **Zone Calibration UI**: Visual tool to set anchor points without editing JSON
5. **Export Reports**: Generate PDF/CSV crowd reports
6. **Alert System**: Notifications when zone capacity exceeded

## Security Notes

- Homography endpoints validate zone_id against config
- Public dashboard has no write access
- Event/layout changes require organizer authentication (add if not present)
- Video uploads use temporary files, auto-deleted after processing

## References

- OpenCV Perspective Transform: https://docs.opencv.org/master/da/d54/group__imgproc__transform.html
- Shapely Documentation: https://shapely.readthedocs.io/
- Leaflet.heat: https://github.com/Leaflet/Leaflet.heat
- Leaflet.js: https://leafletjs.com/
