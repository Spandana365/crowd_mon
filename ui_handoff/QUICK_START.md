# Quick Start Guide - Homography Pipeline

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd c:\Users\LENOVO\Downloads\idp_rtm\ui_handoff
pip install shapely
```

### Step 2: Verify Installation
```bash
python -c "from homography import HomographyTransformer; print('✓ Homography module loaded')"
```

### Step 3: Configure Your Zones
Edit `homography_config.json` - For each zone, you need:
1. **video_anchor_points**: 4 corners of your video frame in pixel coordinates
   ```
   Top-Left, Top-Right, Bottom-Right, Bottom-Left
   e.g., [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
   ```

2. **map_anchor_points**: Same 4 points' geographic coordinates (lat, lng)
   ```
   e.g., [[23.5862, 58.4053], [23.5862, 58.4064], ...]
   ```

3. **border_points**: Irregular polygon boundary (list of lat, lng pairs)
   ```
   e.g., [[23.5862, 58.4053], [23.5862, 58.4064], [23.5856, 58.4064], [23.5856, 58.4053]]
   ```

### Step 4: Run Application
```bash
python app.py
# Navigate to http://localhost:5000
```

### Step 5: Test the Pipeline
1. Go to **Organizer Dashboard** → **Create Event**
2. Create a Layout with zones (use Layout Design tab)
3. Click "Monitoring-Organizer" for your event
4. Upload a test video in the Video Inference section
5. Click "Run Inference"
6. View heatmap in the Heatmap tab

---

## Common Tasks

### How to Get Anchor Points?

You need to identify 4 points in your video that you know the geographic location of. 

**Video Points** (pixel coordinates):
- Use any image editing tool (e.g., Paint, GIMP)
- Mark 4 distinct corners and note their pixel coordinates
- Typically corners of the frame: (0, 0), (width, 0), (width, height), (0, height)

**Map Points** (geographic coordinates):
- Use Google Maps or OpenStreetMap
- Find the real-world GPS coordinates for those 4 points
- Format: [latitude, longitude]

**Border Points** (zone boundary):
- Use OpenStreetMap drawing tool or coordinate list
- List all points around your zone perimeter
- System will create polygon from these points

### Example Configuration

```json
{
  "homography_zones": {
    "main_entrance": {
      "name": "Main Entrance Gate",
      "video_anchor_points": [
        [0, 0],        // Top-left of video frame
        [1920, 0],     // Top-right
        [1920, 1080],  // Bottom-right
        [0, 1080]      // Bottom-left
      ],
      "map_anchor_points": [
        [40.7128, -74.0060],  // Top-left in NYC coords
        [40.7130, -74.0050],  // Top-right
        [40.7120, -74.0050],  // Bottom-right
        [40.7120, -74.0060]   // Bottom-left
      ],
      "border_points": [
        [40.7128, -74.0060],
        [40.7130, -74.0050],
        [40.7120, -74.0050],
        [40.7120, -74.0060]
      ],
      "max_capacity": 200,
      "enabled": true
    }
  }
}
```

### Toggle Heatmap Visualization

In **Organizer Dashboard**:
- **Checkbox "Show heatmap overlay"**: Toggle the Leaflet.heat layer
- **Zones Tab**: Individual circle markers for each detection
- **Heatmap Tab**: Smooth density visualization with color gradient

In **Public Dashboard**:
- **Heatmap Tab**: Auto-updating (no toggle needed)
- **Zones Tab**: Simple zone counts

### Adjust Detection Sensitivity

In the Organizer **Video Inference** section:
- **Sample interval**: Skip frames (higher = faster but less detail)
  - 45 = process every 45th frame
  - Lower = more detections, slower
- **Resize width**: Reduce resolution (faster processing)
  - 512 = resize video to 512px width
  - Lower = faster, less detail
- **Max sampled frames**: Limit total frames processed
  - 40 = process max 40 frames
  - Lower = faster inference

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'shapely'"
**Solution:**
```bash
pip install shapely
```

### Issue: Heatmap not showing
**Checklist:**
1. Is "Show heatmap overlay" checkbox checked? ✓
2. Is heatmap tab active? ✓
3. Did inference complete successfully? Check status message ✓
4. Are there heatmap points? Check metric "Heatmap Points" > 0 ✓
5. Browser console errors? Open DevTools (F12) and check for JS errors

### Issue: Zones appear empty/missing
**Solution:**
1. Go to Layout Design for the event
2. Draw zones on the map
3. Save the layout
4. Return to Monitoring Dashboard
5. Refresh page (F5)

### Issue: Transformation looks incorrect
**Solution:**
1. Verify video_anchor_points are at actual frame corners
2. Verify map_anchor_points are exactly at those GPS locations
3. Check that both point lists are in same order (e.g., TL, TR, BR, BL)
4. Use more precise GPS coordinates (decimal degrees)

### Issue: Slow video processing
**Solution:**
Increase sample_interval or reduce resize_width in inference settings

---

## API Endpoints for Custom Integration

### Get Homography Config
```
GET /api/v1/homography-config
```
Returns full configuration with all zones.

### Transform Pixel Detections
```
POST /api/v1/homography/transform-pixels
Content-Type: application/json

{
  "zone_id": "zone_1",
  "detections": [[960, 540], [100, 200]]
}
```

Response:
```json
{
  "zone_id": "zone_1",
  "transformed_coords": [[23.5859, 58.4060], [23.5850, 58.4050]],
  "points_in_zone": 1,
  "heatmap_data": [{"lat": 23.5859, "lng": 58.4060, "intensity": 0.85}]
}
```

### Batch Transform Multiple Zones
```
POST /api/v1/homography/batch-transform
Content-Type: application/json

{
  "zones": {
    "zone_1": [[960, 540], [100, 200]],
    "zone_2": [[500, 300]],
    "zone_3": [[1000, 800], [1100, 900]]
  }
}
```

### Get Zone Statistics
```
POST /api/v1/homography/zones-summary
Content-Type: application/json

{
  "zones": {
    "zone_1": [[960, 540], [100, 200]],
    "zone_2": [[500, 300]]
  }
}
```

Response:
```json
{
  "zones_summary": [
    {
      "zone_id": "zone_1",
      "zone_name": "North Gate",
      "people_count": 5,
      "capacity": 120,
      "utilization": 0.0417
    },
    {
      "zone_id": "zone_2",
      "zone_name": "Food Court",
      "people_count": 3,
      "capacity": 90,
      "utilization": 0.0333
    }
  ],
  "total_people": 8,
  "total_capacity": 210,
  "overall_utilization": 0.0381
}
```

---

## File Locations

```
ui_handoff/
├── homography.py                  # Transformation module
├── homography_config.json         # Zone configuration
├── app.py                         # Flask app (modified)
├── requirements.txt               # Dependencies (updated)
├── HOMOGRAPHY_README.md          # Full technical documentation
├── QUICK_START.md                # This file
└── templates/
    ├── monitoring_organizer_view.html   # Organizer dashboard (redesigned)
    └── monitoring_public_view.html      # Public dashboard (redesigned)
```

---

## Next: Advanced Customization

See **HOMOGRAPHY_README.md** for:
- Zone calibration tool ideas
- WebSocket real-time updates
- Performance optimization
- Multi-model ensemble
- Export/reporting features

---

## Support

For detailed technical documentation, see `HOMOGRAPHY_README.md`

For backend troubleshooting:
1. Check Flask console output
2. Review browser console (F12)
3. Test API endpoints directly using curl/Postman
