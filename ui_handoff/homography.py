"""
Homography module for mapping video pixels to geographic coordinates.

Provides functionality to:
1. Calculate perspective transformation (homography matrix)
2. Transform pixel coordinates to (lat, lng) coordinates
3. Check if a point lies within an irregular polygon zone
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from shapely.geometry import Point, Polygon


class HomographyTransformer:
    """
    Handles perspective transformation of video pixels to geographic coordinates.
    """

    def __init__(self, video_points: List[Tuple[float, float]], 
                 map_points: List[Tuple[float, float]]):
        """
        Initialize the homography transformer.

        Args:
            video_points: List of 4 anchor points from video frame [[x, y], ...]
            map_points: List of 4 corresponding anchor points on map [[lat, lng], ...]
        """
        if len(video_points) != 4 or len(map_points) != 4:
            raise ValueError("Both video_points and map_points must contain exactly 4 points")

        self.video_points = np.array(video_points, dtype=np.float32)
        self.map_points = np.array(map_points, dtype=np.float32)

        # Calculate homography matrix
        self.matrix = cv2.getPerspectiveTransform(self.video_points, self.map_points)
        if self.matrix is None:
            raise ValueError("Failed to compute homography matrix. Check if points are collinear.")

        # Calculate inverse matrix for reverse transformation
        self.matrix_inv = cv2.getPerspectiveTransform(self.map_points, self.video_points)

    def pixel_to_coords(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Transform pixel coordinates from video to geographic coordinates (lat, lng).

        Args:
            pixel_x: X coordinate in video frame
            pixel_y: Y coordinate in video frame

        Returns:
            Tuple of (latitude, longitude)
        """
        # Create homogeneous coordinate
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

        # Apply perspective transformation
        transformed = cv2.perspectiveTransform(point, self.matrix)

        if transformed is None or transformed.size == 0:
            raise ValueError("Transformation failed")

        lat = float(transformed[0, 0, 0])
        lng = float(transformed[0, 0, 1])

        return (lat, lng)

    def coords_to_pixel(self, lat: float, lng: float) -> Tuple[float, float]:
        """
        Transform geographic coordinates (lat, lng) back to video pixel coordinates.

        Args:
            lat: Latitude
            lng: Longitude

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        point = np.array([[[lat, lng]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix_inv)

        if transformed is None or transformed.size == 0:
            raise ValueError("Inverse transformation failed")

        pixel_x = float(transformed[0, 0, 0])
        pixel_y = float(transformed[0, 0, 1])

        return (pixel_x, pixel_y)

    def batch_transform(self, detections: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Transform multiple pixel coordinates at once.

        Args:
            detections: List of (x, y) pixel coordinates

        Returns:
            List of (lat, lng) tuples
        """
        results = []
        for pixel_x, pixel_y in detections:
            try:
                lat, lng = self.pixel_to_coords(pixel_x, pixel_y)
                results.append((lat, lng))
            except Exception:
                # Skip invalid transformations
                continue
        return results


class ZoneChecker:
    """
    Checks if points lie within irregular polygon zones using shapely.
    """

    def __init__(self, polygon_coords: List[Tuple[float, float]]):
        """
        Initialize zone checker with polygon boundary.

        Args:
            polygon_coords: List of (lat, lng) tuples forming the polygon boundary
        """
        if len(polygon_coords) < 3:
            raise ValueError("Polygon must have at least 3 points")

        self.polygon = Polygon(polygon_coords)
        if not self.polygon.is_valid:
            raise ValueError("Invalid polygon. Check for self-intersections or duplicate points.")

    def is_inside(self, lat: float, lng: float) -> bool:
        """
        Check if a point is inside the zone.

        Args:
            lat: Latitude
            lng: Longitude

        Returns:
            True if point is inside the zone, False otherwise
        """
        point = Point(lat, lng)
        return self.polygon.contains(point)

    def is_inside_or_boundary(self, lat: float, lng: float) -> bool:
        """
        Check if a point is inside or on the boundary of the zone.

        Args:
            lat: Latitude
            lng: Longitude

        Returns:
            True if point is inside or on boundary
        """
        point = Point(lat, lng)
        return self.polygon.contains(point) or self.polygon.touches(point)

    def filter_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filter a list of points to only those inside the zone.

        Args:
            points: List of (lat, lng) tuples

        Returns:
            Filtered list of points inside the zone
        """
        return [p for p in points if self.is_inside(p[0], p[1])]

    def get_polygon_bounds(self) -> Dict[str, float]:
        """
        Get bounding box of the polygon.

        Returns:
            Dict with min_lat, max_lat, min_lng, max_lng
        """
        minx, miny, maxx, maxy = self.polygon.bounds
        return {
            "min_lat": miny,
            "max_lat": maxy,
            "min_lng": minx,
            "max_lng": maxx,
        }


class CrowdHeatmapGenerator:
    """
    Generates heatmap data from detected crowd positions.
    """

    @staticmethod
    def generate_heatmap_points(detections: List[Tuple[float, float]], 
                                intensity_range: Tuple[float, float] = (0.3, 1.0)) -> List[Dict]:
        """
        Convert detection coordinates to heatmap points.

        Args:
            detections: List of (lat, lng) tuples
            intensity_range: Tuple of (min_intensity, max_intensity)

        Returns:
            List of dicts with 'lat', 'lng', and 'intensity' keys
        """
        if not detections:
            return []

        min_intensity, max_intensity = intensity_range

        # Group nearby points for density calculation
        points = []
        for i, (lat, lng) in enumerate(detections):
            intensity = min_intensity + (i % 10) * (max_intensity - min_intensity) / 10
            points.append({
                "lat": lat,
                "lng": lng,
                "intensity": round(intensity, 2),
            })

        return points

    @staticmethod
    def aggregate_detections(detection_frames: List[List[Tuple[float, float]]], 
                             time_decay: bool = True) -> List[Dict]:
        """
        Aggregate detections across multiple frames with optional time decay.

        Args:
            detection_frames: List of detection lists, one per frame
            time_decay: If True, recent frames have higher weight

        Returns:
            List of aggregated heatmap points
        """
        all_points = {}
        total_frames = len(detection_frames)

        for frame_idx, detections in enumerate(detection_frames):
            weight = 1.0
            if time_decay and total_frames > 1:
                # Recent frames have higher weight
                weight = (frame_idx + 1) / total_frames

            for lat, lng in detections:
                key = (round(lat, 6), round(lng, 6))
                if key not in all_points:
                    all_points[key] = {"lat": lat, "lng": lng, "count": 0, "weight": 0}
                all_points[key]["count"] += 1
                all_points[key]["weight"] += weight

        # Normalize weights and convert to intensity
        max_weight = max([p["weight"] for p in all_points.values()]) if all_points else 1.0

        result = []
        for lat_lng, data in all_points.items():
            intensity = (data["weight"] / max_weight) if max_weight > 0 else 0.5
            result.append({
                "lat": data["lat"],
                "lng": data["lng"],
                "intensity": round(intensity, 2),
                "count": data["count"],
            })

        return result


def create_transformer_from_config(config: Dict) -> Optional[HomographyTransformer]:
    """
    Create a HomographyTransformer from a config dictionary.
    Returns None if map coordinates are placeholders (unknown).

    Args:
        config: Dict with 'video_anchor_points' and 'map_anchor_points'

    Returns:
        HomographyTransformer instance or None if coordinates unknown
    """
    video_points = config.get("video_anchor_points", [])
    map_points = config.get("map_anchor_points", [])

    # Check if map points contain placeholders (strings)
    has_placeholders = False
    for point in map_points:
        if isinstance(point, list):
            if any(isinstance(coord, str) for coord in point):
                has_placeholders = True
                break
        elif isinstance(point, str):
            has_placeholders = True
            break
    
    if has_placeholders:
        return None  # Cannot create transformer with unknown coordinates

    # Ensure we have exactly 4 points for transformation
    if len(video_points) != 4 or len(map_points) != 4:
        raise ValueError("Both video_points and map_points must contain exactly 4 points")

    return HomographyTransformer(video_points, map_points)


def create_zone_checker_from_config(zone_config: Dict) -> Optional[ZoneChecker]:
    """
    Create a ZoneChecker from a zone configuration.
    Returns None if border coordinates are placeholders (unknown).

    Args:
        zone_config: Dict with 'border_points' or 'polygon_coords'

    Returns:
        ZoneChecker instance or None if coordinates unknown
    """
    polygon_coords = zone_config.get("border_points") or zone_config.get("polygon_coords", [])

    # Check if border points contain placeholders (strings)
    has_placeholders = False
    for coord in polygon_coords:
        if isinstance(coord, list):
            if any(isinstance(c, str) for c in coord):
                has_placeholders = True
                break
        elif isinstance(coord, str):
            has_placeholders = True
            break
    
    if has_placeholders:
        return None  # Cannot create zone checker with unknown coordinates

    return ZoneChecker(polygon_coords)
