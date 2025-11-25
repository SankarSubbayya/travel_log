"""
Journey Mapping - Create Google Maps paths from travel photos

This module creates visual journey maps by:
1. Extracting GPS coordinates from photos in chronological order
2. Generating Google Maps URLs with route paths
3. Creating interactive map visualizations
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class JourneyMapper:
    """
    Create journey maps from travel photo GPS data.

    Features:
    - Generate Google Maps URLs with multi-point routes
    - Group photos by day/location
    - Create interactive HTML maps
    """

    def __init__(self, qdrant_store=None):
        """
        Initialize journey mapper.

        Args:
            qdrant_store: Optional TravelLogQdrantStore instance
        """
        self.qdrant_store = qdrant_store

    def get_journey_points(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get chronologically sorted journey points from photos.

        Args:
            start_date: Filter photos from this date
            end_date: Filter photos until this date
            limit: Maximum number of points

        Returns:
            List of dicts with {lat, lon, datetime, filename, people}
        """
        if not self.qdrant_store:
            raise ValueError("QdrantStore required for journey mapping")

        # Get all photos with GPS data
        all_photos = self.qdrant_store.get_all_photos(limit=limit)

        # Filter photos with GPS coordinates
        journey_points = []
        for photo in all_photos:
            lat = photo.get('latitude')
            lon = photo.get('longitude')
            dt_str = photo.get('datetime')

            if lat and lon and dt_str:
                # Parse datetime if it's a string
                if isinstance(dt_str, str):
                    try:
                        dt = datetime.fromisoformat(dt_str)
                    except ValueError:
                        continue
                else:
                    dt = dt_str

                # Apply date filters
                if start_date and dt < start_date:
                    continue
                if end_date and dt > end_date:
                    continue

                journey_points.append({
                    'lat': lat,
                    'lon': lon,
                    'datetime': dt,
                    'filename': photo.get('filename', ''),
                    'people': photo.get('people', []),
                    'caption': photo.get('captions', {}).get('title', '')
                })

        # Sort by datetime
        journey_points.sort(key=lambda x: x['datetime'])

        logger.info(f"Found {len(journey_points)} journey points")
        return journey_points

    def create_google_maps_url(
        self,
        points: List[Dict],
        max_waypoints: int = 23
    ) -> str:
        """
        Generate Google Maps URL with route through all points.

        Google Maps allows up to 25 locations (origin + 23 waypoints + destination)

        Args:
            points: List of journey points (from get_journey_points)
            max_waypoints: Maximum waypoints (default 23 for Google Maps)

        Returns:
            Google Maps URL string
        """
        if not points:
            return ""

        if len(points) == 1:
            # Single point - just show location
            p = points[0]
            return f"https://www.google.com/maps/search/?api=1&query={p['lat']},{p['lon']}"

        # If too many points, sample them evenly
        if len(points) > max_waypoints + 2:
            # Keep first, last, and sample the middle
            step = (len(points) - 2) / (max_waypoints)
            indices = [0] + [int(1 + i * step) for i in range(max_waypoints)] + [len(points) - 1]
            sampled_points = [points[i] for i in indices]
            logger.warning(f"Sampling {len(sampled_points)} from {len(points)} points")
            points = sampled_points

        # Build Google Maps Directions URL
        # Format: origin + waypoints + destination
        origin = points[0]
        destination = points[-1]
        waypoints = points[1:-1]

        # Build URL
        origin_str = f"{origin['lat']},{origin['lon']}"
        dest_str = f"{destination['lat']},{destination['lon']}"

        if waypoints:
            waypoint_str = "|".join(f"{p['lat']},{p['lon']}" for p in waypoints)
            url = f"https://www.google.com/maps/dir/?api=1&origin={origin_str}&destination={dest_str}&waypoints={waypoint_str}"
        else:
            url = f"https://www.google.com/maps/dir/?api=1&origin={origin_str}&destination={dest_str}"

        return url

    def group_by_day(self, points: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group journey points by day.

        Args:
            points: List of journey points

        Returns:
            Dict mapping date strings to lists of points
        """
        grouped = {}
        for point in points:
            date_key = point['datetime'].strftime('%Y-%m-%d')
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(point)

        return grouped

    def generate_journey_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Generate complete journey summary with map URLs.

        Args:
            start_date: Filter from this date
            end_date: Filter until this date

        Returns:
            Dict with journey metadata and URLs
        """
        points = self.get_journey_points(start_date, end_date)

        if not points:
            return {
                'total_points': 0,
                'total_days': 0,
                'points': [],
                'map_url': '',
                'daily_maps': {}
            }

        # Group by day
        daily_groups = self.group_by_day(points)

        # Generate daily map URLs
        daily_maps = {}
        for date, day_points in daily_groups.items():
            daily_maps[date] = {
                'points': day_points,
                'map_url': self.create_google_maps_url(day_points),
                'num_photos': len(day_points),
                'people': list(set(p for point in day_points for p in point.get('people', [])))
            }

        # Overall journey map
        overall_map_url = self.create_google_maps_url(points)

        return {
            'total_points': len(points),
            'total_days': len(daily_groups),
            'start_date': points[0]['datetime'].strftime('%Y-%m-%d'),
            'end_date': points[-1]['datetime'].strftime('%Y-%m-%d'),
            'points': points,
            'map_url': overall_map_url,
            'daily_maps': daily_maps
        }

    def create_html_map(
        self,
        points: List[Dict],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Create interactive HTML map with markers and path.

        Uses Leaflet.js for interactive map visualization.

        Args:
            points: List of journey points
            output_path: Optional path to save HTML file

        Returns:
            HTML string
        """
        if not points:
            return "<p>No journey points available</p>"

        # Calculate center point
        center_lat = sum(p['lat'] for p in points) / len(points)
        center_lon = sum(p['lon'] for p in points) / len(points)

        # Generate marker JavaScript
        markers_js = []
        for i, point in enumerate(points):
            date_str = point['datetime'].strftime('%Y-%m-%d %H:%M')
            people_str = ', '.join(point.get('people', [])) or 'Unknown'
            caption = point.get('caption', point.get('filename', ''))

            marker_js = f"""
            L.marker([{point['lat']}, {point['lon']}])
                .bindPopup("<b>{caption}</b><br>Date: {date_str}<br>People: {people_str}")
                .addTo(map);
            """
            markers_js.append(marker_js)

        # Generate path coordinates
        path_coords = [[p['lat'], p['lon']] for p in points]
        path_js = f"var pathCoords = {path_coords};"

        # HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Travel Journey Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ height: 100vh; width: 100%; }}
        .info {{
            padding: 6px 8px;
            background: white;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 10);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        // Add markers
        {''.join(markers_js)}

        // Add path
        {path_js}
        var polyline = L.polyline(pathCoords, {{color: 'red', weight: 3}}).addTo(map);

        // Fit bounds to show all points
        map.fitBounds(polyline.getBounds());

        // Add info box
        var info = L.control({{position: 'topright'}});
        info.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'info');
            div.innerHTML = '<h4>Journey Summary</h4>' +
                           '<b>Total Points:</b> {len(points)}<br>' +
                           '<b>Start:</b> {points[0]['datetime'].strftime('%Y-%m-%d')}<br>' +
                           '<b>End:</b> {points[-1]['datetime'].strftime('%Y-%m-%d')}';
            return div;
        }};
        info.addTo(map);
    </script>
</body>
</html>
        """

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(html)
            logger.info(f"Saved journey map to {output_path}")

        return html


def create_journey_mapper(qdrant_store=None) -> JourneyMapper:
    """
    Factory function to create JourneyMapper instance.

    Args:
        qdrant_store: Optional TravelLogQdrantStore instance

    Returns:
        JourneyMapper instance
    """
    return JourneyMapper(qdrant_store=qdrant_store)
