import folium
from folium import plugins
import webbrowser
import os
import json
from datetime import datetime
from dotenv import load_dotenv

class MapManager:
    def __init__(self, save_dir="maps"):
        """
        Initialize the MapManager with a directory for saving maps.
        
        Args:
            save_dir: Directory where map files will be saved
        """
        self.save_dir = save_dir
        self.current_map = None
        self.defect_markers = []
        self.route_points = []
        os.makedirs(save_dir, exist_ok=True)
        # Load environment variables
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
        self.mapbox_api_key = os.getenv("MAPBOX_API_KEY", "")

    def create_map(self, center_lat, center_lon, zoom_start=15):
        """
        Create a new map centered at the specified coordinates.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            zoom_start: Initial zoom level
        """
        tiles = 'OpenStreetMap'
        if self.mapbox_api_key:
            tiles = f'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={self.mapbox_api_key}'
        self.current_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tiles,
            attr='Mapbox' if self.mapbox_api_key else 'OpenStreetMap'
        )
        self.defect_markers = []
        self.route_points = []

    def add_defect_marker(self, lat, lon, defect_type, severity, image_path=None):
        """
        Add a defect marker to the map.
        
        Args:
            lat: Latitude of the defect
            lon: Longitude of the defect
            defect_type: Type of defect
            severity: Severity level
            image_path: Optional path to defect image
        """
        if not self.current_map:
            return

        # Create popup content
        popup_content = f"""
        <div style='width: 200px'>
            <h4>Defect Information</h4>
            <p><b>Type:</b> {defect_type}</p>
            <p><b>Severity:</b> {severity}</p>
        """
        
        if image_path and os.path.exists(image_path):
            popup_content += f"""
            <img src='{image_path}' style='width: 100%; margin-top: 10px;'>
            """

        popup_content += "</div>"

        # Create marker with custom icon based on severity
        color = self._get_severity_color(severity)
        icon = folium.Icon(color=color, icon='info-sign')
        
        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=icon
        )
        
        marker.add_to(self.current_map)
        self.defect_markers.append({
            'lat': lat,
            'lon': lon,
            'type': defect_type,
            'severity': severity,
            'image_path': image_path
        })

    def add_route_point(self, lat, lon):
        """
        Add a point to the route.
        
        Args:
            lat: Latitude of the point
            lon: Longitude of the point
        """
        if not self.current_map:
            return
            
        self.route_points.append([lat, lon])
        
        # Update the route line
        if len(self.route_points) > 1:
            folium.PolyLine(
                self.route_points,
                color='blue',
                weight=2,
                opacity=0.8
            ).add_to(self.current_map)

    def save_map(self, filename=None):
        """
        Save the current map to an HTML file.
        
        Args:
            filename: Optional custom filename
        """
        if not self.current_map:
            return None
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"map_{timestamp}.html"
            
        filepath = os.path.join(self.save_dir, filename)
        self.current_map.save(filepath)
        return filepath

    def open_map(self, filepath):
        """
        Open the map in the default web browser.
        
        Args:
            filepath: Path to the map HTML file
        """
        if os.path.exists(filepath):
            webbrowser.open('file://' + os.path.abspath(filepath))

    def export_defects_data(self, filename=None):
        """
        Export defect data to a JSON file.
        
        Args:
            filename: Optional custom filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"defects_{timestamp}.json"
            
        filepath = os.path.join(self.save_dir, filename)
        
        data = {
            'defects': self.defect_markers,
            'route': self.route_points
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath

    def _get_severity_color(self, severity):
        """
        Get marker color based on severity level.
        
        Args:
            severity: Severity level (0-1)
            
        Returns:
            Color string for the marker
        """
        if severity >= 0.8:
            return 'red'
        elif severity >= 0.5:
            return 'orange'
        elif severity >= 0.2:
            return 'yellow'
        else:
            return 'green' 