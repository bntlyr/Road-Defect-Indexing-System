import os
import math
import logging
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QMessageBox
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QImage
import numpy as np
from dataclasses import dataclass
import threading
import queue
import time
from collections import defaultdict
from .osm_reader import OSMTileGenerator

@dataclass
class DefectLocation:
    """Represents a defect detection location on the map"""
    latitude: float
    longitude: float
    defect_type: str
    timestamp: float
    image_path: str
    confidence: float

@dataclass
class Cluster:
    """Represents a cluster of defect markers"""
    center_lat: float
    center_lon: float
    defects: List[DefectLocation]
    bounds: Tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon

class MapTile:
    """Represents a single map tile"""
    def __init__(self, x: int, y: int, zoom: int):
        self.x = x
        self.y = y
        self.zoom = zoom
        self.image: Optional[QPixmap] = None
        self.is_loading = False
        self.is_loaded = False

    def get_tile_path(self, cache_dir: str) -> str:
        """Get the local path for this tile"""
        return os.path.join(cache_dir, f"{self.zoom}_{self.x}_{self.y}.png")

    def load_from_cache(self, cache_dir: str) -> bool:
        """Try to load tile from cache"""
        tile_path = self.get_tile_path(cache_dir)
        if os.path.exists(tile_path):
            try:
                self.image = QPixmap(tile_path)
                self.is_loaded = True
                return True
            except Exception as e:
                logging.error(f"Error loading tile from cache: {e}")
        # If not found, use a blank tile
        self.image = QPixmap(256, 256)
        self.image.fill(QColor(220, 220, 220))
        self.is_loaded = True
        return False

class MapView(QWidget):
    """Widget that displays an interactive map with defect markers"""
    tile_loaded = pyqtSignal(int, int, int)  # x, y, zoom

    # Philippines/Luzon bounds
    PH_BOUNDS = {
        'min_lat': 12.0,  # Southern Luzon
        'max_lat': 18.5,  # Northern Luzon
        'min_lon': 119.5,  # Western Luzon
        'max_lon': 124.5,  # Eastern Luzon
    }
    
    # Manila coordinates (center of Luzon)
    MANILA_LAT = 14.5995
    MANILA_LON = 120.9842

    # Zoom level constants
    ZOOM_LEVELS = {
        'street': 17,      # Street level detail
        'neighborhood': 15, # Neighborhood level
        'city': 13,        # City level
        'region': 10,      # Regional level
        'country': 7       # Country level
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        # Initialize map settings with Manila as center
        self.zoom_level = self.ZOOM_LEVELS['street']  # Start at street level
        self.center_lat = self.MANILA_LAT
        self.center_lon = self.MANILA_LON
        self.tile_size = 256
        
        # Use map_tiles directory in src folder
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'map_tiles')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize OSM tile generator with preloaded data
        try:
            self.tile_generator = OSMTileGenerator(self.cache_dir)
        except FileNotFoundError as e:
            logging.error(f"OSM data not found: {e}")
            QMessageBox.critical(self, "Map Error", 
                "OSM data not found. Please run preload_osm.py first to load the map data.")
            self.tile_generator = None
            return

        # Tile bounds
        self.max_zoom = 19  # Maximum zoom for street-level detail
        self.min_zoom = 7   # Minimum zoom to keep Philippines in view

        # Clustering settings
        self.cluster_radius = 50  # pixels
        self.min_cluster_zoom = 12  # minimum zoom level for clustering
        self.clusters: List[Cluster] = []
        self.marker_positions: Dict[DefectLocation, Tuple[float, float]] = {}

        # Initialize view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(QBrush(QColor(240, 240, 240)))  # Light gray background

        # Setup layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        # Initialize tile management
        self.tiles: Dict[Tuple[int, int], QPixmap] = {}
        self.defect_locations: List[DefectLocation] = []
        self.tile_queue = queue.Queue()
        self.tile_loader = threading.Thread(target=self._tile_loader_worker, daemon=True)
        self.tile_loader.start()

        # Setup update timer for real-time tracking
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_tiles)
        self.update_timer.start(100)  # Update every 100ms

        # Connect signals
        self.tile_loaded.connect(self._on_tile_loaded)

        # Add current location marker
        self.current_location_marker = None
        self.current_location_accuracy = 0.0

    def _tile_loader_worker(self):
        """Background worker for loading tiles"""
        while True:
            try:
                tile_key = self.tile_queue.get(timeout=1)
                x, y, zoom = tile_key
                if zoom == self.zoom_level:
                    # Get tile from OSM generator
                    tile_image = self.tile_generator.get_tile(x, y, zoom)
                    if tile_image:
                        # Convert PIL Image to QPixmap
                        data = tile_image.tobytes('raw', 'RGBA')
                        qimage = QImage(data, tile_image.size[0], tile_image.size[1], QImage.Format_RGBA8888)
                        self.tiles[tile_key] = QPixmap.fromImage(qimage)
                        self.tile_loaded.emit(x, y, zoom)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in tile loader: {e}")

    def _on_tile_loaded(self, x: int, y: int, zoom: int):
        """Handle tile loaded signal"""
        if zoom == self.zoom_level:
            self.update_tiles()

    def is_within_ph_bounds(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Philippines/Luzon bounds"""
        return (
            self.PH_BOUNDS['min_lat'] <= lat <= self.PH_BOUNDS['max_lat'] and
            self.PH_BOUNDS['min_lon'] <= lon <= self.PH_BOUNDS['max_lon']
        )

    def set_center(self, lat: float, lon: float, accuracy: float = 0.0):
        """Set the center of the map with Philippines bounds checking"""
        # Clamp coordinates to Philippines/Luzon bounds
        lat = max(self.PH_BOUNDS['min_lat'], min(self.PH_BOUNDS['max_lat'], lat))
        lon = max(self.PH_BOUNDS['min_lon'], min(self.PH_BOUNDS['max_lon'], lon))
        
        self.center_lat = lat
        self.center_lon = lon
        self.current_location_accuracy = accuracy
        self.update_tiles()

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates with Philippines bounds checking"""
        # First check if coordinates are within Philippines bounds
        if not self.is_within_ph_bounds(lat, lon):
            logging.warning(f"Coordinates outside Philippines bounds: lat={lat}, lon={lon}")
            # Return center tile coordinates if outside bounds
            return self.lat_lon_to_tile(self.MANILA_LAT, self.MANILA_LON, zoom)

        n = 2.0 ** zoom
        lat_rad = math.radians(lat)
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        
        # Ensure coordinates are within valid range
        x = max(0, min(x, int(n) - 1))
        y = max(0, min(y, int(n) - 1))
        
        return x, y

    def tile_to_lat_lon(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to latitude/longitude"""
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        return lat, lon

    def update_tiles(self):
        """Update visible tiles based on current view"""
        if not self.isVisible():
            return

        try:
            # Calculate visible tile range
            view_rect = self.view.viewport().rect()
            center_tile_x, center_tile_y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
            
            # Calculate how many tiles we need
            tiles_x = math.ceil(view_rect.width() / self.tile_size) + 2
            tiles_y = math.ceil(view_rect.height() / self.tile_size) + 2
            
            # Calculate tile range with bounds checking
            start_x = center_tile_x - tiles_x // 2
            start_y = center_tile_y - tiles_y // 2
            end_x = start_x + tiles_x
            end_y = start_y + tiles_y
            
            # Request new tiles
            for y in range(start_y, end_y + 1):
                for x in range(start_x, end_x + 1):
                    tile_key = (x, y, self.zoom_level)
                    if tile_key not in self.tiles:
                        self.tile_queue.put(tile_key)

            # Update scene
            self.scene.clear()
            
            # Add tiles to scene
            for y in range(start_y, end_y + 1):
                for x in range(start_x, end_x + 1):
                    tile_key = (x, y, self.zoom_level)
                    tile = self.tiles.get(tile_key)
                    if tile:
                        pos_x = (x - start_x) * self.tile_size
                        pos_y = (y - start_y) * self.tile_size
                        self.scene.addPixmap(tile).setPos(pos_x, pos_y)

            # Add current location marker
            self._draw_current_location()
            
            # Add defect markers
            self._draw_defect_markers()

        except Exception as e:
            logging.error(f"Error updating tiles: {e}")

    def _draw_current_location(self):
        """Draw the current location marker"""
        if self.center_lat == 0.0 and self.center_lon == 0.0:
            return

        x, y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
        center_x, center_y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
        
        # Calculate marker position
        pos_x = (x - center_x + self.view.viewport().width() // (2 * self.tile_size)) * self.tile_size
        pos_y = (y - center_y + self.view.viewport().height() // (2 * self.tile_size)) * self.tile_size
        
        # Draw accuracy circle
        if self.current_location_accuracy > 0:
            accuracy_radius = self.current_location_accuracy * self.tile_size / 111000  # Rough conversion from meters to degrees
            accuracy_circle = self.scene.addEllipse(
                pos_x - accuracy_radius, pos_y - accuracy_radius,
                accuracy_radius * 2, accuracy_radius * 2,
                QPen(QColor(0, 120, 215, 100), 1),
                QBrush(QColor(0, 120, 215, 30))
            )
        
        # Draw location marker
        marker = self.scene.addEllipse(
            pos_x - 5, pos_y - 5, 10, 10,
            QPen(Qt.blue, 2),
            QBrush(QColor(0, 120, 215, 200))
        )
        
        # Add direction indicator
        direction_line = self.scene.addLine(
            pos_x, pos_y,
            pos_x + 15, pos_y,
            QPen(Qt.blue, 2)
        )
        direction_line.setRotation(0)  # Update this with actual heading if available

    def _create_clusters(self) -> List[Cluster]:
        """Create clusters from defect locations based on current view"""
        if not self.defect_locations or self.zoom_level >= self.min_cluster_zoom:
            return []

        # Convert all defect locations to screen coordinates
        screen_positions = {}
        for defect in self.defect_locations:
            x, y = self.lat_lon_to_tile(defect.latitude, defect.longitude, self.zoom_level)
            center_x, center_y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
            screen_x = (x - center_x + self.view.viewport().width() // (2 * self.tile_size)) * self.tile_size
            screen_y = (y - center_y + self.view.viewport().height() // (2 * self.tile_size)) * self.tile_size
            screen_positions[defect] = (screen_x, screen_y)

        # Create clusters using a simple distance-based algorithm
        clusters = []
        used_defects = set()

        for defect, (x1, y1) in screen_positions.items():
            if defect in used_defects:
                continue

            # Start a new cluster
            cluster_defects = [defect]
            used_defects.add(defect)
            min_lat, min_lon = defect.latitude, defect.longitude
            max_lat, max_lon = defect.latitude, defect.longitude

            # Find nearby defects
            for other_defect, (x2, y2) in screen_positions.items():
                if other_defect in used_defects:
                    continue

                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance <= self.cluster_radius:
                    cluster_defects.append(other_defect)
                    used_defects.add(other_defect)
                    min_lat = min(min_lat, other_defect.latitude)
                    min_lon = min(min_lon, other_defect.longitude)
                    max_lat = max(max_lat, other_defect.latitude)
                    max_lon = max(max_lon, other_defect.longitude)

            # Calculate cluster center
            center_lat = sum(d.latitude for d in cluster_defects) / len(cluster_defects)
            center_lon = sum(d.longitude for d in cluster_defects) / len(cluster_defects)

            clusters.append(Cluster(
                center_lat=center_lat,
                center_lon=center_lon,
                defects=cluster_defects,
                bounds=(min_lat, min_lon, max_lat, max_lon)
            ))

        return clusters

    def _draw_defect_markers(self):
        """Draw defect markers or clusters on the map with zoom-based sizing"""
        # Create clusters if needed
        self.clusters = self._create_clusters()

        # Adjust marker size based on zoom level
        marker_size = max(4, min(10, 20 - self.zoom_level))  # Smaller markers at higher zoom
        cluster_size = max(20, min(40, 60 - self.zoom_level))  # Smaller clusters at higher zoom

        if self.zoom_level >= self.ZOOM_LEVELS['street']:
            # Draw individual markers at street level
            for defect in self.defect_locations:
                self._draw_marker(defect, marker_size)
        else:
            # Draw clusters at lower zoom levels
            for cluster in self.clusters:
                self._draw_cluster(cluster, cluster_size)

    def _draw_marker(self, defect: DefectLocation, size: float = 5):
        """Draw a single defect marker with dynamic sizing"""
        x, y = self.lat_lon_to_tile(defect.latitude, defect.longitude, self.zoom_level)
        center_x, center_y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
        
        # Calculate marker position
        pos_x = (x - center_x + self.view.viewport().width() // (2 * self.tile_size)) * self.tile_size
        pos_y = (y - center_y + self.view.viewport().height() // (2 * self.tile_size)) * self.tile_size
        
        # Store position for later use
        self.marker_positions[defect] = (pos_x, pos_y)
        
        # Create marker with dynamic size
        marker = self.scene.addEllipse(
            pos_x - size/2, pos_y - size/2, size, size,
            QPen(Qt.red, max(1, size/5)),  # Thinner pen at higher zoom
            QBrush(QColor(255, 0, 0, 128))
        )
        
        # Add tooltip with more detail at street level
        if self.zoom_level >= self.ZOOM_LEVELS['street']:
            tooltip = (
                f"Type: {defect.defect_type}\n"
                f"Confidence: {defect.confidence:.2f}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(defect.timestamp))}\n"
                f"Location: {defect.latitude:.6f}, {defect.longitude:.6f}"
            )
        else:
            tooltip = (
                f"Type: {defect.defect_type}\n"
                f"Confidence: {defect.confidence:.2f}\n"
                f"Time: {time.strftime('%H:%M:%S', time.localtime(defect.timestamp))}"
            )
        marker.setToolTip(tooltip)

    def _draw_cluster(self, cluster: Cluster, size: float = 20):
        """Draw a cluster marker with dynamic sizing"""
        x, y = self.lat_lon_to_tile(cluster.center_lat, cluster.center_lon, self.zoom_level)
        center_x, center_y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
        
        # Calculate cluster position
        pos_x = (x - center_x + self.view.viewport().width() // (2 * self.tile_size)) * self.tile_size
        pos_y = (y - center_y + self.view.viewport().height() // (2 * self.tile_size)) * self.tile_size
        
        # Create cluster marker with dynamic size
        marker = self.scene.addEllipse(
            pos_x - size/2, pos_y - size/2, size, size,
            QPen(Qt.red, max(1, size/5)),  # Thinner pen at higher zoom
            QBrush(QColor(255, 0, 0, 128))
        )
        
        # Add count text
        text = self.scene.addText(str(len(cluster.defects)))
        text.setDefaultTextColor(Qt.white)
        text.setPos(pos_x - text.boundingRect().width()/2,
                   pos_y - text.boundingRect().height()/2)
        
        # Add tooltip with summary
        defect_types = defaultdict(int)
        for defect in cluster.defects:
            defect_types[defect.defect_type] += 1
        
        tooltip = "Cluster contains:\n"
        for defect_type, count in defect_types.items():
            tooltip += f"{defect_type}: {count}\n"
        tooltip += f"\nTotal: {len(cluster.defects)} defects"
        marker.setToolTip(tooltip)

    def mousePressEvent(self, event):
        """Handle mouse clicks for cluster expansion"""
        if event.button() == Qt.LeftButton and self.zoom_level < self.min_cluster_zoom:
            # Check if a cluster was clicked
            for cluster in self.clusters:
                x, y = self.lat_lon_to_tile(cluster.center_lat, cluster.center_lon, self.zoom_level)
                center_x, center_y = self.lat_lon_to_tile(self.center_lat, self.center_lon, self.zoom_level)
                pos_x = (x - center_x + self.view.viewport().width() // (2 * self.tile_size)) * self.tile_size
                pos_y = (y - center_y + self.view.viewport().height() // (2 * self.tile_size)) * self.tile_size
                
                # Calculate distance to click
                distance = math.sqrt((event.x() - pos_x) ** 2 + (event.y() - pos_y) ** 2)
                if distance <= 30:  # Click radius
                    # Zoom in to the cluster bounds
                    min_lat, min_lon, max_lat, max_lon = cluster.bounds
                    center_lat = (min_lat + max_lat) / 2
                    center_lon = (min_lon + max_lon) / 2
                    self.set_center(center_lat, center_lon)
                    self.set_zoom(self.min_cluster_zoom)
                    return

        super().mousePressEvent(event)

    def add_defect_location(self, lat: float, lon: float, defect_type: str, 
                          image_path: str, confidence: float):
        """Add a new defect location to the map if within Philippines bounds"""
        if self.is_within_ph_bounds(lat, lon):
            defect = DefectLocation(
                latitude=lat,
                longitude=lon,
                defect_type=defect_type,
                timestamp=time.time(),
                image_path=image_path,
                confidence=confidence
            )
            self.defect_locations.append(defect)
            self.update_tiles()
        else:
            logging.warning(f"Defect location outside Philippines bounds: lat={lat}, lon={lon}")

    def set_zoom(self, zoom: int):
        """Set the zoom level with bounds checking and smooth transitions"""
        if self.min_zoom <= zoom <= self.max_zoom:
            # Smooth zoom transition
            current_zoom = self.zoom_level
            self.zoom_level = zoom
            
            # If zooming in significantly, recenter on current view
            if zoom > current_zoom + 2:
                # Get current view center in screen coordinates
                view_center = self.view.viewport().rect().center()
                # Convert to map coordinates
                center_lat, center_lon = self.screen_to_lat_lon(view_center)
                self.set_center(center_lat, center_lon)
            
            self.tiles.clear()  # Clear tile cache when zooming
            self.update_tiles()
        else:
            logging.warning(f"Invalid zoom level: {zoom}. Must be between {self.min_zoom} and {self.max_zoom}")

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming with smooth transitions"""
        # Get mouse position in view coordinates
        mouse_pos = event.pos()
        
        # Calculate zoom factor based on wheel delta
        zoom_delta = 1 if event.angleDelta().y() > 0 else -1
        new_zoom = self.zoom_level + zoom_delta
        
        # Store current mouse position in map coordinates
        old_lat, old_lon = self.screen_to_lat_lon(mouse_pos)
        
        # Apply new zoom
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_level = new_zoom
            
            # Calculate new center to keep mouse position stable
            new_lat, new_lon = self.screen_to_lat_lon(mouse_pos)
            center_lat = self.center_lat + (old_lat - new_lat)
            center_lon = self.center_lon + (old_lon - new_lon)
            
            # Update view
            self.set_center(center_lat, center_lon)
            self.tiles.clear()
            self.update_tiles()

    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        self.update_tiles()

    def showEvent(self, event):
        """Handle widget show"""
        super().showEvent(event)
        self.update_tiles()

    def clear_defects(self):
        """Clear all defect markers"""
        self.defect_locations.clear()
        self.update_tiles()

    def cleanup(self):
        """Clean up resources"""
        self.tile_loader.join(timeout=1)
        self.update_timer.stop()
        self.tiles.clear()
        self.defect_locations.clear()
        self.scene.clear()
        if self.tile_generator:
            self.tile_generator.cleanup()
