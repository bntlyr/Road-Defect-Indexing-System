import os
import math
import logging
import sqlite3
from typing import Dict, Tuple, List, Optional
from PIL import Image, ImageDraw
import threading
from queue import Queue
import time
import json

class OSMTileGenerator:
    """Generates map tiles from preloaded OSM data"""
    
    # Tile size in pixels
    TILE_SIZE = 256
    
    # Colors for different road types
    ROAD_COLORS = {
        'motorway': (255, 0, 0, 255),      # Red
        'trunk': (255, 69, 0, 255),        # Orange-red
        'primary': (255, 140, 0, 255),     # Dark orange
        'secondary': (255, 215, 0, 255),   # Gold
        'tertiary': (255, 255, 0, 255),    # Yellow
        'residential': (200, 200, 200, 255), # Light gray
        'unclassified': (180, 180, 180, 255), # Gray
        'default': (150, 150, 150, 255)    # Dark gray
    }

    def __init__(self, cache_dir: str):
        """Initialize the tile generator with preloaded OSM data"""
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, 'osm_data.db')
        
        # Verify database exists
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"OSM database not found at {self.db_path}. Please run preload_osm.py first.")
        
        # Load bounds information
        bounds_path = os.path.join(cache_dir, 'bounds_info.json')
        if os.path.exists(bounds_path):
            with open(bounds_path, 'r') as f:
                self.bounds = json.load(f)
        else:
            raise FileNotFoundError(f"Bounds information not found at {bounds_path}")
        
        # Initialize data structures
        self.tile_cache = {}
        self.tile_queue = Queue()
        self.worker_thread = None
        self.running = False
        
        # Start tile generation worker
        self.start_worker()

    def start_worker(self):
        """Start the tile generation worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._tile_worker, daemon=True)
        self.worker_thread.start()

    def stop_worker(self):
        """Stop the tile generation worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()

    def _tile_worker(self):
        """Background worker for generating tiles"""
        while self.running:
            try:
                tile_key = self.tile_queue.get(timeout=1)
                if tile_key not in self.tile_cache:
                    self._generate_tile(*tile_key)
            except Queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in tile worker: {e}")

    def _generate_tile(self, x: int, y: int, zoom: int):
        """Generate a single tile from preloaded data"""
        # Create blank tile
        image = Image.new('RGBA', (self.TILE_SIZE, self.TILE_SIZE), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        
        # Calculate tile bounds
        nw_lat, nw_lon = self._tile_to_latlon(x, y, zoom)
        se_lat, se_lon = self._tile_to_latlon(x + 1, y + 1, zoom)
        
        # Query database for ways in this tile
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get ways that intersect with this tile
        c.execute('''
            SELECT DISTINCT w.id, w.type, w.name, n.lat, n.lon
            FROM ways w
            JOIN way_nodes wn ON w.id = wn.way_id
            JOIN nodes n ON wn.node_id = n.id
            WHERE n.lat BETWEEN ? AND ?
            AND n.lon BETWEEN ? AND ?
            ORDER BY w.id, wn.sequence
        ''', (se_lat, nw_lat, nw_lon, se_lon))
        
        # Draw ways
        current_way = None
        points = []
        
        for way_id, way_type, way_name, lat, lon in c.fetchall():
            if current_way != way_id:
                if points:
                    self._draw_way(draw, points, way_type)
                current_way = way_id
                points = []
            
            # Convert lat/lon to pixel coordinates
            px, py = self._latlon_to_pixel(lat, lon, x, y, zoom)
            points.append((px, py))
        
        # Draw last way
        if points:
            self._draw_way(draw, points, way_type)
        
        conn.close()
        
        # Cache the tile
        self.tile_cache[(x, y, zoom)] = image

    def _draw_way(self, draw: ImageDraw, points: List[Tuple[float, float]], way_type: str):
        """Draw a way (road) on the tile"""
        if len(points) < 2:
            return
            
        # Get color for road type
        color = self.ROAD_COLORS.get(way_type, self.ROAD_COLORS['default'])
        
        # Draw the way
        draw.line(points, fill=color, width=2)

    def _latlon_to_pixel(self, lat: float, lon: float, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
        """Convert lat/lon to pixel coordinates within a tile"""
        n = 2.0 ** zoom
        lat_rad = math.radians(lat)
        
        # Calculate tile coordinates
        x = (lon + 180.0) / 360.0 * n
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        
        # Convert to pixel coordinates within tile
        px = (x - tile_x) * self.TILE_SIZE
        py = (y - tile_y) * self.TILE_SIZE
        
        return px, py

    def _tile_to_latlon(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to lat/lon"""
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        return lat, lon

    def get_tile(self, x: int, y: int, zoom: int) -> Optional[Image.Image]:
        """Get a tile, generating it if necessary"""
        tile_key = (x, y, zoom)
        
        # Check cache first
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        
        # Queue tile for generation
        self.tile_queue.put(tile_key)
        
        # Return a blank tile while generating
        return Image.new('RGBA', (self.TILE_SIZE, self.TILE_SIZE), (255, 255, 255, 0))

    def cleanup(self):
        """Clean up resources"""
        self.stop_worker()
        self.tile_cache.clear() 