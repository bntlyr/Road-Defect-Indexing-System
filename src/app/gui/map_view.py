import customtkinter as ctk
from tkintermapview import TkinterMapView
import os
import threading
import time
from typing import Optional, Tuple
import requests
import json
from PIL import Image
import io
import math

class MapView(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Create cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache", "map_tiles")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create map widget
        self.map_widget = TkinterMapView(
            self, 
            width=400, 
            height=300, 
            corner_radius=0,
            max_zoom=19
        )
        self.map_widget.pack(fill="both", expand=True)
        
        # Set default position (Manila coordinates)
        self.map_widget.set_position(14.5995, 120.9842)
        self.map_widget.set_zoom(15)
        
        # Initialize tracking variables
        self.current_marker = None
        self.last_position = None
        self.position_update_interval = 0.5  # seconds
        self.current_zoom = 15  # Default zoom level
        
    def start_tracking(self, gps_reader):
        """Start real-time position tracking"""
        self.gps_reader = gps_reader
        self._update_position()
            
    def stop_tracking(self):
        """Stop real-time position tracking"""
        if hasattr(self, '_update_after_id'):
            self.after_cancel(self._update_after_id)
            
    def _update_position(self):
        """Update position in the main thread"""
        try:
            gps_data = self.gps_reader.get_gps_data()
            if gps_data["latitude"] is not None and gps_data["longitude"] is not None:
                current_pos = (gps_data["latitude"], gps_data["longitude"])
                
                # Only update if position has changed significantly
                if (self.last_position is None or 
                    abs(current_pos[0] - self.last_position[0]) > 0.0001 or 
                    abs(current_pos[1] - self.last_position[1]) > 0.0001):
                    
                    self.last_position = current_pos
                    self.update_location(*current_pos)
                    
        except Exception as e:
            print(f"Error updating position: {e}")
            
        # Schedule next update
        self._update_after_id = self.after(int(self.position_update_interval * 1000), self._update_position)
        
    def update_location(self, lat, lon):
        """Update the map to show the current location"""
        # Update map position
        self.map_widget.set_position(lat, lon)
        
        # Update or create marker
        if self.current_marker:
            self.map_widget.delete_marker(self.current_marker)
        self.current_marker = self.map_widget.set_marker(
            lat, 
            lon, 
            text="Current Location",
            marker_color_circle="blue",
            marker_color_outside="blue",
            text_color="black"
        )
        
    def add_marker(self, lat, lon, text="", color="red"):
        """Add a marker to the map"""
        return self.map_widget.set_marker(
            lat, 
            lon, 
            text=text,
            marker_color_circle=color,
            marker_color_outside=color,
            text_color="black"
        )
        
    def clear_markers(self):
        """Clear all markers except current location"""
        self.map_widget.delete_all_marker()
        if self.current_marker and self.last_position:
            self.map_widget.set_marker(
                self.last_position[0],
                self.last_position[1],
                text="Current Location",
                marker_color_circle="blue",
                marker_color_outside="blue",
                text_color="black"
            )
            
    def cleanup(self):
        """Clean up resources"""
        self.stop_tracking() 