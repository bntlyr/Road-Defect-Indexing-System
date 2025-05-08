import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import psutil
import os
from datetime import datetime
import tkinter.messagebox as messagebox
from tkinter import ttk
from dotenv import load_dotenv
import logging

# Module imports
from ..modules.object_detection import DefectDetector
from ..modules.severity_calculator import SeverityCalculator
from ..modules.camera import Camera
from ..modules.gps_reader import GPSReader
from ..modules.cloud_storage import CloudStorage

# GUI component imports
from .map_view import MapView
from .controls import ControlPanel
from .video_controls import VideoControlPanel
from .stats_log import StatsPanel
from .status import StatusBar
from .settings_window import SettingsWindow
from .upload_window import UploadWindow

# Utility imports
from ..utils.file_manager import FileManager
from ..utils.preprocess_utils import ImagePreprocessor

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.camera = None
        self.gps_reader = GPSReader()
        
        # Hardcoded paths
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "last.pt")
        self.storage_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        # Initialize file manager with storage path
        self.file_manager = FileManager(self.storage_path)
        
        # Initialize image adjustment values
        self.brightness_value = 50
        self.exposure_value = 50
        self.flip_h = False
        self.flip_v = False
        
        # Initialize detector with required parameters
        if not os.path.exists(self.model_path):
            raise Exception(f"Model not found at {self.model_path}")
            
        self.detector = DefectDetector(model_path=self.model_path)
        
        # Hardcoded camera parameters
        self.camera_params = {
            'device_id': 0,
            'resolution': (1920, 1080),
            'fps': 30,
            'focal_length': 3.67,  # mm
            'sensor_width': 4.8,   # mm
            'sensor_height': 3.6   # mm
        }
        
        # Initialize severity calculator with default values (will be updated when camera starts)
        self.severity_calculator = SeverityCalculator(
            camera_width=self.camera_params['resolution'][0],
            camera_height=self.camera_params['resolution'][1],
            focal_length=self.camera_params['focal_length'],
            sensor_width=self.camera_params['sensor_width'],
            sensor_height=self.camera_params['sensor_height']
        )
        
        self.preprocessor = ImagePreprocessor()
        self.cloud_storage = CloudStorage()
        
        # Initialize state variables
        self.is_inferencing = False
        self.camera_thread = None
        self.current_frame = None
        self.detection_results = None
        self.severity_results = None
        
        # Initialize GUI components
        self.setup_window()
        self.setup_components()
        
        # Bind closing event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start streams after a short delay to ensure UI is ready
        self.after(100, self.start_streams)

    def start_streams(self):
        """Start all background processes and updates"""
        self.update_status_bar()
        self.video_loop()
        
    def setup_window(self):
        """Configure the main window"""
        self.title("Road Defect Detection System")
        self.geometry("1200x800")
        self.minsize(1200, 800)  # Set minimum window size
        
        # Configure grid weights for better scaling
        self.grid_columnconfigure(0, weight=4)  # Left column (Video/Stats)
        self.grid_columnconfigure(1, weight=3)  # Right column (Map/Control)
        self.grid_rowconfigure(0, weight=7)     # Top row (Video/Map)
        self.grid_rowconfigure(1, weight=0)     # Bottom row (Stats/Control)
        self.grid_rowconfigure(2, weight=0)     # Status bar

        # Set default font sizes
        self.default_font = ("Arial", 12)
        self.title_font = ("Arial", 14, "bold")
        self.subtitle_font = ("Arial", 12, "bold")

    def setup_components(self):
        """Initialize and place all UI components"""
        # Create main components with larger minimum sizes
        self.video_panel = VideoControlPanel(self)
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_panel.configure(width=900, height=600)  # Larger size
        self.map_view = MapView(self)
        self.map_view.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.map_view.configure(width=700, height=600)  # Larger size
        # Create frames for stats and control to better manage their space
        stats_frame = ctk.CTkFrame(self)
        stats_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        stats_frame.grid_rowconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.configure(width=900, height=140)  # Smaller height
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)
        control_frame.grid_rowconfigure(0, weight=1)
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.configure(width=700, height=140)  # Smaller height
        # Place control panel and stats panel in their frames
        self.stats_panel = StatsPanel(stats_frame)
        self.stats_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.control_panel = ControlPanel(self)
        self.control_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, in_=control_frame)
        # Create status bar
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        # Initialize camera
        self.start_camera()

    def video_loop(self):
        """Main video processing loop"""
        if not hasattr(self, '_is_running'):
            self._is_running = True
            
        if not self._is_running:
            return
            
        try:
            if self.camera and self.camera.is_initialized and self.camera.is_streaming:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                # If camera is no longer initialized, try to restart it
                if frame is None and not self.camera.is_initialized:
                    self.logger.warning("Camera lost connection, attempting to restart...")
                    self.stop_camera()
                    self.start_camera()
                    # Schedule next frame update immediately
                    if self._is_running:
                        self.after(1, self.video_loop)
                    return
                
                if frame is not None:
                    # Get GPS data if available
                    gps_data = None
                    if hasattr(self, 'gps_reader') and self.gps_reader.is_connected and self.gps_reader.has_fix:
                        try:
                            gps_data = self.gps_reader.get_gps_data()
                        except Exception as e:
                            self.logger.error(f"Error getting GPS data: {e}")
                    
                    # Process frame if inferencing is enabled
                    if self.is_inferencing and self.detector:
                        # Update detector's save directory from settings (only when changed)
                        if hasattr(self, 'current_save_dir') and self.detector.save_dir != self.current_save_dir:
                            self.detector.save_dir = self.current_save_dir
                        
                        # Update settings periodically (every 30 frames)
                        if not hasattr(self, '_frame_count'):
                            self._frame_count = 0
                        self._frame_count += 1
                        if self._frame_count >= 30:
                            self.detector.update_settings()
                            self._frame_count = 0
                        
                        # Run inference with GPS data
                        processed_frame, frame_counts = self.detector.detect(frame, gps_data)
                        
                        # Update stats panel with current frame counts (in main thread)
                        if hasattr(self, 'stats_panel'):
                            self.stats_panel.update_stats(frame_counts, gps_data)
                    else:
                        processed_frame = frame
                        # Reset stats panel when not inferencing
                        if hasattr(self, 'stats_panel'):
                            self.stats_panel.update_stats({name: 0 for name in self.detector.class_names}, gps_data)
                    
                    # Update video panel (in main thread)
                    self.video_panel.update_frame(processed_frame)
                    
                    # Update status bar (in main thread)
                    if hasattr(self, 'status_bar'):
                        self.status_bar.update()
            
            # Schedule next frame update immediately
            if self._is_running:
                self.after(1, self.video_loop)
                
        except Exception as e:
            self.logger.error(f"Error in video loop: {e}")
            # On error, try to restart camera immediately
            if self._is_running:
                self.after(1, self.video_loop)

    def save_detection_results(self, frame):
        """Save detection results with metadata"""
        # Get GPS coordinates if available
        gps_data = None
        if hasattr(self, 'gps_reader') and self.gps_reader.is_connected and self.gps_reader.has_fix:
            try:
                gps_data = self.gps_reader.get_location()
            except Exception as e:
                print(f"Error getting GPS data: {e}")
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'gps': gps_data,
            'detections': self.detector.defect_counts,
            'frame_counts': self.detector.frame_counts
        }
        
        # Save image and metadata
        self.file_manager.save_detection(frame, metadata)

    def upload_to_cloud(self):
        """Upload detection results to cloud storage"""
        if self.detector.defect_counts:
            self.cloud_storage.upload_detection(
                self.current_frame,
                self.detector.defect_counts,
                self.detector.frame_counts
            )

    def update_status_bar(self):
        """Update status bar with system metrics"""
        if hasattr(self, 'status_bar'):
            self.status_bar.update()
        # Schedule next update
        if hasattr(self, '_status_after_id'):
            self.after_cancel(self._status_after_id)
        self._status_after_id = self.after(1000, self.update_status_bar)

    def toggle_inferencing(self):
        """Toggle the inferencing state"""
        try:
            # Simply toggle the inference state
            self.is_inferencing = not self.is_inferencing
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.update()
                
        except Exception as e:
            self.logger.error(f"Error toggling inference: {e}")

    def start_camera(self):
        """Initialize and start the camera"""
        try:
            # Get available cameras
            available_cameras = Camera.get_available_cameras()
            if not available_cameras:
                messagebox.showerror("Error", "No cameras detected!")
                return
            
            # Initialize camera with first available camera index
            first_camera_index = list(available_cameras.keys())[0]
            self.camera = Camera(first_camera_index)
            if not self.camera.is_initialized:
                messagebox.showerror("Error", "Failed to initialize camera!")
                return
            
            # Update camera selection combo box
            camera_names = [f"Camera {idx} - {name}" for idx, name in available_cameras.items()]
            self.video_panel.camera_combo.configure(values=camera_names)
            self.video_panel.camera_combo.set(camera_names[0])
            
            # Start video loop
            self.video_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")

    def stop_camera(self):
        """Stop and release the camera"""
        try:
            if self.camera:
                self.camera.is_streaming = False  # Signal camera to stop streaming
                self.camera.release()
                self.camera = None
            self.camera_thread = None
        except Exception as e:
            self.logger.error(f"Error stopping camera: {e}")

    def open_settings_window(self):
        """Open the settings window"""
        SettingsWindow(self)
        
    def open_upload_window(self):
        """Open the upload window"""
        UploadWindow(self)
        
    def calculate_severity(self):
        """Calculate severity of detected defects"""
        if self.detection_results:
            self.severity_results = self.severity_calculator.calculate(
                self.current_frame,
                self.detection_results
            )
            self.stats_panel.update_severity(self.severity_results)
        
    def on_closing(self):
        """Orderly and forceful shutdown to prevent hangs."""
        try:
            # 1. Stop the video loop
            self._is_running = False
            
            # 2. Signal all background threads/resources to stop
            try:
                if hasattr(self, 'gps_reader'):
                    self.gps_reader.is_running = False
                    self.gps_reader.stop()
            except Exception:
                pass
                
            try:
                if hasattr(self, 'detector'):
                    self.detector.cleanup()
            except Exception:
                pass
                
            try:
                if hasattr(self, 'stats_panel'):
                    self.stats_panel.cleanup()
            except Exception:
                pass
                
            try:
                self.stop_camera()
            except Exception:
                pass

            # 3. Destroy all child widgets
            try:
                for widget in self.winfo_children():
                    widget.destroy()
            except Exception:
                pass

            # 4. Destroy the main window
            try:
                self.destroy()
            except Exception:
                pass

        finally:
            # 5. Force kill the process, no matter what
            os._exit(0)
        
    def quit(self):
        """Clean up and quit the application"""
        self.stop_camera()
        super().quit()

    def adjust_brightness(self, value):
        """Adjust the brightness of the camera feed"""
        self.brightness_value = float(value)
        
    def adjust_exposure(self, value):
        """Adjust the exposure of the camera feed"""
        self.exposure_value = float(value)
        
    def flip_horizontal(self):
        """Toggle horizontal flip"""
        self.flip_h = not self.flip_h
        
    def flip_vertical(self):
        """Toggle vertical flip"""
        self.flip_v = not self.flip_v

    def switch_camera(self, camera_name: str):
        """Switch to a different camera"""
        try:
            if not self.camera:
                return
                
            # Extract camera index from name (format: "Camera X - Name")
            try:
                camera_index = int(camera_name.split()[1])
            except:
                self.logger.error(f"Invalid camera name format: {camera_name}")
                return
                
            if self.camera.switch_camera(camera_index):
                # Update resolution combo box with current resolution
                current_resolution = f"{self.camera.camera_width}x{self.camera.camera_height}"
                self.video_panel.resolution_combo.configure(values=self.camera.available_resolutions)
                self.video_panel.resolution_combo.set(current_resolution)
            else:
                messagebox.showerror("Error", f"Failed to switch to camera: {camera_name}")
                
        except Exception as e:
            self.logger.error(f"Error switching camera: {e}")
            messagebox.showerror("Error", f"Failed to switch camera: {str(e)}")

    def change_resolution(self, resolution: str):
        """Change camera resolution"""
        try:
            if not self.camera:
                return
                
            if not self.camera.change_resolution(resolution):
                messagebox.showerror("Error", f"Failed to change resolution to: {resolution}")
                
        except Exception as e:
            self.logger.error(f"Error changing resolution: {e}")
            messagebox.showerror("Error", f"Failed to change resolution: {str(e)}")

    def change_fps(self, fps: str):
        """Change camera FPS"""
        try:
            if not self.camera:
                return
                
            if not self.camera.change_fps(int(fps)):
                messagebox.showerror("Error", f"Failed to change FPS to: {fps}")
                
        except Exception as e:
            self.logger.error(f"Error changing FPS: {e}")
            messagebox.showerror("Error", f"Failed to change FPS: {str(e)}")

    def camera_loop(self):
        """Camera streaming loop"""
        while self.camera and self.camera.is_streaming:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    self.current_frame = frame
                    if self.is_inferencing:
                        self.process_frame(frame)
                time.sleep(1/self.camera.fps)
            except Exception as e:
                print(f"Error in camera loop: {e}")
                time.sleep(0.1)

    def update_camera_list(self):
        """Update the list of available cameras"""
        try:
            # Get available cameras
            available_cameras = Camera.get_available_cameras()
            
            if available_cameras:
                # Create camera list with resolutions
                camera_list = [f"Camera {cam['id']} ({cam['resolution']})" for cam in available_cameras]
                self.video_panel.camera_combo.configure(values=camera_list)
                
                # Set current camera
                current_camera = f"Camera {self.camera.device_id} ({self.camera.max_width}x{self.camera.max_height})"
                if current_camera in camera_list:
                    self.video_panel.camera_combo.set(current_camera)
                else:
                    self.video_panel.camera_combo.set(camera_list[0])
            else:
                self.video_panel.camera_combo.configure(values=["No cameras found"])
                self.video_panel.camera_combo.set("No cameras found")
                
        except Exception as e:
            print(f"Error updating camera list: {e}")

    def toggle_gps_connection(self):
        """Toggle GPS connection"""
        if not hasattr(self, 'gps_reader'):
            messagebox.showerror("Error", "GPS reader not initialized")
            return
            
        if self.gps_reader.is_connected():
            # Disconnect GPS
            self.gps_reader.stop()
            self.gps_button.configure(text="Disconnected", fg_color="gray")
            # Schedule button reset after 2 seconds
            self.after(2000, lambda: self.gps_button.configure(text="Connect GPS", fg_color=["#3B8ED0", "#1F6AA5"]))
        else:
            # Try to connect to GPS
            if self.gps_reader.connect_manually('COM3'):
                self.gps_button.configure(text="Connected", fg_color="gray")
            else:
                messagebox.showerror("Error", "Failed to connect to GPS device")
                self.gps_button.configure(text="Connect GPS", fg_color=["#3B8ED0", "#1F6AA5"])

    def save_settings(self, settings_win):
        new_save_dir = self.save_dir_entry.get()
        if new_save_dir:
            self.current_save_dir = new_save_dir
            os.makedirs(self.current_save_dir, exist_ok=True)
            if self.detector:
                self.detector.update_save_dir(self.current_save_dir)
        settings_win.destroy() 