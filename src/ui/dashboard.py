import sys
import os
import time
import cv2
import threading
import queue
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap

# Use relative imports
from .video_controls import VideoControls
from .main_controls import MainControls
from .statistics import Statistics
from .status_bar import StatusBar
from ..modules.camera import Camera  # Keep this as a relative import
from ..modules.gps_reader import GPSReader  # Add GPS reader import
from ..modules.detection import DefectDetector  # Add defect detector import
from .settings_manager import SettingsManager  # Add settings manager import


class GPSThread(threading.Thread):
    def __init__(self, gps_reader, gps_queue):
        super().__init__()
        self.gps_reader = gps_reader
        self.gps_queue = gps_queue
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.daemon = True  # Set as daemon thread
        self.last_gps_time = time.time()
        self.gps_interval = 0.01  # 100Hz polling rate (10ms)
        self.error_count = 0
        self.max_errors = 5  # Maximum consecutive errors before reducing polling rate
        self.last_valid_gps = None
        self.lock = threading.Lock()  # Add lock for thread safety
        self.priority = 1  # Set high priority for GPS thread

    def run(self):
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_gps_time >= self.gps_interval:
                    if self.gps_reader.is_connected():
                        try:
                            # Use a timeout for GPS reading to prevent blocking
                            lat, lon = self.gps_reader.read_gps_data(timeout=0.001)  # 1ms timeout
                            if lat is not None and lon is not None:
                                with self.lock:
                                    self.error_count = 0  # Reset error count on success
                                    self.last_valid_gps = (lat, lon)  # Store last valid GPS data
                                
                                # Try to put data in queue with timeout
                                try:
                                    self.gps_queue.put_nowait((lat, lon))
                                except queue.Full:
                                    # If queue is full, clear old data and add new
                                    try:
                                        while not self.gps_queue.empty():
                                            self.gps_queue.get_nowait()
                                        self.gps_queue.put_nowait((lat, lon))
                                    except queue.Empty:
                                        pass
                            else:
                                with self.lock:
                                    self.error_count += 1
                                    # If we have last valid GPS data, use it
                                    if self.last_valid_gps:
                                        try:
                                            self.gps_queue.put_nowait(self.last_valid_gps)
                                        except queue.Full:
                                            pass
                        except Exception as e:
                            with self.lock:
                                self.error_count += 1
                                if self.error_count >= self.max_errors:
                                    # Reduce polling rate temporarily if too many errors
                                    time.sleep(0.2)
                                    self.error_count = 0
                    self.last_gps_time = current_time
                else:
                    # Sleep for a very short time to prevent CPU hogging
                    time.sleep(0.001)  # 1ms sleep
            except Exception as e:
                self.logger.error(f"Error in GPS thread: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop

    def stop(self):
        """Stop the GPS thread safely"""
        self.running = False
        # Clear the queue
        while not self.gps_queue.empty():
            try:
                self.gps_queue.get_nowait()
            except queue.Empty:
                break


class GPSDisplayThread(threading.Thread):
    def __init__(self, gps_queue, statistics, status_bar, main_controls):
        super().__init__()
        self.gps_queue = gps_queue
        self.statistics = statistics
        self.status_bar = status_bar
        self.main_controls = main_controls
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.daemon = True  # Set as daemon thread
        self.last_display_time = time.time()
        self.display_interval = 0.2  # Update display every 200ms
        self.last_valid_gps = None
        self.lock = threading.Lock()  # Add lock for thread safety

    def run(self):
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_display_time >= self.display_interval:
                    try:
                        # Get the most recent GPS data with timeout
                        latest_gps = None
                        try:
                            while not self.gps_queue.empty():
                                latest_gps = self.gps_queue.get_nowait()
                        except queue.Empty:
                            pass
                        
                        if latest_gps:
                            with self.lock:
                                self.last_valid_gps = latest_gps
                                lat, lon = latest_gps
                                
                            # Update statistics and status bar using Qt's signal mechanism
                            if self.statistics and self.statistics.isVisible():
                                self.statistics.update_gps(lat, lon)
                            if self.status_bar and self.status_bar.isVisible():
                                self.status_bar.update_gps_status(True, has_fix=True)
                            # Update recording if enabled
                            if self.main_controls and self.main_controls.isVisible() and self.main_controls.is_recording:
                                self.main_controls.update_recording_gps(lat, lon)
                    except Exception as e:
                        self.logger.error(f"Error processing GPS data: {e}")
                    self.last_display_time = current_time
                else:
                    # Sleep for a short time to prevent CPU hogging
                    time.sleep(0.01)  # 10ms sleep
            except Exception as e:
                self.logger.error(f"Error in GPS display thread: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop

    def stop(self):
        """Stop the GPS display thread safely"""
        self.running = False
        # Clear the queue
        while not self.gps_queue.empty():
            try:
                self.gps_queue.get_nowait()
            except queue.Empty:
                break


class DeviceListenerThread(threading.Thread):
    """Thread to monitor for new camera connections"""
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.daemon = True  # Set as daemon thread
        self.last_camera_count = 0
        self.camera_in_use = False  # Flag to track if a camera is currently in use

    def run(self):
        while self.running:
            try:
                # Check if the application is in video upload mode
                if self.dashboard.is_video_upload_mode:
                    time.sleep(1)  # Sleep to avoid busy waiting
                    continue  # Skip camera checking if in video upload mode

                # Check if a camera is currently in use
                if hasattr(self.dashboard, 'video_controls'):
                    selected = self.dashboard.video_controls.camera_combo.currentText()
                    self.camera_in_use = selected != "Video File" and hasattr(self.dashboard, 'camera') and self.dashboard.camera.is_available

                # Only check for new cameras if no camera is currently in use
                if not self.dashboard.is_video_upload_mode and not self.camera_in_use:
                    current_camera_count = self._count_available_cameras()
                    if current_camera_count != self.last_camera_count:
                        self.logger.info(f"Camera count changed from {self.last_camera_count} to {current_camera_count}")
                        # Update camera combo box in the main thread
                        QApplication.instance().processEvents()  # Ensure UI is responsive
                        if hasattr(self.dashboard, 'video_controls'):
                            self.dashboard.video_controls._detect_available_cameras()
                        self.last_camera_count = current_camera_count

            except Exception as e:
                self.logger.error(f"Error in device listener thread: {e}")

            time.sleep(1.0)  # Check every second

    def _count_available_cameras(self):
        """Count the number of available cameras"""
        count = 0
        for i in range(1):  # Check first 2 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                count += 1
                cap.release()
        return count

    def stop(self):
        """Stop the listener thread"""
        self.running = False


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Defect Indexing System")
        self.setMinimumSize(1000, 700)

        # Define the icon path
        icon_path = os.path.join(os.path.dirname(__file__), '..','..', 'public', 'icons', 'icon.png')

        # Set the window icon here
        self.setWindowIcon(QIcon(icon_path))  # Ensure icon is set for the main window
        
        # Initialize all attributes first
        self.video_capture = None
        self.video_file = None
        self.capture_timer = None
        self.video_label = None
        self.video_controls = None
        self.statistics = None
        self.main_controls = None
        self.status_bar = None
        self.detector = None
        self.gps_thread = None
        self.gps_display_thread = None
        self.is_video_upload_mode = False  # Add this flag to track video upload mode
        
        # Initialize settings manager first
        self.settings_manager = SettingsManager()
        
        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.target_fps = 30  # Default target FPS
        self.frame_interval = int(1000 / self.target_fps)  # Default interval in ms
        
        # Initialize UI components first
        self._init_ui()
        self._init_timers()

        # Initialize GPS reader after UI is ready
        self.gps_reader = GPSReader()
        
        # Check initial GPS connection status once
        self._check_initial_gps_status()
        
        # Create a queue for GPS data
        self.gps_queue = queue.Queue(maxsize=10)  # Set a reasonable max size
        
        # Start GPS update thread
        self.gps_thread = GPSThread(self.gps_reader, self.gps_queue)
        self.gps_thread.start()

        # Start GPS display thread
        self.gps_display_thread = GPSDisplayThread(self.gps_queue, self.statistics, self.status_bar, self.main_controls)
        self.gps_display_thread.start()

        # Initialize the camera
        self.camera = Camera()
        
        # Initialize detector with GPS queue after GPS reader is created
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'road_defect.pt')
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            QMessageBox.critical(self, "Error", f"Model file not found at: {model_path}")
            return
            
        self.detector = DefectDetector(
            model_path=model_path,
            gps_reader=self.gps_reader,
            gps_queue=self.gps_queue,
            settings_manager=self.settings_manager,
            dashboard=self  # Pass the dashboard instance
        )
        
        if self.camera.is_available:  # Check if the camera is available
            self.capture_timer = QTimer(self)
            self.capture_timer.timeout.connect(self.update_frame)
            # Use camera's max FPS
            self.target_fps = self.camera.max_fps
            self.frame_interval = int(1000 / self.target_fps)
            self.capture_timer.start(self.frame_interval)
        else:
            print("Camera is not available. The application will run without video stream.")
            # Update status bar
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText("No camera available. Please upload a video.")

        # Commenting out the DeviceListenerThread initialization
        # self.device_listener = DeviceListenerThread(self)
        # self.device_listener.start()

    def _init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #f0f0f0; font-size: 14px; }
            QPushButton { background-color: #4a4a4a; color: #ddd; padding: 8px; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #6a6a6a; }
            QComboBox { background-color: #4a4a4a; color: #ddd; padding: 4px; }
            QLabel#VideoLabel { background-color: #1e1e1e; font-size: 20px; border: 1px solid #444; }
        """)

        # === Video Display Section ===
        self.video_label = QLabel("Video Stream")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # Set minimum size
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make it expand

        # Create a container widget for the video label with proper layout
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        video_layout.addWidget(self.video_label)

        # === Controls Section ===
        self.video_controls = VideoControls(self)
        self.statistics = Statistics()
        self.main_controls = MainControls()

        # Connect MainControls to Dashboard
        self.main_controls.dashboard = self

        # Connect signals from video controls
        self.video_controls.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.video_controls.zoom_slider.valueChanged.connect(self.update_zoom)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)
        controls_layout.addWidget(self.video_controls, 2)
        controls_layout.addWidget(self.statistics, 2)
        controls_layout.addWidget(self.main_controls, 2)

        # === Main Layout ===
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        main_layout.setSpacing(10)  # Add spacing between widgets
        main_layout.addWidget(video_container, 7)     # Top section
        main_layout.addLayout(controls_layout, 3)      # Bottom section
        self.setCentralWidget(central_widget)

        # === Status Bar ===
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)

        # Store references to controls in video_controls
        self.video_controls.dashboard = self
        self.video_controls.main_controls = self.main_controls

    def _init_timers(self):
        """Initialize timers with optimized settings"""
        # Use a high-precision timer for frame updates
        self.capture_timer = QTimer(self)
        self.capture_timer.setTimerType(Qt.PreciseTimer)  # Use precise timer for better timing
        self.capture_timer.timeout.connect(self.update_frame)
        
        # Initialize with target FPS of 14
        self.target_fps = 30
        self.frame_interval = int(1000 / self.target_fps)  # ~71ms per frame
        self.capture_timer.start(self.frame_interval)
        
        # Set up FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_frame_time = time.time()
        self.frame_times = []  # Store last 10 frame times for smoothing

    def _update_time(self):
        self.status_bar.status_time.setText(time.strftime("%H:%M:%S"))

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Force an immediate frame update to adjust video scaling
        self.update_frame()

    def _format_time(self, seconds):
        """Format time in seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_camera_frame(self):
        """Update the camera frame with frame rate control"""
        if not hasattr(self, 'camera') or self.camera is None or not self.camera.is_available:
            return

        try:
            # Get frame from camera
            frame = self.camera.read_frame()
            if frame is None:
                return

            # Calculate frame timing
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.frame_count += 1
            self.frame_times.append(elapsed_time)

            # Keep only last 30 frame times for average calculation
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)

            # Update FPS counter every second
            if elapsed_time >= 1.0:
                self.current_fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.fps_start_time = current_time
                
                # Calculate average frame time
                if self.frame_times:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    target_frame_time = 1.0 / self.target_fps
                    
                    # Adjust frame interval based on performance
                    if avg_frame_time > target_frame_time * 1.1:  # Running too slow
                        self.frame_interval = min(100, self.frame_interval + 2)
                    elif avg_frame_time < target_frame_time * 0.9:  # Running too fast
                        self.frame_interval = max(50, self.frame_interval - 1)
                    
                    # Update timer interval
                    if self.capture_timer:
                        self.capture_timer.setInterval(self.frame_interval)
                
                # Update FPS in status bar
                if hasattr(self, 'status_bar') and self.status_bar is not None:
                    self.status_bar.status_fps.setText(f"FPS: {self.current_fps:.1f}")

            # Apply zoom and flip
            frame = self.camera.digital_zoom(frame, self.camera.zoom_factor)
            frame = self.camera.flip_frame(frame, self.camera.flip_vertical, self.camera.flip_horizontal)

            # Store original frame for recording
            recording_frame = frame.copy()

            # Get GPS data from queue without blocking
            lat, lon = None, None
            try:
                if not self.gps_queue.empty():
                    lat, lon = self.gps_queue.get_nowait()
            except queue.Empty:
                pass

            # Process frame with detector if active
            if self.detector and self.detector.is_active:
                try:
                    # Process frame with detection
                    detection_result = self.detector.detect(frame)
                    
                    # Check if detection result is valid
                    if detection_result is not None and isinstance(detection_result, tuple) and len(detection_result) == 2:
                        processed_frame, counts = detection_result
                        if processed_frame is not None:
                            frame = processed_frame
                            # Update statistics with detection counts
                            if self.statistics:
                                # Convert counts to the format expected by update_defect_count
                                defect_counts = {
                                    'linear': counts.get('Linear-Crack', 0),
                                    'alligator': counts.get('Alligator-Crack', 0),
                                    'pothole': counts.get('pothole', 0)  # Changed to lowercase to match model output
                                }
                                self.statistics.update_defect_count(defect_counts)
                            logging.debug(f"Frame processed with detection. Counts: {counts}")
                        else:
                            logging.warning("Detection returned None frame")
                    else:
                        logging.warning("Invalid detection result format")
                        # Continue with original frame
                        frame = frame
                except Exception as e:
                    logging.error(f"Error in detection: {e}")
                    # If detection fails, continue with original frame
                    frame = frame
            else:
                logging.debug("Detection not active or detector not initialized")

            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            # Create QImage and QPixmap
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Update the video label
            self.video_label.setPixmap(scaled_pixmap)

            # Update GPS data if available
            if lat is not None and lon is not None:
                if self.statistics:
                    self.statistics.update_gps(lat, lon)
                if self.status_bar:
                    self.status_bar.update_gps_status(True, has_fix=True)

            # Handle recording if active
            if hasattr(self, 'main_controls') and self.main_controls and self.main_controls.is_recording:
                if not self.main_controls.is_paused:
                    self.main_controls.update_recording(recording_frame, lat, lon)

        except Exception as e:
            logging.error(f"Error updating camera frame: {e}")
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText(f"Error: {str(e)}")

    def update_frame(self):
        """Update the current frame display"""
        if self.is_video_upload_mode:  # Check if in video upload mode
            # Call the existing video upload logic
            if not self.video_file or not self.video_capture.isOpened():
                return
            
            # Only read frame if video is playing
            if hasattr(self, 'video_controls') and self.video_controls.is_playing:
                ret, frame = self.video_capture.read()
                if ret:
                    # Get current frame number and calculate time
                    frame_number = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                    current_time = frame_number / fps if fps > 0 else 0
                    
                    # Update video time for GPS sync
                    self.video_time = current_time
                    
                    # Update detector's video time for GPS synchronization
                    if self.detector:
                        self.detector.update_video_time(current_time)
                        # Get GPS data for current time
                        lat, lon = self.detector._get_valid_gps_data()
                        
                        # Update statistics with GPS data if available
                        if lat is not None and lon is not None:
                            self.statistics.update_gps(lat, lon)
                            logging.debug(f"Updated statistics with GPS at {current_time:.3f}s: lat={lat}, lon={lon}")
                    
                    # Update FPS counter
                    self.frame_count += 1
                    elapsed = time.time() - self.fps_start_time
                    if elapsed >= 1.0:
                        self.current_fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.fps_start_time = time.time()
                        self.status_bar.status_fps.setText(f"FPS: {self.current_fps:.1f}")
                    
                    # Update status bar
                    self.status_bar.status_time.setText(f"Time: {self._format_time(current_time)}")
                    
                    # Update GPS status - always show "Using GPS Logs" in video mode
                    if self.detector and self.detector.is_video_mode:
                        self.status_bar.update_gps_status(True, True, "Using GPS Logs")
                    else:
                        self.status_bar.update_gps_status(False)
                    
                    # Process frame with detector if active
                    if self.detector and self.detector.is_active:
                        try:
                            # Process frame with detection
                            detection_result = self.detector.detect(frame)
                            
                            # Check if detection result is valid
                            if detection_result is not None and isinstance(detection_result, tuple) and len(detection_result) == 2:
                                processed_frame, counts = detection_result
                                if processed_frame is not None:
                                    frame = processed_frame
                                    # Update statistics with detection counts
                                    if self.statistics:
                                        # Convert counts to the format expected by update_defect_count
                                        defect_counts = {
                                            'linear': counts.get('Linear-Crack', 0),
                                            'alligator': counts.get('Alligator-Crack', 0),
                                            'pothole': counts.get('pothole', 0)  # Changed to lowercase to match model output
                                        }
                                        self.statistics.update_defect_count(defect_counts)
                                    logging.debug(f"Frame processed with detection. Counts: {counts}")
                                else:
                                    logging.warning("Detection returned None frame")
                            else:
                                logging.warning("Invalid detection result format")
                                # Continue with original frame
                                frame = frame
                        except Exception as e:
                            logging.error(f"Error in detection: {e}")
                            # If detection fails, continue with original frame
                            frame = frame
                    else:
                        logging.debug("Detection not active or detector not initialized")
                    
                    # Convert frame to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    
                    # Create QImage and QPixmap
                    image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(image)
                    
                    # Scale pixmap to fit the label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(
                        self.video_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    
                    # Update the video label
                    self.video_label.setPixmap(scaled_pixmap)
                    
                    # Update video controls position
                    if self.video_controls:
                        self.video_controls.update_position(current_time)
                else:
                    # End of video
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning
                    self.status_bar.status_message.setText("Video playback complete. Click play to replay.")
                    if hasattr(self, 'video_controls'):
                        self.video_controls.is_playing = False
                        self.video_controls.play_stop_btn.setText("▶")
        else:
            # In camera mode, use update_camera_frame
            if hasattr(self, 'camera') and self.camera is not None and self.camera.is_available:
                self.update_camera_frame()
            else:
                logging.warning("Camera not available in update_frame")
                # Switch to video file mode if camera is not available
                if hasattr(self, 'video_controls'):
                    self.video_controls.camera_combo.setCurrentText("Video File")
                    self.stop_camera_feed()
                    if hasattr(self, 'status_bar'):
                        self.status_bar.status_message.setText("Please upload a video")

    def switch_to_video_file(self, video_path):
        """Switch to video file mode"""
        self.is_video_upload_mode = True  # Set the flag when switching to video upload mode
        try:
            # Stop current capture if any
            if self.video_capture is not None:
                self.video_capture.release()
            
            # Initialize new video capture
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                raise Exception("Failed to open video file")
            
            self.video_file = video_path
            
            # Disconnect GPS reader if connected
            if self.gps_reader and self.gps_reader.is_connected():
                self.gps_reader.disconnect()  # Call the disconnect method
            
            # Get video properties
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.video_duration = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_fps
            
            # Initialize detector if not already done
            if not self.detector:
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'road_defect.pt')
                self.detector = DefectDetector(model_path=model_path, gps_queue=self.gps_queue)
                logging.info("Initialized detector for video mode")
            
            # Set up GPS log synchronization
            video_dir = os.path.dirname(video_path)
            video_filename = os.path.basename(video_path)
            
            # Try to find matching GPS log file
            if video_filename.startswith('recording_'):
                timestamp = video_filename.split('_')[1] + '_' + video_filename.split('_')[2].split('.')[0]
                gps_log_filename = f"gps_log_{timestamp}.txt"
                gps_log_path = os.path.join(video_dir, gps_log_filename)
                
                if os.path.exists(gps_log_path):
                    logging.info(f"Found GPS log file: {gps_log_path}")
                    self.detector.set_video_mode(video_path, gps_log_path)
                    logging.info(f"Set video mode with GPS log: {gps_log_path}")
                else:
                    logging.warning(f"GPS log file not found: {gps_log_path}")
            
            # Update video controls
            if hasattr(self, 'video_controls') and self.video_controls is not None:
                self.video_controls.set_controls_state(True)
                self.video_controls.upload_btn.setEnabled(True)
                self.video_controls.flip_btn.setEnabled(False)
                self.video_controls.update_video_info(self.video_duration, self.video_fps)
                # Ensure video starts in paused state
                self.video_controls.is_playing = False
                self.video_controls.play_stop_btn.setText("▶")
                # Enable manual playback controls
                self.video_controls.play_stop_btn.setEnabled(True)
                self.video_controls.prev_btn.setEnabled(True)
                self.video_controls.forward_btn.setEnabled(True)
                self.video_controls.position_slider.setEnabled(True)
            
            # Show first frame
            ret, frame = self.video_capture.read()
            if ret:
                # Convert frame to RGB for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get the size of the video label
                label_size = self.video_label.size()
                
                # Scale the frame to fit the label while maintaining aspect ratio
                scaled_pixmap = self.scale_frame_to_label(frame, label_size)
                
                # Update the label with the new frame
                self.video_label.setPixmap(scaled_pixmap)
            else:
                logging.warning("Failed to read the first frame from the video.")
            
            # Stop the capture timer - video starts in stopped state
            if self.capture_timer:
                self.capture_timer.stop()
            
            # Update status
            self.status_bar.status_message.setText("Video loaded. Click Start Detection to begin.")
            
            # Update GPS status to show we're using GPS logs
            self.status_bar.update_gps_status(True, True, "Using GPS Logs")
                
        except Exception as e:
            logging.error(f"Error switching to video file: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load video file: {str(e)}")
            self.pause_video_playback()
            self.is_video_upload_mode = False  # Reset the flag only on error

    def start_video_playback(self):
        """Start video playback"""
        if self.video_capture is not None and self.capture_timer is not None:
            # Set up timer for video playback with correct FPS
            self.capture_timer.setInterval(int(1000 / self.video_fps))
            self.capture_timer.start()
            self.status_bar.status_message.setText(f"Playing: {os.path.basename(self.video_file)}")

    def pause_video_playback(self):
        """Pause video playback"""
        if self.capture_timer is not None:
            self.capture_timer.stop()
            if self.video_file:
                self.status_bar.status_message.setText(f"Paused: {os.path.basename(self.video_file)}")
            else:
                self.status_bar.status_message.setText("Video paused")

    def resume_video_playback(self):
        """Resume video playback"""
        if self.capture_timer is not None:
            self.capture_timer.start()
            if self.video_file:
                self.status_bar.status_message.setText(f"Playing: {os.path.basename(self.video_file)}")
            else:
                self.status_bar.status_message.setText("Video playing")

    def seek_video(self, seconds):
        """Seek video forward or backward by specified seconds"""
        if self.video_capture is not None and self.video_file is not None:
            try:
                # Get current frame position
                current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                
                # Calculate frame offset (use smaller steps for smoother seeking)
                frame_offset = int(fps * seconds)
                new_frame = current_frame + frame_offset
                
                # Ensure we don't go below 0
                new_frame = max(0, new_frame)
                
                # Get total frames to ensure we don't go beyond video length
                total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                new_frame = min(new_frame, total_frames - 1)
                
                # Set new position
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                
                # Update detector's video time for GPS synchronization
                if self.detector:
                    current_time = new_frame / fps
                    self.detector.update_video_time(current_time)
                    # Get and update GPS data for the new position
                    lat, lon = self.detector._get_valid_gps_data()
                    if lat is not None and lon is not None:
                        self.statistics.update_gps(lat, lon)
                        logging.debug(f"Updated statistics after seek to {current_time:.3f}s: lat={lat}, lon={lon}")
                
                # Force an immediate frame update
                self.update_frame()
                
            except Exception as e:
                logging.error(f"Error seeking video: {str(e)}")

    def scale_frame_to_label(self, frame, label_size):
        """Scale frame to fit label while maintaining aspect ratio"""
        try:
            # Convert frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate scaling factors
            width_ratio = label_size.width() / width
            height_ratio = label_size.height() / height
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Scale the image
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                new_width, new_height,
                        Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            return scaled_pixmap
        except Exception as e:
            print(f"Error scaling frame: {str(e)}")
            return None

    def _check_initial_gps_status(self):
        """Check initial GPS connection status once"""
        if not self.gps_reader:
            return

        if self.gps_reader.is_connected():
            lat, lon = self.gps_reader.read_gps_data()
            if lat is not None and lon is not None:
                self.status_bar.update_gps_status(True, has_fix=True)
                self.statistics.update_gps(lat, lon)
            else:
                self.status_bar.update_gps_status(True, has_fix=False)
        else:
            self.status_bar.update_gps_status(False)
            self.statistics.update_gps(0, 0)

    def get_gps_reader(self):
        """Return the GPS reader instance"""
        return self.gps_reader

    def closeEvent(self, event):
        """Handle cleanup when the window is closed"""
        try:
            # Stop GPS threads
            if self.gps_thread:
                self.gps_thread.stop()
                self.gps_thread.join(timeout=1.0)
            if self.gps_display_thread:
                self.gps_display_thread.stop()
                self.gps_display_thread.join(timeout=1.0)
            
            # Stop camera
            if self.camera:
                self.camera.cleanup()
            
            # Stop video capture
            if self.video_capture:
                self.video_capture.release()
            
            # Stop timers
            if self.capture_timer:
                self.capture_timer.stop()
            
            # Cleanup GPS reader
            if self.gps_reader:
                self.gps_reader.cleanup()
            
            # Stop recording if active
            if self.main_controls and self.main_controls.is_recording:
                self.main_controls.stop_recording()
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        
        event.accept()

    def stop_camera_feed(self):
        """Stop the camera feed and clean up resources"""
        try:
            # Stop the capture timer
            if hasattr(self, 'capture_timer') and self.capture_timer is not None:
                self.capture_timer.stop()
            
            # Release the camera
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.cleanup()
            
            # Clear the video label
            if hasattr(self, 'video_label') and self.video_label is not None:
                self.video_label.setText("Please upload a video")
        except Exception as e:
            print(f"Error stopping camera feed: {str(e)}")

    def change_camera(self):
        """Change camera with optimized settings"""
        print("Change camera method called")
        msg = None
        try:
            # Only show dialog when switching from video to camera
            if hasattr(self, 'video_controls') and self.video_controls is not None:
                selected = self.video_controls.camera_combo.currentText()
                if selected != "Video File":
                    msg = QMessageBox()
                    msg.setWindowTitle("Switching Source")
                    msg.setText("Please wait while the camera is being initialized.")
                    msg.setStandardButtons(QMessageBox.NoButton)
                    msg.show()
                    QApplication.processEvents()

            # Stop current capture timer if it exists
            if hasattr(self, 'capture_timer') and self.capture_timer is not None:
                self.capture_timer.stop()
            
            # Clean up video file state
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            
            # Clean up camera if it exists
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.cleanup()
                self.camera = None
            
            self.video_file = None
            self.is_video_upload_mode = False
            
            # Reset video time and GPS sync
            self.video_time = 0
            if hasattr(self, 'detector'):
                self.detector.video_time = 0
                self.detector.current_video_time = 0
                
            # Reset GPS values and status
            if hasattr(self, 'statistics'):
                self.statistics.update_gps(None, None)
            if hasattr(self, 'status_bar'):
                self.status_bar.update_gps_status(False)
            
            # Disconnect from GPS reader if in video mode
            if self.gps_reader and self.gps_reader.is_connected():
                self.gps_reader.disconnect()
            
            # Reconnect GPS when switching from video to camera
            if self.gps_reader:
                try:
                    if self.gps_reader._connect():  # Use _connect instead of connect
                        logging.info("Reconnected to GPS reader for camera mode")
                        if hasattr(self, 'status_bar'):
                            self.status_bar.update_gps_status(True, False, "Waiting for GPS fix...")
                except Exception as e:
                    logging.error(f"Error reconnecting to GPS reader: {e}")
                    if hasattr(self, 'status_bar'):
                        self.status_bar.update_gps_status(False)
                
                # Update detector's GPS reader
                if self.detector:
                    self.detector.gps_reader = self.gps_reader
                    self.detector.is_video_mode = False
                    self.detector.video_gps_data = {}
                    self.detector.video_gps_log = None
                    self.detector.current_video_time = 0
                    self.detector.video_time = 0
                    self.detector.last_gps_data = None
                    logging.info("Reset detector GPS settings for camera mode")
            
            if not hasattr(self, 'video_controls') or self.video_controls is None:
                print("Video controls not initialized")
                return
            
            selected = self.video_controls.camera_combo.currentText()
            
            if selected == "Video File":
                # Stop camera feed and show upload message
                if hasattr(self, 'video_label') and self.video_label is not None:
                    self.video_label.setText("Please upload a video")
                if hasattr(self, 'status_bar') and self.status_bar is not None:
                    self.status_bar.status_message.setText("Please upload a video")
                # Update video controls state for video file mode
                if hasattr(self, 'video_controls') and self.video_controls is not None:
                    self.video_controls.set_controls_state(True)
                    self.video_controls.upload_btn.setEnabled(True)
                    self.video_controls.flip_btn.setEnabled(False)
                    # Reset video controls
                    self.video_controls.is_playing = False
                    self.video_controls.play_stop_btn.setText("▶")
                    self.video_controls.play_stop_btn.setEnabled(False)
                    self.video_controls.prev_btn.setEnabled(False)
                    self.video_controls.forward_btn.setEnabled(False)
                    self.video_controls.position_slider.setEnabled(False)
                return

            # Handle camera selection 
            try:
                camera_index = int(selected.split()[-1])  # Extract number from "Camera X"
                self.camera = Camera(camera_index)
                
                if self.camera.is_available:
                    # Set the camera resolution to 1280x720 for better performance
                    self.camera.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.camera.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    
                    # Set optimized capture properties
                    self.camera.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit buffer size
                    self.camera.capture.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
                    
                    # Initialize timer with optimized settings
                    if hasattr(self, 'capture_timer') and self.capture_timer is not None:
                        self.capture_timer.setTimerType(Qt.PreciseTimer)  # Use precise timer for better timing
                        self.target_fps = 30
                        self.frame_interval = int(1000 / self.target_fps)
                        self.capture_timer.start(self.frame_interval)
                    
                    print("Camera initialized successfully.")
                    
                    # Update status bar with success message
                    if hasattr(self, 'status_bar') and self.status_bar is not None:
                        self.status_bar.status_message.setText("Camera connected")
                
                    # Update video controls for camera mode
                    if hasattr(self, 'video_controls') and self.video_controls is not None:
                        self.video_controls.set_controls_state(False)
                        self.video_controls.upload_btn.setEnabled(False)
                        self.video_controls.flip_btn.setEnabled(True)
                        # Reset video controls
                        self.video_controls.is_playing = False
                        self.video_controls.play_stop_btn.setText("▶")
                        self.video_controls.play_stop_btn.setEnabled(False)
                        self.video_controls.prev_btn.setEnabled(False)
                        self.video_controls.forward_btn.setEnabled(False)
                        self.video_controls.position_slider.setEnabled(False)

                    # Update detect button text based on record mode
                    if hasattr(self, 'main_controls') and self.main_controls is not None:
                        record_mode = self.main_controls.settings_manager.get_setting('record_mode')
                        if record_mode:
                            self.main_controls.detect_btn.setText("Start Recording")
                            self.main_controls.detect_btn.clicked.disconnect()
                            self.main_controls.detect_btn.clicked.connect(self.main_controls.toggle_recording)
                        else:
                            self.main_controls.detect_btn.setText("Start Detection")
                            self.main_controls.detect_btn.clicked.disconnect()
                            self.main_controls.detect_btn.clicked.connect(self.main_controls.toggle_detection)
                else:
                    print("Selected camera is not available.")
                    if hasattr(self, 'status_bar') and self.status_bar is not None:
                        self.status_bar.status_message.setText("Camera not available")
                    # Switch back to video file mode if camera is not available
                    self.video_controls.camera_combo.setCurrentText("Video File")
            except ValueError as e:
                print(f"Invalid camera selection: {e}")
                if hasattr(self, 'status_bar') and self.status_bar is not None:
                    self.status_bar.status_message.setText("Invalid camera selection")
                # Switch back to video file mode on invalid selection
                if hasattr(self, 'video_controls'):
                    self.video_controls.camera_combo.setCurrentText("Video File")
            finally:
                if msg:
                    msg.close()
            
        except Exception as e:
            print(f"Error changing camera: {str(e)}")
            if msg:
                msg.close()
            QMessageBox.warning(self, "Error", f"Failed to change camera: {str(e)}")
            # Switch back to video file mode on error
            if hasattr(self, 'video_controls'):
                self.video_controls.camera_combo.setCurrentText("Video File")

    def flip_camera(self):
        """Handle camera flip functionality"""
        if self.camera.is_available:
            # Toggle between flipped and not flipped states
            current_vertical = self.camera.flip_vertical
            current_horizontal = self.camera.flip_horizontal
            
            # If currently flipped, unflip; if not flipped, flip both
            if current_vertical and current_horizontal:
                self.camera.set_flipped(vertical=False, horizontal=False)
            else:
                self.camera.set_flipped(vertical=True, horizontal=True)
            
            # Force an immediate frame update
            self.update_frame()
        else:
            print("Cannot flip camera; it is not available.")

    def update_zoom(self):
        print("Update zoom method called")
        if self.camera.is_available:
            zoom_factor = self.video_controls.zoom_slider.value() / 100.0
            self.camera.set_zoom(zoom_factor)
            # Update zoom label
            self.video_controls.zoom_label.setText(f"Zoom: {zoom_factor:.2f}x")
            print(f"Zoom set to: {zoom_factor:.2f}")
        else:
            print("Cannot update zoom; camera is not available.")

    def update_gps(self):
        """Update GPS data and UI elements without reinitializing connection"""
        if not self.gps_reader:
            return

        # Only update UI if we have a valid connection
        if self.gps_reader.is_connected():
            lat, lon = self.gps_reader.read_gps_data()
            if lat is not None and lon is not None:
                # Update statistics
                if self.statistics:
                    self.statistics.update_gps(lat, lon)
                # Update status bar
                if self.status_bar:
                    self.status_bar.update_gps_status(True, has_fix=True)
            else:
                # Keep the connected status but show no fix
                if self.status_bar:
                    self.status_bar.update_gps_status(True, has_fix=False)
        else:
            # Only update UI if we're not connected
            if self.statistics:
                self.statistics.update_gps(0, 0)
            if self.status_bar:
                self.status_bar.update_gps_status(False)


def main():
    app = QApplication(sys.argv)

    # Optional: Set App Icon
    icon_path = os.path.join(os.path.dirname(__file__), 'public', 'icons', 'icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    
    window = Dashboard()
    window.show()
    window.setWindowIcon(QIcon(icon_path))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
