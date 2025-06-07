import sys
import os
import time
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QImage, QPixmap

# Use relative imports
from .video_controls import VideoControls
from .main_controls import MainControls
from .statistics import Statistics
from .status_bar import StatusBar
from ..modules.camera import Camera  # Keep this as a relative import
from ..modules.gps_reader import GPSReader  # Add GPS reader import


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Defect Indexing System")
        self.setMinimumSize(1000, 700)
        
        # Initialize all attributes first
        self.video_capture = None
        self.video_file = None
        self.capture_timer = None
        self.video_label = None
        self.video_controls = None
        self.statistics = None
        self.main_controls = None
        self.status_bar = None
        
        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Initialize UI components first
        self._init_ui()
        self._init_timers()

        # Initialize GPS reader after UI is ready
        self.gps_reader = GPSReader()
        
        # Check initial GPS connection status once
        self._check_initial_gps_status()
        
        # Start GPS update timer
        self.gps_timer = QTimer(self)
        self.gps_timer.timeout.connect(self.update_gps)
        self.gps_timer.start(1000)  # Update GPS every second

        # Initialize the camera
        self.camera = Camera()  # Create an instance of the Camera class
        
        if self.camera.is_available:  # Check if the camera is available
            self.capture_timer = QTimer(self)
            self.capture_timer.timeout.connect(self.update_frame)
            self.capture_timer.start(30)  # Capture frame every 30 ms
        else:
            print("Camera is not available. The application will run without video stream.")
            # Update status bar
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText("No camera available. Please upload a video.")

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
        main_layout.addWidget(self.video_label, 7)     # Top section
        main_layout.addLayout(controls_layout, 3)      # Bottom section
        self.setCentralWidget(central_widget)

        # === Status Bar ===
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)

        # Store references to controls in video_controls
        self.video_controls.dashboard = self
        self.video_controls.main_controls = self.main_controls

    def _init_timers(self):
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._update_time)
        self.clock.start(1000)

    def _update_time(self):
        self.status_bar.status_time.setText(time.strftime("%H:%M:%S"))

    def switch_to_video_file(self, file_path):
        """Switch from camera to video file playback"""
        try:
            # Stop current capture timer
            if hasattr(self, 'capture_timer') and self.capture_timer is not None:
                self.capture_timer.stop()
            
            # Release current video capture
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
            
            # Stop camera feed
            self.stop_camera_feed()
            
            # Open the video file
            self.video_capture = cv2.VideoCapture(file_path)
            if not self.video_capture.isOpened():
                QMessageBox.warning(self, "Error", "Failed to open video file")
                return
            
            # Get video FPS
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default to 30 FPS if can't determine
            
            # Read and display the first frame
            ret, frame = self.video_capture.read()
            if ret:
                # Convert the frame to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Update the QLabel with the new image while maintaining aspect ratio
                if hasattr(self, 'video_label') and self.video_label is not None:
                    pixmap = QPixmap.fromImage(q_img)
                    scaled_pixmap = pixmap.scaled(
                        self.video_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.FastTransformation
                    )
                    self.video_label.setPixmap(scaled_pixmap)
            
            # Start video playback timer with correct interval
            self.capture_timer = QTimer(self)
            self.capture_timer.timeout.connect(self.update_frame)
            self.capture_timer.start(int(1000 / fps))  # Convert FPS to milliseconds
            
            # Store the video file path
            self.video_file = file_path
            
            # Update status
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText(f"Video loaded: {os.path.basename(file_path)}")
            
            # Enable video controls
            if hasattr(self, 'video_controls') and self.video_controls is not None:
                self.video_controls.set_controls_state(True)
        except Exception as e:
            print(f"Error switching to video file: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to switch to video file: {str(e)}")

    def pause_video_playback(self):
        """Pause video playback"""
        if self.capture_timer is not None:
            self.capture_timer.stop()
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText("Video paused")

    def resume_video_playback(self):
        """Resume video playback"""
        if self.capture_timer is not None and self.video_capture is not None:
            # Get video FPS
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default to 30 FPS if can't determine
            self.capture_timer.start(int(1000 / fps))  # Convert FPS to milliseconds
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText(f"Playing: {os.path.basename(self.video_file)}")

    def update_frame(self):
        try:
            if self.video_capture is not None:
                # Video file playback
                ret, frame = self.video_capture.read()
                if not ret:
                    # End of video file
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    ret, frame = self.video_capture.read()
            else:
                # Camera capture
                if not self.camera.is_available:
                    print("Camera is not available, skipping frame update.")
                    return
                ret, frame = self.camera.capture.read()

            if ret:
                # Update FPS counter
                self.frame_count += 1
                elapsed_time = time.time() - self.fps_start_time
                if elapsed_time >= 1.0:  # Update FPS every second
                    self.current_fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                    # Update FPS in status bar
                    if hasattr(self, 'status_bar') and self.status_bar is not None:
                        self.status_bar.status_fps.setText(f"FPS: {self.current_fps:.1f}")

                # Apply zoom and flip if using camera
                if self.video_capture is None and self.camera.is_available:
                    # Apply zoom first
                    frame = self.camera.digital_zoom(frame, self.camera.zoom_factor)
                    # Then apply flip
                    frame = self.camera.flip_frame(frame, self.camera.flip_vertical, self.camera.flip_horizontal)
                
                # Run detection if enabled
                if hasattr(self, 'main_controls') and self.main_controls.is_detecting and self.main_controls.detector:
                    frame, counts = self.main_controls.detector.detect(frame)
                    # Update statistics
                    if hasattr(self, 'statistics'):
                        self.statistics.donut_linear.count = counts.get('Linear-Crack', 0)
                        self.statistics.donut_alligator.count = counts.get('Alligator-Crack', 0)
                        self.statistics.donut_pothole.count = counts.get('Pothole', 0)
                        self.statistics.donut_linear.update()
                        self.statistics.donut_alligator.update()
                        self.statistics.donut_pothole.update()

                # Update recording if enabled
                if hasattr(self, 'main_controls') and self.main_controls.is_recording:
                    # Get current GPS coordinates
                    lat, lon = None, None
                    if self.gps_reader and self.gps_reader.is_connected():
                        lat, lon = self.gps_reader.read_gps_data()
                    # Update recording with frame and GPS data
                    self.main_controls.update_recording(frame, lat, lon)

                # Update GPS data based on source
                if hasattr(self, 'video_controls') and self.video_controls is not None:
                    if self.video_controls.camera_combo.currentText() == "Video File":
                        # Get current video position
                        current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                        current_time = current_frame / fps

                        # Try to find matching GPS log file
                        video_dir = os.path.dirname(self.video_file)
                        gps_logs = [f for f in os.listdir(video_dir) if f.startswith('gps_log_') and f.endswith('.txt')]
                        
                        if gps_logs:
                            # Sort by timestamp in filename to get the most recent
                            gps_logs.sort(reverse=True)
                            gps_log_path = os.path.join(video_dir, gps_logs[0])
                            
                            try:
                                with open(gps_log_path, 'r') as f:
                                    # Skip header
                                    next(f)
                                    # Read all GPS data
                                    gps_data = []
                                    for line in f:
                                        timestamp, lat, lon = line.strip().split(',')
                                        gps_data.append((float(timestamp), float(lat), float(lon)))
                                    
                                    # Find closest GPS data point to current time
                                    if gps_data:
                                        closest_point = min(gps_data, key=lambda x: abs(x[0] - current_time))
                                        lat, lon = closest_point[1], closest_point[2]
                                        
                                        # Update statistics and status bar
                                        if self.statistics:
                                            self.statistics.update_gps(lat, lon)
                                        if self.status_bar:
                                            self.status_bar.update_gps_status(True, has_fix=True)
                            except Exception as e:
                                print(f"Error reading GPS log: {str(e)}")
                    else:
                        # Use live GPS data for camera mode
                        if self.gps_reader and self.gps_reader.is_connected():
                            lat, lon = self.gps_reader.read_gps_data()
                            if lat is not None and lon is not None:
                                if self.statistics:
                                    self.statistics.update_gps(lat, lon)
                                if self.status_bar:
                                    self.status_bar.update_gps_status(True, has_fix=True)

                # Convert the frame to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Update the QLabel with the new image while maintaining aspect ratio
                if hasattr(self, 'video_label') and self.video_label is not None:
                    pixmap = QPixmap.fromImage(q_img)
                    scaled_pixmap = pixmap.scaled(
                        self.video_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.FastTransformation
                    )
                    self.video_label.setPixmap(scaled_pixmap)
            else:
                print("Failed to capture frame.")
        except Exception as e:
            print(f"Error updating frame: {str(e)}")

    def change_camera(self):
        print("Change camera method called")
        msg = None
        try:
            # Only show dialog when switching from video to camera
            if hasattr(self, 'video_controls') and self.video_controls is not None:
                selected = self.video_controls.camera_combo.currentText()
                if selected != "Video File":
                    msg = QMessageBox()
                    msg.setWindowTitle("Switching Source")
                    msg.setText("Please wait while the camera is being initialized.       ") #hardcoded space to make the dialog wider
                    msg.setStandardButtons(QMessageBox.NoButton)  # Remove all buttons
                    msg.show()
                    QApplication.processEvents()  # Process events to show dialog

            # Stop video playback if active
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
                self.video_file = None
            
            # Stop current capture timer if it exists
            if hasattr(self, 'capture_timer') and self.capture_timer is not None:
                self.capture_timer.stop()
            
            if not hasattr(self, 'video_controls') or self.video_controls is None:
                print("Video controls not initialized")
                return
                
            selected = self.video_controls.camera_combo.currentText()
            
            if selected == "Video File":
                # Stop camera feed and show upload message
                self.stop_camera_feed()
                if hasattr(self, 'status_bar') and self.status_bar is not None:
                    self.status_bar.status_message.setText("Please upload a video")
                return
            
            # Handle camera selection
            try:
                camera_index = int(selected.split()[-1])  # Extract number from "Camera X"
                self.camera = Camera(camera_index)
                
                if self.camera.is_available:
                    self.capture_timer = QTimer(self)
                    self.capture_timer.timeout.connect(self.update_frame)
                    self.capture_timer.start(30)
                    print("Camera initialized successfully.")
                    
                    # Update status bar with success message
                    if hasattr(self, 'status_bar') and self.status_bar is not None:
                        self.status_bar.status_message.setText(f"Source Switched Successfully")
                    
                    # Enable camera controls
                    if hasattr(self, 'video_controls') and self.video_controls is not None:
                        self.video_controls.set_controls_state(False)

                    # Update detect button text based on record mode
                    if hasattr(self, 'main_controls') and self.main_controls is not None:
                        if self.main_controls.settings_manager.get_setting('record_mode', False):
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
            except ValueError:
                print("Invalid camera selection")
                if hasattr(self, 'status_bar') and self.status_bar is not None:
                    self.status_bar.status_message.setText("Invalid camera selection")
        except Exception as e:
            print(f"Error changing camera: {str(e)}")
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText("Error changing camera")
        finally:
            # Close the waiting dialog if it was shown
            if msg is not None:
                msg.close()
                QApplication.processEvents()  # Process events to close dialog

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

    def stop_video_playback(self):
        """Stop video playback and return to camera view"""
        try:
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
                self.video_file = None
            
            if self.capture_timer is not None:
                self.capture_timer.stop()
            
            # Reinitialize camera
            self.camera = Camera()
            if self.camera.is_available:
                self.capture_timer = QTimer(self)
                self.capture_timer.timeout.connect(self.update_frame)
                self.capture_timer.start(30)
            
            # Update status
            if hasattr(self, 'status_bar') and self.status_bar is not None:
                self.status_bar.status_message.setText("Using Camera 0")
            
            # Update video label
            if hasattr(self, 'video_label') and self.video_label is not None:
                self.video_label.setText("Video Stream")
        except Exception as e:
            print(f"Error stopping video playback: {str(e)}")

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
                
                # Force an immediate frame update
                self.update_frame()
                
            except Exception as e:
                print(f"Error seeking video: {str(e)}")

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

    def get_gps_reader(self):
        """Return the GPS reader instance"""
        return self.gps_reader

    def closeEvent(self, event):
        try:
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
            if hasattr(self, 'camera'):
                self.camera.cleanup()
            if hasattr(self, 'gps_reader'):
                self.gps_reader.cleanup()
            # Clean up cloud manager if it exists
            if hasattr(self, 'main_controls') and self.main_controls is not None:
                if hasattr(self.main_controls, 'cloud_manager') and self.main_controls.cloud_manager is not None:
                    self.main_controls.cloud_manager.close()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Optional: Set App Icon
    icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'public', 'icons', 'icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = Dashboard()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
