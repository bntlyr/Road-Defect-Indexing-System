from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QPushButton, QMessageBox, QDialog, QHBoxLayout
from .setting_dialogs import SettingsDialog
from .directory_selection_dialog import DirectorySelectionDialog
from .settings_manager import SettingsManager
from ..modules.detection import DefectDetector
from ..modules.cloud_connector import CloudConnector
from .cloud_manager import CloudManagerDialog
import os
import cv2
import time
from datetime import datetime
import queue
from PyQt5.QtCore import QThread, pyqtSignal
import logging
import subprocess
import numpy as np

class VideoWriterThread(QThread):
    """Thread for handling video writing operations"""
    frame_written = pyqtSignal()  # Signal to indicate frame has been written

    def __init__(self, video_path, width, height, fps=30.0):
        super().__init__()
        self.video_path = video_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.process = None
        self.pipe = None
        self.use_ffmpeg = False
        self.video_writer = None
        self.last_frame_time = time.time()
        self.frame_interval = 1.0 / fps
        self._initialize_writer()

    def _check_ffmpeg(self):
        """Check if FFMPEG is available in the system"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 timeout=2)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _initialize_writer(self):
        """Initialize video writer using FFMPEG if available, otherwise use OpenCV"""
        try:
            if self._check_ffmpeg():
                self.use_ffmpeg = True
                # FFMPEG command for MP4 encoding with H.264 codec
                command = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{self.width}x{self.height}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(self.fps),
                    '-i', '-',  # Input from pipe
                    '-c:v', 'libx264',  # Use H.264 codec
                    '-preset', 'ultrafast',  # Use ultrafast preset for better performance
                    '-crf', '28',  # Slightly lower quality for better performance
                    '-f', 'mp4',
                    self.video_path
                ]
                
                # Start FFMPEG process
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.pipe = self.process.stdin
                logging.info(f"FFMPEG process started for video: {self.video_path}")
            else:
                # Fallback to OpenCV VideoWriter
                self.use_ffmpeg = False
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.video_path,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                    True  # isColor
                )
                if not self.video_writer.isOpened():
                    raise Exception("Failed to open OpenCV VideoWriter")
                logging.info(f"Using OpenCV VideoWriter for video: {self.video_path}")
            
        except Exception as e:
            logging.error(f"Error initializing video writer: {e}")
            raise

    def run(self):
        while self.running:
            try:
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # Only process frames at the target FPS
                if elapsed >= self.frame_interval:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        if frame is not None:
                            if self.use_ffmpeg and self.pipe is not None:
                                # Write frame to FFMPEG pipe
                                self.pipe.write(frame.tobytes())
                            elif not self.use_ffmpeg and self.video_writer is not None:
                                # Write frame using OpenCV
                                self.video_writer.write(frame)
                            self.last_frame_time = current_time
                    else:
                        # If no frames in queue, sleep for a short time
                        time.sleep(0.001)
                else:
                    # Sleep for the remaining time until next frame
                    time.sleep(max(0.001, self.frame_interval - elapsed))
                    
            except Exception as e:
                logging.error(f"Error writing video frame: {e}")
                time.sleep(0.001)  # Prevent tight loop on error

    def add_frame(self, frame):
        """Add a frame to the write queue"""
        try:
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # If queue is full, remove oldest frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
        except Exception as e:
            logging.error(f"Error adding frame to queue: {e}")

    def stop(self):
        """Stop the thread and clean up"""
        self.running = False
        try:
            # Clear the queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if self.use_ffmpeg:
                # Close FFMPEG pipe and process
                if self.pipe:
                    self.pipe.close()
                if self.process:
                    self.process.wait(timeout=5)  # Wait for FFMPEG to finish
                    if self.process.poll() is None:
                        self.process.terminate()  # Force terminate if still running
                    self.process = None
            else:
                # Release OpenCV VideoWriter
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
            
            logging.info("Video writer thread stopped and cleaned up")
        except Exception as e:
            logging.error(f"Error stopping video writer thread: {e}")

    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.stop()

class MainControls(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Main Control", parent)
        
        # Initialize all attributes
        self.detector = None
        self.is_detecting = False
        self.is_recording = False
        self.is_paused = False  # Add paused state
        self.settings_manager = SettingsManager()
        self.dashboard = None  # Will be set by Dashboard
        self.cloud_connected = False
        self.cloud_connector = None
        self.cloud_manager = None
        
        # Recording attributes
        self.video_writer = None
        self.gps_log_file = None
        self.recording_start_time = None
        self.last_gps_update = None  # Track last GPS update time
        
        # Initialize UI components
        self.detect_btn = None
        self.analysis_btn = None
        self.cloud_btn = None
        self.settings_btn = None
        self.pause_btn = None  # Add pause button
        
        self._setup_ui()
        
        # Try to connect to cloud storage automatically
        self._try_connect_cloud()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Create a horizontal layout for detect and pause buttons
        detect_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("Start Detection")
        self.detect_btn.clicked.connect(self.toggle_detection)
        detect_layout.addWidget(self.detect_btn)

        self.pause_btn = QPushButton("⏸")  # Pause symbol
        self.pause_btn.setFixedWidth(40)  # Make it smaller
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)  # Initially disabled
        self.pause_btn.hide()  # Initially hidden
        detect_layout.addWidget(self.pause_btn)

        layout.addLayout(detect_layout)

        self.analysis_btn = QPushButton("Run Analysis")
        self.analysis_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.analysis_btn)

        self.cloud_btn = QPushButton("Connect Cloud")
        self.cloud_btn.clicked.connect(self.toggle_cloud)
        layout.addWidget(self.cloud_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_btn)

        self.setLayout(layout)

    def _try_connect_cloud(self):
        """Try to connect to cloud storage automatically"""
        try:
            # Use default cloud directory in user's home directory
            cloud_dir = os.path.join(os.path.expanduser("~"), "RoadDefectCloud")
            
            # Initialize cloud connector
            self.cloud_connector = CloudConnector(cloud_dir)
            self.cloud_connected = True
            self.cloud_btn.setText("Manage Cloud Storage")
            
        except Exception as e:
            print(f"Failed to connect to cloud storage automatically: {e}")
            self.cloud_connected = False
            self.cloud_btn.setText("Connect Cloud")

    def toggle_detection(self):
        """Toggle defect detection"""
        try:
            if not self.is_detecting:
                # Check if we're in video file mode or camera mode
                is_video_mode = self.dashboard.video_controls.camera_combo.currentText() == "Video File"
                
                if is_video_mode:
                    # Video file mode
                    if not self.dashboard.video_file:
                        QMessageBox.warning(self, "Warning", "Please upload a video file first.")
                        return

                    # Use the dashboard's detector instance
                    if not self.dashboard.detector:
                        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'road_defect.pt')
                        self.dashboard.detector = DefectDetector(model_path=model_path, gps_queue=self.dashboard.gps_queue)
                        logging.info("Initialized detector for detection")
                    
                    # Ensure detector is properly initialized
                    if not self.dashboard.detector.model:
                        logging.error("Detector model not loaded")
                        raise Exception("Failed to load detection model")
                    
                    # Start video playback
                    self.dashboard.start_video_playback()
                    self.dashboard.video_controls.is_playing = True
                    self.dashboard.video_controls.play_stop_btn.setText("⏸")
                    # Enable manual playback controls while detection is active
                    self.dashboard.video_controls.play_stop_btn.setEnabled(False)
                    self.dashboard.video_controls.prev_btn.setEnabled(True)
                    self.dashboard.video_controls.forward_btn.setEnabled(True)
                    self.dashboard.video_controls.position_slider.setEnabled(True)
                else:
                    # Camera mode
                    if not self.dashboard.camera or not self.dashboard.camera.is_available:
                        QMessageBox.warning(self, "Warning", "Camera is not available.")
                        return

                    # Try to read a test frame to check camera functionality
                    try:
                        ret, _ = self.dashboard.camera.capture.read()
                        if not ret:
                            # Camera error detected, switch to video mode
                            msg = QMessageBox()
                            msg.setIcon(QMessageBox.Warning)
                            msg.setWindowTitle("Camera Error")
                            msg.setText("Unable to capture frames from camera. Switching to video upload mode.")
                            msg.setInformativeText("Please upload a video file to continue detection.")
                            msg.setStandardButtons(QMessageBox.Ok)
                            msg.exec_()
                            
                            # Switch to video file mode
                            self.dashboard.video_controls.camera_combo.setCurrentText("Video File")
                            self.dashboard.stop_camera_feed()
                            if self.dashboard.status_bar:
                                self.dashboard.status_bar.status_message.setText("Please upload a video")
                            return
                    except Exception as e:
                        logging.error(f"Camera error: {str(e)}")
                        # Camera error detected, switch to video mode
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle("Camera Error")
                        msg.setText("Camera error detected. Switching to video upload mode.")
                        msg.setInformativeText("Please upload a video file to continue detection.")
                        msg.setStandardButtons(QMessageBox.Ok)
                        msg.exec_()
                        
                        # Switch to video file mode
                        self.dashboard.video_controls.camera_combo.setCurrentText("Video File")
                        self.dashboard.stop_camera_feed()
                        if self.dashboard.status_bar:
                            self.dashboard.status_bar.status_message.setText("Please upload a video")
                        return

                    # Use the dashboard's detector instance
                    if not self.dashboard.detector:
                        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'road_defect.pt')
                        self.dashboard.detector = DefectDetector(model_path=model_path, gps_queue=self.dashboard.gps_queue)
                        logging.info("Initialized detector for camera mode")
                    
                    # Ensure detector is properly initialized
                    if not self.dashboard.detector.model:
                        logging.error("Detector model not loaded")
                        raise Exception("Failed to load detection model")
                
                # Set detector as active
                self.is_detecting = True
                self.dashboard.detector.is_active = True
                self.dashboard.detector.start_detection()
                self.detect_btn.setText("Stop Detection")
                if self.dashboard and self.dashboard.status_bar:
                    self.dashboard.status_bar.status_message.setText("Detection started")
                logging.info("Detection started")
            else:
                # Stop detection for both modes
                if self.dashboard.video_controls.camera_combo.currentText() == "Video File":
                    # Stop video playback
                    self.dashboard.pause_video_playback()
                    self.dashboard.video_controls.is_playing = False
                    self.dashboard.video_controls.play_stop_btn.setText("▶")
                    # Disable manual playback controls
                    self.dashboard.video_controls.play_stop_btn.setEnabled(True)
                    self.dashboard.video_controls.prev_btn.setEnabled(True)
                    self.dashboard.video_controls.forward_btn.setEnabled(True)
                    self.dashboard.video_controls.position_slider.setEnabled(True)
                
                # Set detector as inactive
                self.is_detecting = False
                self.dashboard.detector.is_active = False
                self.dashboard.detector.stop_detection()  # Call stop_detection to ensure proper cleanup
                self.detect_btn.setText("Start Detection")
                if self.dashboard and self.dashboard.status_bar:
                    self.dashboard.status_bar.status_message.setText("Detection stopped")
                logging.info("Detection stopped")
                
        except Exception as e:
            logging.error(f"Error toggling detection: {e}")
            if self.dashboard and self.dashboard.status_bar:
                self.dashboard.status_bar.status_message.setText("Error toggling detection")

    def _check_detection_running(self):
        """Check if detection is running and show warning if it is"""
        if self.is_detecting:
            QMessageBox.warning(self, "Warning", "Please stop detection first.")
            return True
        return False

    def toggle_recording(self):
        """Toggle video recording and GPS logging"""
        if not self.is_recording:
            # Start recording
            if not self.dashboard or not hasattr(self.dashboard, 'video_controls'):
                logging.error("Dashboard or video controls not available")
                return

            video_controls = self.dashboard.video_controls
            if video_controls.camera_combo.currentText() == "Video File":
                QMessageBox.warning(self, "Warning", "Recording is only available in camera mode.")
                return

            # Get recording output directory
            output_dir = self.settings_manager.get_recording_directory()
            if not output_dir:
                QMessageBox.warning(self, "Warning", "Please set a recording output directory in settings.")
                return

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamp for filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")
            gps_log_path = os.path.join(output_dir, f"gps_log_{timestamp}.txt")

            try:
                # Initialize video writer
                frame = self.dashboard.camera.capture.read()[1]
                height, width = frame.shape[:2]
                
                # Use MP4V codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    30.0,  # FPS
                    (width, height),
                    True  # isColor
                )
                
                if not self.video_writer.isOpened():
                    raise Exception("Failed to open video writer")

                # Initialize GPS log file
                self.gps_log_file = open(gps_log_path, 'w')
                self.gps_log_file.write("timestamp,latitude,longitude\n")
                self.recording_start_time = time.time()

                # Update UI
                self.is_recording = True
                self.is_paused = False
                self.detect_btn.setText("Stop Recording")
                self.detect_btn.clicked.disconnect()
                self.detect_btn.clicked.connect(self.toggle_recording)
                self.pause_btn.show()  # Show pause button
                self.pause_btn.setEnabled(True)
                self.pause_btn.setText("⏸")

                # Update status
                if self.dashboard.status_bar:
                    self.dashboard.status_bar.status_message.setText("Recording...")

            except Exception as e:
                logging.error(f"Error starting recording: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to start recording: {str(e)}")
                self._cleanup_recording()
        else:
            # Stop recording
            self._cleanup_recording()
            
            # Update UI
            self.is_recording = False
            self.is_paused = False
            self.detect_btn.setText("Start Recording")
            self.detect_btn.clicked.disconnect()
            self.detect_btn.clicked.connect(self.toggle_recording)
            self.pause_btn.hide()  # Hide pause button
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("⏸")

            # Update status
            if self.dashboard.status_bar:
                self.dashboard.status_bar.status_message.setText("Recording stopped")

    def _cleanup_recording(self):
        """Clean up recording resources"""
        logging.info("Cleaning up recording resources...")
        
        # Stop video writer thread if it exists
        if hasattr(self, 'video_writer_thread'):
            logging.info("Stopping video writer thread...")
            self.video_writer_thread.stop()
            self.video_writer_thread.wait()  # Wait for thread to finish
            self.video_writer_thread = None

        # Release video writer
        if self.video_writer:
            logging.info("Releasing video writer...")
            try:
                self.video_writer.release()
            except Exception as e:
                logging.error(f"Error releasing video writer: {e}")
            self.video_writer = None

        # Close GPS log file
        if self.gps_log_file:
            logging.info("Closing GPS log file...")
            try:
                self.gps_log_file.close()
            except Exception as e:
                logging.error(f"Error closing GPS log file: {e}")
            self.gps_log_file = None

        self.recording_start_time = None
        self.last_gps_update = None
        logging.info("Recording cleanup completed")

    def update_recording(self, frame, lat, lon):
        """Update recording with current frame and GPS data"""
        if not self.is_recording or self.is_paused:
            return

        try:
            # Write video frame - ensure frame is in BGR format
            if self.video_writer and frame is not None:
                # Convert frame to BGR if it's in RGB
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    self.video_writer.write(frame)
                    logging.debug(f"Wrote frame to video - shape: {frame.shape}, type: {frame.dtype}")

            # Write GPS data
            if self.gps_log_file and lat is not None and lon is not None:
                timestamp = time.time() - self.recording_start_time
                self.gps_log_file.write(f"{timestamp:.3f},{lat},{lon}\n")
                self.gps_log_file.flush()  # Ensure data is written immediately

        except Exception as e:
            logging.error(f"Error updating recording: {str(e)}")
            self.stop_recording()

    def update_recording_gps(self, lat, lon):
        """Update GPS data during recording"""
        if not self.is_recording or not self.gps_log_file:
            return
            
        try:
            current_time = time.time()
            # Only update GPS every 0.5 seconds to avoid too frequent writes
            if self.last_gps_update and (current_time - self.last_gps_update) < 0.5:
                return
                
            # Calculate elapsed time since recording started
            elapsed = current_time - self.recording_start_time
            
            # Write GPS data to log file
            self.gps_log_file.write(f"{elapsed:.3f},{lat:.6f},{lon:.6f}\n")
            self.gps_log_file.flush()  # Ensure data is written immediately
            
            self.last_gps_update = current_time
            
        except Exception as e:
            logging.error(f"Error updating recording GPS data: {e}")

    def toggle_pause(self):
        """Toggle pause state of recording"""
        if self._check_detection_running():
            return
            
        if not self.is_recording:
            return

        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.setText("▶")  # Play symbol
            if self.dashboard.status_bar:
                self.dashboard.status_bar.status_message.setText("Recording paused")
        else:
            self.pause_btn.setText("⏸")  # Pause symbol
            if self.dashboard.status_bar:
                self.dashboard.status_bar.status_message.setText("Recording...")

    def toggle_cloud(self):
        if self._check_detection_running():
            return

        if not self.cloud_connected:
            # Show cloud connection dialog
            dlg = DirectorySelectionDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                cloud_dir, _ = dlg.get_directories()
                # Initialize cloud connector
                self.cloud_connector = CloudConnector(cloud_dir)
                self.cloud_connected = True
                self.cloud_btn.setText("Manage Cloud Storage")
                # Open cloud manager dialog
                self.open_cloud_manager()
        else:
            # Open cloud manager dialog
            self.open_cloud_manager()

    def open_cloud_manager(self):
        """Open the cloud manager dialog"""
        if self.cloud_manager is None:
            self.cloud_manager = CloudManagerDialog(self)
        self.cloud_manager.show()

    def run_analysis(self):
        """Open the analysis dialog"""
        if self._check_detection_running():
            return
            
        from .analysis_dialog import AnalysisDialog
        dialog = AnalysisDialog(self)
        dialog.exec_()

    def open_settings(self):
        """Open settings dialog"""
        if self._check_detection_running():
            return
        settings_dialog = SettingsDialog(self)
        if settings_dialog.exec_() == QDialog.Accepted:
            # Update detect button text based on record mode
            if self.settings_manager.get_setting('record_mode'):
                self.detect_btn.setText("Start Recording")
                self.detect_btn.clicked.disconnect()
                self.detect_btn.clicked.connect(self.toggle_recording)
                self.pause_btn.show()  # Show pause button when in recording mode
            else:
                self.detect_btn.setText("Start Detection")
                self.detect_btn.clicked.disconnect()
                self.detect_btn.clicked.connect(self.toggle_detection)
                self.pause_btn.hide()  # Hide pause button when not in recording mode

    def stop_recording(self):
        """Stop recording and clean up resources"""
        try:
            if self.is_recording:
                self.is_recording = False
                self.recording_start_time = None
                
                # Stop video writer thread
                if self.video_writer_thread:
                    self.video_writer_thread.stop()
                    self.video_writer_thread.join(timeout=1.0)
                    self.video_writer_thread = None
                
                # Close GPS log file
                if self.gps_log_file:
                    self.gps_log_file.close()
                    self.gps_log_file = None
                
                # Update UI
                self.detect_btn.setText("Start Recording")
                self.detect_btn.setStyleSheet("")
                
                # Update status
                if self.dashboard and self.dashboard.status_bar:
                    self.dashboard.status_bar.status_message.setText("Recording stopped")
                
                logging.info("Recording stopped and resources cleaned up")
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            # Ensure recording state is reset even if there's an error
            self.is_recording = False
            self.recording_start_time = None
            if self.detect_btn:
                self.detect_btn.setText("Start Recording")
                self.detect_btn.setStyleSheet("")
