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
            # Get cloud directory from settings or use default
            cloud_dir = self.settings_manager.get_setting('cloud_directory')
            if not cloud_dir:
                # Use default cloud directory in user's home directory
                cloud_dir = os.path.join(os.path.expanduser("~"), "RoadDefectCloud")
            
            # Initialize cloud connector
            self.cloud_connector = CloudConnector(cloud_dir)
            self.cloud_connected = True
            self.cloud_btn.setText("Manage Cloud Storage")
            
            # Update settings with the cloud directory
            self.settings_manager.set_setting('cloud_directory', cloud_dir)
            
        except Exception as e:
            print(f"Failed to connect to cloud storage automatically: {e}")
            self.cloud_connected = False
            self.cloud_btn.setText("Connect Cloud")

    def toggle_detection(self):
        if not self.is_detecting:
            # Check if video is selected and available
            if self.dashboard and hasattr(self.dashboard, 'video_controls'):
                video_controls = self.dashboard.video_controls
                if video_controls.camera_combo.currentText() == "Video File":
                    # Check if video is actually loaded
                    if not hasattr(self.dashboard, 'video_capture'):
                        QMessageBox.warning(self, "Warning", "Please upload a video before starting detection.")
                        return
                    
                    # Check if video capture exists and is opened
                    if (self.dashboard.video_capture is None or 
                        not self.dashboard.video_capture.isOpened() or 
                        not hasattr(self.dashboard, 'video_file') or 
                        self.dashboard.video_file is None):
                        QMessageBox.warning(self, "Warning", "Please upload a video before starting detection.")
                        return

            # Start detection
            if not self.detector:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'road_defect.pt')
                if not os.path.exists(model_path):
                    QMessageBox.warning(self, "Error", "Model file not found. Please ensure the model exists at: " + model_path)
                    return
                
                # Create detector with settings
                output_dir = self.settings_manager.get_setting('output_directory')
                self.detector = DefectDetector(model_path, save_dir=output_dir)
                # Apply confidence threshold
                self.detector.confidence_threshold = self.settings_manager.get_confidence_threshold()
            
            self.is_detecting = True
            self.detect_btn.setText("Stop Detection")

            # If video file is selected, ensure it's playing
            if self.dashboard and hasattr(self.dashboard, 'video_controls'):
                video_controls = self.dashboard.video_controls
                if video_controls.camera_combo.currentText() == "Video File":
                    # Force video to start playing
                    if not video_controls.is_playing:
                        # Update video controls state
                        video_controls.is_playing = True
                        video_controls.play_stop_btn.setText("⏸")  # Show pause icon
                        
                        # Get video FPS and start playback
                        if self.dashboard.video_capture is not None:
                            fps = self.dashboard.video_capture.get(cv2.CAP_PROP_FPS)
                            if fps <= 0:
                                fps = 30  # Default to 30 FPS if can't determine
                            
                            # Start the timer with correct FPS
                            if hasattr(self.dashboard, 'capture_timer'):
                                self.dashboard.capture_timer.start(int(1000 / fps))
                            
                            # Update status
                            if hasattr(self.dashboard, 'status_bar'):
                                self.dashboard.status_bar.status_message.setText(
                                    f"Playing: {os.path.basename(self.dashboard.video_file)}"
                                )
                            
                            # Force an immediate frame update
                            if hasattr(self.dashboard, 'update_frame'):
                                self.dashboard.update_frame()
        else:
            # Stop detection
            self.is_detecting = False
            self.detect_btn.setText("Start Detection")
            
            # Stop video playback if video file is selected
            if self.dashboard and hasattr(self.dashboard, 'video_controls'):
                video_controls = self.dashboard.video_controls
                if video_controls.camera_combo.currentText() == "Video File":
                    # Stop video playback
                    if video_controls.is_playing:
                        video_controls.is_playing = False
                        video_controls.play_stop_btn.setText("▶")  # Show play icon
                        
                        # Stop the timer
                        if hasattr(self.dashboard, 'capture_timer'):
                            self.dashboard.capture_timer.stop()
                        
                        # Update status
                        if hasattr(self.dashboard, 'status_bar'):
                            self.dashboard.status_bar.status_message.setText(
                                f"Paused: {os.path.basename(self.dashboard.video_file)}"
                            )
            
            if self.detector:
                self.detector.cleanup()
                self.detector = None

    def toggle_recording(self):
        """Toggle video recording and GPS logging"""
        if not self.is_recording:
            # Start recording
            if not self.dashboard or not hasattr(self.dashboard, 'video_controls'):
                return

            video_controls = self.dashboard.video_controls
            if video_controls.camera_combo.currentText() == "Video File":
                QMessageBox.warning(self, "Warning", "Recording is only available in camera mode.")
                return

            # Get recording output directory
            output_dir = self.settings_manager.get_setting('recording_output_dir')
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
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

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
                print(f"Error starting recording: {str(e)}")
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

    def toggle_pause(self):
        """Toggle pause state of recording"""
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

    def _cleanup_recording(self):
        """Clean up recording resources"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.gps_log_file:
            self.gps_log_file.close()
            self.gps_log_file = None

        self.recording_start_time = None

    def update_recording(self, frame, lat, lon):
        """Update recording with current frame and GPS data"""
        if not self.is_recording or self.is_paused:
            return

        try:
            # Write video frame
            if self.video_writer:
                self.video_writer.write(frame)

            # Write GPS data
            if self.gps_log_file and lat is not None and lon is not None:
                timestamp = time.time() - self.recording_start_time
                self.gps_log_file.write(f"{timestamp:.3f},{lat},{lon}\n")
                self.gps_log_file.flush()  # Ensure data is written immediately

        except Exception as e:
            print(f"Error updating recording: {str(e)}")
            self.stop_recording()

    def toggle_cloud(self):
        if self.is_detecting:
            QMessageBox.warning(self, "Warning", "Please stop detection before connecting to cloud.")
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
                # Update settings with the cloud directory
                self.settings_manager.set_setting('cloud_directory', cloud_dir)
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
        if self.is_detecting:
            QMessageBox.warning(self, "Warning", "Please stop detection before running analysis.")
            return
        dlg = DirectorySelectionDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            in_dir, out_dir = dlg.get_directories()
            # Proceed with analysis using in_dir and out_dir
            QMessageBox.information(self, "Analysis", f"Input Directory: {in_dir}\nOutput Directory: {out_dir}")

    def open_settings(self):
        """Open settings dialog"""
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
