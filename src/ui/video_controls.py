from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QSlider, QFileDialog, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt
import cv2
import logging
import os

class VideoControls(QGroupBox):
    def __init__(self, dashboard, parent=None):
        super().__init__("Video Control", parent)
        self.dashboard = dashboard  # Store the Dashboard instance
        self.main_controls = None  # Will be set by Dashboard
        
        # Initialize all UI components
        self.camera_combo = None
        self.flip_btn = None
        self.upload_btn = None
        self.zoom_slider = None
        self.zoom_label = None
        self.playback_controls = None
        self.playback_container = None
        self.prev_btn = None
        self.play_stop_btn = None
        self.forward_btn = None
        self.position_slider = None
        self.current_time_label = None
        self.total_time_label = None
        
        # Initialize state variables
        self.is_playing = False  # Track video playback state
        self.is_sliding = False  # Track if the slider is being moved
        self.video_duration = 0  # Track video duration
        self.video_fps = 0  # Track video frames per second
        
        # Initialize the UI
        self._init_ui()
        self._detect_available_cameras()

    def _init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()

        # Set stylesheet for disabled state
        self.setStyleSheet("""
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QPushButton:disabled:hover {
                background-color: #2a2a2a;
                color: #666;
            }
            QSlider:disabled {
                background-color: #2a2a2a;
            }
        """)

        # Camera selection
        self.camera_combo = QComboBox()
        layout.addWidget(QLabel("Source:"))
        layout.addWidget(self.camera_combo)

        # Flip button
        self.flip_btn = QPushButton("Flip Camera")
        self.flip_btn.clicked.connect(self._on_flip_clicked)
        layout.addWidget(self.flip_btn)

        # Upload button
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self._upload_file)
        layout.addWidget(self.upload_btn)

        # Zoom controls
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 500)
        self.zoom_slider.setValue(100)
        self.zoom_label = QLabel("Zoom: 1.00x")
        layout.addWidget(self.zoom_slider)
        layout.addWidget(self.zoom_label)

        # Playback controls container
        self.playback_container = QWidget()
        self.playback_container.setVisible(False)  # Initially hidden
        playback_layout = QHBoxLayout(self.playback_container)
        
        # Previous button
        self.prev_btn = QPushButton("⏪")
        self.prev_btn.clicked.connect(self._rewind)
        playback_layout.addWidget(self.prev_btn)
        
        # Play/Stop button
        self.play_stop_btn = QPushButton("⏸")
        self.play_stop_btn.clicked.connect(self._toggle_playback)
        playback_layout.addWidget(self.play_stop_btn)
        
        # Forward button
        self.forward_btn = QPushButton("⏩")
        self.forward_btn.clicked.connect(self._forward)
        playback_layout.addWidget(self.forward_btn)
        
        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.valueChanged.connect(self._on_slider_value_changed)
        playback_layout.addWidget(self.position_slider)
        
        # Time labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        playback_layout.addLayout(time_layout)
        
        layout.addWidget(self.playback_container)
        self.setLayout(layout)

    def _detect_available_cameras(self):
        """Detect and populate available cameras in the combo box"""
        if self.camera_combo is None:
            return
            
        self.camera_combo.clear()
        available_cameras = []
        
        # Check only first 2 camera indices
        for i in range(2):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        
        # Add video file option last
        available_cameras.append("Video File")
        
        if len(available_cameras) > 1:  # If we have cameras
            self.camera_combo.addItems(available_cameras)
            # Set default to first camera
            self.camera_combo.setCurrentIndex(0)
            # Initially disable upload button
            self.upload_btn.setEnabled(False)
        else:
            # No cameras available, only show video file option
            self.camera_combo.addItem("Video File")
            self.camera_combo.setCurrentIndex(0)
            # Enable upload button since video file is the only option
            self.upload_btn.setEnabled(True)
            # Show playback controls
            self.set_controls_state(True)
            # Notify dashboard to switch to video file mode
            if hasattr(self, 'dashboard') and self.dashboard is not None:
                self.dashboard.stop_camera_feed()
                self.dashboard.video_label.setText("Please upload a video")

    def _on_slider_pressed(self):
        """Handle slider press event"""
        self.is_sliding = True
        if self.dashboard and self.is_playing:
            self.dashboard.pause_video_playback()
            logging.debug("Paused video playback for seeking")

    def _on_slider_released(self):
        """Handle slider release event"""
        self.is_sliding = False
        if self.dashboard:
            # Convert slider value to seconds
            position = self.position_slider.value() / 1000.0
            self.dashboard.seek_video(position)
            logging.debug(f"Seeking video to position: {position:.3f}s")
            if self.is_playing:
                self.dashboard.start_video_playback()
                logging.debug("Resumed video playback after seeking")

    def _on_slider_value_changed(self, value):
        """Handle slider value change event"""
        if self.is_sliding:
            # Update time labels while sliding
            position = value / 1000.0
            self.update_time_labels(position)

    def _toggle_playback(self):
        """Toggle video playback"""
        if not self.dashboard:
            return

        if self.is_playing:
            # Pause playback
            self.dashboard.pause_video_playback()
            self.play_stop_btn.setText("▶")
            self.is_playing = False
        else:
            # Start playback
            self.dashboard.start_video_playback()
            self.play_stop_btn.setText("⏸")
            self.is_playing = True

    def _rewind(self):
        """Rewind video by 10 seconds"""
        if self.dashboard and self.position_slider:
            current_pos = self.position_slider.value() / 1000.0
            new_pos = max(0, current_pos - 1)
            self.dashboard.seek_video(new_pos)

    def _forward(self):
        """Forward video by 10 seconds"""
        if self.dashboard and self.position_slider:
            current_pos = self.position_slider.value() / 1000.0
            new_pos = min(self.video_duration, current_pos + 1)
            self.dashboard.seek_video(new_pos)

    def update_video_info(self, duration, fps):
        """Update video information"""
        if self.position_slider is None:
            return
            
        self.video_duration = duration
        self.video_fps = fps
        self.position_slider.setRange(0, int(duration * 1000))
        self.update_time_labels(0)
        self.setEnabled(True)
        # Ensure video starts in paused state
        self.is_playing = False
        self.play_stop_btn.setText("▶")
        self.play_stop_btn.setEnabled(True)
        
        # Log video info update
        logging.info(f"Updated video info: duration={duration:.3f}s, fps={fps:.2f}")

    def update_position(self, position):
        """Update the position slider and time labels"""
        if self.position_slider is None or self.is_sliding:
            return
            
        # Update slider position
        self.position_slider.setValue(int(position * 1000))
        
        # Update time labels
        self.update_time_labels(position)
        
        # Log position update for debugging
        logging.debug(f"Updated video position: {position:.3f}s")

    def update_time_labels(self, current_time):
        """Update the time labels with current and total time"""
        if self.current_time_label is None or self.total_time_label is None:
            return
            
        # Ensure current_time is not None and is a number
        if current_time is None:
            current_time = 0
            
        # Update current time
        current_minutes = int(current_time // 60)
        current_seconds = int(current_time % 60)
        self.current_time_label.setText(f"{current_minutes:02d}:{current_seconds:02d}")
        
        # Update total time
        if hasattr(self, 'video_duration') and self.video_duration is not None and self.video_duration > 0:
            total_minutes = int(self.video_duration // 60)
            total_seconds = int(self.video_duration % 60)
            self.total_time_label.setText(f"{total_minutes:02d}:{total_seconds:02d}")
        else:
            self.total_time_label.setText("00:00")

    def _upload_file(self):
        """Handle file upload"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mkv);;All Files (*.*)"
            )
            
            if file_path:
                # Pause video playback if it's currently playing
                if self.dashboard:
                    self.dashboard.pause_video_playback()
                    # Ensure video controls are in paused state
                    self.is_playing = False
                    self.play_stop_btn.setText("▶")
                
                # Get video directory and filename
                video_dir = os.path.dirname(file_path)
                video_filename = os.path.basename(file_path)
                
                # Extract timestamp from video filename
                # Assuming format: recording_YYYYMMDD_HHMMSS
                if video_filename.startswith('recording_'):
                    timestamp = video_filename.split('_')[1] + '_' + video_filename.split('_')[2].split('.')[0]
                    gps_log_filename = f"gps_log_{timestamp}.txt"
                    gps_log_path = os.path.join(video_dir, gps_log_filename)
                    
                    if os.path.exists(gps_log_path):
                        logging.info(f"Found GPS log file: {gps_log_path}")
                        # Set video mode in detector to use GPS log
                        if self.dashboard and self.dashboard.detector:
                            self.dashboard.detector.set_video_mode(file_path, gps_log_path)
                            logging.info(f"Set video mode with GPS log: {gps_log_path}")
                    else:
                        logging.warning(f"GPS log file not found: {gps_log_path}")
                
                # Switch to the new video file
                self.dashboard.switch_to_video_file(file_path)
                logging.info(f"Switched to video file: {file_path}")
                
        except Exception as e:
            logging.error(f"Error uploading file: {e}")
            QMessageBox.warning(self, "Error", f"Failed to upload video file: {str(e)}")

    def set_controls_state(self, is_video_file=False):
        """Set the state of the controls based on whether a video file is loaded"""
        self.flip_btn.setEnabled(not is_video_file)
        self.zoom_slider.setEnabled(not is_video_file)
        self.upload_btn.setEnabled(True)  # Always enable upload button
        
        # Show/hide playback controls
        self.playback_container.setVisible(is_video_file)
        
        # Reset playback state when switching sources
        if not is_video_file:
            self.is_playing = False
            self.play_stop_btn.setText("▶")
        else:
            # Enable all playback controls
            self.play_stop_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.forward_btn.setEnabled(True)
            self.position_slider.setEnabled(True)

    def _on_flip_clicked(self):
        """Handle flip camera button click"""
        if hasattr(self, 'dashboard') and self.dashboard is not None:
            self.dashboard.flip_camera()

    def update_zoom(self):
        """Update zoom level"""
        if hasattr(self, 'dashboard') and self.dashboard is not None:
            self.dashboard.update_zoom()  # Call the update_zoom method in the Dashboard
