from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QSlider, QFileDialog, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt
import cv2

class VideoControls(QGroupBox):
    def __init__(self, dashboard, parent=None):
        super().__init__("Video Control", parent)
        self.dashboard = dashboard  # Store the Dashboard instance
        self.main_controls = None  # Will be set by Dashboard
        self.camera_combo = None
        self.flip_btn = None
        self.upload_btn = None
        self.zoom_slider = None
        self.zoom_label = None
        self.playback_controls = None
        self.is_playing = False  # Track video playback state
        self._init_ui()
        self._detect_available_cameras()

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

    def _init_ui(self):
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
        
        layout.addWidget(self.playback_container)

        self.setLayout(layout)

        # Connect signals to methods in the Dashboard
        self.camera_combo.currentIndexChanged.connect(self.change_source)
        self.zoom_slider.valueChanged.connect(self.update_zoom)

    def set_controls_state(self, is_video_file=False):
        """Enable/disable controls based on source type"""
        self.flip_btn.setEnabled(not is_video_file)
        self.zoom_slider.setEnabled(not is_video_file)
        self.upload_btn.setEnabled(is_video_file)
        
        # Show/hide playback controls
        self.playback_container.setVisible(is_video_file)
        
        # Reset playback state when switching sources
        if not is_video_file:
            self.is_playing = False
            self.play_stop_btn.setText("⏸")

    def _toggle_playback(self):
        """Toggle video playback between play and pause"""
        if self.dashboard is None:
            print("Dashboard reference not set")
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_stop_btn.setText("⏸")  # Show pause icon
            self.dashboard.resume_video_playback()
        else:
            self.play_stop_btn.setText("▶")  # Show play icon
            self.dashboard.pause_video_playback()
            # Update status if detection is active
            if self.main_controls and self.main_controls.is_detecting:
                if self.dashboard.status_bar:
                    self.dashboard.status_bar.status_message.setText("Video paused, detection paused")

    def _rewind(self):
        """Rewind video by 5 seconds"""
        if hasattr(self, 'dashboard') and self.dashboard is not None:
            self.dashboard.seek_video(-1)

    def _forward(self):
        """Forward video by 5 seconds"""
        if hasattr(self, 'dashboard') and self.dashboard is not None:
            self.dashboard.seek_video(1)

    def _on_flip_clicked(self):
        """Handle flip button click"""
        if hasattr(self, 'dashboard') and self.dashboard is not None:
            self.dashboard.flip_camera()

    def change_source(self):
        print("Source selection changed")  # Debug print
        if self.dashboard is None:
            print("Dashboard reference not set")
            return

        # Check if detection is active
        if self.main_controls and self.main_controls.is_detecting:
            QMessageBox.warning(self, "Warning", "Please stop detection before changing the video source.")
            # Reset combo box to previous selection
            self.camera_combo.blockSignals(True)
            self.camera_combo.setCurrentText(self.camera_combo.currentText())
            self.camera_combo.blockSignals(False)
            return

        selected = self.camera_combo.currentText()
        if selected == "Video File":
            # Stop camera feed and show upload message
            if self.dashboard:
                self.dashboard.stop_camera_feed()
                self.dashboard.video_label.setText("Please upload a video")
            # Enable upload button only when Video File is selected
            self.upload_btn.setEnabled(True)
            # Show playback controls
            self.set_controls_state(True)
            
            # Disable record mode if it's enabled
            if self.main_controls and self.main_controls.settings_manager:
                if self.main_controls.settings_manager.get_setting('record_mode'):
                    self.main_controls.settings_manager.set_setting('record_mode', False)
                    # Update detect button text and connection
                    self.main_controls.detect_btn.setText("Start Detection")
                    self.main_controls.detect_btn.clicked.disconnect()
                    self.main_controls.detect_btn.clicked.connect(self.main_controls.toggle_detection)
                    self.main_controls.pause_btn.hide()  # Hide pause button
                    # Show notification
                    QMessageBox.information(self, "Record Mode Disabled", 
                                          "Record mode has been automatically disabled as it's not available in video file mode.")
        else:
            # Disable upload button for camera sources
            self.upload_btn.setEnabled(False)
            # Hide playback controls
            self.set_controls_state(False)
            self.dashboard.change_camera()  # Call the change_camera method in the Dashboard

    def update_zoom(self):
        print("Zoom slider changed")  # Debug print
        if hasattr(self, 'dashboard') and self.dashboard is not None:
            self.dashboard.update_zoom()  # Call the update_zoom method in the Dashboard

    def _upload_file(self):
        if not self.upload_btn.isEnabled():
            return

        # Check if detection is active
        if self.main_controls and self.main_controls.is_detecting:
            QMessageBox.warning(self, "Warning", "Please stop detection before uploading a new video.")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            options=options
        )
        if file_name and self.dashboard:
            # Notify the dashboard to switch to video file mode
            self.dashboard.switch_to_video_file(file_name)
            self.set_controls_state(True)  # Show playback controls
            self.is_playing = False  # Start as stopped
            self.play_stop_btn.setText("▶")  # Show play icon
            # Ensure video is paused and timer is stopped
            if self.dashboard.video_capture is not None:
                if self.dashboard.capture_timer is not None:
                    self.dashboard.capture_timer.stop()
                self.dashboard.pause_video_playback()
