import sys
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QComboBox, QPushButton, QSlider, QGroupBox,
    QStatusBar, QSplitter, QFrame, QMessageBox, QDialog,
    QFileDialog, QFormLayout, QDoubleSpinBox, QLineEdit, QInputDialog,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QSize, QRect, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QImage, QPixmap
from src.app.modules.camera import Camera
from src.app.modules.gps_reader import GPSReader
from src.app.modules.detection import DefectDetector
from src.app.modules.cloud_connector import CloudStorage
from src.app.modules.severity_calculator import SeverityCalculator
import time
import os
import threading
import logging
import queue
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    detection_ready = pyqtSignal(np.ndarray, dict, float)
    flip_state_changed = pyqtSignal(bool)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = False
        self._flip_vertical = False
        self._last_flip_time = 0
        self._last_frame_time = time.time()
        self._frame_count = 0
        logging.info("CameraThread initialized")

    def run(self):
        self.running = True
        logging.info("CameraThread started running")
        
        while self.running:
            try:
                # Get frame with detection from camera
                frame, counts = self.camera.get_frame()
                if frame is None:
                    logging.error("Failed to capture frame")
                    self.error_occurred.emit("Failed to capture frame")
                    break

                # Apply vertical flip if enabled
                if self._flip_vertical:
                    frame = cv2.flip(frame, 0)
                    logging.debug("Frame flipped vertically")

                # Update FPS calculation
                self._frame_count += 1
                current_time = time.time()
                if current_time - self._last_frame_time >= 1.0:
                    fps = self._frame_count / (current_time - self._last_frame_time)
                    logging.debug(f"Current FPS: {fps:.1f}")
                    self._last_frame_time = current_time
                    self._frame_count = 0

                # Emit frame with detection results
                if self.camera.detecting:
                    self.detection_ready.emit(frame, counts, fps)
                else:
                    self.frame_ready.emit(frame)

            except Exception as e:
                logging.error(f"Camera thread error: {str(e)}")
                self.error_occurred.emit(str(e))
                break

            time.sleep(1/30)  # Limit to ~30 FPS

    def toggle_flip(self):
        """Toggle both vertical and horizontal flip states with debounce."""
        current_time = time.time()
        if current_time - self._last_flip_time > 0.5:  # 500ms debounce
            self._last_flip_time = current_time
            self._flip_vertical = not self._flip_vertical
            self._flip_horizontal = not self._flip_horizontal
            self.flip_state_changed.emit((self._flip_vertical, self._flip_horizontal))
            logging.info(f"Flip toggled to: vertical={'on' if self._flip_vertical else 'off'}, "
                         f"horizontal={'on' if self._flip_horizontal else 'off'}")

    def stop(self):
        self.running = False
        self.wait()

class GPSThread(QThread):
    gps_update = pyqtSignal(float, float)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.gps = GPSReader()
        self.running = False
        self.last_update = 0
        self.update_interval = 1.0  # Update every second
        self.connection_attempts = 0
        self.max_connection_attempts = 3

    def run(self):
        """Start GPS reading"""
        self.running = True
        self.status_update.emit("Searching for GPS...")
        
        while self.running:
            try:
                if not self.gps.is_connected():
                    if self.connection_attempts < self.max_connection_attempts:
                        # Try to reconnect
                        if self.gps.port and self.gps._connect():
                            self.status_update.emit(f"Connected to GPS on {self.gps.port}")
                            self.connection_attempts = 0
                        else:
                            self.connection_attempts += 1
                            self.status_update.emit(f"No GPS device found (Attempt {self.connection_attempts}/{self.max_connection_attempts})")
                            time.sleep(2)  # Wait before retrying
                            continue
                    else:
                        self.status_update.emit("GPS connection failed after multiple attempts")
                        time.sleep(5)  # Longer wait after max attempts
                        self.connection_attempts = 0
                        continue
                
                # Read GPS data
                lat, lon = self.gps.get_gps_data()
                if lat is not None and lon is not None:
                    self.gps_update.emit(lat, lon)
                    self.status_update.emit("GPS Fix")
                    self.connection_attempts = 0  # Reset attempts on successful read
                else:
                    self.status_update.emit("No Fix")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.status_update.emit(f"Error: {str(e)}")
                time.sleep(2)  # Wait before retrying

    def stop(self):
        """Stop GPS reading"""
        self.running = False
        if self.gps:
            self.gps.cleanup()  # Clean up the GPS connection
        self.wait()

    def is_connected(self):
        """Check if GPS is connected"""
        return self.gps.is_connected()

    def get_available_ports(self):
        """Get list of available GPS ports"""
        return self.gps.get_available_ports()

    def connect_manually(self, port):
        """Manually connect to a specific GPS port"""
        if self.gps.connect_manually(port):
            self.connection_attempts = 0  # Reset attempts on manual connection
            self.status_update.emit(f"Connected to GPS on {port}")
            return True
        else:
            self.status_update.emit("Failed to connect to GPS")
            return False

class DonutWidget(QFrame):
    def __init__(self, title, count, parent=None):
        super().__init__(parent)
        self.title = title
        self.count = count
        self.setMinimumSize(150, 150)  # Set minimum size for better visibility

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set the background color
        painter.setBrush(self.palette().window())
        painter.drawRect(self.rect())

        width = self.width()
        height = self.height()

        padding_bottom = 20
        available_height = height - padding_bottom
        outer_radius = min(width, available_height) * 0.4
        hole_radius = outer_radius * 0.5

        # Center the donut
        rect = QRect(
           int ((width - 2 * outer_radius) // 2),
           int( (available_height - 2 * outer_radius) // 2),
           int ( 2 * outer_radius),
            int (2 * outer_radius)
        )

        # Draw the donut fill with a more visible color
        painter.setBrush(QColor(70, 130, 180))  # Steel blue
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect)

        # Draw the hole
        hole_rect = QRect(
            int(self.rect().center().x() - hole_radius),
            int (rect.top() + outer_radius - hole_radius),
           int( 2 * hole_radius),
           int ( 2 * hole_radius)
        )
        painter.setBrush(self.palette().window())
        painter.drawEllipse(hole_rect)

        # Draw count with larger font
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont("Arial", 14, QFont.Bold)  # Increased font size
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, str(self.count))

        # Draw title with larger font
        painter.setFont(QFont("Arial", 12))  # Increased font size
        title_rect = QRect(0, height - padding_bottom, width, padding_bottom)
        painter.drawText(title_rect, Qt.AlignCenter | Qt.AlignVCenter, self.title)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        # Get current settings from parent
        self.parent = parent
        self.camera = parent.camera if parent and parent.camera else None
        
        # Initialize UI
        self.setup_ui()
        
        # Load current settings
        self.load_current_settings()
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #f0f0f0;
            }
            QLabel {
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: #ddd;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QSlider {
                background: transparent;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4a4a4a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QLineEdit {
                background-color: #3b3b3b;
                color: #f0f0f0;
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QDoubleSpinBox {
                background-color: #3b3b3b;
                color: #f0f0f0;
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create form layout for settings
        form_layout = QFormLayout()
        
        # Output directory selection
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_output_dir)
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.browse_button)
        form_layout.addRow("Output Directory:", self.output_dir_layout)
        
        # Confidence threshold slider
        self.confidence_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(25)  # Default 0.25
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        self.confidence_label = QLabel("0.25")
        self.confidence_layout.addWidget(self.confidence_slider)
        self.confidence_layout.addWidget(self.confidence_label)
        form_layout.addRow("Confidence Threshold:", self.confidence_layout)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def load_current_settings(self):
        """Load current settings from camera"""
        if self.camera:
            settings = self.camera.get_current_settings()
            if settings:
                # Load confidence threshold
                confidence = int(settings['confidence_threshold'] * 100)
                self.confidence_slider.setValue(confidence)
                self.update_confidence_label(confidence)
                
                # Load output directory
                if settings['save_dir']:
                    self.output_dir_edit.setText(settings['save_dir'])

    def update_confidence_label(self, value):
        """Update confidence label when slider changes"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")

    def browse_output_dir(self):
        """Open directory browser for output path"""
        current_dir = self.output_dir_edit.text() or os.path.expanduser("~")
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            current_dir,
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def save_settings(self):
        """Save settings and apply them"""
        try:
            if self.camera:
                # Get new settings
                confidence = self.confidence_slider.value() / 100.0
                new_dir = self.output_dir_edit.text()
                
                # Update settings through camera
                if self.camera.update_settings(confidence, new_dir):
                    logging.info("Settings updated successfully")
                    self.accept()
                else:
                    QMessageBox.warning(self, "Error", "Failed to update settings")
            else:
                QMessageBox.warning(self, "Error", "Camera not initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

class DirectorySelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Directories")
        self.setMinimumWidth(600)
        self.setMinimumHeight(200)
        
        # Initialize directories
        self.input_dir = ""
        self.output_dir = ""
        
        # Setup UI
        self.setup_ui()
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #f0f0f0;
            }
            QLabel {
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: #ddd;
                padding: 8px;
                border: none;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton#proceedButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                min-width: 120px;
            }
            QPushButton#proceedButton:hover {
                background-color: #45a049;
            }
            QLineEdit {
                background-color: #3b3b3b;
                color: #f0f0f0;
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QGroupBox {
                border: 1px solid #555;
                margin-top: 10px;
                color: #f0f0f0;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Input directory selection
        input_group = QGroupBox("Input Directory")
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Select input directory containing images...")
        input_browse = QPushButton("Browse...")
        input_browse.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(input_browse)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Output directory selection
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Select output directory for processed images...")
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.setObjectName("proceedButton")
        self.proceed_button.clicked.connect(self.accept)
        self.proceed_button.setEnabled(False)  # Initially disabled
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.proceed_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals to validate directory selection
        self.input_edit.textChanged.connect(self.validate_directories)
        self.output_edit.textChanged.connect(self.validate_directories)

    def validate_directories(self):
        """Validate that both directories are selected and different"""
        input_dir = self.input_edit.text()
        output_dir = self.output_edit.text()
        
        # Basic validation
        if not input_dir or not output_dir:
            self.proceed_button.setEnabled(False)
            return False
            
        # Check if directories are the same
        if input_dir == output_dir:
            self.proceed_button.setEnabled(False)
            return False
            
        # Check if input directory exists
        if not os.path.exists(input_dir):
            self.proceed_button.setEnabled(False)
            return False
            
        # Check if output directory exists or can be created
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                self.proceed_button.setEnabled(False)
                return False
        
        # If we get here, the directories are valid
        self.proceed_button.setEnabled(True)
        return True

    def accept(self):
        """Override accept to ensure directories are valid"""
        if not self.validate_directories():
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Please ensure:\n"
                "1. Both directories are selected\n"
                "2. Input and output directories are different\n"
                "3. Input directory exists\n"
                "4. Output directory can be created"
            )
            return
        super().accept()

    def browse_input(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            self.input_edit.text() or os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self.input_dir = dir_path
            self.input_edit.setText(dir_path)
            self.validate_directories()

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_edit.text() or os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_edit.setText(dir_path)
            self.validate_directories()

    def get_directories(self):
        """Return the selected directories"""
        return self.input_dir, self.output_dir

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Defect Detection Dashboard")
        self.setMinimumSize(1000, 700)

        # Initialize state variables
        self.detecting = False
        logging.info("Initializing Dashboard")

        # Cloud storage state
        self.cloud_storage = None
        self.cloud_connected = False

        # Initialize all UI controls first
        self.initialize_controls()
        self.initialize_buttons()
        self.initialize_statistics()

        # Initialize GPS first since detector will use it
        logging.info("Initializing GPS thread...")
        self.gps_thread = GPSThread()
        self.gps_thread.gps_update.connect(self.update_gps)
        self.gps_thread.status_update.connect(self.update_gps_status)
        logging.info("GPS thread initialized")

        # Construct the model path dynamically
        if getattr(sys, 'frozen', False):
            # If the application is frozen (running as an executable)
            model_path = os.path.join(sys._MEIPASS, 'models', 'road_defect_v2.pt')
        else:
            # If running in a normal Python environment
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'road_defect_v2.pt')

        logging.info(f"Model path: {model_path}")

        # Initialize camera with model and GPS reader
        try:
            logging.info("Initializing camera with GPS reader...")
            self.camera = Camera(model_path=model_path, gps_reader=self.gps_thread.gps)
            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.detection_ready.connect(self.update_detection)
            self.camera_thread.error_occurred.connect(self.handle_camera_error)
            self.camera_thread.flip_state_changed.connect(self.update_flip_button)
            logging.info("Camera and camera thread initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize camera: {str(e)}")
            QMessageBox.critical(self, "Camera Error", f"Failed to initialize camera: {str(e)}")
            self.camera = None
            self.camera_thread = None

        # Setup UI and connections
        self.setup_ui()
        self.setup_connections()

        # Start CPU/GPU update timer
        self.cpu_gpu_timer = QTimer(self)
        self.cpu_gpu_timer.timeout.connect(self.update_cpu_gpu)
        self.cpu_gpu_timer.start(1000)

        # Start GPS thread
        logging.info("Starting GPS thread...")
        self.gps_thread.start()  # Ensure GPS thread starts
        logging.info("GPS thread started")

        # Start camera if available
        if self.camera_thread:
            self.start_camera()

    def initialize_controls(self):
        """Initialize all control widgets"""
        # Camera selection combo box
        self.camera_combo = QComboBox()
        
        # Flip button - starts in normal state
        self.flip_button = QPushButton("Flip Camera")
        self.flip_button.clicked.connect(self._handle_flip_click)
        
        # Zoom controls
        self.zoom_label = QLabel("Zoom: 1.00x")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(100)  # Start at 1.0x zoom
        self.zoom_slider.setMaximum(500)  # 5.0x zoom max
        self.zoom_slider.setValue(100)  # default 1.0x
        self.zoom_slider.setTracking(False)  # Only update when slider is released
        
        # Brightness controls
        self.brightness_label = QLabel("Brightness: 50")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(50)
        self.brightness_slider.setTracking(False)  # Only update when slider is released
        self.brightness_slider.setEnabled(False)  # Initially disabled until we check camera support
        
        # Exposure controls
        self.exposure_label = QLabel("Exposure: 50")
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(0)
        self.exposure_slider.setMaximum(100)
        self.exposure_slider.setValue(50)
        self.exposure_slider.setTracking(False)  # Only update when slider is released
        self.exposure_slider.setEnabled(False)  # Initially disabled until we check camera support

    def _handle_flip_click(self):
        """Handle flip button click with debounce"""
        if not hasattr(self, '_last_flip_time'):
            self._last_flip_time = 0
        
        current_time = time.time()
        if current_time - self._last_flip_time < 0.5:  # 500ms debounce
            return
            
        self._last_flip_time = current_time
        
        # Toggle flip state
        is_flipped = self.camera.toggle_flip()
        
        # Update button appearance based on flip state
        if is_flipped:
            self.flip_button.setText("Normal View")
            self.flip_button.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
        else:
            self.flip_button.setText("Flip Camera")
            self.flip_button.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)

    def initialize_statistics(self):
        """Initialize statistics widgets"""
        # Initialize donut widgets for statistics
        self.linear_donut = DonutWidget("Linear Cracks", 0)
        self.alligator_donut = DonutWidget("Alligator Cracks", 0)
        self.potholes_donut = DonutWidget("Potholes", 0)

        # Initialize GPS labels
        self.long_label = QLabel("Longitude: 0.0")
        self.lat_label = QLabel("Latitude: 0.0")

    def initialize_buttons(self):
        """Initialize all buttons with their styles"""
        # Detection button
        self.detect_button = QPushButton("Start Detection")
        self.detect_button.setObjectName("detect_button")
        self.detect_button.setStyleSheet("""
            #detect_button {
                background-color: #77dd77;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #5cb85c;
            }
            #detect_button:hover {
                background-color: #5cb85c;
                border-color: #4cae4c;
            }
            #detect_button:pressed {
                background-color: #4cae4c;
                border-color: #3d8b3d;
            }
        """)
        self.detect_button.clicked.connect(self.toggle_detection)

        # GPS button
        self.connect_gps_button = QPushButton("Connect GPS")
        self.connect_gps_button.setObjectName("gps_button")
        self.connect_gps_button.setStyleSheet("""
            #gps_button {
                background-color: #4a4a4a;
                color: #ddd;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #3a3a3a;
            }
            #gps_button:hover {
                background-color: #5a5a5a;
                border-color: #4a4a4a;
            }
            #gps_button:pressed {
                background-color: #3a3a3a;
                border-color: #2a2a2a;
            }
        """)
        self.connect_gps_button.clicked.connect(self.toggle_gps)

        # Analysis button
        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setObjectName("analysis_button")
        self.run_analysis_button.setStyleSheet("""
            #analysis_button {
                background-color: #4a90e2;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #357abd;
            }
            #analysis_button:hover {
                background-color: #357abd;
                border-color: #2a5f94;
            }
            #analysis_button:pressed {
                background-color: #2a5f94;
                border-color: #1f4b7a;
            }
        """)
        self.run_analysis_button.clicked.connect(self.run_analysis)

        # Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.setObjectName("settings_button")
        self.settings_button.setStyleSheet("""
            #settings_button {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #5a6268;
            }
            #settings_button:hover {
                background-color: #5a6268;
                border-color: #4a4f54;
            }
            #settings_button:pressed {
                background-color: #4a4f54;
                border-color: #3a3f44;
            }
        """)
        self.settings_button.clicked.connect(self.show_settings)

        # Upload Data button
        self.upload_button = QPushButton("Upload Data")
        self.upload_button.setObjectName("upload_button")
        self.upload_button.setStyleSheet("""
            #upload_button {
                background-color: #f4b942;
                color: #222;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #c4902f;
            }
            #upload_button:hover {
                background-color: #e1a53a;
                border-color: #b07d22;
            }
            #upload_button:pressed {
                background-color: #b07d22;
                border-color: #8a5f18;
            }
        """)
        self.upload_button.clicked.connect(self.handle_upload_data)

        # Connect Cloud button
        self.connect_cloud_button = QPushButton("Connect Cloud")
        self.connect_cloud_button.setObjectName("connect_cloud_button")
        self.connect_cloud_button.setStyleSheet("""
            #connect_cloud_button {
                background-color: #7ec0ee;
                color: #222;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #4a90e2;
            }
            #connect_cloud_button:hover {
                background-color: #5dade2;
                border-color: #357abd;
            }
            #connect_cloud_button:pressed {
                background-color: #357abd;
                border-color: #2a5f94;
            }
        """)
        self.connect_cloud_button.clicked.connect(self.handle_connect_cloud)

    def handle_upload_data(self):
        """Process images and upload data to cloud."""
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            # Get input directory
            input_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Input Directory",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )
            if not input_dir:
                logger.info("No input directory selected")
                return

            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory",
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )
            if not output_dir:
                logger.info("No output directory selected")
                return

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Input directory: {input_dir}")
            logger.info(f"Output directory: {output_dir}")

            # Get list of image files
            image_files = [f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            logger.info(f"Found {len(image_files)} image files")
            if not image_files:
                QMessageBox.warning(self, "No Images", "No image files found in the selected directory.")
                return

            # Create progress dialog
            progress = QProgressDialog("Processing images...", "Cancel", 0, len(image_files), self)
            progress.setWindowTitle("Processing Images")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)  # Show immediately
            progress.setAutoClose(True)
            progress.setAutoReset(True)

            # Initialize timing variables
            start_time = time.time()
            processed_count = 0
            total_images = len(image_files)

            # Model path
            if getattr(sys, 'frozen', False):
                model_path = os.path.join(sys._MEIPASS, 'models', 'road_defect.pt')
            else:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'road_defect.pt')

            logger.info(f"Looking for model at: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found at {model_path}")

            # Initialize detector if not already initialized
            if not hasattr(self, 'detector') or self.detector is None:
                logger.info("Initializing detector...")
                try:
                    from src.app.modules.detection import DefectDetector
                    from src.app.modules.severity_calculator import SeverityCalculator
                    
                    # Initialize both detector and severity calculator
                    self.detector = DefectDetector(model_path=model_path)
                    self.severity_calculator = SeverityCalculator(
                        camera_width=1280,  # Default camera resolution
                        camera_height=720,
                        model_path=model_path
                    )
                    
                    if not self.detector or not self.severity_calculator:
                        raise RuntimeError("Failed to initialize detector or severity calculator")
                    logger.info("Detector and severity calculator initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize detector: {str(e)}")
                    raise

            # Initialize camera calibration parameters
            camera_matrix = np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ])
            distortion_coeffs = np.zeros(5)

            # Process each image
            processed_files = []  # Track successfully processed files
            for idx, filename in enumerate(image_files):
                if progress.wasCanceled():
                    logger.info("Processing cancelled by user")
                    break

                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                logger.info(f"Processing image {idx + 1}/{total_images}: {filename}")

                try:
                    # Update progress dialog
                    elapsed_time = time.time() - start_time
                    if processed_count > 0:
                        avg_speed = elapsed_time / processed_count
                        eta = avg_speed * (total_images - processed_count)
                        progress.setLabelText(
                            f"Processing: {filename}\n"
                            f"Speed: {avg_speed:.2f}s per image\n"
                            f"ETA: {eta:.1f}s remaining"
                        )
                    else:
                        progress.setLabelText(f"Processing: {filename}")

                    # Process image with severity calculator
                    final_img, defect_pixels, defect_ratio, metadata = self.severity_calculator.process_image(
                        input_path,
                        camera_matrix,
                        distortion_coeffs,
                        output_path,
                        distance_to_object_m=1.0,  # Default distance
                        confidence_threshold=0.15
                    )

                    if final_img is not None and len(defect_pixels) > 0:
                        # Save processed image
                        save_success = cv2.imwrite(output_path, final_img)
                        if save_success:
                            processed_files.append(output_path)
                            processed_count += 1
                            logger.info(f"Successfully processed and saved: {filename}")

                            # Update statistics from metadata
                            if metadata:
                                self.update_defect_stats(
                                    metadata['DefectCounts'].get('Linear-Crack', 0),
                                    metadata['DefectCounts'].get('Alligator-Crack', 0),
                                    metadata['DefectCounts'].get('Pothole', 0)
                                )
                                
                                # Save metadata as JSON
                                json_path = os.path.splitext(output_path)[0] + '_metadata.json'
                                with open(json_path, 'w') as f:
                                    json.dump(metadata, f, indent=4)
                                logger.info(f"Saved metadata to: {json_path}")
                        else:
                            logger.error(f"Failed to save processed image: {output_path}")
                    else:
                        logger.info(f"No defects detected in {filename}, skipping...")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)

                # Update progress bar
                progress.setValue(idx + 1)
                QApplication.processEvents()

            # Calculate final statistics
            total_time = time.time() - start_time
            avg_speed = total_time / processed_count if processed_count > 0 else 0

            logger.info(f"Processing complete. Processed {processed_count} out of {total_images} images")
            logger.info(f"Total time: {total_time:.1f}s, Average speed: {avg_speed:.2f}s per image")

            # Show completion message
            progress.close()
            if processed_count > 0:
                QMessageBox.information(
                    self,
                    "Processing Complete",
                    f"Successfully processed {processed_count} out of {total_images} images.\n\n"
                    f"Total processing time: {total_time:.1f} seconds\n"
                    f"Average speed: {avg_speed:.2f} seconds per image\n"
                    f"Output directory: {output_dir}"
                )
            else:
                error_msg = (
                    "No images were processed successfully.\n\n"
                    "Please check the following:\n"
                    "1. The input directory contains valid image files (.jpg, .jpeg, .png)\n"
                    "2. The images can be read by OpenCV\n"
                    "3. The detector is properly initialized\n"
                    "4. Check the application logs for detailed error messages\n\n"
                    f"Input directory: {input_dir}\n"
                    f"Output directory: {output_dir}\n"
                    f"Total images found: {total_images}"
                )
                logger.error(error_msg)
                QMessageBox.warning(self, "Processing Complete", error_msg)

        except Exception as e:
            logger.error(f"Script failed: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Processing Error",
                f"An error occurred during processing: {str(e)}\n\n"
                "Please check the application logs for more details."
            )

    def handle_connect_cloud(self):
        """Initialize cloud connection and handle upload of all files in directory"""
        try:
            # If already connected, handle upload
            if self.cloud_connected and self.cloud_storage:
                # Get directory to upload from
                upload_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Directory to Upload",
                    os.path.expanduser("~"),
                    QFileDialog.ShowDirsOnly
                )
                
                if not upload_dir:
                    logging.info("No directory selected for upload")
                    return

                # Get list of all files in directory
                all_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
                logging.info(f"Found {len(all_files)} files in directory: {upload_dir}")
                
                if not all_files:
                    QMessageBox.warning(
                        self,
                        "Upload Error",
                        f"No files found in the selected directory:\n{upload_dir}"
                    )
                    return

                # Create progress dialog for upload
                progress = QProgressDialog("Uploading files to cloud...", "Cancel", 0, len(all_files), self)
                progress.setWindowTitle("Cloud Upload")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setAutoClose(True)
                progress.setAutoReset(True)

                # Upload each file
                success_count = 0
                for idx, filename in enumerate(all_files):
                    if progress.wasCanceled():
                        break

                    file_path = os.path.join(upload_dir, filename)
                    progress.setLabelText(f"Uploading: {filename}\nFrom: {upload_dir}")

                    try:
                        # Read file content
                        with open(file_path, 'rb') as f:
                            file_content = f.read()

                        # Determine content type
                        if filename.lower().endswith(('.jpg', '.jpeg')):
                            content_type = 'image/jpeg'
                        elif filename.lower().endswith('.png'):
                            content_type = 'image/png'
                        elif filename.lower().endswith('.json'):
                            content_type = 'application/json'
                        else:
                            content_type = 'application/octet-stream'

                        # Create blob name with folder path
                        folder_path = self.cloud_storage.settings['folder_path'].rstrip('/')
                        blob_name = f"{folder_path}/{filename}"

                        # Upload file
                        blob = self.cloud_storage.bucket.blob(blob_name)
                        blob.upload_from_string(
                            file_content,
                            content_type=content_type
                        )
                        
                        success_count += 1
                        logging.info(f"Successfully uploaded: {filename}")

                    except Exception as e:
                        logging.error(f"Error uploading {filename}: {str(e)}", exc_info=True)

                    # Update progress
                    progress.setValue(idx + 1)
                    QApplication.processEvents()

                # Show upload completion message
                if success_count > 0:
                    QMessageBox.information(
                        self,
                        "Upload Complete",
                        f"Successfully uploaded {success_count} out of {len(all_files)} files to cloud.\n\n"
                        f"Uploaded from: {upload_dir}"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Upload Complete",
                        f"No files were uploaded successfully from:\n{upload_dir}\n\n"
                        "Please check the application logs for more details."
                    )

            else:
                # Initialize cloud connection
                self.cloud_storage = CloudStorage()
                if self.cloud_storage.is_initialized:
                    self.cloud_connected = True
                    self.connect_cloud_button.setText("Upload to Cloud")
                    self.connect_cloud_button.setStyleSheet("""
                        #connect_cloud_button {
                            background-color: #4CAF50;
                            color: white;
                            font-weight: bold;
                            padding: 12px;
                            border-radius: 6px;
                            border: 2px solid #45a049;
                        }
                        #connect_cloud_button:hover {
                            background-color: #45a049;
                            border-color: #3d8b40;
                        }
                        #connect_cloud_button:pressed {
                            background-color: #3d8b40;
                            border-color: #2e6b31;
                        }
                    """)
                    QMessageBox.information(self, "Cloud", "Connected to cloud successfully.")
                else:
                    self.cloud_connected = False
                    QMessageBox.warning(self, "Cloud", "Failed to connect to cloud. Check credentials and settings.")

        except Exception as e:
            self.cloud_connected = False
            logging.error(f"Cloud operation failed: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Cloud Error", f"Error during cloud operation: {str(e)}")

    def handle_send_data(self):
        """Send current frame and dummy data to cloud"""
        if not self.cloud_connected or not self.cloud_storage:
            QMessageBox.warning(self, "Cloud", "Cloud not connected. Please connect first.")
            return
        try:
            # Try to get current frame from camera
            frame = None
            defect_counts = {}
            frame_counts = {}
            if self.camera and hasattr(self.camera, "capture"):
                ret, frame = self.camera.capture.read()
                if not ret or frame is None:
                    QMessageBox.warning(self, "Send Data", "No frame available to send.")
                    return
                # Convert frame to JPEG bytes
                _, jpeg_bytes = cv2.imencode('.jpg', frame)
                image_bytes = jpeg_bytes.tobytes()
            else:
                QMessageBox.warning(self, "Send Data", "Camera not initialized.")
                return

            # Dummy defect/frame counts (replace with real data if available)
            defect_counts = {"Linear-Crack": 1, "Alligator-Crack": 2, "Pothole": 0}
            frame_counts = {"total": 1}

            # Upload to cloud
            success = self.cloud_storage.upload_detection(
                image=jpeg_bytes,  # Pass numpy array as in CloudStorage
                defect_counts=defect_counts,
                frame_counts=frame_counts
            )
            if success:
                QMessageBox.information(self, "Send Data", "Data sent to cloud successfully.")
            else:
                QMessageBox.warning(self, "Send Data", "Failed to send data to cloud.")
        except Exception as e:
            QMessageBox.critical(self, "Send Data", f"Error sending data: {str(e)}")

    def setup_ui(self):
        """Setup the main UI layout"""
        # Dark theme styling
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #f0f0f0;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #555;
                margin-top: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: none;
                padding: 8px;
                color: #ddd;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #6a6a6a;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: #ddd;
                padding: 4px;
            }
            QSlider {
                background: transparent;
            }
            QLabel {
                font-size: 14px;
            }
            #detect_button {
                font-weight: bold;
                color: white;
            }
        """)

        # Central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Video | Map layout
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Video view
        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; font-size: 20px;")
        self.splitter.addWidget(self.video_label)

        # Commenting out the map view initialization
        # self.map_view = MapView()
        # self.splitter.addWidget(self.map_view)  # Comment this line to avoid the error

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.splitter, stretch=8)
        layout.addLayout(self.create_controls_section(), stretch=2)
        main_widget.setLayout(layout)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("font-size: 16px; background-color: #1a1a1a; color: #ffffff;")
        self.status_labels = {
            "time": QLabel("00:00:00"),
            "cpu": QLabel("CPU: 0%"),
            "gpu": QLabel("GPU: 0%"),
            "fps": QLabel("FPS: 0"),
            "gps": QLabel("GPS: Disconnected")
        }
        for label in self.status_labels.values():
            label.setStyleSheet("margin: 0 10px;")
            self.status.addWidget(label)

        self.update_detect_button_style()

    def setup_connections(self):
        """Set up signal connections"""
        # Connect flip button
        self.flip_button.clicked.connect(self._handle_flip_click)
        
        # Set initial flip button state
        self.flip_button.setText("Flip Camera")
        self.flip_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)

        # Camera connections
        if self.camera:
            available_cameras = self.camera.get_available_cameras()
            self.camera_combo.clear()
            self.camera_combo.addItems([f"Camera {i}" for i in available_cameras])
            self.camera_combo.currentIndexChanged.connect(self.change_camera)

            # Check camera capabilities and update UI accordingly
            self._update_camera_controls()

        # Control connections
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.exposure_slider.valueChanged.connect(self.update_exposure)

    def _update_camera_controls(self):
        """Update camera control UI based on camera capabilities"""
        if not self.camera:
            return

        # Check brightness support
        if hasattr(self.camera, 'supports_brightness') and self.camera.supports_brightness:
            self.brightness_slider.setEnabled(True)
            self.brightness_slider.setValue(self.camera.brightness)
            self.brightness_label.setText(f"Brightness: {self.camera.brightness}")
            self.brightness_label.setStyleSheet("color: #f0f0f0;")
        else:
            self.brightness_slider.setEnabled(False)
            self.brightness_label.setText("Brightness: Not Supported")
            self.brightness_label.setStyleSheet("color: #666666;")

        # Check exposure support
        if hasattr(self.camera, 'supports_exposure') and self.camera.supports_exposure:
            self.exposure_slider.setEnabled(True)
            self.exposure_slider.setValue(self.camera.exposure)
            self.exposure_label.setText(f"Exposure: {self.camera.exposure}")
            self.exposure_label.setStyleSheet("color: #f0f0f0;")
        else:
            self.exposure_slider.setEnabled(False)
            self.exposure_label.setText("Exposure: Not Supported")
            self.exposure_label.setStyleSheet("color: #666666;")

    def change_camera(self, index):
        """Change camera with blocking dialog"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Switching Cameras")
        dialog.setText("Switching cameras, please wait...")
        dialog.setStandardButtons(QMessageBox.NoButton)
        dialog.show()
        QApplication.processEvents()
        
        try:
            if self.camera_thread and self.camera_thread.isRunning():
                self.stop_camera()
            
            # Get the model path from the current camera
            model_path = self.camera.model_path if self.camera else None
            
            # Create new camera instance with the same model
            self.camera = Camera(index, model_path=model_path)
            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.detection_ready.connect(self.update_detection)
            self.camera_thread.error_occurred.connect(self.handle_camera_error)
            
            # Update camera controls based on new camera capabilities
            self._update_camera_controls()
            
            # Set flip button state
            self.flip_button.setText("Flip Camera")
            self.flip_button.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            
            self.start_camera()
            dialog.accept()
        except Exception as e:
            dialog.close()
            QMessageBox.critical(self, "Camera Error", f"Failed to switch camera: {str(e)}")

    def update_frame(self, frame):
        """Update the video display with a regular frame"""
        self.update_video_display(frame)

    def update_detection(self, frame, counts, fps):
        """Update the video display with a frame containing detections and FPS"""
        logging.info(f"Update detection called with FPS: {fps}, Counts: {counts}")
        try:
            # Update video display immediately with the frame
            self.update_video_display(frame)
            
            # Update statistics with proper class names
            linear_cracks = counts.get('Linear-Crack', 0)
            alligator_cracks = counts.get('Alligator-Crack', 0)
            potholes = counts.get('Pothole', 0)
            
            # Update donut widgets
            self.linear_donut.count = linear_cracks
            self.alligator_donut.count = alligator_cracks
            self.potholes_donut.count = potholes
            
            # Force immediate update of donut widgets
            self.linear_donut.repaint()
            self.alligator_donut.repaint()
            self.potholes_donut.repaint()
            
            # Update status bar with detection info
            self.status_labels["fps"].setText(f"FPS: {fps:.1f}")
            
            # If we have GPS coordinates and defects were detected, update the map
            # if self.gps_thread.is_connected():
            #     lat, lon = self.gps_thread.get_coordinates()
            #     for defect_type, count in counts.items():
            #         if count > 0:
            #             # Get the latest saved image path for this defect
            #             image_path = self.camera.detector.get_latest_image_path(defect_type)
            #             if image_path:
            #                 self.map_view.add_defect_location(
            #                     lat=lat,
            #                     lon=lon,
            #                     defect_type=defect_type,
            #                     image_path=image_path,
            #                     confidence=count  # Using count as a simple confidence metric
            #                 )
            
            # Log the counts for debugging
            logging.info(f"Detection counts - Linear: {linear_cracks}, Alligator: {alligator_cracks}, Potholes: {potholes}")
            
        except Exception as e:
            logging.error(f"Error updating detection display: {e}")

    def update_video_display(self, frame):
        """Common method to update the video display"""
        try:
            # Convert frame to QImage (no color conversion, use BGR888)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Update the label immediately
            self.video_label.setPixmap(scaled_pixmap)
            self.video_label.repaint()  # Force immediate repaint
            
        except Exception as e:
            logging.error(f"Error updating video display: {e}")

    def handle_camera_error(self, error_msg):
        QMessageBox.warning(self, "Camera Error", error_msg)
        self.stop_camera()

    def toggle_detection(self):
        """Toggle detection on/off"""
        if not self.camera or not self.camera.detector:
            logging.warning("Camera or detector not initialized")
            QMessageBox.warning(self, "Detection Error", "Camera or detector not initialized")
            return

        self.detecting = not self.detecting
        logging.info(f"Detection toggled: {'on' if self.detecting else 'off'}")
        self.update_detect_button_style()
        
        if self.camera:
            self.camera.toggle_detection(self.detecting)
            logging.info(f"Camera detection state: {self.detecting}")
            
            if not self.camera_thread.isRunning():
                logging.info("Starting camera thread")
                self.start_camera()

    def toggle_gps(self):
        """Toggle GPS connection"""
        if self.gps_thread.isRunning():
            self.gps_thread.stop()
            self.connect_gps_button.setText("Connect GPS")
            self.status_labels["gps"].setText("GPS: Disconnected")
        else:
            # Show port selection dialog
            ports = self.gps_thread.get_available_ports()
            if not ports:
                QMessageBox.warning(self, "GPS Error", "No GPS devices found")
                return
                
            port, ok = QInputDialog.getItem(
                self, "Select GPS Port", 
                "Choose a GPS port:", 
                ports, 0, False
            )
            
            if ok and port:
                if self.gps_thread.connect_manually(port):
                    self.gps_thread.start()
                    self.connect_gps_button.setText("Disconnect GPS")
                else:
                    QMessageBox.warning(self, "GPS Error", "Failed to connect to GPS device")

    def run_analysis(self):
        """Run analysis on a directory of images and upload results to cloud."""
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            # Show directory selection dialog
            dialog = DirectorySelectionDialog(self)
            if dialog.exec_() != QDialog.Accepted:
                logger.info("Directory selection cancelled")
                return

            # Get selected directories
            input_dir = os.path.abspath(dialog.get_directories()[0])  # Get absolute path
            output_dir = os.path.abspath(dialog.get_directories()[1])  # Get absolute path
            logger.info(f"Selected input directory (absolute path): {input_dir}")
            logger.info(f"Selected output directory (absolute path): {output_dir}")

            # Verify input directory exists and is accessible
            if not os.path.exists(input_dir):
                logger.error(f"Input directory does not exist: {input_dir}")
                QMessageBox.critical(self, "Error", f"Input directory does not exist:\n{input_dir}")
                return

            if not os.access(input_dir, os.R_OK):
                logger.error(f"No read permission for input directory: {input_dir}")
                QMessageBox.critical(self, "Error", f"No read permission for input directory:\n{input_dir}")
                return

            # List all files in directory for debugging
            all_files = os.listdir(input_dir)
            logger.info(f"All files in directory: {all_files}")

            # Get list of image files with more detailed logging
            image_files = []
            for f in all_files:
                full_path = os.path.join(input_dir, f)
                if os.path.isfile(full_path):  # Only check files, not directories
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(f)
                        logger.info(f"Found image file: {f}")
                    else:
                        logger.debug(f"Skipping non-image file: {f}")
                else:
                    logger.debug(f"Skipping directory: {f}")

            logger.info(f"Found {len(image_files)} image files: {image_files}")
            
            if not image_files:
                error_msg = (
                    "No image files found in the selected directory.\n\n"
                    f"Directory: {input_dir}\n"
                    "Supported formats: .png, .jpg, .jpeg\n\n"
                    "Please ensure:\n"
                    "1. The directory contains image files\n"
                    "2. The files have correct extensions (.png, .jpg, .jpeg)\n"
                    "3. You have read permissions for the files"
                )
                logger.error(error_msg)
                QMessageBox.warning(self, "No Images", error_msg)
                return

            # Store the selected output directory as a class attribute
            self.selected_output_dir = output_dir
            logger.info(f"Selected output directory stored: {self.selected_output_dir}")

            # Create output directory if it doesn't exist
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Output directory created/verified: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to create output directory:\n{output_dir}\n\nError: {str(e)}")
                return

            # Initialize list to track processed files
            processed_files = []

            # Create progress dialog
            progress = QProgressDialog("Processing images...", "Cancel", 0, len(image_files), self)
            progress.setWindowTitle("Processing Images")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)  # Show immediately
            progress.setAutoClose(True)
            progress.setAutoReset(True)

            # Initialize timing variables
            start_time = time.time()
            processed_count = 0
            total_images = len(image_files)

            # Model path
            if getattr(sys, 'frozen', False):
                model_path = os.path.join(sys._MEIPASS, 'models', 'road_defect.pt')
            else:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'road_defect.pt')

            logger.info(f"Looking for model at: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found at {model_path}")

            # Initialize detector if not already initialized
            if not hasattr(self, 'detector') or self.detector is None:
                logger.info("Initializing detector...")
                try:
                    from src.app.modules.detection import DefectDetector
                    from src.app.modules.severity_calculator import SeverityCalculator
                    
                    # Initialize both detector and severity calculator
                    self.detector = DefectDetector(model_path=model_path)
                    self.severity_calculator = SeverityCalculator(
                        camera_width=1280,  # Default camera resolution
                        camera_height=720,
                        model_path=model_path
                    )
                    
                    if not self.detector or not self.severity_calculator:
                        raise RuntimeError("Failed to initialize detector or severity calculator")
                    logger.info("Detector and severity calculator initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize detector: {str(e)}")
                    raise

            # Initialize camera calibration parameters
            camera_matrix = np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ])
            distortion_coeffs = np.zeros(5)

            # Process each image
            processed_files = []  # Track successfully processed files
            for idx, filename in enumerate(image_files):
                if progress.wasCanceled():
                    logger.info("Processing cancelled by user")
                    break

                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                logger.info(f"Processing image {idx + 1}/{total_images}: {filename}")

                try:
                    # Update progress dialog
                    elapsed_time = time.time() - start_time
                    if processed_count > 0:
                        avg_speed = elapsed_time / processed_count
                        eta = avg_speed * (total_images - processed_count)
                        progress.setLabelText(
                            f"Processing: {filename}\n"
                            f"Speed: {avg_speed:.2f}s per image\n"
                            f"ETA: {eta:.1f}s remaining"
                        )
                    else:
                        progress.setLabelText(f"Processing: {filename}")

                    # Process image with severity calculator
                    final_img, defect_pixels, defect_ratio, metadata = self.severity_calculator.process_image(
                        input_path,
                        camera_matrix,
                        distortion_coeffs,
                        output_path,
                        distance_to_object_m=1.0,  # Default distance
                        confidence_threshold=0.16
                    )

                    if final_img is not None and len(defect_pixels) > 0:
                        # Save processed image
                        save_success = cv2.imwrite(output_path, final_img)
                        if save_success:
                            processed_files.append(output_path)
                            processed_count += 1
                            logger.info(f"Successfully processed and saved: {filename}")

                            # Update statistics from metadata
                            if metadata:
                                self.update_defect_stats(
                                    metadata['DefectCounts'].get('Linear-Crack', 0),
                                    metadata['DefectCounts'].get('Alligator-Crack', 0),
                                    metadata['DefectCounts'].get('Pothole', 0)
                                )
                                
                                # Save metadata as JSON
                                json_path = os.path.splitext(output_path)[0] + '_metadata.json'
                                with open(json_path, 'w') as f:
                                    json.dump(metadata, f, indent=4)
                                logger.info(f"Saved metadata to: {json_path}")
                        else:
                            logger.error(f"Failed to save processed image: {output_path}")
                    else:
                        logger.info(f"No defects detected in {filename}, skipping...")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)

                # Update progress bar
                progress.setValue(idx + 1)
                QApplication.processEvents()

            # Calculate final statistics
            total_time = time.time() - start_time
            avg_speed = total_time / processed_count if processed_count > 0 else 0

            logger.info(f"Processing complete. Processed {processed_count} out of {total_images} images")
            logger.info(f"Total time: {total_time:.1f}s, Average speed: {avg_speed:.2f}s per image")

            # Show completion message
            progress.close()
            if processed_count > 0:
                QMessageBox.information(
                    self,
                    "Processing Complete",
                    f"Successfully processed {processed_count} out of {total_images} images.\n\n"
                    f"Total processing time: {total_time:.1f} seconds\n"
                    f"Average speed: {avg_speed:.2f} seconds per image\n"
                    f"Output directory: {output_dir}"
                )
            else:
                error_msg = (
                    "No images were processed successfully.\n\n"
                    "Please check the following:\n"
                    "1. The input directory contains valid image files (.jpg, .jpeg, .png)\n"
                    "2. The images can be read by OpenCV\n"
                    "3. The detector is properly initialized\n"
                    "4. Check the application logs for detailed error messages\n\n"
                    f"Input directory: {input_dir}\n"
                    f"Output directory: {output_dir}\n"
                    f"Total images found: {total_images}"
                )
                logger.error(error_msg)
                QMessageBox.warning(self, "Processing Complete", error_msg)

        except Exception as e:
            logger.error(f"Script failed: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Processing Error",
                f"An error occurred during processing: {str(e)}\n\n"
                "Please check the application logs for more details."
            )

    def show_settings(self):
        """Show settings dialog"""
        if not self.camera or not self.camera.detector:
            QMessageBox.warning(self, "Settings Error", "Camera or detector not initialized")
            return
            
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Settings were saved successfully
            logging.info("Settings updated successfully")

    def update_gps(self, lat, lon):
        """Update GPS coordinates in the UI"""
        logging.info(f"GPS update received - Lat: {lat}, Lon: {lon}")
        self.lat_label.setText(f"Latitude: {lat:.6f}")
        self.long_label.setText(f"Longitude: {lon:.6f}")

    def update_gps_status(self, status):
        """Update GPS status in the UI"""
        logging.info(f"GPS status update: {status}")
        self.status_labels["gps"].setText(f"GPS: {status}")  # Update GPS status in the status bar
        if status == "GPS Fix":
            self.connect_gps_button.setText("Disconnect GPS")
        elif status == "No Fix":
            self.connect_gps_button.setText("Connect GPS")

    def start_camera(self):
        if self.camera_thread and not self.camera_thread.isRunning():
            self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()

    def closeEvent(self, event):
        """Handle application close"""
        self.stop_camera()
        # self.gps_thread.stop()
        if self.camera:
            self.camera.cleanup()
        # self.map_view.clear_defects()  # Clear map markers
        event.accept()

    def update_detect_button_style(self):
        """Update the detection button style based on state"""
        if self.detecting:
            self.detect_button.setText("Stop Detection")
            self.detect_button.setStyleSheet("""
                #detect_button {
                    background-color: #ff6f61;
                    color: white;
                    font-weight: bold;
                    padding: 12px;
                    border-radius: 6px;
                    border: 2px solid #e05d50;
                }
                #detect_button:hover {
                    background-color: #e05d50;
                    border-color: #c04c40;
                }
                #detect_button:pressed {
                    background-color: #c04c40;
                    border-color: #a03b30;
                }
            """)
        else:
            self.detect_button.setText("Start Detection")
            self.detect_button.setStyleSheet("""
                #detect_button {
                    background-color: #77dd77;
                    color: white;
                    font-weight: bold;
                    padding: 12px;
                    border-radius: 6px;
                    border: 2px solid #5cb85c;
                }
                #detect_button:hover {
                    background-color: #5cb85c;
                    border-color: #4cae4c;
                }
                #detect_button:pressed {
                    background-color: #4cae4c;
                    border-color: #3d8b3d;
                }
            """)

    def update_zoom(self, value):
        """Update zoom setting"""
        if self.camera:
            zoom_factor = value / 100.0  # Convert slider value to zoom factor (100 = 1.0x)
            self.camera.set_zoom(zoom_factor)
            self.zoom_label.setText(f"Zoom: {zoom_factor:.2f}x")
            logging.info(f"Zoom updated to: {zoom_factor:.2f}x")

    def update_brightness(self, value):
        """Update brightness setting"""
        if self.camera:
            success = self.camera.set_brightness(value)
            if success:
                self.brightness_label.setText(f"Brightness: {value}")
                logging.info(f"Brightness updated to: {value}")
            else:
                logging.warning(f"Failed to update brightness to: {value}")
                # Reset slider to last successful value
                self.brightness_slider.setValue(self.camera.brightness)

    def update_exposure(self, value):
        """Update exposure setting"""
        if self.camera:
            success = self.camera.set_exposure(value)
            if success:
                self.exposure_label.setText(f"Exposure: {value}")
                logging.info(f"Exposure updated to: {value}")
            else:
                logging.warning(f"Failed to update exposure to: {value}")
                # Reset slider to last successful value
                self.exposure_slider.setValue(self.camera.exposure)

    def update_defect_stats(self, linear, alligator, potholes):
        # Update the defect counts and redraw donuts
        self.linear_donut.count = linear
        self.alligator_donut.count = alligator
        self.potholes_donut.count = potholes
        self.linear_donut.update()
        self.alligator_donut.update()
        self.potholes_donut.update()

    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        if not hasattr(self, '_last_fps_update'):
            self._last_fps_update = current_time
            self._frame_count = 0
            return 0
        
        self._frame_count += 1
        elapsed = current_time - self._last_fps_update
        
        if elapsed >= 1.0:  # Update FPS every second
            fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_update = current_time
            return fps
        
        return getattr(self, '_last_fps', 0)

    def create_controls_section(self):
        layout = QHBoxLayout()

        # Create a group box for video controls
        video_group = QGroupBox("Video Control")
        video_layout = QVBoxLayout()

        # Camera selection combo box
        video_layout.addWidget(QLabel("Camera:"))
        video_layout.addWidget(self.camera_combo)

        # Flip button
        video_layout.addWidget(self.flip_button)

        # Zoom slider with label
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_label)
        video_layout.addLayout(zoom_layout)

        # Brightness slider with label
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_label)
        video_layout.addLayout(brightness_layout)

        # Exposure slider with label
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("Exposure:"))
        exposure_layout.addWidget(self.exposure_slider)
        exposure_layout.addWidget(self.exposure_label)
        video_layout.addLayout(exposure_layout)

        # Add disabled slider styles
        video_group.setStyleSheet("""
            QSlider:disabled {
                background: #2b2b2b;
            }
            QSlider::groove:disabled {
                background: #3b3b3b;
                border: 1px solid #555;
            }
            QSlider::handle:disabled {
                background: #555;
                border: 1px solid #666;
            }
        """)

        video_group.setLayout(video_layout)

        # Statistics with Donut Chart and GPS Frames
        stats_group = QGroupBox("Statistics Logs")
        stats_layout = QVBoxLayout()

        # Donuts for each defect class
        donuts_layout = QHBoxLayout()
        donuts_layout.addWidget(self.linear_donut)
        donuts_layout.addWidget(self.alligator_donut)
        donuts_layout.addWidget(self.potholes_donut)
        stats_layout.addLayout(donuts_layout)

        # GPS Display with separate frames
        gps_layout = QHBoxLayout()
        self.long_frame = QFrame()
        self.long_frame.setFrameShape(QFrame.StyledPanel)
        self.long_label = QLabel("Longitude: 0.0")
        long_layout = QVBoxLayout()
        long_layout.addWidget(self.long_label)
        self.long_frame.setLayout(long_layout)

        self.lat_frame = QFrame()
        self.lat_frame.setFrameShape(QFrame.StyledPanel)
        self.lat_label = QLabel("Latitude: 0.0")
        lat_layout = QVBoxLayout()
        lat_layout.addWidget(self.lat_label)
        self.lat_frame.setLayout(lat_layout)

        gps_layout.addWidget(self.long_frame)
        gps_layout.addWidget(self.lat_frame)
        stats_layout.addLayout(gps_layout)
        stats_group.setLayout(stats_layout)

        # Main Controls
        main_group = QGroupBox("Main Control")
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.detect_button)
        main_layout.addWidget(self.connect_gps_button)
        main_layout.addWidget(self.run_analysis_button)
        main_layout.addWidget(self.settings_button)
        main_layout.addWidget(self.connect_cloud_button)  # Add connect cloud button
        main_group.setLayout(main_layout)

        layout.addWidget(video_group, stretch=1)
        layout.addWidget(stats_group, stretch=1)
        layout.addWidget(main_group, stretch=1)

        return layout

    def update_flip_button(self, is_flipped):
        """Update flip button text and style based on state"""
        try:
            # Block signals temporarily to prevent multiple updates
            self.flip_button.blockSignals(True)
            
            # Update button state
            self.flip_button.setChecked(is_flipped)
            self.flip_button.setText("Normal View" if is_flipped else "Flip Camera")
            
            # Update button style
            if is_flipped:
                self.flip_button.setStyleSheet("""
                    QPushButton {
                        background-color: #6c757d;
                        color: white;
                        border: none;
                        padding: 5px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #5a6268;
                    }
                """)
            else:
                self.flip_button.setStyleSheet("""
                    QPushButton {
                        background-color: #6c757d;
                        color: white;
                        border: none;
                        padding: 5px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #5a6268;
                    }
                """)
            
            logging.info(f"Flip button UI updated - {'Normal View' if is_flipped else 'Flip Camera'}")
            
        finally:
            # Re-enable signals
            self.flip_button.blockSignals(False)

    def update_cpu_gpu(self):
        # Update CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.status_labels["cpu"].setText(f"CPU: {cpu_percent:.1f}%")
        # Update GPU usage if available
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.status_labels["gpu"].setText(f"GPU: {util.gpu}%")
            except Exception:
                self.status_labels["gpu"].setText("GPU: N/A")
        else:
            self.status_labels["gpu"].setText("GPU: N/A")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())
