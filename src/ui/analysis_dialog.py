from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QTreeView, QFileSystemModel, QSplitter, QFrame,
    QScrollArea, QTextEdit, QMessageBox, QListWidget, QComboBox,
    QFileDialog, QCheckBox, QProgressDialog, QListWidgetItem
)
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QPixmap, QImage
import os
import json
from datetime import datetime
import logging
from PIL import Image
import piexif
import cv2
from PyQt5.QtWidgets import QApplication
import numpy as np
from src.ui.settings_manager import SettingsManager
import time

class AnalysisDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Defect Analysis")
        self.setMinimumSize(1600, 900)
        
        # Initialize attributes
        self.current_image_path = None
        self.current_metadata = None
        self.image_list = []
        self.current_index = -1
        self.local_path = None
        self.show_delete_warning = True
        self.settings = SettingsManager()
        
        # Initialize UI components
        self.file_list = None
        self.image_label = None
        self.metadata_text = None
        self.prev_btn = None
        self.next_btn = None
        self.delete_btn = None
        self.analyze_btn = None
        self.folder_btn = None
        self.select_all_btn = None
        
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (File List)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.clicked.connect(self.open_local_folder)
        folder_layout.addWidget(self.folder_btn)
        
        # Add Select All button
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.toggle_select_all)
        folder_layout.addWidget(self.select_all_btn)
        
        left_layout.addLayout(folder_layout)
        
        # File list with checkboxes
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        
        # Center panel (Image Preview)
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        # Image preview
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        center_layout.addWidget(self.image_label)
        
        # Right panel (Metadata View)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Metadata text
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setStyleSheet("background-color: #1e1e1e; color: #f0f0f0; border: 1px solid #444;")
        right_layout.addWidget(self.metadata_text)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)  # Left panel (wider)
        splitter.setStretchFactor(1, 3)  # Center panel
        splitter.setStretchFactor(2, 2)  # Right panel
        
        # Control buttons at bottom
        control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.delete_btn = QPushButton("Delete Selected")
        self.analyze_btn = QPushButton("Analyze Selected")
        
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.next_btn.clicked.connect(self.show_next_image)
        self.delete_btn.clicked.connect(self.delete_selected_images)
        self.analyze_btn.clicked.connect(self.analyze_selected_images)
        
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.delete_btn)
        control_layout.addWidget(self.analyze_btn)
        
        # Add widgets to main layout
        layout.addWidget(splitter)
        layout.addLayout(control_layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
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
                background-color: #6a6a6a;
            }
            QListWidget {
                background-color: #1e1e1e;
                color: #f0f0f0;
                border: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #4a4a4a;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #444;
            }
        """)

    def open_local_folder(self):
        """Open a local folder for viewing detections"""
        folder = QFileDialog.getExistingDirectory(self, "Select Detection Folder")
        if folder:
            self.local_path = folder
            self.refresh_file_list()

    def refresh_file_list(self):
        """Refresh the file list with contents from current directory"""
        self.file_list.clear()
        
        if not self.local_path:
            return
            
        for filename in os.listdir(self.local_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                item = QListWidgetItem(filename)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.file_list.addItem(item)

    def toggle_select_all(self):
        """Toggle selection of all files"""
        if self.file_list.count() == 0:
            return
            
        # Check if all items are currently selected
        all_selected = all(self.file_list.item(i).checkState() == Qt.Checked 
                          for i in range(self.file_list.count()))
        
        # Set all items to the opposite state
        new_state = Qt.Unchecked if all_selected else Qt.Checked
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(new_state)

    def on_file_selected(self, item):
        """Handle file selection from the list"""
        filename = item.text()
        
        if not self.local_path:
            return
            
        file_path = os.path.join(self.local_path, filename)
        if os.path.exists(file_path):
            image = QImage(file_path)
            self.display_image(image)
            
            # Get metadata from EXIF
            try:
                exif_dict = piexif.load(file_path)
                if 'Exif' in exif_dict and piexif.ExifIFD.UserComment in exif_dict['Exif']:
                    metadata_str = exif_dict['Exif'][piexif.ExifIFD.UserComment].decode('ascii', 'replace')
                    if metadata_str.startswith('ASCII'):
                        metadata_str = metadata_str[6:]  # Remove ASCII prefix
                    try:
                        metadata = json.loads(metadata_str)
                        self.display_metadata(metadata, filename)
                    except json.JSONDecodeError:
                        # Try to parse as a simple string if JSON parsing fails
                        self.metadata_text.setText(f"Raw metadata: {metadata_str}")
                else:
                    self.metadata_text.setText("No metadata found in image")
            except Exception as e:
                self.metadata_text.setText(f"Error reading metadata: {str(e)}")

    def display_image(self, image):
        """Display image in the preview"""
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def display_metadata(self, metadata, filename):
        """Display metadata in the text view"""
        if not metadata:
            self.metadata_text.setText("No metadata available")
            return

        try:
            text = f"File: {filename}\n\n"
            
            # Format detection results
            if 'detections' in metadata:
                text += "Detection Results:\n"
                for defect_type, count in metadata['detections'].items():
                    text += f"  {defect_type}: {count}\n"
                text += "\n"
            
            # Format GPS data
            if 'gps_data' in metadata:
                text += "GPS Data:\n"
                for key, value in metadata['gps_data'].items():
                    text += f"  {key}: {value}\n"
                text += "\n"
            
            # Format other metadata
            for key, value in metadata.items():
                if key not in ['detections', 'gps_data']:
                    if isinstance(value, dict):
                        text += f"{key}:\n"
                        for subkey, subvalue in value.items():
                            text += f"  {subkey}: {subvalue}\n"
                    else:
                        text += f"{key}: {value}\n"
            
            self.metadata_text.setText(text)
        except Exception as e:
            self.metadata_text.setText(f"Error formatting metadata: {str(e)}")

    def show_previous_image(self):
        """Show previous image in the list"""
        current_row = self.file_list.currentRow()
        if current_row > 0:
            self.file_list.setCurrentRow(current_row - 1)
            self.on_file_selected(self.file_list.item(current_row - 1))

    def show_next_image(self):
        """Show next image in the list"""
        current_row = self.file_list.currentRow()
        if current_row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(current_row + 1)
            self.on_file_selected(self.file_list.item(current_row + 1))

    def get_selected_files(self):
        """Get list of selected files"""
        selected_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_files.append(item.text())
        return selected_files

    def delete_selected_images(self):
        """Delete selected images and their associated metadata files."""
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "No Selection", "Please select files to delete")
            return
            
        # Show warning dialog if enabled
        if self.show_delete_warning:
            warning_dialog = QMessageBox(self)
            warning_dialog.setIcon(QMessageBox.Warning)
            warning_dialog.setWindowTitle("Confirm Delete")
            warning_dialog.setText(f"Are you sure you want to delete {len(selected_files)} selected file(s)?")
            warning_dialog.setInformativeText("This action cannot be undone.")
            
            # Add checkbox for "Don't show again"
            dont_show_checkbox = QCheckBox("Don't show this warning again for this session")
            warning_dialog.setCheckBox(dont_show_checkbox)
            
            warning_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            warning_dialog.setDefaultButton(QMessageBox.No)
            
            reply = warning_dialog.exec_()
            
            # Update warning preference
            self.show_delete_warning = not dont_show_checkbox.isChecked()
            
            if reply != QMessageBox.Yes:
                return
        
        # Delete selected files and their metadata
        for filename in selected_files:
            try:
                file_path = os.path.join(self.local_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                # Delete the associated metadata.json file
                metadata_file_path = os.path.splitext(file_path)[0] + '_metadata.json'
                if os.path.exists(metadata_file_path):
                    os.remove(metadata_file_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete {filename}: {str(e)}")
        
        self.refresh_file_list()
        QMessageBox.information(self, "Success", f"Successfully deleted {len(selected_files)} file(s)")

    def analyze_selected_images(self):
        """Analyze selected images using severity calculator"""
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self, "No Selection", "Please select files to analyze")
            return
            
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return
            
        # Create progress dialog
        progress = QProgressDialog("Analyzing files...", "Cancel", 0, len(selected_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Analysis Progress")
        progress.setMinimumDuration(0)  # Show immediately
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        
        # Initialize severity calculator
        from ..modules.severity_calculator import SeverityCalculator
        calculator = SeverityCalculator(
            camera_width=1920,
            camera_height=1080,
            model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'road_defect.pt')
        )
        
        # Camera calibration parameters
        camera_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ])
        distortion_coeffs = np.zeros(5)
        
        success_count = 0
        error_count = 0
        
        # Start time measurement
        start_time = time.time()
        
        for i, filename in enumerate(selected_files):
            if progress.wasCanceled():
                break
                
            # Update progress
            progress.setValue(i)
            progress.setLabelText(f"Analyzing {filename}... ({i+1}/{len(selected_files)})")
            QApplication.processEvents()  # Ensure UI updates   
            
            try:
                # Read image
                image_path = os.path.join(self.local_path, filename)
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to read image: {filename}")
                
                # Process image
                processed_image, _, _, metadata = calculator.process_image(
                    image_path=image_path,
                    camera_matrix=camera_matrix,
                    distortion_coeffs=distortion_coeffs,
                    save_path=os.path.join(output_dir, f"processed_{filename}"),
                    distance_to_object_m=1.0,
                    confidence_threshold=self.settings.get_confidence_threshold()
                )
                
                if processed_image is not None:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing {filename}: {str(e)}")
        
        # End time measurement
        end_time = time.time()
        total_time = end_time - start_time  # Total time in seconds
        
        # Set final progress value
        progress.setValue(len(selected_files))
        
        # Show results
        result_message = f"Analysis Complete:\n"
        result_message += f"Successfully analyzed: {success_count} file(s)\n"
        if error_count > 0:
            result_message += f"Failed to analyze: {error_count} file(s)\n"
        
        # Add total time taken to the message
        result_message += f"Total time taken: {total_time:.2f} seconds for {success_count} file(s)"
        
        if error_count == 0:
            QMessageBox.information(self, "Analysis Complete", result_message)
        else:
            QMessageBox.warning(self, "Analysis Complete", result_message)

