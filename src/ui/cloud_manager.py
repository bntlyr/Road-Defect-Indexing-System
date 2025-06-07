from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QTreeView, QFileSystemModel, QSplitter, QFrame,
    QScrollArea, QTextEdit, QMessageBox, QListWidget, QComboBox,
    QFileDialog, QCheckBox, QProgressDialog
)
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QPixmap, QImage
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from ..modules.cloud_connector import CloudConnector, CloudStorage
import piexif
from PIL import Image

class CloudManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Storage Manager")
        self.setMinimumSize(1600, 900)  # Increased window size
        
        # Initialize all attributes
        self.cloud_storage = None
        self.current_image_path = None
        self.current_metadata = None
        self.image_list = []
        self.current_index = -1
        self.is_cloud_view = True
        self.local_path = None
        self.show_delete_warning = True  # Flag to control warning display
        self.show_upload_warning = True  # Flag to control upload warning display
        self.show_duplicate_warning = True  # Flag to control duplicate warning display
        
        # Initialize UI components
        self.file_list = None
        self.image_label = None
        self.metadata_text = None
        self.prev_btn = None
        self.next_btn = None
        self.delete_btn = None
        self.upload_btn = None
        self.view_selector = None
        
        # Load environment variables
        load_dotenv()
        
        self.setup_ui()
        self.initialize_cloud_connection()

    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (File List)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # View selector
        view_layout = QHBoxLayout()
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Cloud Storage", "Local Storage"])
        self.view_selector.currentIndexChanged.connect(self.on_view_changed)
        view_layout.addWidget(self.view_selector)
        
        # Local folder button
        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.clicked.connect(self.open_local_folder)
        view_layout.addWidget(self.folder_btn)
        
        left_layout.addLayout(view_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        
        # Center panel (Image Preview)
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        # Image preview
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)  # Increased size
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
        self.delete_btn = QPushButton("Delete")
        self.upload_btn = QPushButton("Upload to Cloud")
        
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.next_btn.clicked.connect(self.show_next_image)
        self.delete_btn.clicked.connect(self.delete_current_image)
        self.upload_btn.clicked.connect(self.upload_detections)
        
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.delete_btn)
        control_layout.addWidget(self.upload_btn)
        
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
            QComboBox {
                background-color: #4a4a4a;
                color: #ddd;
                padding: 5px;
                border: 1px solid #666;
                border-radius: 4px;
            }
            QComboBox:hover {
                background-color: #6a6a6a;
            }
        """)

    def initialize_cloud_connection(self):
        """Initialize cloud connection using .env file"""
        try:
            # Initialize cloud storage
            self.cloud_storage = CloudStorage()
            
            if self.cloud_storage.is_initialized:
                # Load cloud contents
                self.refresh_file_list()
                QMessageBox.information(self, "Success", "Successfully connected to cloud storage")
            else:
                QMessageBox.warning(self, "Warning", "Failed to connect to cloud storage. Check your .env file and credentials.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error initializing cloud connection: {str(e)}")

    def on_view_changed(self, index):
        """Handle view switching between cloud and local storage"""
        self.is_cloud_view = index == 0
        self.upload_btn.setEnabled(not self.is_cloud_view)
        self.refresh_file_list()

    def open_local_folder(self):
        """Open a local folder for viewing detections"""
        folder = QFileDialog.getExistingDirectory(self, "Select Detection Folder")
        if folder:
            self.local_path = folder
            self.refresh_file_list()

    def refresh_file_list(self):
        """Refresh the file list with contents from current view"""
        self.file_list.clear()
        
        if self.is_cloud_view:
            if not self.cloud_storage or not self.cloud_storage.is_initialized:
                return
            detections = self.cloud_storage.list_detections()
            for detection in detections:
                if detection.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.file_list.addItem(detection)
        else:
            if not self.local_path:
                return
            for filename in os.listdir(self.local_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.file_list.addItem(filename)

    def on_file_selected(self, item):
        """Handle file selection from the list"""
        filename = item.text()
        
        if self.is_cloud_view:
            if not self.cloud_storage or not self.cloud_storage.is_initialized:
                return
            image_data, _ = self.cloud_storage.download_detection(filename)
            if image_data:
                image = QImage.fromData(image_data)
                self.display_image(image)
                
                # Get metadata from JSON file - use same filename pattern
                base_name = filename.rsplit('.', 1)[0]  # Remove extension
                metadata_filename = f"{base_name}_metadata.json"
                metadata = self.cloud_storage.download_metadata(metadata_filename)
                if metadata:
                    self.display_metadata(metadata, filename)
                else:
                    self.metadata_text.setText(f"No metadata found in cloud storage: {metadata_filename}")
        else:
            if not self.local_path:
                return
            file_path = os.path.join(self.local_path, filename)
            if os.path.exists(file_path):
                image = QImage(file_path)
                self.display_image(image)
                
                # Get metadata from JSON file in the same directory
                base_name = filename.rsplit('.', 1)[0]  # Remove extension
                metadata_filename = f"{base_name}_metadata.json"
                metadata_path = os.path.join(self.local_path, metadata_filename)
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            if metadata:
                                self.display_metadata(metadata, filename)
                            else:
                                self.metadata_text.setText("Empty metadata file")
                    except json.JSONDecodeError as e:
                        self.metadata_text.setText(f"Error parsing metadata JSON: {str(e)}")
                    except Exception as e:
                        self.metadata_text.setText(f"Error reading metadata: {str(e)}")
                else:
                    self.metadata_text.setText(f"No metadata file found: {metadata_filename}")

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
            
            # Handle different metadata formats
            if isinstance(metadata, dict):
                # Format detection results
                if 'detection_results' in metadata:
                    text += "Detection Results:\n"
                    for defect_type, count in metadata['detection_results'].items():
                        text += f"  {defect_type}: {count}\n"
                    text += "\n"
                
                # Format frame information
                if 'frame_info' in metadata:
                    text += "Frame Information:\n"
                    for key, value in metadata['frame_info'].items():
                        text += f"  {key}: {value}\n"
                    text += "\n"
                
                # Format GPS data
                if 'gps_data' in metadata:
                    text += "GPS Data:\n"
                    for key, value in metadata['gps_data'].items():
                        text += f"  {key}: {value}\n"
                    text += "\n"
                
                # Format other metadata
                for key, value in metadata.items():
                    if key not in ['detection_results', 'frame_info', 'gps_data']:
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
            # Trigger the selection to update the view
            self.on_file_selected(self.file_list.item(current_row - 1))

    def show_next_image(self):
        """Show next image in the list"""
        current_row = self.file_list.currentRow()
        if current_row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(current_row + 1)
            # Trigger the selection to update the view
            self.on_file_selected(self.file_list.item(current_row + 1))

    def delete_current_image(self):
        """Delete the currently selected image and its metadata"""
        current_item = self.file_list.currentItem()
        if not current_item:
            return
            
        filename = current_item.text()
        base_name = filename.rsplit('.', 1)[0]  # Remove extension
        metadata_filename = f"{base_name}_metadata.json"
        
        # Show warning dialog if enabled
        if self.show_delete_warning:
            warning_dialog = QMessageBox(self)
            warning_dialog.setIcon(QMessageBox.Warning)
            warning_dialog.setWindowTitle("Confirm Delete")
            warning_dialog.setText(f"Are you sure you want to delete:\n\n{filename}\n{metadata_filename}")
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
        
        if self.is_cloud_view:
            if self.cloud_storage and self.cloud_storage.is_initialized:
                try:
                    # Delete both files from cloud storage
                    self.cloud_storage.delete_detection(filename)
                    self.cloud_storage.delete_detection(metadata_filename)
                    self.refresh_file_list()
                    QMessageBox.information(self, "Success", "Files deleted successfully from cloud storage")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to delete files from cloud storage: {str(e)}")
        else:
            try:
                # Delete image file
                file_path = os.path.join(self.local_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Delete metadata file
                metadata_path = os.path.join(self.local_path, metadata_filename)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                self.refresh_file_list()
                QMessageBox.information(self, "Success", "Files deleted successfully from local storage")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete files from local storage: {str(e)}")

    def upload_detections(self):
        """Upload detections from local folder to cloud storage"""
        # Check if we're in cloud view
        if self.is_cloud_view:
            QMessageBox.warning(self, "Invalid Operation", 
                              "Please switch to Local Storage view and select a folder first.")
            return

        if not self.local_path or not self.cloud_storage or not self.cloud_storage.is_initialized:
            QMessageBox.warning(self, "Invalid Operation", 
                              "Please select a local folder first.")
            return
            
        # Get list of image files and their corresponding metadata files
        image_files = []
        metadata_files = []
        for filename in os.listdir(self.local_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(filename)
                # Check for corresponding metadata file
                metadata_filename = f"{filename.rsplit('.', 1)[0]}_metadata.json"
                metadata_path = os.path.join(self.local_path, metadata_filename)
                if os.path.exists(metadata_path):
                    metadata_files.append(metadata_filename)
                else:
                    print(f"Warning: No metadata file found for {filename}")
        
        if not image_files:
            QMessageBox.warning(self, "No Files", "No image files found in the selected folder")
            return

        # Show upload warning dialog first
        warning_dialog = QMessageBox(self)
        warning_dialog.setIcon(QMessageBox.Warning)
        warning_dialog.setWindowTitle("Confirm Upload")
        warning_dialog.setText(f"Are you sure you want to upload {len(image_files)} image(s) and their metadata to cloud storage?")
        warning_dialog.setInformativeText("This will upload both the image files and their corresponding metadata JSON files.")
        
        # Add checkbox for "Don't show again"
        dont_show_checkbox = QCheckBox("Don't show this warning again for this session")
        warning_dialog.setCheckBox(dont_show_checkbox)
        
        warning_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        warning_dialog.setDefaultButton(QMessageBox.No)
        
        reply = warning_dialog.exec_()
        
        # Update warning preference
        self.show_upload_warning = not dont_show_checkbox.isChecked()
        
        if reply != QMessageBox.Yes:
            return

        # Check for duplicates after user confirms upload
        existing_files = self.cloud_storage.list_detections()
        duplicates = []
        for filename in image_files:
            if filename in existing_files:
                duplicates.append(filename)
        
        duplicate_handling = None
        if duplicates:
            duplicate_dialog = QMessageBox(self)
            duplicate_dialog.setIcon(QMessageBox.Warning)
            duplicate_dialog.setWindowTitle("Duplicate Files Found")
            duplicate_dialog.setText(f"Found {len(duplicates)} file(s) that already exist in cloud storage.")
            duplicate_dialog.setInformativeText("How would you like to handle these files?")
            
            # Add buttons for different options
            overwrite_button = duplicate_dialog.addButton("Overwrite All", QMessageBox.ActionRole)
            skip_button = duplicate_dialog.addButton("Skip All", QMessageBox.ActionRole)
            cancel_button = duplicate_dialog.addButton("Cancel", QMessageBox.RejectRole)
            
            # Add checkbox for "Don't show again"
            dont_show_checkbox = QCheckBox("Don't show this warning again for this session")
            duplicate_dialog.setCheckBox(dont_show_checkbox)
            
            duplicate_dialog.exec_()
            
            # Update warning preference
            self.show_duplicate_warning = not dont_show_checkbox.isChecked()
            
            clicked_button = duplicate_dialog.clickedButton()
            if clicked_button == cancel_button:
                return
            elif clicked_button == overwrite_button:
                duplicate_handling = "overwrite"
            else:  # skip_button
                duplicate_handling = "skip"
            
        # Create progress dialog
        progress = QProgressDialog("Uploading files...", "Cancel", 0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Upload Progress")
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for i, filename in enumerate(image_files):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Uploading {filename}...")
            
            # Handle duplicates
            if filename in duplicates:
                if duplicate_handling == "skip":
                    skipped_count += 1
                    continue
                elif duplicate_handling == "overwrite":
                    # Delete existing files before uploading
                    try:
                        self.cloud_storage.delete_detection(filename)
                        metadata_filename = f"{filename.rsplit('.', 1)[0]}_metadata"
                        self.cloud_storage.delete_detection(metadata_filename)
                    except Exception as e:
                        print(f"Error deleting existing file {filename}: {str(e)}")
            
            try:
                # Upload image file
                file_path = os.path.join(self.local_path, filename)
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                    self.cloud_storage.upload_detection(image_data, {}, filename)
                
                # Upload corresponding metadata file
                metadata_filename = f"{filename.rsplit('.', 1)[0]}_metadata"
                metadata_path = os.path.join(self.local_path, metadata_filename)
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.cloud_storage.upload_metadata(metadata_filename, metadata)
                    success_count += 1
                else:
                    error_count += 1
                    print(f"Warning: No metadata file found for {filename}")
                    
            except Exception as e:
                error_count += 1
                print(f"Error uploading {filename}: {str(e)}")
        
        progress.setValue(len(image_files))
        
        # Show results
        result_message = f"Upload Complete:\n"
        result_message += f"Successfully uploaded: {success_count} file(s)\n"
        if skipped_count > 0:
            result_message += f"Skipped: {skipped_count} file(s)\n"
        if error_count > 0:
            result_message += f"Failed to upload: {error_count} file(s)"
            
        if error_count == 0 and skipped_count == 0:
            QMessageBox.information(self, "Upload Complete", result_message)
        else:
            QMessageBox.warning(self, "Upload Complete", result_message)
        
        # Refresh the file list
        self.refresh_file_list() 