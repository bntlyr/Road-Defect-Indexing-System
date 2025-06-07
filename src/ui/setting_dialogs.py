from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSlider, QFileDialog, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt
from .settings_manager import SettingsManager
import os

class SettingsDialog(QDialog):
    """Pure‑UI settings — no backend connections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_manager = parent.settings_manager
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        # Initialize UI components
        self._setup_ui()
        
        # Load current settings after UI is set up
        self._load_current_settings()

    def _setup_ui(self):
        """Set up the settings dialog UI"""
        layout = QVBoxLayout(self)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)  # Default to 0.5
        self.conf_value_label = QLabel("0.5")
        self.conf_slider.valueChanged.connect(self._update_conf_label)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        layout.addLayout(conf_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_directory)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)
        
        # Recording output directory
        rec_dir_layout = QHBoxLayout()
        rec_dir_label = QLabel("Recording Output Directory:")
        self.rec_output_dir_edit = QLineEdit()
        self.rec_output_dir_edit.setReadOnly(True)
        rec_browse_btn = QPushButton("Browse")
        rec_browse_btn.clicked.connect(self._browse_recording_directory)
        rec_dir_layout.addWidget(rec_dir_label)
        rec_dir_layout.addWidget(self.rec_output_dir_edit)
        rec_dir_layout.addWidget(rec_browse_btn)
        layout.addLayout(rec_dir_layout)
        
        # Record mode checkbox
        self.record_mode_checkbox = QCheckBox("Enable Record Mode")
        self.record_mode_checkbox.stateChanged.connect(self._on_record_mode_changed)
        layout.addWidget(self.record_mode_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def _load_current_settings(self):
        """Load current settings into the dialog"""
        try:
            # Load current settings with proper type handling
            conf_value = self.settings_manager.get_setting('confidence_threshold')
            self.conf_slider.setValue(int(conf_value * 100))
            
            output_dir = self.settings_manager.get_setting('output_dir')
            self.output_dir_edit.setText(str(output_dir))
            
            rec_output_dir = self.settings_manager.get_setting('recording_output_dir')
            self.rec_output_dir_edit.setText(str(rec_output_dir))
            
            record_mode = self.settings_manager.get_setting('record_mode')
            self.record_mode_checkbox.setChecked(bool(record_mode))
            
            # Enable/disable recording output directory based on record mode
            self.rec_output_dir_edit.setEnabled(bool(record_mode))
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Set default values if loading fails
            self.conf_slider.setValue(50)  # 0.5
            self.output_dir_edit.setText('')
            self.rec_output_dir_edit.setText('')
            self.record_mode_checkbox.setChecked(False)
            self.rec_output_dir_edit.setEnabled(False)

    def _update_conf_label(self, value):
        """Update the confidence threshold label"""
        self.conf_value_label.setText(f"{value/100:.2f}")

    def _browse_directory(self):
        """Open directory browser for output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _browse_recording_directory(self):
        """Open directory browser for recording output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Recording Output Directory")
        if dir_path:
            self.rec_output_dir_edit.setText(dir_path)

    def _on_record_mode_changed(self, state):
        """Handle record mode checkbox state change"""
        self.rec_output_dir_edit.setEnabled(state == Qt.Checked)

    def _save_settings(self):
        """Save settings and close dialog"""
        try:
            # Save settings with proper type conversion
            self.settings_manager.set_setting('confidence_threshold', self.conf_slider.value() / 100)
            self.settings_manager.set_setting('output_dir', self.output_dir_edit.text())
            self.settings_manager.set_setting('recording_output_dir', self.rec_output_dir_edit.text())
            self.settings_manager.set_setting('record_mode', self.record_mode_checkbox.isChecked())
            
            # Close dialog
            self.accept()
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to save settings: {str(e)}")

    def _apply_dark_theme(self):
        self.setStyleSheet(
            """
            QDialog {background:#2b2b2b; color:#f0f0f0;}
            QPushButton {background:#4a4a4a; color:#ddd; padding:6px; border:none; border-radius:4px;}
            QPushButton:hover {background:#5a5a5a;}
            QLineEdit {background:#3b3b3b; color:#f0f0f0; border:1px solid #555; border-radius:4px; padding:4px;}
            QSlider::groove:horizontal {height:8px; background:#4a4a4a; border:1px solid #999; border-radius:4px;}
            QSlider::handle:horizontal {width:18px; background:#4a90e2; border:1px solid #5c5c5c; margin:-2px 0; border-radius:9px;}
            QCheckBox {color:#f0f0f0;}
            QCheckBox::indicator {width:18px; height:18px; border:1px solid #555; border-radius:4px;}
            QCheckBox::indicator:checked {background:#4a90e2; border:1px solid #4a90e2;}
            """
        )
