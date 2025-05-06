import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import logging
from dotenv import load_dotenv
import threading
import time

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Window setup
        self.title("Settings")
        self.geometry("600x800")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        
        # Load current settings
        self.current_settings = self.load_settings()
        
        # Create main frame with scrollbar
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create settings sections
        self.create_storage_settings()
        self.create_detection_settings()
        self.create_cloud_settings()
        
        # Create save button
        self.save_button = ctk.CTkButton(
            self.main_frame,
            text="Save Settings",
            command=self.save_settings,
            height=40
        )
        self.save_button.pack(pady=20)
        
        # Center window
        self.center_window()
        
    def load_settings(self):
        """Load settings from file"""
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
        return {}
        
    def create_storage_settings(self):
        """Create storage settings section"""
        # Storage section
        storage_frame = ctk.CTkFrame(self.main_frame)
        storage_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        ctk.CTkLabel(storage_frame, text="Storage Settings", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Default input path
        ctk.CTkLabel(storage_frame, text="Default Input Path:").pack(anchor="w", padx=10)
        input_path_frame = ctk.CTkFrame(storage_frame)
        input_path_frame.pack(fill="x", padx=10, pady=5)
        
        self.input_path_var = ctk.StringVar(value=self.current_settings.get('input_path', 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "input")))
        input_path_entry = ctk.CTkEntry(input_path_frame, textvariable=self.input_path_var)
        input_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        browse_button = ctk.CTkButton(input_path_frame, text="Browse", command=lambda: self.browse_directory('input'))
        browse_button.pack(side="right")
        
        # Default output path
        ctk.CTkLabel(storage_frame, text="Default Output Path:").pack(anchor="w", padx=10)
        output_path_frame = ctk.CTkFrame(storage_frame)
        output_path_frame.pack(fill="x", padx=10, pady=5)
        
        self.output_path_var = ctk.StringVar(value=self.current_settings.get('output_path',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output")))
        output_path_entry = ctk.CTkEntry(output_path_frame, textvariable=self.output_path_var)
        output_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        browse_button = ctk.CTkButton(output_path_frame, text="Browse", command=lambda: self.browse_directory('output'))
        browse_button.pack(side="right")
        
        # Organization options
        ctk.CTkLabel(storage_frame, text="Organize Output By:").pack(anchor="w", padx=10)
        organization_frame = ctk.CTkFrame(storage_frame)
        organization_frame.pack(fill="x", padx=10, pady=5)
        
        self.organization_var = ctk.StringVar(value=self.current_settings.get('organization_type', 'date'))
        ctk.CTkRadioButton(organization_frame, text="Date", variable=self.organization_var, value="date").pack(side="left", padx=10)
        ctk.CTkRadioButton(organization_frame, text="Location", variable=self.organization_var, value="location").pack(side="left", padx=10)
        
        # Auto-delete option
        self.auto_delete_var = ctk.BooleanVar(value=self.current_settings.get('auto_delete_raw', False))
        auto_delete_check = ctk.CTkCheckBox(
            storage_frame,
            text="Auto-delete raw images after processing",
            variable=self.auto_delete_var
        )
        auto_delete_check.pack(anchor="w", padx=10, pady=5)
        
    def create_detection_settings(self):
        """Create detection settings section"""
        # Detection section
        detection_frame = ctk.CTkFrame(self.main_frame)
        detection_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        ctk.CTkLabel(detection_frame, text="Detection Settings", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Confidence threshold
        ctk.CTkLabel(detection_frame, text="Confidence Threshold:").pack(anchor="w", padx=10)
        self.confidence_var = ctk.DoubleVar(value=self.current_settings.get('confidence_threshold', 0.5))
        confidence_scale = ctk.CTkSlider(
            detection_frame,
            from_=0.0,
            to=1.0,
            variable=self.confidence_var,
            number_of_steps=100
        )
        confidence_scale.pack(fill="x", padx=10, pady=5)
        
        # Confidence value label
        self.confidence_label = ctk.CTkLabel(detection_frame, text=f"Value: {self.confidence_var.get():.2f}")
        self.confidence_label.pack(anchor="w", padx=10)
        self.confidence_var.trace_add('write', self.update_confidence_label)
        
        # Class filter
        ctk.CTkLabel(detection_frame, text="Detect Classes:").pack(anchor="w", padx=10)
        self.class_vars = {}
        classes = ['Linear-Crack', 'Alligator-Crack', 'pothole']
        for class_name in classes:
            var = ctk.BooleanVar(value=self.current_settings.get(f'detect_{class_name}', True))
            self.class_vars[class_name] = var
            ctk.CTkCheckBox(
                detection_frame,
                text=class_name.replace('_', ' ').title(),
                variable=var
            ).pack(anchor="w", padx=10, pady=2)
            
    def create_cloud_settings(self):
        """Create cloud settings section"""
        # Cloud section
        cloud_frame = ctk.CTkFrame(self.main_frame)
        cloud_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        ctk.CTkLabel(cloud_frame, text="Cloud Settings", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Auto-upload
        self.auto_upload_var = ctk.BooleanVar(value=self.current_settings.get('auto_upload', False))
        auto_upload_check = ctk.CTkCheckBox(
            cloud_frame,
            text="Auto-upload detections to cloud",
            variable=self.auto_upload_var
        )
        auto_upload_check.pack(anchor="w", padx=10, pady=5)
        
    def update_confidence_label(self, *args):
        """Update the confidence threshold label"""
        self.confidence_label.configure(text=f"Value: {self.confidence_var.get():.2f}")
        
    def browse_directory(self, path_type):
        """Open directory dialog to select path"""
        directory = filedialog.askdirectory(title=f"Select {path_type.title()} Directory")
        if directory:
            if path_type == 'input':
                self.input_path_var.set(directory)
            else:
                self.output_path_var.set(directory)
                
    def save_settings(self):
        """Save settings"""
        try:
            # Create settings dictionary
            settings = {
                'input_path': self.input_path_var.get(),
                'output_path': self.output_path_var.get(),
                'organization_type': self.organization_var.get(),
                'auto_delete_raw': self.auto_delete_var.get(),
                'confidence_threshold': self.confidence_var.get(),
                'auto_upload': self.auto_upload_var.get()
            }
            
            # Add class detection settings
            for class_name, var in self.class_vars.items():
                settings[f'detect_{class_name}'] = var.get()
            
            # Save settings to file
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            # Update environment variables
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".env")
            env_content = []
            
            # Read existing .env file if it exists
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    env_content = f.readlines()
            
            # Update or add settings to .env
            env_settings = {
                'INPUT_PATH': self.input_path_var.get(),
                'OUTPUT_PATH': self.output_path_var.get(),
                'ORGANIZATION_TYPE': self.organization_var.get(),
                'AUTO_DELETE_RAW': str(self.auto_delete_var.get()).lower(),
                'CONFIDENCE_THRESHOLD': str(self.confidence_var.get()),
                'AUTO_UPLOAD': str(self.auto_upload_var.get()).lower()
            }
            
            # Add class detection settings
            for class_name, var in self.class_vars.items():
                env_settings[f'DETECT_{class_name.upper()}'] = str(var.get()).lower()
            
            # Update .env content
            new_env_content = []
            for line in env_content:
                key = line.split('=')[0].strip() if '=' in line else ''
                if key not in env_settings:
                    new_env_content.append(line)
            
            # Add new settings
            for key, value in env_settings.items():
                new_env_content.append(f"{key}={value}\n")
            
            # Write updated .env file
            with open(env_path, 'w') as f:
                f.writelines(new_env_content)
            
            # Reload environment variables
            load_dotenv(env_path, override=True)
            
            # Update parent window settings
            if hasattr(self.parent, 'update_settings'):
                self.parent.update_settings(settings)
            
            # Show success message
            messagebox.showinfo("Success", "Settings saved successfully")
            
            # Close window
            self.destroy()
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            
    def center_window(self):
        """Center the window on the screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}') 