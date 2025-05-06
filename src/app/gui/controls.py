import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os

class ControlPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        # Register for GPS callbacks
        if hasattr(self.parent, 'gps_reader'):
            self.parent.gps_reader.add_callback(self.on_gps_update)

    def setup_ui(self):
        # Create control buttons
        self.inference_button = ctk.CTkButton(
            self, 
            text="Start Inferencing",
            command=self.toggle_inferencing
        )
        self.inference_button.pack(pady=5, padx=10, fill="x")

        self.upload_button = ctk.CTkButton(
            self,
            text="Upload to Cloud",
            command=self.parent.open_upload_window
        )
        self.upload_button.pack(pady=5, padx=10, fill="x")

        self.severity_button = ctk.CTkButton(
            self,
            text="Calculate Severity",
            command=self.parent.calculate_severity
        )
        self.severity_button.pack(pady=5, padx=10, fill="x")

        # Add GPS connection button
        self.gps_button = ctk.CTkButton(
            self,
            text="Connect GPS",
            command=self.toggle_gps_connection
        )
        self.gps_button.pack(pady=5, padx=10, fill="x")

        self.settings_button = ctk.CTkButton(
            self,
            text="Settings",
            command=self.parent.open_settings_window
        )
        self.settings_button.pack(pady=5, padx=10, fill="x")

        self.exit_button = ctk.CTkButton(
            self,
            text="Exit",
            command=self.parent.on_closing,
            fg_color="#ff4444",
            hover_color="#cc0000"
        )
        self.exit_button.pack(pady=5, padx=10, fill="x") 

    def on_gps_update(self, gps_data, error=None):
        """Handle GPS updates and errors"""
        if error:
            # Show error and reset button state
            messagebox.showerror("GPS Error", error)
            self.gps_button.configure(text="Connect GPS", fg_color=["#3B8ED0", "#1F6AA5"])
        elif gps_data and self.parent.gps_reader.has_fix:
            self.gps_button.configure(text="Connected", fg_color="gray")

    def toggle_inferencing(self):
        """Toggle the inferencing state"""
        # Simply toggle the inference state
        self.parent.is_inferencing = not self.parent.is_inferencing
        
        # Update button text
        if self.parent.is_inferencing:
            self.inference_button.configure(text="Stop Inferencing")
        else:
            self.inference_button.configure(text="Start Inferencing")

    def toggle_gps_connection(self):
        """Toggle GPS connection"""
        if not hasattr(self.parent, 'gps_reader'):
            messagebox.showerror("Error", "GPS reader not initialized")
            return
            
        if self.parent.gps_reader.is_connected():
            # Disconnect GPS
            self.parent.gps_reader.stop()
            self.gps_button.configure(text="Connect GPS", fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            # Try to connect to GPS
            self.gps_button.configure(text="Trying to connect...", fg_color=["#3B8ED0", "#1F6AA5"])
            if not self.parent.gps_reader.connect_manually('COM3'):
                # If connection fails immediately, reset button state
                self.gps_button.configure(text="Connect GPS", fg_color=["#3B8ED0", "#1F6AA5"]) 