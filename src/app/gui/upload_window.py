import customtkinter as ctk
import json
import os
from tkinter import messagebox
import threading
import time
from dotenv import load_dotenv
import requests
from google.cloud import storage

class UploadWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Cloud Upload")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Load environment variables
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
        
        # Setup UI
        self.setup_ui()
        
        # Make window modal
        self.transient(parent)
        self.grab_set()
        
        # Check cloud connection status
        self.check_cloud_status()
        
        # Center window
        self.center_window()
        
    def center_window(self):
        """Center the window on the screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
    def setup_ui(self):
        # Create main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Cloud Status
        status_frame = ctk.CTkFrame(main_frame)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(status_frame, text="Cloud Connection Status", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Status indicator
        self.status_frame = ctk.CTkFrame(status_frame)
        self.status_frame.pack(fill="x", padx=5, pady=5)
        
        self.status_indicator = ctk.CTkLabel(self.status_frame, text="●", font=("Arial", 20))
        self.status_indicator.pack(side="left", padx=5)
        
        self.status_text = ctk.CTkLabel(self.status_frame, text="Checking connection...")
        self.status_text.pack(side="left", padx=5)
        
        # Connection Details
        details_frame = ctk.CTkFrame(main_frame)
        details_frame.pack(fill="x", padx=5, pady=5)
        
        self.details_text = ctk.CTkTextbox(details_frame, height=100)
        self.details_text.pack(fill="x", padx=5, pady=5)
        self.details_text.configure(state="disabled")
        
        # Upload Progress
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready to upload")
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        self.progress_bar.set(0)
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        refresh_button = ctk.CTkButton(
            button_frame, 
            text="Refresh Status", 
            command=self.check_cloud_status,
            fg_color="gray",
            hover_color="darkgray"
        )
        refresh_button.pack(side="left", padx=5, pady=10, expand=True)
        
        self.upload_button = ctk.CTkButton(
            button_frame, 
            text="Upload", 
            command=self.start_upload,
            state="disabled"
        )
        self.upload_button.pack(side="right", padx=5, pady=10, expand=True)
        
    def check_cloud_status(self):
        """Check cloud connection status"""
        try:
            # Reset status
            self.status_indicator.configure(text="●", text_color="gray")
            self.status_text.configure(text="Checking connection...")
            self.details_text.configure(state="normal")
            self.details_text.delete("1.0", "end")
            self.details_text.configure(state="disabled")
            self.upload_button.configure(state="disabled")
            
            # Check credentials file
            credentials_path = os.path.join(os.path.dirname(__file__), "..", "config", "credentials.json")
            if not os.path.exists(credentials_path):
                self.update_status("error", "Credentials file not found", 
                                 f"Please ensure credentials.json exists at:\n{credentials_path}")
                return
                
            # Try to read credentials
            try:
                with open(credentials_path, 'r') as f:
                    creds = json.load(f)
                    project_id = creds.get('project_id', 'Unknown')
                    client_email = creds.get('client_email', 'Unknown')
            except Exception as e:
                self.update_status("error", "Invalid credentials file", str(e))
                return
                
            # Check cloud connection
            try:
                # Initialize storage client
                client = storage.Client.from_service_account_json(credentials_path)
                
                # Get bucket name from environment
                bucket_name = os.getenv('GOOGLE_CLOUD_BUCKET_NAME')
                if not bucket_name:
                    self.update_status("error", "Missing bucket configuration", 
                                     "GOOGLE_CLOUD_BUCKET_NAME not found in environment variables")
                    return
                
                # Try to access bucket
                bucket = client.bucket(bucket_name)
                if not bucket.exists():
                    self.update_status("error", "Bucket not found", 
                                     f"Bucket '{bucket_name}' does not exist or is not accessible")
                    return
                
                # If we get here, everything is working
                self.update_status("success", "Connected to Cloud Storage", 
                                 f"Project: {project_id}\nAccount: {client_email}\nBucket: {bucket_name}")
                
                # Enable upload button
                self.upload_button.configure(state="normal")
                
            except Exception as e:
                self.update_status("error", "Cloud connection failed", str(e))
            
        except Exception as e:
            self.update_status("error", "Connection check failed", str(e))
            
    def update_status(self, status, message, details=""):
        """Update status indicators"""
        if status == "success":
            self.status_indicator.configure(text="●", text_color="green")
        elif status == "error":
            self.status_indicator.configure(text="●", text_color="red")
        else:
            self.status_indicator.configure(text="●", text_color="gray")
            
        self.status_text.configure(text=message)
        
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", details)
        self.details_text.configure(state="disabled")
        
    def start_upload(self):
        """Start the upload process in a separate thread"""
        # Start upload in a separate thread
        upload_thread = threading.Thread(target=self.upload_process)
        upload_thread.daemon = True
        upload_thread.start()
        
    def upload_process(self):
        """Simulate upload process (replace with actual upload logic)"""
        try:
            self.progress_label.configure(text="Uploading...")
            self.progress_bar.set(0)
            
            # Simulate upload progress
            for i in range(101):
                time.sleep(0.05)  # Simulate work
                self.progress_bar.set(i/100)
                self.progress_label.configure(text=f"Uploading... {i}%")
                self.update()
                
            self.progress_label.configure(text="Upload complete!")
            messagebox.showinfo("Success", "Upload completed successfully!")
            
        except Exception as e:
            self.progress_label.configure(text="Upload failed")
            messagebox.showerror("Error", f"Upload failed: {e}")
            
    def __del__(self):
        """Clean up resources"""
        try:
            # Release any resources if needed
            pass
        except Exception as e:
            print(f"Error during cleanup: {e}") 