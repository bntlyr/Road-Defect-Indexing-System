import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import psutil
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import platform
import os
from prediction import DefectDetector
from severity_calculator import SeverityCalculator
from tkinter import filedialog
from getGPS import find_gps_port, read_gps
import serial
import pynmea2


matplotlib.use('Agg')

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Initialize GPS-related variables
        self.gps_port = None
        self.gps_thread = None
        self.gps_data = {"latitude": None, "longitude": None}
        self.gps_running = False

        # Initialize FPS-related variables first
        self.fps = 30  # Default FPS
        self.available_fps = [15, 30, 60]  # Available FPS options
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0

        self.title("Road Defect Detection")
        self.geometry("1400x800")
        self.minsize(1100, 700)
        self.roi_visible = True
        self.layout_switched = False
        self.inferencing = False
        self.roi_selecting = False

        self.current_camera_index = 0
        self.cameras = self.list_available_cameras()
        
        # Initialize camera resolution defaults
        self.camera_width = 640  # Default width
        self.camera_height = 480  # Default height

        # Initialize capture object
        self.cap = self.open_camera(self.current_camera_index)
        
        # Get actual camera resolution if available
        if self.cap and self.cap.isOpened():
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configure main window grid
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=12)  # Adjusted weight for top section
        self.grid_rowconfigure(1, weight=2)   # Adjusted weight for bottom section
        self.grid_rowconfigure(2, weight=0)   # Status bar

        # --- Top Section ---
        self.top_container = ctk.CTkFrame(self, corner_radius=10)
        self.top_container.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.top_container.grid_columnconfigure((0, 1), weight=1)
        self.top_container.grid_rowconfigure(0, weight=0)
        self.top_container.grid_rowconfigure(1, weight=1)

        # Section Description for Video Stream
        self.video_stream_label = ctk.CTkLabel(
            self.top_container, text="Video Stream", font=("Arial", 14, "bold"), anchor="w"
        )
        self.video_stream_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))

        self.video_stream_frame = ctk.CTkFrame(self.top_container, corner_radius=10)
        self.video_stream_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 5))
        self.video_stream_frame.grid_rowconfigure(0, weight=1)
        self.video_stream_frame.grid_columnconfigure(0, weight=1)

        # Section Description for Map View
        self.map_view_label = ctk.CTkLabel(
            self.top_container, text="Map View", font=("Arial", 14, "bold"), anchor="w"
        )
        self.map_view_label.grid(row=0, column=1, sticky="w", padx=10, pady=(5, 0))

        self.map_view_frame = ctk.CTkFrame(self.top_container, corner_radius=10)
        self.map_view_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 5))

        # --- Bottom Section ---
        self.bottom_frame = ctk.CTkFrame(self, corner_radius=10)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=(0, 5))
        self.bottom_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=0)
        self.bottom_frame.grid_rowconfigure(1, weight=1)

        self.video_controls_label = ctk.CTkLabel(
            self.bottom_frame, text="Video Controls", font=("Arial", 14, "bold"), anchor="w"
        )
        self.video_controls_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))

        self.video_control_frame = ctk.CTkFrame(self.bottom_frame, corner_radius=10)
        self.video_control_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 5))

        self.defect_log_label = ctk.CTkLabel(
            self.bottom_frame, text="Defect Log", font=("Arial", 14, "bold"), anchor="w"
        )
        self.defect_log_label.grid(row=0, column=1, sticky="w", padx=10, pady=(5, 0))

        self.defect_log_frame = ctk.CTkFrame(self.bottom_frame, corner_radius=10)
        self.defect_log_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 5))

        self.control_settings_label = ctk.CTkLabel(
            self.bottom_frame, text="Control Settings", font=("Arial", 14, "bold"), anchor="w"
        )
        self.control_settings_label.grid(row=0, column=2, sticky="w", padx=10, pady=(5, 0))

        self.control_settings_frame = ctk.CTkFrame(self.bottom_frame, corner_radius=10)
        self.control_settings_frame.grid(row=1, column=2, sticky="nsew", padx=10, pady=(0, 5))

        # --- Status Bar ---
        self.status_bar = ctk.CTkLabel(self, text="🟥 00:00:00 | GPU: 0% | CPU: 0%", anchor="w", height=30)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # --- Setup Widgets ---
        self.setup_video_stream()
        self.setup_map_view()
        self.setup_video_controls()
        self.setup_defect_logs()
        self.setup_control_settings()

        # --- Variables ---
        self.is_streaming = True
        self.roi_points = []

        # Add detection-related variables
        self.detector = None
        self.default_save_dir = os.path.join(os.path.expanduser("~"), "RoadDefectDetections")
        self.current_save_dir = self.default_save_dir

        # Start streams
        self.update_status_bar()
        self.video_loop()

        # Configure height for bottom frame
        self.bottom_frame.configure(height=200)
        self.bottom_frame.grid_propagate(False)

    def setup_video_stream(self):
        self.video_canvas = tk.Canvas(self.video_stream_frame, bg="black", highlightthickness=2, highlightbackground="white")
        self.video_canvas.pack(expand=True, fill="both", padx=10, pady=10)
        self.video_canvas.bind("<Button-3>", self.select_roi_point)
        
        # Configure the canvas to expand and fill its container
        self.video_stream_frame.grid_rowconfigure(0, weight=1)
        self.video_stream_frame.grid_columnconfigure(0, weight=1)

    def setup_map_view(self):
        self.map_label = ctk.CTkLabel(self.map_view_frame, text="Map View (Placeholder)", anchor="center")
        self.map_label.pack(expand=True, fill="both", padx=10, pady=10)

    def setup_video_controls(self):
        available_cameras = list(self.cameras.values())
        if not available_cameras:
            available_cameras = ["No Camera Found"]

        self.video_device_combo = ctk.CTkComboBox(self.video_control_frame, values=available_cameras, command=self.switch_camera)
        self.video_device_combo.set(available_cameras[0])
        self.video_device_combo.pack(pady=5, padx=10, fill="x")

        # Add resolution selection with dynamic values
        self.resolution_combo = ctk.CTkComboBox(
            self.video_control_frame, 
            values=self.available_resolutions,
            command=self.change_resolution
        )
        self.resolution_combo.set(f"{self.camera_width}x{self.camera_height}")
        self.resolution_combo.pack(pady=5, padx=10, fill="x")

        # Add FPS selection
        self.fps_combo = ctk.CTkComboBox(
            self.video_control_frame,
            values=[str(fps) for fps in self.available_fps],
            command=self.change_fps
        )
        self.fps_combo.set(str(self.fps))
        self.fps_combo.pack(pady=5, padx=10, fill="x")

        # Add GPS controls
        self.gps_frame = ctk.CTkFrame(self.video_control_frame, fg_color="transparent")
        self.gps_frame.pack(pady=5, padx=10, fill="x")
        
        self.gps_status_label = ctk.CTkLabel(self.gps_frame, text="GPS: Not Connected", font=("Arial", 12))
        self.gps_status_label.pack(side="left", padx=(0, 5))
        
        self.gps_button = ctk.CTkButton(self.gps_frame, text="Connect GPS", command=self.toggle_gps, width=100)
        self.gps_button.pack(side="right")

        self.roi_controls_frame = ctk.CTkFrame(self.video_control_frame, fg_color="transparent")
        self.roi_controls_frame.pack(pady=5, padx=10, fill="x")

        self.roi_button = ctk.CTkButton(self.roi_controls_frame, text="Start ROI Selection", command=self.activate_roi_mode)
        self.roi_button.pack(side="left", expand=True, fill="x", padx=(0,5))

        self.roi_eye_button = ctk.CTkButton(self.roi_controls_frame, text="👁️", width=50, command=self.toggle_roi_visibility)
        self.roi_eye_button.pack(side="left")

        self.switch_layout_button = ctk.CTkButton(self.video_control_frame, text="Switch Layout", command=self.switch_layout)
        self.switch_layout_button.pack(pady=5, padx=10, fill="x")

    def setup_defect_logs(self):
        # Create a frame to hold all defect charts
        self.defect_charts_frame = ctk.CTkFrame(self.defect_log_frame, fg_color="transparent")
        self.defect_charts_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Initialize defect data (simplified to only track totals)
        self.defect_data = {
            "Linear-Crack": 0,
            "Alligator-Crack": 0,
            "Pothole": 0,
            "Patch": 0
        }
        
        # Create charts for each defect type
        self.defect_charts = {}
        for i, (defect_type, count) in enumerate(self.defect_data.items()):
            # Create frame for each chart
            chart_frame = ctk.CTkFrame(self.defect_charts_frame, fg_color="transparent")
            chart_frame.grid(row=0, column=i, padx=5, pady=2, sticky="nsew")
            
            # Add title label above chart
            title_label = ctk.CTkLabel(chart_frame, text=defect_type, font=("Arial", 11, "bold"))
            title_label.pack(pady=(0, 2))
            
            # Create figure and axis with adjusted size
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0, hspace=0)
            fig.patch.set_facecolor('#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            # Create simple pie chart with one segment
            sizes = [1]
            colors = ['#1f77b4']
            ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5))
            
            # Add total count in center
            ax.text(0, 0, "0", ha='center', va='center', fontsize=12, color='white')
            
            # Create canvas and add to frame
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill="both", padx=2)
            
            self.defect_charts[defect_type] = {
                "fig": fig,
                "ax": ax,
                "canvas": canvas,
                "title_label": title_label
            }
            
            # Configure grid weights
            self.defect_charts_frame.grid_columnconfigure(i, weight=1)

    def update_defect_statistics(self, frame_counts, gps_data=None):
        """Update the defect statistics in the GUI with frame-specific counts"""
        # Color mapping for different defect types (BGR format)
        defect_colors = {
            'Linear-Crack': (0, 255, 0),      # Green
            'Alligator-Crack': (255, 0, 0),   # Blue
            'Pothole': (0, 0, 255),          # Red
            'Patch': (255, 255, 0)           # Cyan
        }
        
        # Update the charts with new counts
        for defect_type, chart_data in self.defect_charts.items():
            count = frame_counts.get(defect_type, 0)
            
            # Clear previous chart
            chart_data["ax"].clear()
            
            # Get color for this defect type (convert BGR to RGB)
            color = defect_colors[defect_type]
            rgb_color = (color[2]/255, color[1]/255, color[0]/255)  # Convert to RGB and normalize
            
            # Create simple pie chart with one segment
            sizes = [1]
            colors = [rgb_color]
            chart_data["ax"].pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5))
            
            # Update total count in center
            chart_data["ax"].text(0, 0, str(count), ha='center', va='center', fontsize=12, color='white')
            
            # Set background color
            chart_data["ax"].set_facecolor('#2b2b2b')
            chart_data["fig"].patch.set_facecolor('#2b2b2b')
            
            # Update canvas
            chart_data["canvas"].draw()
        
        # Update status bar with GPS info if available
        if gps_data and gps_data["latitude"] is not None and gps_data["longitude"] is not None:
            gps_text = f"GPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}"
            self.status_bar.configure(text=f"🟥 {time.strftime('%H:%M:%S')} | {gps_text}")

    def setup_control_settings(self):
        self.inference_button = ctk.CTkButton(self.control_settings_frame, text="Start Inferencing", command=self.toggle_inferencing)
        self.inference_button.pack(pady=5, padx=10, fill="x")

        self.upload_button = ctk.CTkButton(self.control_settings_frame, text="Upload to Cloud", command=self.open_upload_window)
        self.upload_button.pack(pady=5, padx=10, fill="x")

        self.severity_button = ctk.CTkButton(self.control_settings_frame, text="Calculate Severity", command=self.calculate_severity)
        self.severity_button.pack(pady=5, padx=10, fill="x")

        self.settings_button = ctk.CTkButton(self.control_settings_frame, text="Settings", command=self.open_settings_window)
        self.settings_button.pack(pady=5, padx=10, fill="x")

        self.exit_button = ctk.CTkButton(self.control_settings_frame, text="Exit", command=self.quit, fg_color="#ff4444", hover_color="#cc0000")
        self.exit_button.pack(pady=5, padx=10, fill="x")

        # Initialize severity calculator
        self.severity_calculator = SeverityCalculator(
            camera_width=self.camera_width,
            camera_height=self.camera_height
        )
        self.last_detection = None

    def open_camera(self, index):
        try:
            # Try with the default backend
            cap = cv2.VideoCapture(index)

            # If the default backend doesn't work, try MSMF (Windows)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

            # Try DirectShow backend (Windows)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

            # Finally, try the V4L2 backend (Linux)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

            # If none of the backends work
            if not cap.isOpened():
                raise Exception(f"Camera {index} cannot be opened.")

            # Test common resolutions
            test_resolutions = [
                (1920, 1080),  # Full HD
                (1280, 720),   # HD
                (640, 480),    # VGA
                (320, 240),    # QVGA
                (160, 120)     # QQVGA
            ]

            # Store available resolutions
            self.available_resolutions = []
            current_width = 0
            current_height = 0

            for width, height in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Get actual resolution that was set
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # If the resolution was set successfully
                if actual_width == width and actual_height == height:
                    resolution_str = f"{width}x{height}"
                    if resolution_str not in self.available_resolutions:
                        self.available_resolutions.append(resolution_str)
                    
                    # Keep track of the highest resolution
                    if width * height > current_width * current_height:
                        current_width = width
                        current_height = height

            # Set to highest available resolution
            if current_width > 0 and current_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
                print(f"Camera resolution set to: {current_width}x{current_height}")

            # Set other camera properties for better quality
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
            cap.set(cv2.CAP_PROP_FPS, self.fps)  # Set FPS based on user selection

            return cap
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def switch_camera(self, camera_name):
        for idx, name in self.cameras.items():
            if name == camera_name:
                self.current_camera_index = idx
                break
        if self.cap:
            self.cap.release()
        self.cap = self.open_camera(self.current_camera_index)
        
        # Update resolution combobox with current resolution and available resolutions
        if self.cap and self.cap.isOpened():
            current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_resolution = f"{current_width}x{current_height}"
            
            # Update the resolution combo box with available resolutions
            self.resolution_combo.configure(values=self.available_resolutions)
            self.resolution_combo.set(current_resolution)
            
            # Update stored camera resolution
            self.camera_width = current_width
            self.camera_height = current_height

    def list_available_cameras(self):
        available_cameras = {}
        system = platform.system()

        index = 0
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if system == "Windows" else 0)
            if not cap.isOpened():
                cap.release()
                break

            name = f"Camera {index}"
            if system == "Windows":
                name = self.get_camera_name_windows(index)

            available_cameras[index] = name
            cap.release()
            index += 1

        return available_cameras


    def get_camera_name_windows(self, index):
        try:
            output = subprocess.check_output('powershell "Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -like \'*camera*\' } | Select-Object -ExpandProperty Name"', shell=True)
            names = output.decode().strip().split('\r\n')
            if index < len(names):
                return names[index]
        except:
            pass
        return f"Camera {index}"

    def video_loop(self):
        if self.is_streaming:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Update camera resolution in case it changed
                    self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process frame with detector if inferencing is active
                    if self.inferencing and self.detector:
                        # Process the frame to get detections
                        frame, defect_counts = self.detector.process_frame(
                            frame, 
                            self.roi_points if self.roi_points else None,
                            self.gps_data if self.gps_running else None
                        )
                        
                        # Update defect charts immediately after processing
                        self.update_defect_statistics(defect_counts, self.gps_data if self.gps_running else None)

                    # Get canvas size
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()

                    # Calculate scaling factors while maintaining aspect ratio
                    scale_x = canvas_width / self.camera_width
                    scale_y = canvas_height / self.camera_height
                    
                    # Use the smaller scale to ensure the entire image fits
                    scale = min(scale_x, scale_y)
                    
                    # Calculate new dimensions
                    new_width = int(self.camera_width * scale)
                    new_height = int(self.camera_height * scale)
                    
                    # Resize the frame
                    if new_width > 0 and new_height > 0:
                        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    else:
                        frame_resized = frame

                    img_pil = Image.fromarray(frame_resized)

                    # Draw ROI points overlay if visible
                    if self.roi_points and self.roi_visible:
                        draw = ImageDraw.Draw(img_pil)
                        scale_x = new_width / self.camera_width
                        scale_y = new_height / self.camera_height

                        if len(self.roi_points) >= 2:
                            for i in range(len(self.roi_points)-1):
                                p1 = (self.roi_points[i][0] * scale_x, self.roi_points[i][1] * scale_y)
                                p2 = (self.roi_points[i+1][0] * scale_x, self.roi_points[i+1][1] * scale_y)
                                draw.line([p1, p2], fill="red", width=3)
                        if len(self.roi_points) == 4:
                            p1 = (self.roi_points[3][0] * scale_x, self.roi_points[3][1] * scale_y)
                            p2 = (self.roi_points[0][0] * scale_x, self.roi_points[0][1] * scale_y)
                            draw.line([p1, p2], fill="red", width=3)

                        for pt in self.roi_points:
                            x, y = pt[0] * scale_x, pt[1] * scale_y
                            draw.ellipse((x-5, y-5, x+5, y+5), fill="red")

                    self.current_image = ImageTk.PhotoImage(img_pil)

                    # Calculate centering offsets
                    offset_x = (canvas_width - new_width) // 2
                    offset_y = (canvas_height - new_height) // 2

                    # Clear canvas and draw the image centered
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(offset_x, offset_y, anchor="nw", image=self.current_image)

                    # Update FPS counter
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_update >= 1.0:  # Update FPS every second
                        self.current_fps = self.frame_count / (current_time - self.last_fps_update)
                        self.frame_count = 0
                        self.last_fps_update = current_time

                else:
                    print("[WARNING] Frame read failed, trying to reconnect...")
                    self.cap.release()
                    self.cap = self.open_camera(self.current_camera_index)

            else:
                print("[WARNING] Camera not opened, retrying...")
                self.cap = self.open_camera(self.current_camera_index)

            # Calculate delay based on target FPS
            delay = max(1, int(1000 / self.fps))
            self.after(delay, self.video_loop)

    def activate_roi_mode(self):
        if self.roi_points and not self.roi_selecting:
            self.roi_points.clear()
            self.roi_button.configure(text="Start ROI Selection")
        else:
            self.roi_points.clear()
            self.roi_selecting = True
            self.roi_button.configure(text="Selecting... (Right Click)")

    def select_roi_point(self, event):
        if self.roi_selecting and len(self.roi_points) < 4:
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            # Calculate frame size and offsets based on aspect ratio
            frame_aspect = self.camera_width / self.camera_height
            canvas_aspect = canvas_width / canvas_height

            if frame_aspect > canvas_aspect:
                new_width = canvas_width
                new_height = int(canvas_width / frame_aspect)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * frame_aspect)

            offset_x = (canvas_width - new_width) // 2
            offset_y = (canvas_height - new_height) // 2

            # Adjust click coordinates relative to the frame
            x = event.x - offset_x
            y = event.y - offset_y

            if 0 <= x <= new_width and 0 <= y <= new_height:
                # Convert coordinates back to original camera resolution
                real_x = int(x * self.camera_width / new_width)
                real_y = int(y * self.camera_height / new_height)
                self.roi_points.append((real_x, real_y))
                if len(self.roi_points) == 4:
                    self.roi_selecting = False
                    self.roi_button.configure(text="Clear ROI Selection")

    def toggle_roi_visibility(self):
        self.roi_visible = not self.roi_visible
        self.roi_eye_button.configure(text="👁️" if self.roi_visible else "🚫")

    def open_upload_window(self):
        upload_win = tk.Toplevel(self)
        upload_win.title("Upload To Cloud")
        upload_win.geometry("300x200")
        tk.Label(upload_win, text="Upload to Cloud Feature").pack(pady=20)

    def open_settings_window(self):
        settings_win = ctk.CTkToplevel(self)
        settings_win.title("Settings")
        settings_win.geometry("400x300")
        settings_win.attributes('-topmost', True)  # Make window always on top
        
        # Configure grid
        settings_win.grid_columnconfigure(0, weight=1)
        settings_win.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        settings_frame = ctk.CTkFrame(settings_win, corner_radius=10)
        settings_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        settings_frame.grid_columnconfigure(0, weight=1)
        
        # Save directory settings
        save_dir_label = ctk.CTkLabel(settings_frame, text="Save Directory", font=("Arial", 14, "bold"))
        save_dir_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        
        save_dir_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        save_dir_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        save_dir_frame.grid_columnconfigure(0, weight=1)
        
        self.save_dir_entry = ctk.CTkEntry(save_dir_frame, placeholder_text=self.current_save_dir)
        self.save_dir_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        browse_button = ctk.CTkButton(save_dir_frame, text="Browse", command=self.browse_save_dir, width=80)
        browse_button.grid(row=0, column=1)
        
        # Save button
        save_button = ctk.CTkButton(settings_frame, text="Save Settings", command=lambda: self.save_settings(settings_win))
        save_button.grid(row=2, column=0, pady=20)

    def browse_save_dir(self):
        directory = filedialog.askdirectory(initialdir=self.current_save_dir)
        if directory:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, directory)

    def save_settings(self, settings_win):
        new_save_dir = self.save_dir_entry.get()
        if new_save_dir:
            self.current_save_dir = new_save_dir
            os.makedirs(self.current_save_dir, exist_ok=True)
            if self.detector:
                self.detector.save_dir = self.current_save_dir
        settings_win.destroy()

    def toggle_inferencing(self):
        self.inferencing = not self.inferencing
        self.inference_button.configure(text="Stop Inferencing" if self.inferencing else "Start Inferencing")
        
        if self.inferencing:
            # Initialize detector if not already initialized
            if not self.detector:
                model_path = "best.pt"  # Update this path to your model
                self.detector = DefectDetector(model_path, self.current_save_dir, self.update_defect_statistics)
        else:
            # Clean up detector
            self.detector = None

    def update_status_bar(self):
        cpu = psutil.cpu_percent()
        gpu = 0  # Placeholder for GPU usage
        now = time.strftime("%H:%M:%S")
        fps_text = f"FPS: {self.current_fps:.1f}" if self.current_fps > 0 else "FPS: --"
        self.status_bar.configure(text=f"🟥 {now} | {fps_text} | GPU: {gpu}% | CPU: {cpu}%")
        self.after(1000, self.update_status_bar)

    def switch_layout(self):
        if not self.layout_switched:
            # Remove frames and labels from grid
            self.video_stream_label.grid_forget()
            self.video_stream_frame.grid_forget()
            self.map_view_label.grid_forget()
            self.map_view_frame.grid_forget()
            
            # Re-add labels and frames in switched positions
            self.map_view_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
            self.map_view_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 5))
            
            self.video_stream_label.grid(row=0, column=1, sticky="w", padx=10, pady=(5, 0))
            self.video_stream_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 5))
        else:
            # Remove frames and labels from grid
            self.video_stream_label.grid_forget()
            self.video_stream_frame.grid_forget()
            self.map_view_label.grid_forget()
            self.map_view_frame.grid_forget()
            
            # Re-add labels and frames in original positions
            self.video_stream_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
            self.video_stream_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 5))
            
            self.map_view_label.grid(row=0, column=1, sticky="w", padx=10, pady=(5, 0))
            self.map_view_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 5))

        self.layout_switched = not self.layout_switched

    def change_resolution(self, resolution_str):
        if self.cap and self.cap.isOpened():
            width, height = map(int, resolution_str.split('x'))
            
            # Try to set the new resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify if the resolution was actually set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Update the combobox with the actual resolution
            actual_resolution = f"{actual_width}x{actual_height}"
            if actual_resolution != resolution_str:
                self.resolution_combo.set(actual_resolution)
            
            # Update stored camera resolution
            self.camera_width = actual_width
            self.camera_height = actual_height
            
            print(f"Camera resolution set to: {actual_width}x{actual_height}")

    def change_fps(self, fps_str):
        self.fps = int(fps_str)
        self.fps_combo.set(fps_str)

    def calculate_severity(self):
        # Close any existing severity window
        if hasattr(self, 'severity_window') and self.severity_window:
            self.severity_window.destroy()
        
        # Initialize detector if not already initialized
        if not self.detector:
            model_path = "best.pt"  # Update this path to your model
            self.detector = DefectDetector(model_path, self.current_save_dir, self.update_defect_statistics)
        
        # Show severity window
        self.show_severity_window(None, None)

    def show_severity_window(self, processed_image, severity_level):
        # Create a new window
        severity_win = ctk.CTkToplevel(self)
        severity_win.title("Severity Calculation")
        severity_win.geometry("1200x800")
        severity_win.attributes('-topmost', True)  # Make window always on top
        severity_win.focus_force()  # Force focus to the window
        
        # Configure grid
        severity_win.grid_columnconfigure(0, weight=1)
        severity_win.grid_rowconfigure(0, weight=0)  # Header
        severity_win.grid_rowconfigure(1, weight=1)  # Content
        severity_win.grid_rowconfigure(2, weight=0)  # Footer
        
        # Header
        header_frame = ctk.CTkFrame(severity_win, corner_radius=10)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ctk.CTkLabel(header_frame, text="Severity Calculation Process", font=("Arial", 20, "bold"))
        header_label.grid(row=0, column=0, pady=10)
        
        # Content
        content_frame = ctk.CTkFrame(severity_win, corner_radius=10)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=0)  # Process steps
        content_frame.grid_rowconfigure(1, weight=1)  # Image display
        
        # Process steps
        steps_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        steps_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        steps = [
            "1. Image Selection",
            "2. Defect Detection",
            "3. Area Calculation",
            "4. Pinhole Camera Model",
            "5. Severity Analysis",
            "6. Results Visualization"
        ]
        
        self.step_labels = []
        self.progress_bars = []
        
        for i, step in enumerate(steps):
            step_frame = ctk.CTkFrame(steps_frame, fg_color="transparent")
            step_frame.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            
            label = ctk.CTkLabel(step_frame, text=step, font=("Arial", 12))
            label.grid(row=0, column=0, sticky="w", padx=5)
            
            progress = ctk.CTkProgressBar(step_frame, width=200)
            progress.grid(row=0, column=1, sticky="ew", padx=5)
            progress.set(0)
            
            self.step_labels.append(label)
            self.progress_bars.append(progress)
        
        # Image display
        image_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create two canvas for main and sub images
        self.main_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=2, highlightbackground="white")
        self.main_canvas.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        
        self.sub_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=2, highlightbackground="white")
        self.sub_canvas.pack(side="right", expand=True, fill="both", padx=5, pady=5)
        
        # Add labels for the canvases
        main_label = ctk.CTkLabel(image_frame, text="Main Image", font=("Arial", 12, "bold"))
        main_label.place(relx=0.25, rely=0.02, anchor="center")
        
        sub_label = ctk.CTkLabel(image_frame, text="Sub Images", font=("Arial", 12, "bold"))
        sub_label.place(relx=0.75, rely=0.02, anchor="center")
        
        # Footer
        footer_frame = ctk.CTkFrame(severity_win, corner_radius=10)
        footer_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        footer_frame.grid_columnconfigure(0, weight=1)
        
        # Folder selection
        folder_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
        folder_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # Input folder selection
        input_folder_frame = ctk.CTkFrame(folder_frame, fg_color="transparent")
        input_folder_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        input_folder_frame.grid_columnconfigure(0, weight=1)
        
        input_label = ctk.CTkLabel(input_folder_frame, text="Input Folder:", font=("Arial", 12))
        input_label.grid(row=0, column=0, sticky="w", padx=5)
        
        self.folder_entry = ctk.CTkEntry(input_folder_frame, placeholder_text="Select folder with detected images")
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        
        input_browse_button = ctk.CTkButton(input_folder_frame, text="Browse", command=self.browse_severity_folder, width=80)
        input_browse_button.grid(row=0, column=2)
        
        # Output folder selection
        output_folder_frame = ctk.CTkFrame(folder_frame, fg_color="transparent")
        output_folder_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        output_folder_frame.grid_columnconfigure(0, weight=1)
        
        output_label = ctk.CTkLabel(output_folder_frame, text="Output Folder:", font=("Arial", 12))
        output_label.grid(row=0, column=0, sticky="w", padx=5)
        
        self.output_folder_entry = ctk.CTkEntry(output_folder_frame, placeholder_text="Select folder to save severity results")
        self.output_folder_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        
        output_browse_button = ctk.CTkButton(output_folder_frame, text="Browse", command=self.browse_output_folder, width=80)
        output_browse_button.grid(row=0, column=2)
        
        # Process button
        process_button = ctk.CTkButton(footer_frame, text="Process Images", command=self.process_severity_images)
        process_button.grid(row=2, column=0, pady=10)
        
        # Store window reference
        self.severity_window = severity_win
        
        # Set default folders
        self.folder_entry.insert(0, self.current_save_dir)
        self.output_folder_entry.insert(0, os.path.join(self.current_save_dir, "severity_results"))

    def browse_severity_folder(self):
        directory = filedialog.askdirectory(initialdir=self.current_save_dir)
        if directory:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, directory)
            # Force the severity window back to top
            self.severity_window.lift()
            self.severity_window.focus_force()

    def browse_output_folder(self):
        directory = filedialog.askdirectory(initialdir=self.current_save_dir)
        if directory:
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, directory)
            # Force the severity window back to top
            self.severity_window.lift()
            self.severity_window.focus_force()

    def process_severity_images(self):
        input_folder_path = self.folder_entry.get()
        output_folder_path = self.output_folder_entry.get()
        
        if not input_folder_path or not os.path.isdir(input_folder_path):
            tk.messagebox.showerror("Error", "Please select a valid input folder", parent=self.severity_window)
            return
            
        if not output_folder_path:
            tk.messagebox.showerror("Error", "Please select an output folder", parent=self.severity_window)
            return
            
        # Create output folder if it doesn't exist
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Get all image files in the folder
        image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            tk.messagebox.showerror("Error", "No images found in the selected folder", parent=self.severity_window)
            return
        
        # Process each image
        total_images = len(image_files)
        severity_stats = {
            "Low": 0,
            "Moderate": 0,
            "High": 0,
            "Critical": 0
        }
        
        # Create a CSV file to store severity results
        import csv
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_folder_path, f"severity_results_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Image Type', 'Image Name', 'Severity Level', 'Severity Category', 'Real-World Area (m²)'])
        
        # Process main images first
        main_images = [f for f in image_files if not any(f.startswith(prefix) for prefix in ['LC_', 'AC_', 'PH_', 'P_'])]
        sub_images = [f for f in image_files if any(f.startswith(prefix) for prefix in ['LC_', 'AC_', 'PH_', 'P_'])]
        
        total_images = len(main_images) + len(sub_images)
        processed_count = 0
        
        # Process main images
        for i, image_file in enumerate(main_images):
            # Update progress for each step
            for j in range(len(self.progress_bars)):
                self.progress_bars[j].set((i + j/len(self.progress_bars)) / total_images)
                self.step_labels[j].configure(text_color="green")
            self.severity_window.update()
            
            # Process the image
            image_path = os.path.join(input_folder_path, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get detection from the image
            if self.detector:
                # Process the frame to get detections
                processed_frame, defect_counts = self.detector.process_frame(frame_rgb)
                
                # If we have detections, calculate severity
                if hasattr(self.detector, 'last_detection') and self.detector.last_detection is not None:
                    # Calculate severity using the new calculator
                    severity_level, real_world_area = self.severity_calculator.calculate_severity(
                        frame_rgb,
                        self.detector.last_detection,
                        distance_to_object=1.0
                    )
                    
                    # Determine severity category
                    if severity_level < 0.25:
                        severity_category = "Low"
                        severity_stats["Low"] += 1
                    elif severity_level < 0.5:
                        severity_category = "Moderate"
                        severity_stats["Moderate"] += 1
                    elif severity_level < 0.75:
                        severity_category = "High"
                        severity_stats["High"] += 1
                    else:
                        severity_category = "Critical"
                        severity_stats["Critical"] += 1
                    
                    # Save the processed image with severity visualization
                    output_image_name = f"severity_{severity_category}_{image_file}"
                    output_image_path = os.path.join(output_folder_path, output_image_name)
                    cv2.imwrite(output_image_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    
                    # Write to CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([
                            "Main",
                            image_file,
                            f"{severity_level:.4f}",
                            severity_category,
                            f"{real_world_area:.4f}"
                        ])
                    
                    # Update the display
                    self.update_severity_display(processed_frame, is_main=True)
            
            processed_count += 1
        
        # Process sub images
        for i, image_file in enumerate(sub_images):
            # Update progress for each step
            for j in range(len(self.progress_bars)):
                self.progress_bars[j].set((processed_count + i + j/len(self.progress_bars)) / total_images)
                self.step_labels[j].configure(text_color="green")
            self.severity_window.update()
            
            # Process the image
            image_path = os.path.join(input_folder_path, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get detection from the image
            if self.detector:
                # Process the frame to get detections
                processed_frame, defect_counts = self.detector.process_frame(frame_rgb)
                
                # If we have detections, calculate severity
                if hasattr(self.detector, 'last_detection') and self.detector.last_detection is not None:
                    # Calculate severity using the new calculator
                    severity_level, real_world_area = self.severity_calculator.calculate_severity(
                        frame_rgb,
                        self.detector.last_detection,
                        distance_to_object=1.0
                    )
                    
                    # Determine severity category
                    if severity_level < 0.25:
                        severity_category = "Low"
                        severity_stats["Low"] += 1
                    elif severity_level < 0.5:
                        severity_category = "Moderate"
                        severity_stats["Moderate"] += 1
                    elif severity_level < 0.75:
                        severity_category = "High"
                        severity_stats["High"] += 1
                    else:
                        severity_category = "Critical"
                        severity_stats["Critical"] += 1
                    
                    # Save the processed image with severity visualization
                    output_image_name = f"severity_{severity_category}_{image_file}"
                    output_image_path = os.path.join(output_folder_path, output_image_name)
                    cv2.imwrite(output_image_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                    
                    # Write to CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([
                            "Sub",
                            image_file,
                            f"{severity_level:.4f}",
                            severity_category,
                            f"{real_world_area:.4f}"
                        ])
                    
                    # Update the display
                    self.update_severity_display(processed_frame, is_main=False)
            
            processed_count += 1
        
        # Show statistics
        self.show_severity_statistics(severity_stats, total_images)
        
        # Show completion message
        tk.messagebox.showinfo("Processing Complete", 
            f"Severity analysis completed!\n\n"
            f"Results saved to:\n{output_folder_path}\n\n"
            f"CSV file: severity_results_{timestamp}.csv",
            parent=self.severity_window)

    def update_severity_display(self, image, is_main=True):
        # Convert OpenCV image to PIL format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        # Get canvas size
        canvas = self.main_canvas if is_main else self.sub_canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Calculate scaling factors
        scale_x = canvas_width / pil_image.width
        scale_y = canvas_height / pil_image.height
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(
            (canvas_width - new_width) // 2,
            (canvas_height - new_height) // 2,
            anchor="nw",
            image=photo
        )
        canvas.image = photo  # Keep a reference
        
        self.severity_window.update()

    def show_severity_statistics(self, stats, total_images):
        # Create statistics window
        stats_win = ctk.CTkToplevel(self.severity_window)
        stats_win.title("Severity Statistics")
        stats_win.geometry("400x300")
        stats_win.attributes('-topmost', True)  # Make window always on top
        stats_win.focus_force()  # Force focus to the window
        
        # Configure grid
        stats_win.grid_columnconfigure(0, weight=1)
        stats_win.grid_rowconfigure(0, weight=0)  # Header
        stats_win.grid_rowconfigure(1, weight=1)  # Content
        
        # Header
        header_label = ctk.CTkLabel(stats_win, text="Severity Distribution", font=("Arial", 16, "bold"))
        header_label.grid(row=0, column=0, pady=10)
        
        # Content
        content_frame = ctk.CTkFrame(stats_win, corner_radius=10)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        for i, (severity, count) in enumerate(stats.items()):
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            
            # Create progress bar
            progress = ctk.CTkProgressBar(content_frame, width=200)
            progress.grid(row=i, column=0, sticky="ew", padx=10, pady=5)
            progress.set(percentage / 100)
            
            # Create label
            label = ctk.CTkLabel(
                content_frame,
                text=f"{severity}: {count} ({percentage:.1f}%)",
                font=("Arial", 12)
            )
            label.grid(row=i, column=1, sticky="w", padx=10, pady=5)

    def toggle_gps(self):
        if not self.gps_running:
            # Try to find and connect to GPS
            self.gps_port = find_gps_port()
            if self.gps_port:
                self.gps_running = True
                self.gps_thread = threading.Thread(target=self._gps_reader, daemon=True)
                self.gps_thread.start()
                self.gps_button.configure(text="Disconnect GPS")
                self.gps_status_label.configure(text="GPS: Connected")
            else:
                tk.messagebox.showerror("GPS Error", "Could not find GPS device")
        else:
            # Disconnect GPS
            self.gps_running = False
            if self.gps_thread:
                self.gps_thread.join(timeout=1)
            self.gps_button.configure(text="Connect GPS")
            self.gps_status_label.configure(text="GPS: Not Connected")

    def _gps_reader(self):
        try:
            with serial.Serial(self.gps_port, 9600, timeout=1) as ser:
                while self.gps_running:
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                self.gps_data["latitude"] = msg.latitude
                                self.gps_data["longitude"] = msg.longitude
                                # Update status label with coordinates
                                self.gps_status_label.configure(
                                    text=f"GPS: {msg.latitude:.6f}, {msg.longitude:.6f}"
                                )
                        except pynmea2.ParseError:
                            continue
        except serial.SerialException:
            self.gps_running = False
            self.gps_button.configure(text="Connect GPS")
            self.gps_status_label.configure(text="GPS: Connection Error")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()
