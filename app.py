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
from tkinter import filedialog


matplotlib.use('Agg')

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

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

        self.gps_device_combo = ctk.CTkComboBox(self.video_control_frame, values=["GPS 0", "GPS 1"])
        self.gps_device_combo.pack(pady=5, padx=10, fill="x")

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

    def update_defect_charts(self, defect_counts):
        # Update data for each defect type
        for defect_type, chart_data in self.defect_charts.items():
            total = defect_counts.get(defect_type, 0)
            
            # Clear previous chart
            chart_data["ax"].clear()
            
            # Create simple pie chart with one segment
            sizes = [1]
            colors = ['#1f77b4']
            chart_data["ax"].pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5))
            
            # Update total count in center
            chart_data["ax"].text(0, 0, str(total), ha='center', va='center', fontsize=12, color='white')
            
            # Set background color
            chart_data["ax"].set_facecolor('#2b2b2b')
            chart_data["fig"].patch.set_facecolor('#2b2b2b')
            
            # Update canvas
            chart_data["canvas"].draw()

    def add_defect(self, defect_type, severity):
        """Add a new defect to the statistics"""
        if defect_type in self.defect_data:
            self.defect_data[defect_type]["total"] += 1
            self.defect_data[defect_type]["severity"][severity] += 1

    def setup_control_settings(self):
        self.inference_button = ctk.CTkButton(self.control_settings_frame, text="Start Inferencing", command=self.toggle_inferencing)
        self.inference_button.pack(pady=5, padx=10, fill="x")

        self.upload_button = ctk.CTkButton(self.control_settings_frame, text="Upload to Cloud", command=self.open_upload_window)
        self.upload_button.pack(pady=5, padx=10, fill="x")

        self.settings_button = ctk.CTkButton(self.control_settings_frame, text="Settings", command=self.open_settings_window)
        self.settings_button.pack(pady=5, padx=10, fill="x")

        self.exit_button = ctk.CTkButton(self.control_settings_frame, text="Exit", command=self.quit, fg_color="#ff4444", hover_color="#cc0000")
        self.exit_button.pack(pady=5, padx=10, fill="x")

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
            cap.set(cv2.CAP_PROP_FPS, 30)       # Try to set 30 FPS

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
                        frame, defect_counts = self.detector.process_frame(frame, self.roi_points if self.roi_points else None)
                        # Update defect charts with new counts
                        self.update_defect_charts(defect_counts)

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

                else:
                    print("[WARNING] Frame read failed, trying to reconnect...")
                    self.cap.release()
                    self.cap = self.open_camera(self.current_camera_index)

            else:
                print("[WARNING] Camera not opened, retrying...")
                self.cap = self.open_camera(self.current_camera_index)

            self.after(30, self.video_loop)

    def update_defect_charts(self, defect_counts):
        # Update data for each defect type
        for defect_type, chart_data in self.defect_charts.items():
            total = defect_counts.get(defect_type, 0)
            
            # Clear previous chart
            chart_data["ax"].clear()
            
            # Create simple pie chart with one segment
            sizes = [1]
            colors = ['#1f77b4']
            chart_data["ax"].pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5))
            
            # Update total count in center
            chart_data["ax"].text(0, 0, str(total), ha='center', va='center', fontsize=12, color='white')
            
            # Set background color
            chart_data["ax"].set_facecolor('#2b2b2b')
            chart_data["fig"].patch.set_facecolor('#2b2b2b')
            
            # Update canvas
            chart_data["canvas"].draw()

    # ---- Other methods (ROI, Upload window, Settings window, etc) ----
    # (Same as your code, no need to repeat here for brevity)

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
                self.detector = DefectDetector(model_path, self.current_save_dir)
        else:
            # Clean up detector
            self.detector = None

    def update_status_bar(self):
        cpu = psutil.cpu_percent()
        gpu = 0
        now = time.strftime("%H:%M:%S")
        self.status_bar.configure(text=f"🟥 {now} | GPU: {gpu}% | CPU: {cpu}%")
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

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()
