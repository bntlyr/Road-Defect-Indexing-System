import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class VideoControlPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        # Video canvas
        self.video_canvas = tk.Canvas(
            self,
            bg="black",
            highlightthickness=2,
            highlightbackground="white"
        )
        self.video_canvas.pack(expand=True, fill="both", padx=10, pady=10)

        # Camera controls
        self.camera_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.camera_frame.pack(fill="x", padx=10, pady=5)

        # Camera selection
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Camera:")
        self.camera_label.pack(side="left", padx=5)

        self.camera_combo = ctk.CTkComboBox(
            self.camera_frame,
            values=["Camera 0"],
            command=self.parent.switch_camera
        )
        self.camera_combo.pack(side="left", padx=5)

        # Resolution selection
        self.resolution_label = ctk.CTkLabel(self.camera_frame, text="Resolution:")
        self.resolution_label.pack(side="left", padx=5)

        self.resolution_combo = ctk.CTkComboBox(
            self.camera_frame,
            values=["1920x1080", "1280x720"],
            command=self.parent.change_resolution
        )
        self.resolution_combo.pack(side="left", padx=5)

        # FPS selection
        self.fps_label = ctk.CTkLabel(self.camera_frame, text="FPS:")
        self.fps_label.pack(side="left", padx=5)

        self.fps_combo = ctk.CTkComboBox(
            self.camera_frame,
            values=["60", "30"],
            command=self.parent.change_fps
        )
        self.fps_combo.set("30")
        self.fps_combo.pack(side="left", padx=5)

        # Image controls frame
        self.image_controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.image_controls_frame.pack(fill="x", padx=10, pady=5)

        # Brightness control
        self.brightness_label = ctk.CTkLabel(self.image_controls_frame, text="Brightness:")
        self.brightness_label.pack(side="left", padx=5)
        
        self.brightness_slider = ctk.CTkSlider(
            self.image_controls_frame,
            from_=0,
            to=100,
            command=self.parent.adjust_brightness
        )
        self.brightness_slider.set(50)
        self.brightness_slider.pack(side="left", padx=5, fill="x", expand=True)

        # Exposure control
        self.exposure_label = ctk.CTkLabel(self.image_controls_frame, text="Exposure:")
        self.exposure_label.pack(side="left", padx=5)
        
        self.exposure_slider = ctk.CTkSlider(
            self.image_controls_frame,
            from_=0,
            to=100,
            command=self.parent.adjust_exposure
        )
        self.exposure_slider.set(50)
        self.exposure_slider.pack(side="left", padx=5, fill="x", expand=True)

        # Flip controls frame
        self.flip_controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.flip_controls_frame.pack(fill="x", padx=10, pady=5)

        # Flip horizontal button
        self.flip_h_button = ctk.CTkButton(
            self.flip_controls_frame,
            text="Flip Horizontal",
            command=self.parent.flip_horizontal
        )
        self.flip_h_button.pack(side="left", padx=5)

        # Flip vertical button
        self.flip_v_button = ctk.CTkButton(
            self.flip_controls_frame,
            text="Flip Vertical",
            command=self.parent.flip_vertical
        )
        self.flip_v_button.pack(side="left", padx=5)

    def update_frame(self, frame):
        if frame is not None:
            # Apply image adjustments
            if hasattr(self.parent, 'brightness_value'):
                frame = cv2.convertScaleAbs(frame, alpha=self.parent.brightness_value/50.0)
            if hasattr(self.parent, 'exposure_value'):
                frame = cv2.convertScaleAbs(frame, beta=self.parent.exposure_value-50)
            if hasattr(self.parent, 'flip_h') and self.parent.flip_h:
                frame = cv2.flip(frame, 1)
            if hasattr(self.parent, 'flip_v') and self.parent.flip_v:
                frame = cv2.flip(frame, 0)

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get canvas size
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            # If canvas size is not yet available, use default size
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 640
                canvas_height = 480
            
            # Calculate scaling factors
            scale_x = canvas_width / frame_rgb.shape[1]
            scale_y = canvas_height / frame_rgb.shape[0]
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(frame_rgb.shape[1] * scale)
            new_height = int(frame_rgb.shape[0] * scale)
            
            # Ensure minimum dimensions
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                (canvas_width - new_width) // 2,
                (canvas_height - new_height) // 2,
                anchor="nw",
                image=photo
            )
            self.video_canvas.image = photo  # Keep a reference 