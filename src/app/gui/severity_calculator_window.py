import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import os
from ..modules.severity_calculator import SeverityCalculator, extract_image_metadata, apply_pinhole_camera_model, convert_to_grayscale, enhance_contrast, apply_canny_edge_detection, apply_superpixel_segmentation, overlay_segmentation_on_image
import cv2
import numpy as np

class SeverityCalculatorWindow(ctk.CTkToplevel):
    STEPS = [
        "Load Image and GPS Metadata",
        "Camera Calibration and Undistortion",
        "Grayscale Conversion",
        "Region Processing (per bounding box)",
        "Superpixel Segmentation",
        "Severity Metrics Calculation",
        "Overlay and Visualization",
        "Metadata Generation and Saving"
    ]

    def __init__(self, parent, input_dir, default_output_dir):
        super().__init__(parent)
        self.title("Severity Calculator")
        self.geometry("900x700")
        self.resizable(True, True)
        self.parent = parent
        self.input_dir = input_dir
        self.output_dir = default_output_dir
        self.processed_images = []
        self.current_image_index = 0
        self.processing_thread = None
        self.progress_vars = [tk.StringVar(value="Pending") for _ in self.STEPS]
        self._setup_ui()
        # Center and set on top of parent
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.attributes('-topmost', True)
        self.after(100, lambda: self.attributes('-topmost', False))
        self._center_window()

    def _center_window(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = (ws // 2) - (w // 2)
        y = (hs // 2) - (h // 2)
        self.geometry(f'{w}x{h}+{x}+{y}')

    def _setup_ui(self):
        # Output directory selector
        dir_frame = ctk.CTkFrame(self)
        dir_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(dir_frame, text="Output Directory:").pack(side="left", padx=5)
        self.dir_var = tk.StringVar(value=self.output_dir)
        self.dir_entry = ctk.CTkEntry(dir_frame, textvariable=self.dir_var, width=400)
        self.dir_entry.pack(side="left", padx=5)
        browse_btn = ctk.CTkButton(dir_frame, text="Browse", command=self._browse_dir)
        browse_btn.pack(side="left", padx=5)
        self.start_btn = ctk.CTkButton(dir_frame, text="Start Processing", command=self._toggle_processing)
        self.start_btn.pack(side="right", padx=5)

        # Progress bar and step labels
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(fill="x", padx=10, pady=10)
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        self.step_labels = []
        for i, step in enumerate(self.STEPS):
            row = ctk.CTkFrame(progress_frame)
            row.pack(fill="x", pady=2)
            label = ctk.CTkLabel(row, text=step, anchor="w")
            label.pack(side="left", padx=5)
            status = ctk.CTkLabel(row, textvariable=self.progress_vars[i], width=80)
            status.pack(side="right", padx=5)
            self.step_labels.append((label, status))

        # Image viewer
        viewer_frame = ctk.CTkFrame(self)
        viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(viewer_frame, text="No image to display", width=600, height=400)
        self.image_label.pack(expand=True)
        nav_frame = ctk.CTkFrame(viewer_frame)
        nav_frame.pack(pady=5)
        self.prev_btn = ctk.CTkButton(nav_frame, text="Previous", command=self._show_prev_image)
        self.prev_btn.pack(side="left", padx=10)
        self.index_label = ctk.CTkLabel(nav_frame, text="")
        self.index_label.pack(side="left", padx=10)
        self.next_btn = ctk.CTkButton(nav_frame, text="Next", command=self._show_next_image)
        self.next_btn.pack(side="left", padx=10)

    def _browse_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.output_dir)
        if new_dir:
            self.output_dir = new_dir
            self.dir_var.set(new_dir)

    def _toggle_processing(self):
        if not hasattr(self, '_processing_active') or not self._processing_active:
            self._processing_active = True
            self.start_btn.configure(text="Stop Processing", fg_color="gray", state="normal")
            self.processing_thread = threading.Thread(target=self._process_all_images, daemon=True)
            self.processing_thread.start()
        else:
            self._processing_active = False
            self.start_btn.configure(text="Start Processing", fg_color="transparent", state="normal")

    def _safe_update_button(self, text, fg_color, state):
        if self.start_btn.winfo_exists():
            self.start_btn.configure(text=text, fg_color=fg_color, state=state)

    def _process_all_images(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.processed_images = []
        self.current_image_index = 0
        camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
        distortion_coeffs = np.zeros(5)
        calculator = SeverityCalculator(1280, 720)
        try:
            for idx, image_file in enumerate(image_files):
                if not getattr(self, '_processing_active', True):
                    break
                image_path = os.path.join(self.input_dir, image_file)
                save_path = os.path.join(self.output_dir, f"processed_{os.path.basename(image_file)}")
                for i, step_name in enumerate(self.STEPS):
                    if not getattr(self, '_processing_active', True):
                        self.progress_vars[i].set("Stopped")
                        self._update_progress(i / len(self.STEPS))
                        self._safe_update()
                        return
                    self.progress_vars[i].set("In Progress")
                    self._update_progress(i / len(self.STEPS))
                    self._safe_update()
                try:
                    overlay_img, bbox_pixel_areas, avg_bbox_pixel_area, metadata = calculator.process_image(
                        image_path, camera_matrix, distortion_coeffs, save_path, distance_to_object_m=1.0)
                    if overlay_img is not None:
                        overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(overlay_img_rgb)
                        self.processed_images.append(pil_img)
                        for i in range(len(self.STEPS)):
                            self.progress_vars[i].set("Done")
                            self._update_progress((i + 1) / len(self.STEPS))
                            self._safe_update()
                    else:
                        for i in range(len(self.STEPS)):
                            self.progress_vars[i].set("Error: Processing failed")
                            self._update_progress((i + 1) / len(self.STEPS))
                            self._safe_update()
                except Exception as e:
                    for i in range(len(self.STEPS)):
                        self.progress_vars[i].set(f"Error: {e}")
                        self._update_progress((i + 1) / len(self.STEPS))
                        self._safe_update()
                if not getattr(self, '_processing_active', True):
                    break
            self.current_image_index = 0
            self._show_image()
        except Exception as e:
            for i in range(len(self.STEPS)):
                self.progress_vars[i].set(f"Error: {e}")
            self._update_progress(0)
            self._safe_update()
        finally:
            self._processing_active = False
            self.after(0, self._safe_update_button, "Start Processing", "transparent", "normal")
            self._update_progress(0)

    def _update_progress(self, value):
        self.progress_bar.set(value)

    def _safe_update(self):
        self.update_idletasks()
        self.update()

    def _show_image(self):
        if not self.processed_images:
            self.image_label.configure(text="No image to display", image=None)
            self.index_label.configure(text="")
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            return
        img = self.processed_images[self.current_image_index]
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk  # Keep reference
        # Update index label
        total = len(self.processed_images)
        idx = self.current_image_index + 1
        self.index_label.configure(text=f"Image {idx} of {total}")
        # Enable/disable navigation buttons
        if self.current_image_index == 0:
            self.prev_btn.configure(state="disabled")
        else:
            self.prev_btn.configure(state="normal")
        if self.current_image_index == total - 1:
            self.next_btn.configure(state="disabled")
        else:
            self.next_btn.configure(state="normal")

    def _show_prev_image(self):
        if not self.processed_images or self.current_image_index == 0:
            return
        self.current_image_index -= 1
        self._show_image()

    def _show_next_image(self):
        if not self.processed_images or self.current_image_index == len(self.processed_images) - 1:
            return
        self.current_image_index += 1
        self._show_image() 