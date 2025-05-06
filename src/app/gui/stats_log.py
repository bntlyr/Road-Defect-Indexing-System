import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import queue
from datetime import datetime

class StatsPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

        # Thread-safe update queue
        self.update_queue = queue.Queue()
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()
        self.after(100, self._process_queue)

    def setup_ui(self):
        # Create a frame to hold all defect charts
        self.defect_charts_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.defect_charts_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Initialize defect data
        self.defect_data = {
            "Linear-Crack": 0,
            "Alligator-Crack": 0,
            "pothole": 0,
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
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            fig.patch.set_facecolor('#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            # Create simple pie chart
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

    def update_stats(self, frame_counts, gps_data=None):
        """Queue the update for the worker thread"""
        self.update_queue.put((frame_counts, None))

    def _update_worker(self):
        """Background worker for updating charts and GPS label"""
        while True:
            try:
                try:
                    frame_counts, gps_data = self.update_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if frame_counts is None:
                    break
                self._update_charts_and_gps(frame_counts, gps_data)
                self.update_queue.task_done()
            except Exception as e:
                print(f"Error in stats update worker: {e}")

    def _update_charts_and_gps(self, frame_counts, gps_data):
        # Color mapping for different defect types (BGR format)
        defect_colors = {
            'Linear-Crack': (0, 255, 0),      # Green
            'Alligator-Crack': (255, 0, 0),   # Blue
            'pothole': (0, 0, 255)           # Red
        }
        # Update the charts with new counts
        for defect_type, chart_data in self.defect_charts.items():
            count = frame_counts.get(defect_type, 0)
            chart_data["ax"].clear()
            color = defect_colors[defect_type]
            rgb_color = (color[2]/255, color[1]/255, color[0]/255)
            sizes = [1]
            colors = [rgb_color]
            chart_data["ax"].pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5))
            chart_data["ax"].text(0, 0, str(count), ha='center', va='center', fontsize=12, color='white')
            chart_data["ax"].set_facecolor('#2b2b2b')
            chart_data["fig"].patch.set_facecolor('#2b2b2b')
            chart_data["canvas"].draw()

    def _process_queue(self):
        # This method is called periodically in the main thread to keep the UI responsive
        self.after(100, self._process_queue)

    def cleanup(self):
        # Signal update worker to stop
        self.update_queue.put((None, None))
        self.update_thread.join()
        # Close all matplotlib figures
        for chart_data in self.defect_charts.values():
            plt.close(chart_data["fig"]) 