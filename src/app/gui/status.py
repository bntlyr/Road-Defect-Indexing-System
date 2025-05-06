import customtkinter as ctk
import psutil
import time
import threading
from typing import Optional
import queue

class StatusBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Create status frames for each indicator
        self.camera_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.gps_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.gps_signal_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.memory_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cpu_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        # Create status indicators (dots)
        self.camera_dot = ctk.CTkLabel(self.camera_frame, text="●", font=("Arial", 16))
        self.gps_dot = ctk.CTkLabel(self.gps_frame, text="●", font=("Arial", 16))
        self.gps_signal_dot = ctk.CTkLabel(self.gps_signal_frame, text="●", font=("Arial", 16))
        self.memory_dot = ctk.CTkLabel(self.memory_frame, text="●", font=("Arial", 16))
        self.cpu_dot = ctk.CTkLabel(self.cpu_frame, text="●", font=("Arial", 16))
        
        # Create status labels
        self.camera_status = ctk.CTkLabel(self.camera_frame, text="Camera: Disconnected")
        self.gps_status = ctk.CTkLabel(self.gps_frame, text="GPS: Disconnected")
        self.gps_signal = ctk.CTkLabel(self.gps_signal_frame, text="Signal: --")
        self.memory_status = ctk.CTkLabel(self.memory_frame, text="Memory: --")
        self.cpu_status = ctk.CTkLabel(self.cpu_frame, text="CPU: --")
        
        # Layout
        self.camera_frame.pack(side="left", padx=10)
        self.gps_frame.pack(side="left", padx=10)
        self.gps_signal_frame.pack(side="left", padx=10)
        self.memory_frame.pack(side="left", padx=10)
        self.cpu_frame.pack(side="left", padx=10)
        
        # Pack dots and labels
        self.camera_dot.pack(side="left", padx=(0, 5))
        self.camera_status.pack(side="left")
        
        self.gps_dot.pack(side="left", padx=(0, 5))
        self.gps_status.pack(side="left")
        
        self.gps_signal_dot.pack(side="left", padx=(0, 5))
        self.gps_signal.pack(side="left")
        
        self.memory_dot.pack(side="left", padx=(0, 5))
        self.memory_status.pack(side="left")
        
        self.cpu_dot.pack(side="left", padx=(0, 5))
        self.cpu_status.pack(side="left")
        
        # Initialize update queue and thread
        self.update_queue = queue.Queue()
        self.update_thread: Optional[threading.Thread] = None
        self.is_updating = False
        self.update_interval = 1.0  # seconds
        
        # Start update thread
        self.start_updates()
        
    def start_updates(self):
        """Start the status update thread"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.is_updating = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            # Schedule the first update in the main thread
            self.after(0, self._schedule_update)
            
    def _schedule_update(self):
        """Schedule the next update in the main thread"""
        if self.is_updating:
            self._process_update()
            self.after(int(self.update_interval * 1000), self._schedule_update)
            
    def stop_updates(self):
        """Stop the status update thread"""
        self.is_updating = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
            
    def _update_loop(self):
        """Update status in a loop"""
        while self.is_updating:
            try:
                # Get status data
                status_data = self._get_status_data()
                # Put data in queue for main thread to process
                self.update_queue.put(status_data)
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error updating status: {e}")
                time.sleep(1)
                
    def _get_status_data(self):
        """Get current status data"""
        data = {}
        
        # Get camera status
        if hasattr(self.master, 'camera') and self.master.camera is not None:
            if self.master.camera.is_streaming:
                data['camera'] = ("Camera: Connected", "green")
            else:
                data['camera'] = ("Camera: Ready", "orange")
        else:
            data['camera'] = ("Camera: Disconnected", "red")
            
        # Get GPS status
        if hasattr(self.master, 'gps_reader'):
            if self.master.gps_reader.is_connected():
                if self.master.gps_reader.has_fix:
                    signal_quality = self.master.gps_reader.get_signal_quality()
                    data['gps'] = ("GPS: Connected", "green")
                    
                    # Update signal quality with color coding
                    if signal_quality > 0.7:
                        signal_color = "green"
                        signal_text = "Strong"
                    elif signal_quality > 0.4:
                        signal_color = "orange"
                        signal_text = "Medium"
                    else:
                        signal_color = "red"
                        signal_text = "Weak"
                        
                    data['gps_signal'] = (f"Signal: {signal_text} ({signal_quality:.0%})", signal_color)
                else:
                    data['gps'] = ("GPS: No Fix", "orange")
                    data['gps_signal'] = ("Signal: Searching...", "orange")
            else:
                data['gps'] = ("GPS: Disconnected", "red")
                data['gps_signal'] = ("Signal: --", "gray")
        else:
            data['gps'] = ("GPS: Not Available", "gray")
            data['gps_signal'] = ("Signal: --", "gray")
            
        # Get system stats
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            # Color code memory usage
            if memory.percent > 90:
                memory_color = "red"
            elif memory.percent > 70:
                memory_color = "orange"
            else:
                memory_color = "green"
                
            # Color code CPU usage
            if cpu > 90:
                cpu_color = "red"
            elif cpu > 70:
                cpu_color = "orange"
            else:
                cpu_color = "green"
                
            data['memory'] = (f"Memory: {memory.percent}%", memory_color)
            data['cpu'] = (f"CPU: {cpu}%", cpu_color)
        except Exception as e:
            print(f"Error getting system stats: {e}")
            data['memory'] = ("Memory: --", "gray")
            data['cpu'] = ("CPU: --", "gray")
            
        return data
        
    def _process_update(self):
        """Process status update in main thread"""
        try:
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                
                # Update camera status
                if 'camera' in data:
                    text, color = data['camera']
                    self.camera_status.configure(text=text)
                    self.camera_dot.configure(text_color=color)
                    
                # Update GPS status
                if 'gps' in data:
                    text, color = data['gps']
                    self.gps_status.configure(text=text)
                    self.gps_dot.configure(text_color=color)
                    
                # Update GPS signal
                if 'gps_signal' in data:
                    text, color = data['gps_signal']
                    self.gps_signal.configure(text=text)
                    self.gps_signal_dot.configure(text_color=color)
                    
                # Update system stats
                if 'memory' in data:
                    text, color = data['memory']
                    self.memory_status.configure(text=text)
                    self.memory_dot.configure(text_color=color)
                    
                if 'cpu' in data:
                    text, color = data['cpu']
                    self.cpu_status.configure(text=text)
                    self.cpu_dot.configure(text_color=color)
                    
        except Exception as e:
            print(f"Error processing status update: {e}")
            
    def __del__(self):
        """Clean up resources"""
        self.stop_updates() 