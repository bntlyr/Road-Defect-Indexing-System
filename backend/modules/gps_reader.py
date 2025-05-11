import serial
import serial.tools.list_ports
import pynmea2
import threading
import time
import queue
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
import tkinter.messagebox as messagebox

@dataclass
class GPSData:
    """Data class for GPS information"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    fix_quality: Optional[int] = None
    num_satellites: Optional[int] = None
    timestamp: Optional[float] = None

class GPSReader:
    def __init__(self):
        self.gps_port = None
        self.gps_thread = None
        self.gps_data = GPSData()
        self.is_running = False
        self.baudrate = 9600
        self.has_fix = False
        self.data_queue = queue.Queue(maxsize=10)  # Buffer for GPS data
        self.logger = logging.getLogger(__name__)
        self.last_valid_position: Optional[Tuple[float, float]] = None
        self.position_update_interval = 0.1  # seconds
        self.min_signal_quality = 0.5  # Minimum signal quality (0-1)
        self.auto_detect = True  # Enable auto-detection by default
        self.detect_thread = None
        self.is_detecting = True
        self._callbacks = []  # List of callback functions to notify of GPS updates
        self.serial_port = None

    def add_callback(self, callback):
        """Add a callback function to be called when GPS data updates"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a callback function"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks of GPS data updates"""
        for callback in self._callbacks:
            try:
                callback(self.gps_data)
            except Exception as e:
                self.logger.error(f"Error in GPS callback: {e}")

    def find_gps_port(self) -> Optional[str]:
        """Scan and return the first port that receives valid GPS NMEA data."""
        self.logger.info("Scanning serial ports for GPS device...")
        ports = serial.tools.list_ports.comports()

        for port in ports:
            try:
                with serial.Serial(port.device, self.baudrate, timeout=1) as ser:
                    for _ in range(10):  # Try 10 lines
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            self.logger.info(f"GPS device found on {port.device}")
                            return port.device
            except Exception as e:
                self.logger.debug(f"Failed to check port {port.device}: {e}")
                continue

        self.logger.warning("GPS device not found.")
        return None

    def start(self) -> bool:
        """Start GPS reading in a separate thread."""
        if not self.is_running:
            if self.auto_detect:
                self.start_auto_detect()
                self._notify_callbacks()  # Notify when starting auto-detect
                return True
            else:
                self.gps_port = self.find_gps_port()
                if self.gps_port:
                    self._start_reading()
                    self._notify_callbacks()  # Notify when starting manual connection
                    return True
                else:
                    self._notify_callbacks()  # Notify when no GPS found
        return False

    def start_auto_detect(self):
        """Start auto-detection of GPS devices"""
        if not self.is_detecting:
            self.is_detecting = True
            self.detect_thread = threading.Thread(target=self._auto_detect_loop, daemon=True)
            self.detect_thread.start()

    def stop_auto_detect(self):
        """Stop auto-detection of GPS devices"""
        self.is_detecting = False
        if self.detect_thread:
            self.detect_thread.join(timeout=1)

    def _auto_detect_loop(self):
        """Continuously scan for GPS devices"""
        while self.is_detecting:
            if not self.is_running:
                port = self.find_gps_port()
                if port:
                    self.gps_port = port
                    self._start_reading()
            time.sleep(5)  # Check every 5 seconds

    def connect_manually(self, port: str) -> bool:
        """Manually connect to a specific GPS port"""
        try:
            # Stop any existing connection
            self.stop()
            # Try to open the port directly like getGPS.py
            self.serial_port = serial.Serial(port, self.baudrate, timeout=1)
            # Test if we can read data
            for _ in range(10):
                line = self.serial_port.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    self.gps_port = port
                    self._start_reading()
                    return True
            # If we get here, we couldn't read valid GPS data
            self.serial_port.close()
            self.serial_port = None
            messagebox.showerror("GPS Connection Error", f"No valid GPS data found on {port}. Please check your device.")
            return False
        except Exception as e:
            messagebox.showerror("GPS Connection Error", f"Failed to connect to port {port}: {e}")
            if hasattr(self, 'serial_port') and self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            return False

    def _start_reading(self):
        """Start reading from the GPS device"""
        if not self.gps_port:
            self.logger.error("Cannot start reading: No GPS port specified")
            return
            
        # Start the reading thread
        self.is_running = True
        self.gps_thread = threading.Thread(target=self._gps_reader, daemon=True)
        self.gps_thread.start()

    def stop(self):
        """Stop GPS reading."""
        self.is_running = False
        self.stop_auto_detect()
        if self.gps_thread:
            self.gps_thread.join(timeout=1)
        if hasattr(self, 'serial_port') and self.serial_port:
            self.serial_port.close()
            self.serial_port = None
        # Clear the data queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        self._notify_callbacks()

    def _gps_reader(self):
        """Read GPS data in a loop."""
        try:
            while self.is_running and self.serial_port and self.serial_port.is_open:
                try:
                    line = self.serial_port.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                # Update GPS data
                                self.gps_data.latitude = msg.latitude
                                self.gps_data.longitude = msg.longitude
                                
                                # Update additional data if available
                                if hasattr(msg, 'altitude'):
                                    self.gps_data.altitude = msg.altitude
                                if hasattr(msg, 'spd_over_grnd'):
                                    self.gps_data.speed = msg.spd_over_grnd
                                if hasattr(msg, 'true_course'):
                                    self.gps_data.heading = msg.true_course
                                if hasattr(msg, 'gps_qual'):
                                    self.gps_data.fix_quality = msg.gps_qual
                                if hasattr(msg, 'num_sats'):
                                    self.gps_data.num_satellites = msg.num_sats
                                
                                self.gps_data.timestamp = time.time()
                                self.has_fix = True
                                self._notify_callbacks()
                        except pynmea2.ParseError:
                            continue
                except serial.SerialException:
                    break
        except Exception as e:
            self.logger.error(f"Error in GPS reader thread: {e}")
            # Show error in window
            if hasattr(self, '_callbacks') and self._callbacks:
                for callback in self._callbacks:
                    try:
                        callback(None, error=str(e))
                    except:
                        pass
        finally:
            if hasattr(self, 'serial_port') and self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            self.is_running = False
            self.has_fix = False
            self._notify_callbacks()

    def _is_valid_position(self, msg) -> bool:
        """Check if the GPS position is valid."""
        if not hasattr(msg, 'latitude') or not hasattr(msg, 'longitude'):
            return False
            
        # Check for valid coordinates
        if msg.latitude is None or msg.longitude is None:
            return False
            
        return True

    def _is_position_significant(self, position: Tuple[float, float]) -> bool:
        """Check if the new position is significantly different from the last one."""
        if self.last_valid_position is None:
            return True
            
        # Check if position has changed by at least 0.0001 degrees
        # (approximately 11 meters at the equator)
        return (abs(position[0] - self.last_valid_position[0]) > 0.0001 or
                abs(position[1] - self.last_valid_position[1]) > 0.0001)

    def get_gps_data(self) -> GPSData:
        """Get the current GPS data."""
        try:
            # Try to get the latest data from the queue
            return self.data_queue.get_nowait()
        except queue.Empty:
            # Return the last known position if queue is empty
            return self.gps_data

    def is_connected(self) -> bool:
        """Check if GPS is connected and running."""
        return self.is_running and self.gps_port is not None

    def get_signal_quality(self) -> float:
        """Get the current GPS signal quality (0-1)."""
        if not self.is_connected():
            return 0.0
            
        try:
            with serial.Serial(self.gps_port, self.baudrate, timeout=1) as ser:
                for _ in range(5):  # Try 5 lines
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA'):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, 'gps_qual'):
                                return float(msg.gps_qual) / 9.0  # Normalize to 0-1
                        except pynmea2.ParseError:
                            continue
        except Exception:
            pass
        return 0.0

    def get_available_ports(self) -> List[str]:
        """Get a list of available serial ports."""
        return [port.device for port in serial.tools.list_ports.comports()]

    def set_auto_detect(self, enabled: bool):
        """Enable or disable auto-detection."""
        self.auto_detect = enabled
        if enabled:
            self.start_auto_detect()
        else:
            self.stop_auto_detect()

    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get the current position as (latitude, longitude)."""
        if self.gps_data.latitude is not None and self.gps_data.longitude is not None:
            return (self.gps_data.latitude, self.gps_data.longitude)
        return None

    def get_altitude(self) -> Optional[float]:
        """Get the current altitude in meters."""
        return self.gps_data.altitude

    def get_speed(self) -> Optional[float]:
        """Get the current speed in knots."""
        return self.gps_data.speed

    def get_heading(self) -> Optional[float]:
        """Get the current heading in degrees."""
        return self.gps_data.heading

    def get_fix_quality(self) -> Optional[int]:
        """Get the current fix quality."""
        return self.gps_data.fix_quality

    def get_num_satellites(self) -> Optional[int]:
        """Get the number of satellites in view."""
        return self.gps_data.num_satellites

if __name__ == "__main__":
    gps_reader = GPSReader()
    if gps_reader.start():
        print("GPS reading started.")
    else:
        print("Failed to start GPS reading.")import serial
import serial.tools.list_ports
import pynmea2
import threading
import time
import queue
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
import tkinter.messagebox as messagebox

@dataclass
class GPSData:
    """Data class for GPS information"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    fix_quality: Optional[int] = None
    num_satellites: Optional[int] = None
    timestamp: Optional[float] = None

class GPSReader:
    def __init__(self):
        self.gps_port = None
        self.gps_thread = None
        self.gps_data = GPSData()
        self.is_running = False
        self.baudrate = 9600
        self.has_fix = False
        self.data_queue = queue.Queue(maxsize=10)  # Buffer for GPS data
        self.logger = logging.getLogger(__name__)
        self.last_valid_position: Optional[Tuple[float, float]] = None
        self.position_update_interval = 0.1  # seconds
        self.min_signal_quality = 0.5  # Minimum signal quality (0-1)
        self.auto_detect = True  # Enable auto-detection by default
        self.detect_thread = None
        self.is_detecting = True
        self._callbacks = []  # List of callback functions to notify of GPS updates
        self.serial_port = None

    def add_callback(self, callback):
        """Add a callback function to be called when GPS data updates"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a callback function"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks of GPS data updates"""
        for callback in self._callbacks:
            try:
                callback(self.gps_data)
            except Exception as e:
                self.logger.error(f"Error in GPS callback: {e}")

    def find_gps_port(self) -> Optional[str]:
        """Scan and return the first port that receives valid GPS NMEA data."""
        self.logger.info("Scanning serial ports for GPS device...")
        ports = serial.tools.list_ports.comports()

        for port in ports:
            try:
                with serial.Serial(port.device, self.baudrate, timeout=1) as ser:
                    for _ in range(10):  # Try 10 lines
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            self.logger.info(f"GPS device found on {port.device}")
                            return port.device
            except Exception as e:
                self.logger.debug(f"Failed to check port {port.device}: {e}")
                continue

        self.logger.warning("GPS device not found.")
        return None

    def start(self) -> bool:
        """Start GPS reading in a separate thread."""
        if not self.is_running:
            if self.auto_detect:
                self.start_auto_detect()
                self._notify_callbacks()  # Notify when starting auto-detect
                return True
            else:
                self.gps_port = self.find_gps_port()
                if self.gps_port:
                    self._start_reading()
                    self._notify_callbacks()  # Notify when starting manual connection
                    return True
                else:
                    self._notify_callbacks()  # Notify when no GPS found
        return False

    def start_auto_detect(self):
        """Start auto-detection of GPS devices"""
        if not self.is_detecting:
            self.is_detecting = True
            self.detect_thread = threading.Thread(target=self._auto_detect_loop, daemon=True)
            self.detect_thread.start()

    def stop_auto_detect(self):
        """Stop auto-detection of GPS devices"""
        self.is_detecting = False
        if self.detect_thread:
            self.detect_thread.join(timeout=1)

    def _auto_detect_loop(self):
        """Continuously scan for GPS devices"""
        while self.is_detecting:
            if not self.is_running:
                port = self.find_gps_port()
                if port:
                    self.gps_port = port
                    self._start_reading()
            time.sleep(5)  # Check every 5 seconds

    def connect_manually(self, port: str) -> bool:
        """Manually connect to a specific GPS port"""
        try:
            # Stop any existing connection
            self.stop()
            # Try to open the port directly like getGPS.py
            self.serial_port = serial.Serial(port, self.baudrate, timeout=1)
            # Test if we can read data
            for _ in range(10):
                line = self.serial_port.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    self.gps_port = port
                    self._start_reading()
                    return True
            # If we get here, we couldn't read valid GPS data
            self.serial_port.close()
            self.serial_port = None
            messagebox.showerror("GPS Connection Error", f"No valid GPS data found on {port}. Please check your device.")
            return False
        except Exception as e:
            messagebox.showerror("GPS Connection Error", f"Failed to connect to port {port}: {e}")
            if hasattr(self, 'serial_port') and self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            return False

    def _start_reading(self):
        """Start reading from the GPS device"""
        if not self.gps_port:
            self.logger.error("Cannot start reading: No GPS port specified")
            return
            
        # Start the reading thread
        self.is_running = True
        self.gps_thread = threading.Thread(target=self._gps_reader, daemon=True)
        self.gps_thread.start()

    def stop(self):
        """Stop GPS reading."""
        self.is_running = False
        self.stop_auto_detect()
        if self.gps_thread:
            self.gps_thread.join(timeout=1)
        if hasattr(self, 'serial_port') and self.serial_port:
            self.serial_port.close()
            self.serial_port = None
        # Clear the data queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        self._notify_callbacks()

    def _gps_reader(self):
        """Read GPS data in a loop."""
        try:
            while self.is_running and self.serial_port and self.serial_port.is_open:
                try:
                    line = self.serial_port.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                # Update GPS data
                                self.gps_data.latitude = msg.latitude
                                self.gps_data.longitude = msg.longitude
                                
                                # Update additional data if available
                                if hasattr(msg, 'altitude'):
                                    self.gps_data.altitude = msg.altitude
                                if hasattr(msg, 'spd_over_grnd'):
                                    self.gps_data.speed = msg.spd_over_grnd
                                if hasattr(msg, 'true_course'):
                                    self.gps_data.heading = msg.true_course
                                if hasattr(msg, 'gps_qual'):
                                    self.gps_data.fix_quality = msg.gps_qual
                                if hasattr(msg, 'num_sats'):
                                    self.gps_data.num_satellites = msg.num_sats
                                
                                self.gps_data.timestamp = time.time()
                                self.has_fix = True
                                self._notify_callbacks()
                        except pynmea2.ParseError:
                            continue
                except serial.SerialException:
                    break
        except Exception as e:
            self.logger.error(f"Error in GPS reader thread: {e}")
            # Show error in window
            if hasattr(self, '_callbacks') and self._callbacks:
                for callback in self._callbacks:
                    try:
                        callback(None, error=str(e))
                    except:
                        pass
        finally:
            if hasattr(self, 'serial_port') and self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            self.is_running = False
            self.has_fix = False
            self._notify_callbacks()

    def _is_valid_position(self, msg) -> bool:
        """Check if the GPS position is valid."""
        if not hasattr(msg, 'latitude') or not hasattr(msg, 'longitude'):
            return False
            
        # Check for valid coordinates
        if msg.latitude is None or msg.longitude is None:
            return False
            
        return True

    def _is_position_significant(self, position: Tuple[float, float]) -> bool:
        """Check if the new position is significantly different from the last one."""
        if self.last_valid_position is None:
            return True
            
        # Check if position has changed by at least 0.0001 degrees
        # (approximately 11 meters at the equator)
        return (abs(position[0] - self.last_valid_position[0]) > 0.0001 or
                abs(position[1] - self.last_valid_position[1]) > 0.0001)

    def get_gps_data(self) -> GPSData:
        """Get the current GPS data."""
        try:
            # Try to get the latest data from the queue
            return self.data_queue.get_nowait()
        except queue.Empty:
            # Return the last known position if queue is empty
            return self.gps_data

    def is_connected(self) -> bool:
        """Check if GPS is connected and running."""
        return self.is_running and self.gps_port is not None

    def get_signal_quality(self) -> float:
        """Get the current GPS signal quality (0-1)."""
        if not self.is_connected():
            return 0.0
            
        try:
            with serial.Serial(self.gps_port, self.baudrate, timeout=1) as ser:
                for _ in range(5):  # Try 5 lines
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA'):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, 'gps_qual'):
                                return float(msg.gps_qual) / 9.0  # Normalize to 0-1
                        except pynmea2.ParseError:
                            continue
        except Exception:
            pass
        return 0.0

    def get_available_ports(self) -> List[str]:
        """Get a list of available serial ports."""
        return [port.device for port in serial.tools.list_ports.comports()]

    def set_auto_detect(self, enabled: bool):
        """Enable or disable auto-detection."""
        self.auto_detect = enabled
        if enabled:
            self.start_auto_detect()
        else:
            self.stop_auto_detect()

    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get the current position as (latitude, longitude)."""
        if self.gps_data.latitude is not None and self.gps_data.longitude is not None:
            return (self.gps_data.latitude, self.gps_data.longitude)
        return None

    def get_altitude(self) -> Optional[float]:
        """Get the current altitude in meters."""
        return self.gps_data.altitude

    def get_speed(self) -> Optional[float]:
        """Get the current speed in knots."""
        return self.gps_data.speed

    def get_heading(self) -> Optional[float]:
        """Get the current heading in degrees."""
        return self.gps_data.heading

    def get_fix_quality(self) -> Optional[int]:
        """Get the current fix quality."""
        return self.gps_data.fix_quality

    def get_num_satellites(self) -> Optional[int]:
        """Get the number of satellites in view."""
        return self.gps_data.num_satellites

if __name__ == "__main__":
    gps_reader = GPSReader()
    if gps_reader.start():
        print("GPS reading started.")
    else:
        print("Failed to start GPS reading.")