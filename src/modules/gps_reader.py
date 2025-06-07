import serial
import serial.tools.list_ports
import pynmea2
import logging
import time
import os

BAUDRATE = 9600
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

class GPSReader:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        # Then initialize other attributes
        self.port = None
        self.latitude = None
        self.longitude = None
        self.serial = None  # Add serial connection attribute
        self.last_retry_time = 0
        self.retry_count = 0
        
        # Try COM3 first, then COM4, then scan other ports
        if self._try_connect_port('COM3'):
            self.logger.info("Connected to GPS on COM3")
            if not self._connect():
                self.logger.error("Failed to establish persistent connection on COM3")
        elif self._try_connect_port('COM4'):
            self.logger.info("Connected to GPS on COM4")
            if not self._connect():
                self.logger.error("Failed to establish persistent connection on COM4")
        else:
            # If both COM3 and COM4 fail, scan other ports
            self.port = self.find_gps_port()
            if self.port:
                if not self._connect():
                    self.logger.error(f"Failed to establish persistent connection on {self.port}")

    def _try_connect_port(self, port):
        """Try to connect to a specific port and verify it's a GPS device"""
        try:
            # Check if port is already in use
            if self._is_port_in_use(port):
                self.logger.warning(f"Port {port} is already in use")
                return False

            with serial.Serial(port, BAUDRATE, timeout=1) as ser:
                # Try to read a few lines to verify it's a GPS
                for _ in range(5):  # Reduced number of attempts for faster response
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        self.port = port
                        return True
        except serial.SerialException as e:
            self.logger.debug(f"Failed to connect to {port}: {e}")
            if "Access is denied" in str(e):
                self.logger.warning(f"Access denied to port {port}. Port might be in use by another application.")
        except Exception as e:
            self.logger.debug(f"Unexpected error connecting to {port}: {e}")
        return False

    def _is_port_in_use(self, port):
        """Check if a port is already in use by another process"""
        try:
            # Try to open the port exclusively
            with serial.Serial(port, BAUDRATE, timeout=0.1) as ser:
                return False
        except serial.SerialException:
            return True

    def _connect(self):
        """Establish a persistent connection to the GPS device with retry logic"""
        current_time = time.time()
        
        # Check if we should retry
        if self.retry_count >= MAX_RETRIES:
            if current_time - self.last_retry_time > 30:  # Reset retry count after 30 seconds
                self.retry_count = 0
            else:
                return False

        try:
            if self.serial is not None:
                try:
                    self.serial.close()
                except:
                    pass
                self.serial = None
            
            # Try to connect with a shorter timeout
            self.serial = serial.Serial(self.port, BAUDRATE, timeout=0.5)
            
            # Verify we can read data
            for _ in range(3):  # Try to read a few lines to verify connection
                line = self.serial.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    self.logger.info(f"Successfully connected to GPS on {self.port}")
                    self.retry_count = 0  # Reset retry count on successful connection
                    return True
            
            # If we get here, we couldn't read valid GPS data
            self.serial.close()
            self.serial = None
            return False
            
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to GPS: {e}")
            self.serial = None
            self.retry_count += 1
            self.last_retry_time = current_time
            time.sleep(RETRY_DELAY)  # Wait before retrying
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to GPS: {e}")
            self.serial = None
            return False

    def find_gps_port(self):
        """Scan and return the first port that receives valid GPS NMEA data."""
        self.logger.info("Scanning serial ports for GPS device...")
        ports = serial.tools.list_ports.comports()
        
        # Sort ports to prioritize COM3
        ports = sorted(ports, key=lambda x: 0 if x.device == 'COM3' else 1)
        
        for port in ports:
            if self._try_connect_port(port.device):
                self.logger.info(f"GPS device found on {port.device}")
                return port.device

        self.logger.warning("GPS device not found.")
        return None

    def read_gps_data(self):
        """Read GPS data and update latitude and longitude."""
        if not self.port:
            return None, None

        try:
            # Ensure connection is active
            if self.serial is None or not self.serial.is_open:
                if not self._connect():
                    return None, None

            # Read data from the persistent connection
            line = self.serial.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                try:
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                        self.latitude = msg.latitude
                        self.longitude = msg.longitude
                        return self.latitude, self.longitude
                except pynmea2.ParseError as e:
                    self.logger.error(f"Parse error: {e}")
        except serial.SerialException as e:
            self.logger.error(f"Serial error: {e}")
            # Only reset connection if we get a critical error
            if "Access is denied" in str(e) or "device disconnected" in str(e).lower():
                self.serial = None
                self._connect()
        return None, None

    def get_gps_data(self):
        """Alias for read_gps_data to maintain compatibility."""
        return self.read_gps_data()

    def is_connected(self):
        """Check if GPS is connected."""
        # Only check if the port and serial connection exist and are open
        return self.port is not None and self.serial is not None and self.serial.is_open

    def get_available_ports(self):
        """Get list of available serial ports."""
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect_manually(self, port):
        """Manually connect to a specific port with retry logic"""
        try:
            # Reset retry count for manual connection
            self.retry_count = 0
            
            # Test the port first
            with serial.Serial(port, BAUDRATE, timeout=1) as ser:
                for _ in range(10):
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        self.port = port
                        return self._connect()  # Use persistent connection
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to port {port}: {e}")
            if "Access is denied" in str(e):
                self.logger.warning(f"Access denied to port {port}. Please ensure no other application is using this port.")
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to port {port}: {e}")
        return False

    def cleanup(self):
        """Clean up the serial connection"""
        if self.serial is not None:
            try:
                self.serial.close()
            except:
                pass
            self.serial = None
        self.port = None
        self.retry_count = 0

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and test GPS reader
    gps_reader = GPSReader()
    port = gps_reader.find_gps_port()
    
    if port:
        print(f"GPS found on port: {port}")
        print("Reading GPS data (press Ctrl+C to stop)...")
        try:
            while True:
                lat, lon = gps_reader.read_gps_data()
                if lat is not None and lon is not None:
                    print(f"Location: {lat}, {lon}")
        except KeyboardInterrupt:
            print("\nGPS reading stopped.")
    else:
        print("No GPS device found. Available ports:", gps_reader.get_available_ports())