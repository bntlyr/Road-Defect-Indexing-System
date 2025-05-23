import serial
import serial.tools.list_ports
import pynmea2
import logging

BAUDRATE = 9600

class GPSReader:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        # Then initialize other attributes
        self.port = None
        self.latitude = None
        self.longitude = None
        self.serial = None  # Add serial connection attribute
        
        # Try COM3 first since we know GPS is always there
        if self._try_connect_port('COM3'):
            self.logger.info("Connected to GPS on COM3")
        else:
            # If COM3 fails, scan other ports
            self.port = self.find_gps_port()
            if self.port:
                self._connect()

    def _try_connect_port(self, port):
        """Try to connect to a specific port and verify it's a GPS device"""
        try:
            with serial.Serial(port, BAUDRATE, timeout=1) as ser:
                # Try to read a few lines to verify it's a GPS
                for _ in range(5):  # Reduced number of attempts for faster response
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        self.port = port
                        return True
        except Exception as e:
            self.logger.debug(f"Failed to connect to {port}: {e}")
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

    def _connect(self):
        """Establish a persistent connection to the GPS device"""
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
            line = self.serial.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                self.logger.info(f"Connected to GPS on {self.port}")
                return True
            else:
                self.serial.close()
                self.serial = None
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to GPS: {e}")
            self.serial = None
            return False

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
            self.serial = None  # Reset connection on error
        return None, None

    def get_gps_data(self):
        """Alias for read_gps_data to maintain compatibility."""
        return self.read_gps_data()

    def is_connected(self):
        """Check if GPS is connected."""
        return self.port is not None and self.serial is not None and self.serial.is_open

    def get_available_ports(self):
        """Get list of available serial ports."""
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect_manually(self, port):
        """Manually connect to a specific port."""
        try:
            # Test the port first
            with serial.Serial(port, BAUDRATE, timeout=1) as ser:
                for _ in range(10):
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        self.port = port
                        return self._connect()  # Use persistent connection
        except Exception as e:
            self.logger.error(f"Failed to connect to port {port}: {e}")
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