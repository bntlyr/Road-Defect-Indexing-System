import serial
import serial.tools.list_ports
import pynmea2
import threading
import time
import logging

BAUDRATE = 9600
UPDATE_INTERVAL = 1.0  # Update GPS data every second

class GPSReader:
    def __init__(self):
        self.port = None
        self.latitude = None
        self.longitude = None
        self.last_update = 0
        self.is_running = False
        self.gps_thread = None
        self.lock = threading.Lock()
        
        # Start GPS reading in background
        self.start_gps_reading()

    def find_gps_port(self):
        """Scan and return the first port that receives valid GPS NMEA data."""
        logging.info("Scanning serial ports for GPS device...")
        ports = serial.tools.list_ports.comports()

        for port in ports:
            try:
                with serial.Serial(port.device, BAUDRATE, timeout=1) as ser:
                    # Try to read multiple lines to ensure stable connection
                    valid_lines = 0
                    for _ in range(20):  # Try more lines for better reliability
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            valid_lines += 1
                            if valid_lines >= 3:  # Require at least 3 valid lines
                                logging.info(f"GPS device found on {port.device}")
                                return port.device
            except Exception as e:
                logging.debug(f"Failed to connect to {port.device}: {e}")
                continue

        logging.warning("GPS device not found.")
        return None

    def _gps_reading_thread(self):
        """Background thread for continuous GPS reading."""
        while self.is_running:
            if not self.port:
                self.port = self.find_gps_port()
                if not self.port:
                    time.sleep(5)  # Wait before retrying
                    continue

            try:
                with serial.Serial(self.port, BAUDRATE, timeout=1) as ser:
                    while self.is_running:
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            try:
                                msg = pynmea2.parse(line)
                                if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                    with self.lock:
                                        self.latitude = msg.latitude
                                        self.longitude = msg.longitude
                                        self.last_update = time.time()
                                        logging.debug(f"GPS Update: {self.latitude}, {self.longitude}")
                            except pynmea2.ParseError as e:
                                logging.debug(f"Parse error: {e}")
            except serial.SerialException as e:
                logging.error(f"Serial error: {e}")
                self.port = None  # Reset port to trigger reconnection
                time.sleep(5)  # Wait before retrying

    def start_gps_reading(self):
        """Start the GPS reading thread."""
        if not self.is_running:
            self.is_running = True
            self.gps_thread = threading.Thread(target=self._gps_reading_thread, daemon=True)
            self.gps_thread.start()

    def stop_gps_reading(self):
        """Stop the GPS reading thread."""
        self.is_running = False
        if self.gps_thread:
            self.gps_thread.join()

    def get_gps_data(self):
        """Get the latest GPS data with validation."""
        with self.lock:
            # Check if GPS data is stale (older than 5 seconds)
            if time.time() - self.last_update > 5:
                logging.warning("GPS data is stale")
                return None, None
            return self.latitude, self.longitude

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_gps_reading()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gps_reader = GPSReader()
    try:
        while True:
            lat, lon = gps_reader.get_gps_data()
            if lat is not None and lon is not None:
                print(f"GPS: {lat}, {lon}")
            time.sleep(1)
    except KeyboardInterrupt:
        gps_reader.stop_gps_reading()