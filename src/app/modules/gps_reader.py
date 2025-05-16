import serial
import serial.tools.list_ports
import pynmea2

BAUDRATE = 9600

class GPSReader:
    def __init__(self):
        self.port = self.find_gps_port()
        self.latitude = None
        self.longitude = None

    def find_gps_port(self):
        """Scan and return the first port that receives valid GPS NMEA data."""
        print("Scanning serial ports for GPS device...")
        ports = serial.tools.list_ports.comports()

        for port in ports:
            try:
                with serial.Serial(port.device, BAUDRATE, timeout=1) as ser:
                    for _ in range(10):  # Try 10 lines
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            print(f"GPS device found on {port.device}")
                            return port.device
            except Exception as e:
                continue

        print("GPS device not found.")
        return None

    def read_gps_data(self):
        """Read GPS data and update latitude and longitude."""
        if self.port:
            try:
                with serial.Serial(self.port, BAUDRATE, timeout=1) as ser:
                    print(f"Reading GPS data from {self.port}...")
                    while True:
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            try:
                                msg = pynmea2.parse(line)
                                if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                    self.latitude = msg.latitude
                                    self.longitude = msg.longitude
                                    return self.latitude, self.longitude
                            except pynmea2.ParseError as e:
                                print(f"Parse error: {e}")
            except serial.SerialException as e:
                print(f"Serial error: {e}")
        return None, None

if __name__ == "__main__":
    gps_reader = GPSReader()
    port = gps_reader.find_gps_port()
    while port:
        print(type(gps_reader.read_gps_data()))