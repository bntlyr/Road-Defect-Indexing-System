import serial
import serial.tools.list_ports
import pynmea2
import logging
import time
import os

BAUDRATE = 9600
MAX_RETRIES = 5  # Increased from 3 to 5
RETRY_DELAY = 2  # Increased from 1 to 2 seconds
CONNECTION_TIMEOUT = 3  # Added timeout for connection attempts

class GPSReader:
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        # Then initialize other attributes
        self.port = None
        self.latitude = None
        self.longitude = None
        self.serial = None
        self.last_retry_time = 0
        self.retry_count = 0
        self.last_valid_data_time = 0
        self.connection_verified = False
        self.connected = False
        self.connection = None

        
        # Try to connect to GPS
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize GPS connection with improved reliability"""
        # First, get all available ports
        available_ports = self.get_available_ports()
        self.logger.info(f"Available ports: {available_ports}")
        
        # Try to connect to each port in order
        for port in available_ports:
            self.logger.info(f"Attempting to connect to port {port}")
            if self._try_connect_port(port):
                self.logger.info(f"Successfully connected to GPS on {port}")
                if self._connect():
                    return
                else:
                    self.logger.error(f"Failed to establish persistent connection on {port}")
        
        self.logger.warning("Failed to connect to GPS on any available port")

    def _try_connect_port(self, port):
        """Try to connect to a specific port and verify it's a GPS device"""
        try:
            # Check if port is already in use
            if self._is_port_in_use(port):
                self.logger.warning(f"Port {port} is already in use")
                return False

            with serial.Serial(port, BAUDRATE, timeout=CONNECTION_TIMEOUT) as ser:
                # Try to read a few lines to verify it's a GPS
                valid_data_count = 0
                start_time = time.time()
                
                while time.time() - start_time < CONNECTION_TIMEOUT:
                    try:
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                            try:
                                msg = pynmea2.parse(line)
                                if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                    valid_data_count += 1
                                    if valid_data_count >= 2:  # Require at least 2 valid readings
                                        self.port = port
                                        return True
                            except pynmea2.ParseError:
                                continue
                    except serial.SerialException:
                        break
                    
                    time.sleep(0.1)  # Small delay between reads
                
                self.logger.debug(f"Port {port} did not provide valid GPS data")
                return False
                
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
        """Establish a persistent connection to the GPS device with improved retry logic"""
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
            
            # Try to connect with a timeout
            self.serial = serial.Serial(self.port, BAUDRATE, timeout=CONNECTION_TIMEOUT)
            
            # Verify we can read data
            valid_data_count = 0
            start_time = time.time()
            
            while time.time() - start_time < CONNECTION_TIMEOUT:
                try:
                    line = self.serial.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                                valid_data_count += 1
                                if valid_data_count >= 2:  # Require at least 2 valid readings
                                    self.logger.info(f"Successfully connected to GPS on {self.port}")
                                    self.retry_count = 0
                                    self.connection_verified = True
                                    self.connected = True
                                    self.last_valid_data_time = time.time()
                                    self.connection = self.serial
                                    return True
                        except pynmea2.ParseError:
                            continue
                except serial.SerialException:
                    break
                
                time.sleep(0.1)  # Small delay between reads
            
            # If we get here, we couldn't read valid GPS data
            self.serial.close()
            self.serial = None
            self.connection_verified = False
            return False
            
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to GPS: {e}")
            self.serial = None
            self.connection_verified = False
            self.retry_count += 1
            self.last_retry_time = current_time
            time.sleep(RETRY_DELAY)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to GPS: {e}")
            self.serial = None
            self.connection_verified = False
            return False

    def get_available_ports(self):
        """Get list of available serial ports with detailed information"""
        ports = []
        for port in serial.tools.list_ports.comports():
            try:
                # Try to get more information about the port
                port_info = {
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'manufacturer': port.manufacturer,
                    'product': port.product,
                    'serial_number': port.serial_number
                }
                self.logger.debug(f"Found port: {port_info}")
                ports.append(port.device)
            except Exception as e:
                self.logger.debug(f"Error getting port info for {port.device}: {e}")
                ports.append(port.device)
        
        # Sort ports to prioritize common GPS port names
        priority_ports = ['COM3', 'COM4', 'COM5', 'COM6']  # Add more if needed
        sorted_ports = []
        
        # First add priority ports if they exist
        for priority_port in priority_ports:
            if priority_port in ports:
                sorted_ports.append(priority_port)
                ports.remove(priority_port)
        
        # Then add any remaining ports
        sorted_ports.extend(ports)
        
        return sorted_ports

    def _validate_nmea_checksum(self, sentence):
        """Validate NMEA sentence checksum"""
        try:
            # Remove any whitespace and newlines
            sentence = sentence.strip()
            
            # Check if sentence starts with $ and contains *
            if not sentence.startswith('$') or '*' not in sentence:
                return False
                
            # Split sentence and checksum
            sentence_part, checksum = sentence.split('*')
            
            # Calculate checksum
            calculated_checksum = 0
            for char in sentence_part[1:]:  # Skip the $ character
                calculated_checksum ^= ord(char)
                
            # Compare checksums
            return calculated_checksum == int(checksum, 16)
        except Exception:
            return False

    def _clean_nmea_sentence(self, sentence):
        """Clean and validate NMEA sentence"""
        try:
            # Remove any non-printable characters
            sentence = ''.join(char for char in sentence if char.isprintable())
            
            # Basic format validation
            if not sentence.startswith('$'):
                return None
                
            # Check for minimum length
            if len(sentence) < 10:  # Minimum valid NMEA sentence length
                return None
                
            # Validate checksum
            if not self._validate_nmea_checksum(sentence):
                return None
                
            return sentence
        except Exception:
            return None

    def _read_nmea_sentence(self, timeout=None):
        """Read a complete NMEA sentence with optional timeout"""
        if not self.is_connected():
            return None

        try:
            # Set read timeout if specified
            if timeout is not None:
                self.serial.timeout = timeout
            else:
                self.serial.timeout = 1.0  # Default timeout

            # Read a single line with timeout
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    return None

                # Clean and validate the sentence
                cleaned_line = self._clean_nmea_sentence(line)
                if cleaned_line:
                    return cleaned_line

            except serial.SerialTimeoutException:
                return None
            except Exception as e:
                logging.debug(f"Error reading NMEA sentence: {e}")
                return None

        except Exception as e:
            logging.debug(f"Error in _read_nmea_sentence: {e}")
            return None

    def read_gps_data(self, timeout=None):
        """Read GPS data with optional timeout for non-blocking operation"""
        if not self.is_connected():
            return None, None

        try:
            # Read NMEA sentence with timeout
            sentence = self._read_nmea_sentence(timeout=timeout)
            if not sentence:
                return None, None

            # Parse the sentence
            try:
                parsed_data = pynmea2.parse(sentence)
                if isinstance(parsed_data, pynmea2.GGA):
                    if parsed_data.gps_qual > 0:  # Only use data with valid fix
                        self.latitude = parsed_data.latitude
                        self.longitude = parsed_data.longitude
                        self.last_valid_data_time = time.time()
                        return self.latitude, self.longitude
                elif isinstance(parsed_data, pynmea2.RMC):
                    if parsed_data.status == 'A':  # Only use data with valid fix
                        self.latitude = parsed_data.latitude
                        self.longitude = parsed_data.longitude
                        self.last_valid_data_time = time.time()
                        return self.latitude, self.longitude
            except pynmea2.ParseError as e:
                logging.debug(f"Parse error: {e}")
            except Exception as e:
                logging.debug(f"Error parsing GPS data: {e}")

        except Exception as e:
            logging.debug(f"Error reading GPS data: {e}")

        return None, None

    def get_gps_data(self):
        """Alias for read_gps_data to maintain compatibility."""
        return self.read_gps_data()

    def is_connected(self):
        """Check if GPS is connected and verified"""
        return (self.port is not None and 
                self.serial is not None and 
                self.serial.is_open and 
                self.connection_verified and 
                time.time() - self.last_valid_data_time < 10)  # Consider disconnected if no valid data for 10 seconds

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

    def disconnect(self):
        """Disconnect from the GPS device."""
        if self.is_connected():
            # Logic to disconnect from the GPS device
            # This could involve closing serial connections, etc.
            if self.connection:
                self.connection.close()  # Example, adjust based on your implementation
            self.connected = False  # Update the connection status

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