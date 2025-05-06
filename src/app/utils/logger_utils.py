import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    def __init__(self, name: str, log_dir: str = "logs"):
        """
        Initialize the Logger with a name and log directory.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create handlers
        self._setup_handlers()
        
    def _setup_handlers(self) -> None:
        """Set up file and console handlers for logging."""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message: Debug message
        """
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message: Info message
        """
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Warning message
        """
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: Error message
        """
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """
        Log a critical message.
        
        Args:
            message: Critical message
        """
        self.logger.critical(message)
        
    def exception(self, message: str) -> None:
        """
        Log an exception message with traceback.
        
        Args:
            message: Exception message
        """
        self.logger.exception(message)

class SystemLogger(Logger):
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the SystemLogger for system-wide logging.
        
        Args:
            log_dir: Directory for log files
        """
        super().__init__("system", log_dir)
        
    def log_system_start(self) -> None:
        """Log system startup."""
        self.info("System starting up")
        
    def log_system_shutdown(self) -> None:
        """Log system shutdown."""
        self.info("System shutting down")
        
    def log_camera_status(self, status: str) -> None:
        """
        Log camera status.
        
        Args:
            status: Camera status message
        """
        self.info(f"Camera status: {status}")
        
    def log_gps_status(self, status: str) -> None:
        """
        Log GPS status.
        
        Args:
            status: GPS status message
        """
        self.info(f"GPS status: {status}")
        
    def log_detection(self, defect_type: str, confidence: float, location: Optional[tuple] = None) -> None:
        """
        Log defect detection.
        
        Args:
            defect_type: Type of defect detected
            confidence: Detection confidence
            location: Optional GPS location (lat, lon)
        """
        message = f"Detected {defect_type} with confidence {confidence:.2f}"
        if location:
            message += f" at location {location}"
        self.info(message)
        
    def log_error(self, component: str, error: str) -> None:
        """
        Log component error.
        
        Args:
            component: Component name
            error: Error message
        """
        self.error(f"{component} error: {error}")
        
    def log_performance(self, component: str, metric: str, value: float) -> None:
        """
        Log performance metric.
        
        Args:
            component: Component name
            metric: Metric name
            value: Metric value
        """
        self.debug(f"{component} {metric}: {value:.2f}") 