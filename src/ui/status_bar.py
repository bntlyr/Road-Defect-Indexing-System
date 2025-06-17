from PyQt5.QtWidgets import QStatusBar, QLabel
from PyQt5.QtCore import Qt

class StatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # Left side widgets container
        left_widget = QLabel()
        left_widget.setStyleSheet("padding: 2px 5px;")
        
        # Time display (left side)
        self.status_time = QLabel("00:00:00")
        self.status_time.setStyleSheet("padding: 2px 5px;")
        self.addWidget(self.status_time)

        # FPS counter (left side)
        self.status_fps = QLabel("FPS: 0.0")
        self.status_fps.setStyleSheet("padding: 2px 5px;")
        self.addWidget(self.status_fps)

        # GPS status (left side)
        self.status_gps = QLabel("GPS: Disconnected")
        self.status_gps.setStyleSheet("padding: 2px 5px; color: #ff0000;")  # Red for disconnected
        self.addWidget(self.status_gps)

        # Status message (right side)
        self.status_message = QLabel("Ready")
        self.status_message.setStyleSheet("padding: 2px 5px;")
        self.addPermanentWidget(self.status_message)  # This will stick to the right

    def update_gps_status(self, is_connected, has_fix=False, custom_message=None):
        """Update GPS connection status"""
        if is_connected:
            if custom_message:
                self.status_gps.setText(custom_message)
                self.status_gps.setStyleSheet("padding: 2px 5px; color: #00ff00;")  # Red for disconnected
            elif has_fix:
                self.status_gps.setText("GPS: Connected")
                self.status_gps.setStyleSheet("padding: 2px 5px; color: #00ff00;")  # Red for disconnected
            else:
                self.status_gps.setText("GPS: No Fix")
                self.status_gps.setStyleSheet("padding: 2px 5px; color: #00ff00;")  # Red for disconnected
        else:
            self.status_gps.setText("GPS: Disconnected")
