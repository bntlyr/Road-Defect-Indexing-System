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
        self.status_gps.setStyleSheet("padding: 2px 5px;")
        self.addWidget(self.status_gps)

        # Status message (right side)
        self.status_message = QLabel("Ready")
        self.status_message.setStyleSheet("padding: 2px 5px;")
        self.addPermanentWidget(self.status_message)  # This will stick to the right

    def update_gps_status(self, connected, has_fix=False):
        """Update GPS status display"""
        if connected:
            if has_fix:
                self.status_gps.setText("GPS: Connected")
                self.status_gps.setStyleSheet("padding: 2px 5px; color: #00ff00;")  # Green for connected
            else:
                self.status_gps.setText("GPS: Connected (No Fix)")
                self.status_gps.setStyleSheet("padding: 2px 5px; color: #ffff00;")  # Yellow for no fix
        else:
            self.status_gps.setText("GPS: Disconnected")
            self.status_gps.setStyleSheet("padding: 2px 5px; color: #ff0000;")  # Red for disconnected
