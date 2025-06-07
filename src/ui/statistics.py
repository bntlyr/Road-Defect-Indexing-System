from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QFrame

class DonutWidget(QFrame):
    def __init__(self, title: str, count: int = 0, parent=None):
        super().__init__(parent)
        self.title = title
        self.count = count
        self.setMinimumSize(150, 150)

    def update(self):
        self.repaint()  # Trigger a repaint of the widget

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(self.palette().window())
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        width = self.width()
        height = self.height()
        padding_bottom = 22
        outer_radius = int(min(width, height - padding_bottom) * 0.4)
        inner_radius = int(outer_radius * 0.55)
        center_x = width // 2
        center_y = (height - padding_bottom) // 2

        painter.setBrush(QColor(70, 130, 180))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - outer_radius, center_y - outer_radius,
                            outer_radius * 2, outer_radius * 2)

        painter.setBrush(self.palette().window())
        painter.drawEllipse(center_x - inner_radius, center_y - inner_radius,
                            inner_radius * 2, inner_radius * 2)

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(center_x - outer_radius, center_y - 12,
                         outer_radius * 2, 24, Qt.AlignCenter, str(self.count))

        painter.setFont(QFont("Arial", 12))
        painter.drawText(0, height - padding_bottom, width, padding_bottom,
                         Qt.AlignCenter, self.title)

class Statistics(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Statistics Logs", parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        donut_layout = QHBoxLayout()
        self.donut_linear = DonutWidget("Linear Cracks", 0)
        self.donut_alligator = DonutWidget("Alligator Cracks", 0)
        self.donut_pothole = DonutWidget("Potholes", 0)
        donut_layout.addWidget(self.donut_linear)
        donut_layout.addWidget(self.donut_alligator)
        donut_layout.addWidget(self.donut_pothole)
        layout.addLayout(donut_layout)

        self.lat_label = QLabel("Latitude: 0.0")
        self.lon_label = QLabel("Longitude: 0.0")
        self.lat_label.setStyleSheet("font-size: 16px; color: #f0f0f0; padding: 5px; border: 1px solid #ffffff; border-radius: 4px;")
        self.lon_label.setStyleSheet("font-size: 16px; color: #f0f0f0; padding: 5px; border: 1px solid #ffffff; border-radius: 4px;")
        gps_layout = QHBoxLayout()
        gps_layout.addWidget(self.lon_label)
        gps_layout.addWidget(self.lat_label)
        layout.addLayout(gps_layout)

        self.setLayout(layout)

    def update_gps(self, lat, lon):
        """Update GPS coordinates display"""
        self.lat_label.setText(f"Latitude: {lat:.6f}")
        self.lon_label.setText(f"Longitude: {lon:.6f}")
