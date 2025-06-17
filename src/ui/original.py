import sys
import time
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QComboBox,
    QPushButton,
    QSlider,
    QGroupBox,
    QStatusBar,
    QSplitter,
    QFrame,
    QMessageBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QImage, QPixmap, QIcon

# ==========================================================
# Helper / Re‑usable Widgets
# ==========================================================

class DonutWidget(QFrame):
    """Simple donut (ring) chart for displaying a single number."""

    def __init__(self, title: str, count: int = 0, parent=None):
        super().__init__(parent)
        self.title = title
        self.count = count
        self.setMinimumSize(150, 150)

    # ------------------------------------------------------------------
    # Painting                                                        
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background (transparent with our window palette)
        painter.setBrush(self.palette().window())
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        # Ring geometry
        width = self.width()
        height = self.height()
        padding_bottom = 22  # leave room for title text
        outer_radius = int(min(width, height - padding_bottom) * 0.4)
        inner_radius = int(outer_radius * 0.55)
        center_x = width // 2
        center_y = (height - padding_bottom) // 2

        # Draw outer ring
        painter.setBrush(QColor(70, 130, 180))  # steel‑blue
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - outer_radius, center_y - outer_radius, outer_radius * 2, outer_radius * 2)

        # Draw inner hole
        painter.setBrush(self.palette().window())
        painter.drawEllipse(center_x - inner_radius, center_y - inner_radius, inner_radius * 2, inner_radius * 2)

        # Count text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(QRect(center_x - outer_radius, center_y - 12, outer_radius * 2, 24), Qt.AlignCenter, str(self.count))

        # Title text
        painter.setFont(QFont("Arial", 12))
        painter.drawText(QRect(0, height - padding_bottom, width, padding_bottom), Qt.AlignCenter, self.title)


class SettingsDialog(QDialog):
    """Pure‑UI settings — no backend connections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        self._setup_ui()
        self._apply_dark_theme()

    # ..................................................................
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Output directory field
        self.output_dir_edit = QLineEdit(readOnly=True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dir)
        h = QHBoxLayout()
        h.addWidget(self.output_dir_edit, 1)
        h.addWidget(browse_btn)
        layout.addLayout(h)

        # Dummy slider for confidence (visual only)
        form = QFormLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_value_lbl = QLabel("0.25")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value_lbl.setText(f"{v/100:.2f}"))
        s_h = QHBoxLayout()
        s_h.addWidget(self.conf_slider)
        s_h.addWidget(self.conf_value_lbl)
        form.addRow("Confidence Threshold:", s_h)
        layout.addLayout(form)

        # Buttons
        btn_h = QHBoxLayout()
        btn_h.addStretch(1)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_h.addWidget(save_btn)
        btn_h.addWidget(cancel_btn)
        layout.addLayout(btn_h)

    def _browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", "~")
        if directory:
            self.output_dir_edit.setText(directory)

    def _apply_dark_theme(self):
        self.setStyleSheet(
            """
            QDialog {background:#2b2b2b; color:#f0f0f0;}
            QPushButton {background:#4a4a4a; color:#ddd; padding:6px; border:none; border-radius:4px;}
            QPushButton:hover {background:#5a5a5a;}
            QLineEdit {background:#3b3b3b; color:#f0f0f0; border:1px solid #555; border-radius:4px; padding:4px;}
            QSlider::groove:horizontal {height:8px; background:#4a4a4a; border:1px solid #999; border-radius:4px;}
            QSlider::handle:horizontal {width:18px; background:#4a90e2; border:1px solid #5c5c5c; margin:-2px 0; border-radius:9px;}
            """
        )


class DirectorySelectionDialog(QDialog):
    """Utility dialog reused in analysis; purely UI, no backend work."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Directories")
        self.setMinimumSize(600, 200)
        self._in_dir = ""
        self._out_dir = ""
        self._setup_ui()
        self._apply_dark_theme()

    # ..................................................................
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Input dir
        self.in_edit = QLineEdit(readOnly=True)
        in_btn = QPushButton("Browse…")
        in_btn.clicked.connect(self._browse_in)
        in_h = QHBoxLayout()
        in_h.addWidget(self.in_edit, 1)
        in_h.addWidget(in_btn)
        layout.addRow = layout.addLayout  # alias
        layout.addLayout(in_h)

        # Output dir
        self.out_edit = QLineEdit(readOnly=True)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._browse_out)
        out_h = QHBoxLayout()
        out_h.addWidget(self.out_edit, 1)
        out_h.addWidget(out_btn)
        layout.addLayout(out_h)

        # Buttons
        btn_h = QHBoxLayout()
        btn_h.addStretch(1)
        ok_btn = QPushButton("Proceed")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_h.addWidget(cancel_btn)
        btn_h.addWidget(ok_btn)
        layout.addLayout(btn_h)

    def _browse_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Directory", "~")
        if d:
            self._in_dir = d
            self.in_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", "~")
        if d:
            self._out_dir = d
            self.out_edit.setText(d)

    def get_directories(self):
        return self._in_dir, self._out_dir

    def _apply_dark_theme(self):
        self.setStyleSheet(
            """
            QDialog {background:#2b2b2b; color:#f0f0f0;}
            QPushButton {background:#4a4a4a; color:#ddd; padding:6px; border:none; border-radius:4px;}
            QPushButton:hover {background:#5a5a5a;}
            QLineEdit {background:#3b3b3b; color:#f0f0f0; border:1px solid #555; border-radius:4px; padding:4px;}
            """
        )


# ==========================================================
# Main Dashboard (Frontend‑only Stub)
# ==========================================================

class Dashboard(QMainWindow):
    """Frontend‑only refactor of the original dashboard.

    All backend‑heavy modules (camera, GPS, detection, cloud, etc.) have been
    completely removed or stubbed, leaving a fully functional PyQt5 GUI that
    you can further wire to your own services.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Defect Indexing System")
        self.setMinimumSize(1000, 700)

        # Internal state purely for UI toggling
        self._detecting = False
        self._gps_connected = False

        self._init_controls()
        self._init_ui()
        self._init_timers()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _init_controls(self):
        # Video related
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])

        self.flip_btn = QPushButton("Flip Camera")
        self.flip_btn.clicked.connect(self._on_flip)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 500)
        self.zoom_slider.setValue(100)
        self.zoom_label = QLabel("Zoom: 1.00x")
        self.zoom_slider.valueChanged.connect(lambda v: self.zoom_label.setText(f"Zoom: {v/100:.2f}x"))

        # New file upload button
        self.upload_btn = QPushButton("Upload File")
        self.upload_btn.clicked.connect(self._upload_file)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)
        self.brightness_label = QLabel("Brightness: 50")
        self.brightness_slider.valueChanged.connect(lambda v: self.brightness_label.setText(f"Brightness: {v}"))

        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(0, 100)
        self.exposure_slider.setValue(50)
        self.exposure_label = QLabel("Exposure: 50")
        self.exposure_slider.valueChanged.connect(lambda v: self.exposure_label.setText(f"Exposure: {v}"))

        # Main buttons
        self.detect_btn = QPushButton("Start Detection")
        self.detect_btn.clicked.connect(self._toggle_detection)

        self.gps_btn = QPushButton("Connect GPS")
        self.gps_btn.clicked.connect(self._toggle_gps)

        self.analysis_btn = QPushButton("Run Analysis")
        self.analysis_btn.clicked.connect(self._run_analysis)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(lambda: SettingsDialog(self).exec_())

        self.cloud_btn = QPushButton("Connect Cloud")
        self.cloud_btn.clicked.connect(self._connect_cloud_placeholder)

        # Statistics donuts
        self.donut_linear = DonutWidget("Linear Cracks", 0)
        self.donut_alligator = DonutWidget("Alligator Cracks", 0)
        self.donut_pothole = DonutWidget("Potholes", 0)

        self.lat_label = QLabel("Latitude: 0.0")
        self.lon_label = QLabel("Longitude: 0.0")

    def _init_ui(self):
        self.setStyleSheet(
            """
            QWidget {background:#2b2b2b; color:#f0f0f0; font-size:14px;}
            QPushButton {background:#4a4a4a; color:#ddd; padding:8px; border:none; border-radius:4px;}
            QPushButton:hover {background:#6a6a6a;}
            QComboBox {background:#4a4a4a; color:#ddd; padding:4px;}
            """
        )

        # Central split view (video placeholder)
        splitter = QSplitter(Qt.Horizontal)
        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#1e1e1e; font-size:20px;")
        splitter.addWidget(self.video_label)

        # Controls section under the splitter
        controls_layout = self._build_controls_section()

        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(splitter, 8)
        v.addLayout(controls_layout, 2)
        self.setCentralWidget(central)

        # Status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_time = QLabel("00:00:00")
        self._status_fps = QLabel("FPS: 0")
        self._status_gps = QLabel("GPS: Disconnected")
        for w in (self._status_time, self._status_fps, self._status_gps):
            w.setStyleSheet("margin:0 10px;")
            sb.addWidget(w)

    def _build_controls_section(self):
        # Video group ---------------------------------------------------
        video_group = QGroupBox("Video Control")
        v_layout = QVBoxLayout(video_group)
        v_layout.addWidget(QLabel("Camera:"))
        v_layout.addWidget(self.camera_combo)
        v_layout.addWidget(self.flip_btn)

        # Add the file upload button here
        v_layout.addWidget(self.upload_btn)

        hl = QHBoxLayout(); hl.addWidget(QLabel("Zoom:")); hl.addWidget(self.zoom_slider); hl.addWidget(self.zoom_label)
        v_layout.addLayout(hl)

        # Stats group ---------------------------------------------------
        stats_group = QGroupBox("Statistics Logs")
        s_layout = QVBoxLayout(stats_group)
        d_layout = QHBoxLayout()
        for d in (self.donut_linear, self.donut_alligator, self.donut_pothole):
            d_layout.addWidget(d)
        s_layout.addLayout(d_layout)

        # Update latitude and longitude labels for better visibility
        self.lat_label.setStyleSheet("font-size: 16px; color: #f0f0f0; padding: 5px; border: 1px solid #ffffff; border-radius: 4px;")  # White border for latitude
        self.lon_label.setStyleSheet("font-size: 16px; color: #f0f0f0; padding: 5px; border: 1px solid #ffffff; border-radius: 4px;")  # White border for longitude
        gps_layout = QHBoxLayout()
        gps_layout.addWidget(self.lon_label)
        gps_layout.addWidget(self.lat_label)
        s_layout.addLayout(gps_layout)

        # Main group ----------------------------------------------------
        main_group = QGroupBox("Main Control")
        m_layout = QVBoxLayout(main_group)
        for b in (self.detect_btn, self.gps_btn, self.analysis_btn, self.cloud_btn, self.settings_btn):
            m_layout.addWidget(b)

        # Combine -------------------------------------------------------
        root = QHBoxLayout()
        root.addWidget(video_group, 1)
        root.addWidget(stats_group, 1)
        root.addWidget(main_group, 1)
        return root

    def _init_timers(self):
        # Update clock every second
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick)
        self._clock.start(1000)

    # ------------------------------------------------------------------
    # Slots / Event handlers
    # ------------------------------------------------------------------
    def _tick(self):
        self._status_time.setText(time.strftime("%H:%M:%S"))

    def _on_flip(self):
        QMessageBox.information(self, "Flip", "Flip camera request sent (stub).")

    def _toggle_detection(self):
        self._detecting = not self._detecting
        self.detect_btn.setText("Stop Detection" if self._detecting else "Start Detection")
        self.detect_btn.setStyleSheet("background:#ff6f61; color:white;" if self._detecting else "")
        # Purely UI — no backend.

    def _toggle_gps(self):
        self._gps_connected = not self._gps_connected
        self.gps_btn.setText("Disconnect GPS" if self._gps_connected else "Connect GPS")
        self._status_gps.setText("GPS: Connected" if self._gps_connected else "GPS: Disconnected")
        # No actual GPS handling.

    def _run_analysis(self):
        dlg = DirectorySelectionDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            in_dir, out_dir = dlg.get_directories()
            QMessageBox.information(self, "Analysis", f"Pretend analysis will run on:\nInput: {in_dir}\nOutput: {out_dir}")

    def _connect_cloud_placeholder(self):
        QMessageBox.information(self, "Cloud", "Cloud connection functionality is not implemented in this frontend‑only build.")

    def _upload_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            QMessageBox.information(self, "File Selected", f"You selected: {file_name}")


# ==========================================================
# Main entry‑point
# ==========================================================

def main():
    app = QApplication(sys.argv)
    # Set window icon using dynamic relative path
    icon_path = os.path.join('..', '..', 'public', 'icons','icon.png')
    window = Dashboard()
    window.setWindowIcon(QIcon(icon_path))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()