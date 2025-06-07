import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QProgressBar, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from .ui.dashboard import Dashboard

class LoadingScreen(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app  # Store the QApplication instance
        self.setWindowTitle("Road Defect System")
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        
        # Load and set the icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'public', 'icons', 'icon.png')
        self.icon_label = QLabel()
        pixmap = QPixmap(icon_path)
        scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(scaled_pixmap)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)
        
        # Add title
        title_label = QLabel("Road Defect System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(300, 4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #5865F2;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.status_label)
        
        # Set window background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #36393f;
            }
        """)
        
        # Initialize progress
        self.progress = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.initialize_components)
        self.timer.start(100)  # Check initialization status every 100ms
        
        # Initialize dashboard reference
        self.dashboard = None
        self.initialization_steps = [
            ("Initializing UI components...", 20),
            ("Setting up camera...", 40),
            ("Initializing GPS...", 60),
            ("Loading detection models...", 80),
            ("Finalizing setup...", 100)
        ]
        self.current_step = 0
        
    def initialize_components(self):
        """Initialize dashboard components and update progress"""
        if self.current_step >= len(self.initialization_steps):
            self.timer.stop()
            QTimer.singleShot(100, self.launch_dashboard)  # Use singleShot to ensure proper event handling
            return
            
        status, target_progress = self.initialization_steps[self.current_step]
        self.status_label.setText(status)
        
        # Simulate component initialization
        if self.progress < target_progress:
            self.progress += 1
            self.progress_bar.setValue(self.progress)
        else:
            self.current_step += 1
    
    def launch_dashboard(self):
        """Launch the main dashboard window"""
        try:
            self.dashboard = Dashboard()
            self.dashboard.show()
            self.close()  # Close loading screen after dashboard is shown
        except Exception as e:
            print(f"Error launching dashboard: {str(e)}")
            self.app.quit()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(54, 57, 63, 255))
        painter.drawRoundedRect(self.rect(), 10, 10)

def main():
    app = QApplication(sys.argv)
    loading_screen = LoadingScreen(app)
    loading_screen.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
