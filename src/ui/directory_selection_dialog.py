from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout, QLabel, QFileDialog

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

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Input dir
        self.in_edit = QLineEdit(readOnly=True)
        in_btn = QPushButton("Browse…")
        in_btn.clicked.connect(self._browse_in)
        in_h = QHBoxLayout()
        in_h.addWidget(self.in_edit, 1)
        in_h.addWidget(in_btn)
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
