import sys
import os

# Ensure stdout is initialized
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')

# Now import the rest of your application
from PyQt5 import QtWidgets
from src.app.components.dashboard import Dashboard

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dashboard = Dashboard()
    dashboard.show()
    sys.exit(app.exec_())
