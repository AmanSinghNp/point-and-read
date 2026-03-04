"""
main.py — Entry point for the Point & Read GUI application.

Usage:
    python main.py
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # Set a nice default font
    app.setFont(QFont("Inter", 10))

    # Set application metadata
    app.setApplicationName("Point & Read")
    app.setOrganizationName("TextRecognition")
    app.setApplicationVersion("1.0.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
