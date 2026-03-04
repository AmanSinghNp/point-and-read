"""
output_panel.py — Right panel: recognized text display + TTS controls.

Provides:
    display_text(str)   — slot to show recognized text
    show_spinner()      — show loading animation
    hide_spinner()      — hide loading animation
"""

import os
import pyttsx3

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QMovie, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSlider, QApplication, QSizePolicy,
)


# ── TTS Worker (runs in its own thread) ─────────────────────────
class TTSWorker(QObject):
    """Runs pyttsx3 speech in a background thread to avoid UI freeze."""
    finished = pyqtSignal()

    def __init__(self, text: str, rate: int = 150):
        super().__init__()
        self.text = text
        self.rate = rate

    @pyqtSlot()
    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            engine.say(self.text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            self.finished.emit()


class OutputPanel(QWidget):
    """Right panel showing recognized text, spinner, TTS, and clipboard copy."""
    submit_correction_requested = pyqtSignal(str)
    append_notes_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tts_thread: QThread | None = None
        self._tts_worker: TTSWorker | None = None
        self._build_ui()

    # ── UI Construction ─────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("Recognized Text")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #1E2A3A;")
        layout.addWidget(title)

        # Loading spinner (shown during inference)
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner_label.setFixedHeight(80)
        self.spinner_label.hide()

        spinner_path = os.path.join(os.path.dirname(__file__), "..", "assets", "spinner.gif")
        spinner_path = os.path.abspath(spinner_path)
        if os.path.exists(spinner_path):
            self._movie = QMovie(spinner_path)
            # Use a fixed square size to avoid stretching
            from PyQt6.QtCore import QSize
            self._movie.setScaledSize(QSize(64, 64))
            self.spinner_label.setMovie(self._movie)
        else:
            self._movie = None
            self.spinner_label.setText("Processing...")
            self.spinner_label.setStyleSheet("color: #4A90D9; font-size: 16px;")

        layout.addWidget(self.spinner_label)

        # Status label (shows "Processing..." text)
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #4A90D9; font-size: 14px; font-weight: 500;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Text display
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)
        self.text_edit.setFont(QFont("Inter", 16))
        self.text_edit.setPlaceholderText("Recognized text will appear here...")
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background: #FFFFFF;
                color: #1E2A3A;
                border: 1px solid #E0E5EC;
                padding: 16px;
                selection-background-color: #4A90D9;
            }
        """)
        layout.addWidget(self.text_edit)

        # Confidence display
        self.confidence_label = QLabel()
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.confidence_label.setStyleSheet("""
            font-size: 13px;
            font-weight: 500;
            color: #8A9BB0;
            padding: 4px 0;
        """)
        self.confidence_label.hide()
        layout.addWidget(self.confidence_label)

        # ── TTS speed slider ────────────────────────────────────────
        speed_row = QHBoxLayout()
        speed_label = QLabel("Speech Speed:")
        speed_label.setStyleSheet("color: #8A9BB0; font-size: 13px;")
        speed_row.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(80, 300)
        self.speed_slider.setValue(150)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(20)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E5EC;
            }
            QSlider::handle:horizontal {
                background: #4A90D9;
                width: 18px;
                height: 18px;
                margin: -6px 0;
            }
            QSlider::sub-page:horizontal {
                background: #4A90D9;
            }
        """)

        self.speed_value_label = QLabel("150 wpm")
        self.speed_value_label.setStyleSheet("color: #8A9BB0; font-size: 13px; min-width: 60px;")
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_value_label.setText(f"{v} wpm")
        )

        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_value_label)
        layout.addLayout(speed_row)

        # ── Action buttons ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_read = QPushButton("Read Aloud")
        self.btn_read.setObjectName("btnRead")
        self.btn_read.clicked.connect(self._read_aloud)

        self.btn_copy = QPushButton("Copy to Clipboard")
        self.btn_copy.setObjectName("btnCopy")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)

        self.btn_append = QPushButton("Append to Notes")
        self.btn_append.setObjectName("btnAppendNotes")
        self.btn_append.clicked.connect(self._append_to_notes)

        for btn in (self.btn_read, self.btn_copy, self.btn_append):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(40)
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

        # Correction submission button (human-in-the-loop dataset logging)
        self.btn_submit_correction = QPushButton("Submit Correction")
        self.btn_submit_correction.setObjectName("btnSubmitCorrection")
        self.btn_submit_correction.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_submit_correction.setMinimumHeight(38)
        self.btn_submit_correction.clicked.connect(self._submit_correction)
        layout.addWidget(self.btn_submit_correction)

    # ── Public Slots ────────────────────────────────────────────────
    @pyqtSlot(str, float)
    def display_text(self, text: str, confidence: float = -1.0):
        """Show recognized text and hide spinner.

        Args:
            text: The recognized text to display.
            confidence: Confidence score (0.0–1.0). If negative, hides the label.
        """
        self.hide_spinner()
        self.text_edit.setPlainText(text)

        if confidence >= 0:
            pct = confidence * 100
            # Color-code: green (>80%), orange (50-80%), red (<50%)
            if pct >= 80:
                color = "#2ECC71"
            elif pct >= 50:
                color = "#F39C12"
            else:
                color = "#E74C3C"
            self.confidence_label.setText(f"Confidence: {pct:.1f}%")
            self.confidence_label.setStyleSheet(f"""
                font-size: 13px;
                font-weight: 600;
                color: {color};
                padding: 4px 0;
            """)
            self.confidence_label.show()
        else:
            self.confidence_label.hide()

    @pyqtSlot(list)
    def display_text_results(self, results: list):
        """Show multi-line recognition results (combined text + average confidence)."""
        combined = "\n".join(r["text"] for r in results)
        avg_conf = (
            sum(r["confidence"] for r in results) / len(results)
            if results else 0.0
        )
        self.display_text(combined, avg_conf)

    def show_spinner(self):
        """Show loading animation during inference."""
        self.text_edit.clear()
        self.spinner_label.show()
        self.status_label.setText("Recognizing handwriting...")
        self.status_label.show()
        if self._movie:
            self._movie.start()

    def hide_spinner(self):
        """Stop and hide the loading animation."""
        if self._movie:
            self._movie.stop()
        self.spinner_label.hide()
        self.status_label.hide()

    # ── TTS ─────────────────────────────────────────────────────────
    def _read_aloud(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            return

        # Don't start a new read if one is running
        if self._tts_thread is not None and self._tts_thread.isRunning():
            return

        rate = self.speed_slider.value()
        self._tts_worker = TTSWorker(text, rate)
        self._tts_thread = QThread()
        self._tts_worker.moveToThread(self._tts_thread)
        self._tts_thread.started.connect(self._tts_worker.run)
        self._tts_worker.finished.connect(self._tts_thread.quit)
        self._tts_worker.finished.connect(self._on_tts_done)
        self._tts_thread.start()
        self.btn_read.setText("Speaking...")
        self.btn_read.setEnabled(False)

    def _on_tts_done(self):
        self.btn_read.setText("Read Aloud")
        self.btn_read.setEnabled(True)

    # ── Clipboard ───────────────────────────────────────────────────
    def _copy_to_clipboard(self):
        text = self.text_edit.toPlainText().strip()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.btn_copy.setText("Copied!")
            # Reset button text after 2 seconds
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.btn_copy.setText("Copy to Clipboard"))

    def _submit_correction(self):
        self.submit_correction_requested.emit(self.text_edit.toPlainText())

    def _append_to_notes(self):
        self.append_notes_requested.emit(self.text_edit.toPlainText())
