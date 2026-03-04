"""
input_panel.py â€” Left panel: drag-and-drop image zone + webcam capture.

Signals:
    image_ready(np.ndarray)  â€” emitted when an image is loaded or snapped.
    snap_ready(np.ndarray)   â€” emitted only when user clicks Snap (for auto-fire in Live Mode).
"""

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMimeData
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSizePolicy, QCheckBox,
)


def numpy_to_qpixmap(image: np.ndarray, max_width: int = 520, max_height: int = 400) -> QPixmap:
    """Convert a numpy image (grayscale or BGR) to QPixmap, scaled to fit."""
    if len(image.shape) == 2:
        h, w = image.shape
        qimg = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, ch = image.shape
        if ch == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            qimg = QImage(rgb.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(
        max_width, max_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


# Blur threshold for snap rejection (Laplacian variance)
BLUR_THRESHOLD = 100

# Motion stability: mean abs diff between frames (lower = more stable)
MOTION_THRESHOLD = 5.0
MOTION_STABLE_FRAMES = 3

DROP_ZONE_STYLE = """
    QLabel {
        border: 2px dashed #4A90D9;
        background: #FFFFFF;
        color: #8A9BB0;
        font-size: 15px;
        padding: 20px;
    }
"""


def _blur_score(image: np.ndarray) -> float:
    """Compute Laplacian variance as blur metric. Higher = sharper."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _motion_score(current: np.ndarray, previous: np.ndarray) -> float:
    """Mean absolute difference between frames. Lower = more stable."""
    if previous is None or current is None:
        return 0.0
    if current.shape != previous.shape:
        return float("inf")  # Treat as unstable
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if len(current.shape) == 3 else current
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if len(previous.shape) == 3 else previous
    return float(np.abs(curr_gray.astype(float) - prev_gray.astype(float)).mean())


class InputPanel(QWidget):
    """Left panel with drag-and-drop zone, file picker, and webcam capture."""

    image_ready = pyqtSignal(np.ndarray)
    snap_ready = pyqtSignal(np.ndarray)  # Emitted only on Snap (for Live Mode auto-fire)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: np.ndarray | None = None
        self._webcam_active = False
        self._cap = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._read_frame)
        self._live_mode = True
        self._apply_spell_check = True
        self._prev_frame: np.ndarray | None = None
        self._motion_stable_count = 0

        self._build_ui()

    # â”€â”€ UI Construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("Input Image")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #1E2A3A;")
        layout.addWidget(title)

        # Drop zone
        self.drop_label = QLabel("Drag & drop an image here\nor use the buttons below")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setMinimumSize(480, 320)
        self.drop_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.drop_label.setStyleSheet(DROP_ZONE_STYLE)
        self.drop_label.setAcceptDrops(True)
        layout.addWidget(self.drop_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_open = QPushButton("Open File")
        self.btn_open.setObjectName("btnOpen")
        self.btn_open.clicked.connect(self._open_file)

        self.btn_webcam = QPushButton("Capture from Webcam")
        self.btn_webcam.setObjectName("btnWebcam")
        self.btn_webcam.clicked.connect(self._toggle_webcam)

        self.btn_snap = QPushButton("Snap")
        self.btn_snap.setObjectName("btnSnap")
        self.btn_snap.clicked.connect(self._snap_frame)
        self.btn_snap.setEnabled(False)

        for btn in (self.btn_open, self.btn_webcam, self.btn_snap):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(40)
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)
        # Live Mode + Spell Check toggles
        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(16)

        self.live_mode_check = QCheckBox("Live Mode (snap -> OCR automatically)")
        self.live_mode_check.setChecked(True)
        self.live_mode_check.setStyleSheet("font-size: 13px; color: #1E2A3A;")
        self.live_mode_check.toggled.connect(self._on_live_mode_toggled)
        toggles_row.addWidget(self.live_mode_check)

        self.spell_check_check = QCheckBox("Spell Check")
        self.spell_check_check.setChecked(True)
        self.spell_check_check.setStyleSheet("font-size: 13px; color: #1E2A3A;")
        self.spell_check_check.toggled.connect(self._on_spell_check_toggled)
        toggles_row.addWidget(self.spell_check_check)
        toggles_row.addStretch()

        layout.addLayout(toggles_row)

        # Enable drag-and-drop on the whole widget
        self.setAcceptDrops(True)

    # â”€â”€ Drag & Drop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet(self.drop_label.styleSheet().replace(
                "border: 2px dashed #4A90D9", "border: 2px solid #3A7BC8"
            ))

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet(self.drop_label.styleSheet().replace(
            "border: 2px solid #3A7BC8", "border: 2px dashed #4A90D9"
        ))

    def dropEvent(self, event: QDropEvent):
        self.drop_label.setStyleSheet(self.drop_label.styleSheet().replace(
            "border: 2px solid #3A7BC8", "border: 2px dashed #4A90D9"
        ))
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_image_from_path(path)

    # â”€â”€ File Picker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if path:
            self._load_image_from_path(path)

    def _load_image_from_path(self, path: str):
        image = cv2.imread(path)
        if image is None:
            self.drop_label.setText(f"Failed to load:\n{path}")
            return
        self._stop_webcam()
        self._current_image = image
        self.drop_label.setPixmap(numpy_to_qpixmap(image))
        self.image_ready.emit(image)

    # â”€â”€ Webcam â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _toggle_webcam(self):
        if self._webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.drop_label.setText("Could not open webcam")
            return
        self._webcam_active = True
        self._prev_frame = None
        self._motion_stable_count = 0
        self.btn_webcam.setText("Stop Webcam")
        self.btn_snap.setEnabled(False)  # Enabled once 3 stable frames in _read_frame
        self.btn_open.setEnabled(False)
        self._timer.start()

    def _stop_webcam(self):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._webcam_active = False
        self.btn_webcam.setText("Capture from Webcam")
        self.btn_snap.setEnabled(False)
        self.btn_open.setEnabled(True)

    def _read_frame(self):
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            # Motion stability gate
            motion = _motion_score(frame, self._prev_frame)
            if motion < MOTION_THRESHOLD:
                self._motion_stable_count = min(
                    self._motion_stable_count + 1, MOTION_STABLE_FRAMES
                )
            else:
                self._motion_stable_count = 0
            self._prev_frame = frame.copy()

            self._current_image = frame
            self.btn_snap.setEnabled(self._motion_stable_count >= MOTION_STABLE_FRAMES)
            if self._motion_stable_count >= MOTION_STABLE_FRAMES:
                self.drop_label.setPixmap(numpy_to_qpixmap(frame))
            else:
                # Show frame with "Hold still..." overlay
                overlay = frame.copy()
                cv2.putText(
                    overlay, "Hold still...",
                    (50, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,
                )
                self.drop_label.setPixmap(numpy_to_qpixmap(overlay))

    def _on_live_mode_toggled(self, checked: bool):
        self._live_mode = checked

    def _on_spell_check_toggled(self, checked: bool):
        self._apply_spell_check = checked

    def is_live_mode(self) -> bool:
        """Return True if Live Mode is on (snap auto-fires OCR)."""
        return self._live_mode

    def is_spell_check_enabled(self) -> bool:
        """Return True if spell-check post-processing is enabled."""
        return self._apply_spell_check

    def _snap_frame(self):
        if self._current_image is not None:
            if self._motion_stable_count < MOTION_STABLE_FRAMES:
                # Overlay already shows "Hold still..." â€” just reject the snap
                return
            snapshot = self._current_image.copy()
            if _blur_score(snapshot) < BLUR_THRESHOLD:
                self._stop_webcam()
                self.drop_label.setText("Image too blurry, try again")
                self.drop_label.setStyleSheet(DROP_ZONE_STYLE.replace("#4A90D9", "#E74C3C"))
                return
            self._stop_webcam()
            self.drop_label.setStyleSheet(DROP_ZONE_STYLE)
            self.drop_label.setPixmap(numpy_to_qpixmap(snapshot))
            self._current_image = snapshot
            self.image_ready.emit(snapshot)
            self.snap_ready.emit(snapshot)

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def show_annotations(self, annotated_image: np.ndarray) -> None:
        """Display an image with bounding boxes overlaid (e.g. detected text lines)."""
        self.drop_label.setPixmap(numpy_to_qpixmap(annotated_image))

    def get_current_image(self) -> np.ndarray | None:
        """Return the current loaded/snapped image, or None."""
        return self._current_image

    # â”€â”€ Cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def closeEvent(self, event):
        self._stop_webcam()
        super().closeEvent(event)

