"""
main_window.py â€” QMainWindow with horizontal split layout.

Wires InputPanel â†” InferenceWorker â†” OutputPanel using Qt signals/threads.
Includes a model selector dropdown for switching TrOCR model sizes.
"""

import os
import sys

import numpy as np

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt6.QtGui import QFont, QIcon, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox, QApplication, QStatusBar, QLabel,
    QComboBox,
)

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from gui.input_panel import InputPanel
from gui.output_panel import OutputPanel
from predictor import predict_page, set_model, get_available_models
from trocr.config import DEFAULT_MODEL_WEBCAM
from detection.line_detector import LineDetector


# â”€â”€ Inference Worker (runs in QThread) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class InferenceWorker(QObject):
    """Runs TrOCR inference in a background thread (multi-line detection)."""
    result_ready = pyqtSignal(list, object, object)  # results, preprocessed, metadata
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        image: np.ndarray,
        *,
        from_webcam: bool = False,
        detector_backend: str = "auto",
        apply_spell_check: bool = True,
    ):
        super().__init__()
        self.image = image
        self.from_webcam = from_webcam
        self.detector_backend = detector_backend
        self.apply_spell_check = apply_spell_check
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = predict_page(
                self.image,
                use_webcam_config=self.from_webcam,
                detector_backend=self.detector_backend,
                apply_spell_check=self.apply_spell_check,
                include_crops=True,
                return_preprocessed=True,
            )
            results, preprocessed, metadata = result
            if self.from_webcam:
                from gui.input_panel import _blur_score
                gray = (
                    cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    if len(self.image.shape) == 3
                    else self.image
                )
                metadata["blur_score"] = _blur_score(self.image)
                metadata["brightness"] = float(np.mean(gray))
            self.signals.result_ready.emit(results, preprocessed, metadata)
        except Exception as e:
            self.signals.error_occurred.emit(str(e))


# â”€â”€ Model Loader Worker (runs in QThread) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ModelLoaderRunnable(QRunnable):
    """Loads a TrOCR model in a background thread to avoid UI freeze."""
    def __init__(self, model_key: str):
        super().__init__()
        self.model_key = model_key
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            set_model(self.model_key)
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error_occurred.emit(str(e))


# â”€â”€ Theme Stylesheet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
THEME = """
    QMainWindow, QWidget {
        background: #F5F7FA;
        color: #1E2A3A;
        font-family: 'Inter', sans-serif;
    }

    QPushButton {
        background: #4A90D9;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
    }
    QPushButton:hover {
        background: #3A7BC8;
    }
    QPushButton:pressed {
        background: #2E6DB5;
    }
    QPushButton:disabled {
        background: #C4CDD9;
        color: #8A9BB0;
    }

    QPushButton#btnRecognize {
        background: #4A90D9;
        font-size: 16px;
        font-weight: 600;
        padding: 14px 28px;
        min-height: 48px;
    }
    QPushButton#btnRecognize:hover {
        background: #3A7BC8;
    }
    QPushButton#btnRecognize:disabled {
        background: #C4CDD9;
        color: #8A9BB0;
    }

    QComboBox {
        background: #FFFFFF;
        color: #1E2A3A;
        border: 1px solid #C4CDD9;
        padding: 6px 12px;
        font-size: 13px;
        min-width: 160px;
    }
    QComboBox:hover {
        border-color: #4A90D9;
    }
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    QComboBox QAbstractItemView {
        background: #FFFFFF;
        color: #1E2A3A;
        selection-background-color: #4A90D9;
        selection-color: #FFFFFF;
    }

    QSplitter::handle {
        background: #C4CDD9;
        width: 3px;
    }
    QSplitter::handle:hover {
        background: #4A90D9;
    }

    QStatusBar {
        background: #FFFFFF;
        color: #8A9BB0;
        font-size: 12px;
        border-top: 1px solid #E0E5EC;
    }
"""


# â”€â”€ Model display names â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_DISPLAY = {
    "small": "TrOCR Small  (~240 MB)",
    "base":  "TrOCR Base   (~900 MB)",
    "large": "TrOCR Large  (~1.7 GB)",
    "fine-tuned (local)": "Fine-tuned (local)",
}

DETECTOR_DISPLAY = {
    "auto": "Auto (YOLO->OpenCV)",
    "yolo": "YOLO",
    "opencv": "OpenCV",
}


class MainWindow(QMainWindow):
    """Main application window â€” horizontal split with Recognize action."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point & Read â€” Handwriting Recognition")
        self.setMinimumSize(1100, 650)
        self.resize(1280, 720)

        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)
        
        self._current_image: np.ndarray | None = None
        self._image_from_webcam: bool = False
        self._detector_backend: str = "auto"
        self._last_results: list[dict] = []

        # Apply theme
        self.setStyleSheet(THEME)

        self._build_ui()
        self._start_model_warmup()

    # â”€â”€ UI Construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 8)
        root_layout.setSpacing(10)

        # Header row: title + model selector
        header_row = QHBoxLayout()

        header = QLabel("Point & Read")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            color: #1E2A3A;
            padding: 4px 0;
        """)
        header_row.addWidget(header)

        header_row.addStretch()

        # Model selector
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-size: 13px; color: #8A9BB0; padding-right: 4px;")
        header_row.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelSelector")
        available = get_available_models()
        default_idx = 0
        for i, key in enumerate(available):
            display = MODEL_DISPLAY.get(key, key)
            self.model_combo.addItem(display, key)
            if key == DEFAULT_MODEL_WEBCAM:
                default_idx = i
            elif key == "base" and default_idx == 0:
                default_idx = i
        self.model_combo.setCurrentIndex(default_idx)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        header_row.addWidget(self.model_combo)

        detector_label = QLabel("Detector:")
        detector_label.setStyleSheet(
            "font-size: 13px; color: #8A9BB0; padding: 0 4px 0 12px;"
        )
        header_row.addWidget(detector_label)

        self.detector_combo = QComboBox()
        self.detector_combo.setObjectName("detectorSelector")
        for key in ("auto", "yolo", "opencv"):
            self.detector_combo.addItem(DETECTOR_DISPLAY[key], key)
        self.detector_combo.setCurrentIndex(0)
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        header_row.addWidget(self.detector_combo)

        root_layout.addLayout(header_row)

        # Splitter: left = input, right = output
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.input_panel = InputPanel()
        self.output_panel = OutputPanel()
        splitter.addWidget(self.input_panel)
        splitter.addWidget(self.output_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        root_layout.addWidget(splitter, stretch=1)

        # Recognize button (centered, prominent) â€” disabled until image is loaded
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_recognize = QPushButton("Recognize Handwriting")
        self.btn_recognize.setObjectName("btnRecognize")
        self.btn_recognize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.clicked.connect(self._on_recognize)
        btn_row.addWidget(self.btn_recognize)
        btn_row.addStretch()
        root_layout.addLayout(btn_row)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready â€” load an image or start webcam")

        # Connect input signals
        self.input_panel.image_ready.connect(self._on_image_ready)
        self.input_panel.snap_ready.connect(self._on_snap_ready)
        self.output_panel.submit_correction_requested.connect(
            self._on_submit_correction_requested
        )
        self.output_panel.append_notes_requested.connect(
            self._on_append_notes_requested
        )

    # â”€â”€ Slots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @pyqtSlot(np.ndarray)
    def _on_image_ready(self, image: np.ndarray):
        """Called when input panel has a new image (file/drop or snap)."""
        self._current_image = image
        self._image_from_webcam = False  # Will be set True by _on_snap_ready if from snap
        self.btn_recognize.setEnabled(True)
        self.status_bar.showMessage("Image loaded â€” click Recognize Handwriting to process")

    @pyqtSlot(np.ndarray)
    def _on_snap_ready(self, image: np.ndarray):
        """Called when user snaps from webcam. Auto-fires inference if Live Mode is on."""
        self._current_image = image
        self._image_from_webcam = True
        self.btn_recognize.setEnabled(True)
        if self.input_panel.is_live_mode():
            self.output_panel.show_spinner()
            self.status_bar.showMessage("Snapped â€” recognizing...")
            self._run_inference()
        else:
            self.status_bar.showMessage("Snapped â€” click Recognize Handwriting to process")

    def _on_model_changed(self, index: int):
        """Switch model when dropdown selection changes."""
        model_key = self.model_combo.itemData(index)
        if model_key is None:
            return

        # Disable UI during model load
        self.btn_recognize.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.status_bar.showMessage(f"Loading {self.model_combo.currentText().strip()}...")

        # Load in background thread
        worker = ModelLoaderRunnable(model_key)
        worker.signals.finished.connect(self._on_model_loaded)
        worker.signals.error_occurred.connect(self._on_model_load_error)
        self.thread_pool.start(worker)

    def _on_detector_changed(self, index: int):
        """Switch detector backend used by predict_page()."""
        backend = self.detector_combo.itemData(index)
        if backend is None:
            return
        self._detector_backend = str(backend)
        self.status_bar.showMessage(
            f"Detector set to {self.detector_combo.currentText().strip()}"
        )

    def _start_model_warmup(self):
        """Load the default model in the background so first inference is fast."""
        default_key = DEFAULT_MODEL_WEBCAM
        available = get_available_models()
        if default_key not in available:
            default_key = available[0] if available else "base"
        # Prevent overlapping model loads while warmup is running.
        self.model_combo.setEnabled(False)
        self.btn_recognize.setEnabled(False)
        worker = ModelLoaderRunnable(default_key)
        worker.signals.finished.connect(self._on_warmup_complete)
        worker.signals.error_occurred.connect(self._on_warmup_error)
        self.thread_pool.start(worker)
        self.status_bar.showMessage("Loading model in background...")

    @pyqtSlot()
    def _on_warmup_complete(self):
        """Called when model warm-up finishes."""
        self.model_combo.setEnabled(True)
        if self._current_image is not None:
            self.btn_recognize.setEnabled(True)
        self.status_bar.showMessage("Ready - load an image or start webcam")

    @pyqtSlot(str)
    def _on_warmup_error(self, _error: str):
        """Called when model warm-up fails; fallback is lazy load on first OCR run."""
        self.model_combo.setEnabled(True)
        if self._current_image is not None:
            self.btn_recognize.setEnabled(True)
        self.status_bar.showMessage("Model warm-up failed - will load on first recognition")

    @pyqtSlot()
    def _on_model_loaded(self):
        """Called when model loading completes successfully (user changed model)."""
        self.model_combo.setEnabled(True)
        if self._current_image is not None:
            self.btn_recognize.setEnabled(True)
        model_name = self.model_combo.currentText().strip()
        self.status_bar.showMessage(f"{model_name} loaded â€” ready")

    @pyqtSlot(str)
    def _on_model_load_error(self, error: str):
        """Called when model loading fails."""
        self.model_combo.setEnabled(True)
        if self._current_image is not None:
            self.btn_recognize.setEnabled(True)
        self.status_bar.showMessage(f"Model load error: {error}")
        QMessageBox.critical(self, "Model Error", f"Failed to load model:\n\n{error}")

    def _on_recognize(self):
        """Launch inference in a background QThread."""
        if self._current_image is None:
            QMessageBox.information(self, "No Image", "Please load an image first.")
            return
        self._run_inference()

    def _check_brightness(self, image: np.ndarray) -> bool:
        """Return True if mean brightness is in acceptable range (50â€“200)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        mean_val = float(np.mean(gray))
        return 50 <= mean_val <= 200

    def _run_inference(self):
        """Shared logic: show spinner, launch InferenceWorker in background thread."""
        if self._current_image is None:
            return

        if not self._check_brightness(self._current_image):
            mean_val = float(np.mean(
                cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
                if len(self._current_image.shape) == 3
                else self._current_image
            ))
            reply = QMessageBox.warning(
                self,
                "Brightness Warning",
                f"Image may be too dark or bright (mean pixel: {mean_val:.0f}). "
                "OCR accuracy may suffer. Proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.output_panel.show_spinner()
        self.btn_recognize.setEnabled(False)
        self.status_bar.showMessage("Running inference...")

        worker = InferenceRunnable(
            self._current_image,
            from_webcam=self._image_from_webcam,
            detector_backend=self._detector_backend,
            apply_spell_check=self.input_panel.is_spell_check_enabled(),
        )
        worker.signals.result_ready.connect(self._on_inference_done)
        worker.signals.error_occurred.connect(self._on_inference_error)
        self.thread_pool.start(worker)

    @pyqtSlot(list, object, object)
    def _on_inference_done(
        self, results: list, preprocessed: np.ndarray | None, metadata: dict | None
    ):
        """Called when InferenceWorker completes successfully."""
        self._last_results = results or []
        self.output_panel.display_text_results(results)
        self.btn_recognize.setEnabled(True)

        combined = "".join(r["text"] for r in results)
        avg_conf = (
            sum(r["confidence"] for r in results) / len(results)
            if results else 0.0
        )
        pct = avg_conf * 100
        meta = metadata or {}
        detector_used = str(meta.get("detector_backend", "opencv")).upper()
        detector_requested = str(meta.get("detector_requested", self._detector_backend))
        runtime = str(meta.get("inference_runtime", "pytorch")).upper()

        # Build status message; add specific feedback when confidence is low
        status = (
            f"Recognized {len(results)} line(s), {len(combined)} characters "
            f"- Confidence: {pct:.1f}% - Detector: {detector_used} - Runtime: {runtime}"
        )
        if detector_requested == "auto":
            status += " (auto)"
        if meta.get("spell_corrected"):
            status += " - Spell-corrected"
        if avg_conf < 0.4:
            feedback = self._low_confidence_feedback(avg_conf, meta)
            if feedback:
                status += " - " + feedback
        self.status_bar.showMessage(status)

        # Draw colour-coded bounding boxes on preprocessed image
        if results and preprocessed is not None:
            annotated = LineDetector().annotate_with_confidence(preprocessed, results)
            self.input_panel.show_annotations(annotated)

    @pyqtSlot(str)
    def _on_submit_correction_requested(self, edited_text: str):
        """Save user text corrections as training samples."""
        if not self._last_results:
            self.status_bar.showMessage("No OCR output available to save yet")
            return

        try:
            from data_logger import save_corrections

            summary = save_corrections(self._last_results, edited_text)
            saved = int(summary.get("saved", 0))
            changed = int(summary.get("changed", 0))
            if saved > 0:
                self.status_bar.showMessage(
                    f"Saved {saved} corrected sample(s) to personal dataset"
                )
            elif changed > 0:
                self.status_bar.showMessage(
                    "Detected corrections, but no crops were available to save"
                )
            else:
                self.status_bar.showMessage("No text changes detected")
        except Exception as exc:
            self.status_bar.showMessage(f"Correction save failed: {exc}")

    @pyqtSlot(str)
    def _on_append_notes_requested(self, text: str):
        """Append corrected OCR text to notes.md with timestamp."""
        try:
            from note_logger import append_markdown_entry

            path = append_markdown_entry(text)
            self.status_bar.showMessage(f"Appended text to {path}")
        except Exception as exc:
            self.status_bar.showMessage(f"Append to notes failed: {exc}")

    def _low_confidence_feedback(self, avg_conf: float, metadata: dict) -> str:
        """Return specific feedback message for low-confidence results."""
        if metadata.get("fallback_used") and metadata.get("num_regions", 0) == 0:
            return "Couldn't detect lines â€” try a closer shot"
        blur = metadata.get("blur_score")
        if blur is not None and 100 <= blur <= 150:
            return "Try holding the camera steadier"
        brightness = metadata.get("brightness")
        if brightness is not None and (brightness < 60 or brightness > 210):
            return "Improve lighting for better results"
        return "Low contrast or unclear text detected"

    @pyqtSlot(str)
    def _on_inference_error(self, error: str):
        """Called when InferenceWorker fails."""
        self.output_panel.hide_spinner()
        self.btn_recognize.setEnabled(True)
        self.status_bar.showMessage(f"Error: {error}")
        QMessageBox.critical(self, "Inference Error", f"Recognition failed:\n\n{error}")

    # â”€â”€ Cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def closeEvent(self, event):
        self.input_panel.close()
        super().closeEvent(event)

