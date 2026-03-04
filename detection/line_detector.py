"""
line_detector.py — OpenCV-based text line detection.

Uses morphological operations to segment an image into individual text lines.
No external model download required — purely OpenCV.

Usage:
    from detection.line_detector import LineDetector

    detector = LineDetector()
    boxes = detector.detect(image)           # list of (x, y, w, h)
    crops = detector.detect_and_crop(image)  # list of cropped np.ndarrays
    annotated = detector.annotate(image, boxes)  # image with drawn boxes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class DetectionMode(Enum):
    """Detection granularity: line-level or word-level."""
    LINE = "line"
    WORD = "word"


def _default_yolo_model_candidates() -> list[str]:
    """Default YOLO detector weight locations (highest priority first)."""
    env_path = os.environ.get("POINTREAD_YOLO_MODEL", "").strip()
    candidates = []
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            os.path.join("runs", "detect", "iam_line_detector", "weights", "best.pt"),
            os.path.join("runs", "detect", "train", "weights", "best.pt"),
        ]
    )
    return candidates


_YOLO_MODEL_CACHE: dict[str, object] = {}


class YOLOLineDetector:
    """YOLO-based line detector.

    Expects a model trained with one class ("line") and returns crops sorted in
    reading order (top-to-bottom, then left-to-right).
    """

    def __init__(self, model_path: str | None = None, conf: float = 0.25):
        self.conf = float(conf)
        self.model_path = self.resolve_model_path(model_path)
        self._model = None
        self._available = self.model_path is not None

    @staticmethod
    def resolve_model_path(model_path: str | None = None) -> str | None:
        if model_path:
            return model_path if os.path.isfile(model_path) else None
        for path in _default_yolo_model_candidates():
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def is_available(model_path: str | None = None) -> bool:
        resolved = YOLOLineDetector.resolve_model_path(model_path)
        if resolved is None:
            return False
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_loaded(self) -> bool:
        if not self._available or self.model_path is None:
            return False
        if self._model is not None:
            return True
        if self.model_path in _YOLO_MODEL_CACHE:
            self._model = _YOLO_MODEL_CACHE[self.model_path]
            return True
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            _YOLO_MODEL_CACHE[self.model_path] = self._model
            return True
        except Exception:
            self._available = False
            return False

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return line boxes as (x, y, w, h) sorted top-to-bottom."""
        if image is None or image.size == 0:
            return []
        if not self._ensure_loaded():
            return []

        try:
            results = self._model(image, conf=self.conf, verbose=False)[0]
            xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.empty((0, 4))
        except Exception:
            return []

        h, w = image.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for box in xyxy:
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            # Small padding to avoid clipping ascenders/descenders.
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            bw = x2 - x1
            bh = y2 - y1
            if bw > 1 and bh > 1:
                boxes.append((x1, y1, bw, bh))

        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    def detect_and_crop(
        self, image: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        boxes = self.detect(image)
        crops: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        for x, y, w, h in boxes:
            crop = image[y : y + h, x : x + w]
            if crop.size > 0:
                crops.append((crop, (x, y, w, h)))
        return crops


@dataclass
class DetectionConfig:
    """Tuneable parameters for line detection.

    Attributes:
        horizontal_kernel_scale: Fraction of image width for the horizontal
            dilation kernel.  Larger = more aggressive merging of characters
            into lines.  0.3–0.5 works well for typical handwriting.
        vertical_kernel: Height of the vertical dilation kernel.  Helps merge
            ascenders/descenders into the same line blob.
        min_line_height: Minimum bounding-box height (pixels) to keep.
            Filters out noise / tiny specks.
        max_line_height_ratio: Maximum line height as a fraction of image
            height.  Boxes taller than this are probably the whole image.
        min_line_width_ratio: Minimum line width as fraction of image width.
            Filters out very narrow noise blobs.
        margin: Extra pixels to add around each cropped line (padding).
        merge_vertical_gap: If two boxes are within this many pixels
            vertically, merge them into one (catches broken lines).
        adaptive_block_size: Block size for adaptive threshold (odd, >= 3).
        adaptive_C: Constant subtracted from mean in adaptive threshold.
            Lower = less aggressive for low-contrast images.
        mode: LINE or WORD detection.
    """
    horizontal_kernel_scale: float = 0.08
    vertical_kernel: int = 2
    min_line_height: int = 15
    max_line_height_ratio: float = 0.85
    min_line_width_ratio: float = 0.05
    margin: int = 8
    merge_vertical_gap: int = 10
    adaptive_block_size: int = 15
    adaptive_C: int = 10
    mode: DetectionMode = DetectionMode.LINE

    @classmethod
    def webcam(cls) -> "DetectionConfig":
        """Preset for webcam images: lower contrast, thinner lines."""
        return cls(
            adaptive_block_size=11,
            adaptive_C=3,
            min_line_width_ratio=0.02,
            min_line_height=12,
            max_line_height_ratio=0.5,
        )


def crop_to_square(crop: np.ndarray, pad_color: int = 255) -> np.ndarray:
    """
    Pad a crop to square with white background — preserves letter proportions for TrOCR.

    Args:
        crop: Grayscale (H, W) or BGR (H, W, C) numpy array.
        pad_color: Fill value for padding. Default 255 (white).

    Returns:
        Square image with content centred.
    """
    h, w = crop.shape[:2]
    size = max(h, w)

    if len(crop.shape) == 3:
        square = np.full((size, size, crop.shape[2]), pad_color, dtype=np.uint8)
    else:
        square = np.full((size, size), pad_color, dtype=np.uint8)

    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off : y_off + h, x_off : x_off + w] = crop
    return square


def tight_crop_with_padding(
    image: np.ndarray, bbox: tuple[int, int, int, int], margin_px: int = 8
) -> np.ndarray:
    """
    Crop tightly around a bbox with small fixed margin, then pad to square.

    Args:
        image: Source image (grayscale or BGR).
        bbox: (x, y, w, h) bounding box.
        margin_px: Extra pixels around bbox. Default 8.

    Returns:
        Square crop ready for TrOCR.
    """
    x, y, w, h = bbox
    ih, iw = image.shape[:2]

    x1 = max(0, x - margin_px)
    y1 = max(0, y - margin_px)
    x2 = min(iw, x + w + margin_px)
    y2 = min(ih, y + h + margin_px)

    crop = image[y1:y2, x1:x2]
    return crop_to_square(crop)


def invert_if_needed(crop: np.ndarray) -> np.ndarray:
    """
    Ensure dark text on light background for TrOCR.
    If more pixels are dark than light, invert the crop.
    """
    if crop is None or crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    dark_pixels = np.sum(gray < 128)
    light_pixels = np.sum(gray >= 128)
    if dark_pixels > light_pixels:
        return cv2.bitwise_not(crop)
    return crop


def resize_for_trocr(crop: np.ndarray, size: int = 384) -> np.ndarray:
    """
    Resize crop to TrOCR's expected input size (384x384) using bicubic interpolation.
    """
    if crop is None or crop.size == 0:
        return crop
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


class LineDetector:
    """Detects text lines in an image using OpenCV morphological operations.

    Pipeline:
        1. Convert to grayscale
        2. Adaptive threshold → binary
        3. Dilate horizontally to merge characters into line blobs
        4. Dilate vertically (small) to merge ascenders/descenders
        5. Find contours → bounding boxes
        6. Filter by size, merge overlapping/close boxes
        7. Sort top-to-bottom
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and binarize with adaptive threshold."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Illumination normalization: flatten slow shadow gradients.
        sigma = max(15.0, gray.shape[1] / 20.0)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        normalized = cv2.divide(gray, bg, scale=255)

        # Slight blur to reduce local noise
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

        # Adaptive threshold — works better than Otsu for uneven lighting
        block = self.config.adaptive_block_size
        if block % 2 == 0:
            block += 1
        block = max(3, block)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block,
            C=self.config.adaptive_C,
        )

        # Remove tiny speckles and border noise that often become ghost boxes.
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        # Remove large border-connected blobs (page edges, hard shadows).
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        img_area = binary.shape[0] * binary.shape[1]
        for label_idx in range(1, num_labels):
            x, y, bw, bh, area = stats[label_idx]
            touches_border = (
                x == 0
                or y == 0
                or x + bw >= binary.shape[1]
                or y + bh >= binary.shape[0]
            )
            large_blob = area > int(img_area * 0.01) or bh > int(binary.shape[0] * 0.6)
            if touches_border and large_blob:
                binary[labels == label_idx] = 0

        h, w = binary.shape
        border = max(2, int(min(h, w) * 0.01))
        binary[:border, :] = 0
        binary[h - border :, :] = 0
        binary[:, :border] = 0
        binary[:, w - border :] = 0
        return binary

    def _dilate_to_lines(self, binary: np.ndarray) -> np.ndarray:
        """Dilate to merge characters into line blobs (LINE) or word blobs (WORD)."""
        h, w = binary.shape

        if self.config.mode == DetectionMode.WORD:
            # Small horizontal kernel: merge chars into words, not across words
            h_kernel_width = min(20, max(5, int(w * 0.03)))
            h_kernel_height = 1
        else:
            # Line mode: wide-and-thin kernel to merge text horizontally per line.
            h_kernel_width = max(16, min(96, int(w * self.config.horizontal_kernel_scale)))
            h_kernel_height = 2

        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (h_kernel_width, h_kernel_height)
        )
        dilated = cv2.dilate(binary, h_kernel, iterations=1)

        if self.config.mode == DetectionMode.LINE and self.config.vertical_kernel > 1:
            v_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, self.config.vertical_kernel)
            )
            dilated = cv2.dilate(dilated, v_kernel, iterations=1)

        return dilated

    def _find_boxes(self, dilated: np.ndarray, img_h: int, img_w: int) -> list[tuple[int, int, int, int]]:
        """Find contours and convert to bounding boxes, filtering by size."""
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter too small
            if h < self.config.min_line_height:
                continue
            if w < img_w * self.config.min_line_width_ratio:
                continue

            # Filter too large (likely the whole image)
            if h > img_h * self.config.max_line_height_ratio:
                continue

            boxes.append((x, y, w, h))

        return boxes

    def _merge_close_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Merge boxes that are vertically close (broken line fragments). Skip in WORD mode."""
        if not boxes or self.config.mode == DetectionMode.WORD:
            return boxes

        # Sort by Y position
        boxes = sorted(boxes, key=lambda b: b[1])

        merged = [boxes[0]]
        for x, y, w, h in boxes[1:]:
            px, py, pw, ph = merged[-1]

            # Check if this box overlaps or is close vertically with the previous
            prev_bottom = py + ph
            vertical_gap = y - prev_bottom

            # Check horizontal overlap
            h_overlap = min(x + w, px + pw) - max(x, px)

            if vertical_gap <= self.config.merge_vertical_gap and h_overlap > 0:
                # Merge: expand the previous box
                new_x = min(x, px)
                new_y = min(y, py)
                new_right = max(x + w, px + pw)
                new_bottom = max(y + h, py + ph)
                merged[-1] = (new_x, new_y, new_right - new_x, new_bottom - new_y)
            else:
                merged.append((x, y, w, h))

        return merged

    def _sort_boxes_reading_order(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Sort boxes for stable reading order.

        LINE mode: top-to-bottom.
        WORD mode: group words into visual rows, then sort each row left-to-right.
        """
        if not boxes:
            return []

        if self.config.mode != DetectionMode.WORD:
            return sorted(boxes, key=lambda b: b[1])

        sorted_by_y = sorted(boxes, key=lambda b: b[1])
        heights = [h for _, _, _, h in sorted_by_y]
        median_h = float(np.median(heights)) if heights else 0.0
        line_tol = max(8.0, median_h * 0.5)

        rows: list[dict[str, object]] = []
        for box in sorted_by_y:
            x, y, w, h = box
            center_y = y + h / 2.0

            best_row_idx = None
            best_dist = None
            for idx, row in enumerate(rows):
                top = float(row["top"])
                bottom = float(row["bottom"])
                row_center = (top + bottom) / 2.0
                overlaps_vertically = (y <= bottom + line_tol) and (y + h >= top - line_tol)
                if not overlaps_vertically:
                    continue

                dist = abs(center_y - row_center)
                if best_row_idx is None or dist < best_dist:
                    best_row_idx = idx
                    best_dist = dist

            if best_row_idx is None:
                rows.append({"top": float(y), "bottom": float(y + h), "boxes": [box]})
                continue

            row = rows[best_row_idx]
            row["top"] = min(float(row["top"]), float(y))
            row["bottom"] = max(float(row["bottom"]), float(y + h))
            row["boxes"].append(box)

        rows.sort(key=lambda r: float(r["top"]))
        ordered: list[tuple[int, int, int, int]] = []
        for row in rows:
            row_boxes = sorted(row["boxes"], key=lambda b: b[0])
            ordered.extend(row_boxes)

        return ordered

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect text line bounding boxes in the image.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            List of (x, y, w, h) tuples in reading order.
            LINE mode: top-to-bottom.
            WORD mode: top-to-bottom rows, left-to-right within each row.
        """
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]

        binary = self._preprocess(image)
        dilated = self._dilate_to_lines(binary)
        boxes = self._find_boxes(dilated, img_h, img_w)
        boxes = self._merge_close_boxes(boxes)
        return self._sort_boxes_reading_order(boxes)

    def detect_and_crop(
        self, image: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """Detect text lines and return cropped images with their bounding boxes.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            List of (cropped_image, (x, y, w, h)) tuples in reading order.
        """
        boxes = self.detect(image)
        img_h, img_w = image.shape[:2]
        margin = self.config.margin

        results = []
        for x, y, w, h in boxes:
            # Add margin, clamped to image bounds
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_w, x + w + margin)
            y2 = min(img_h, y + h + margin)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                results.append((crop, (x, y, w, h)))

        return results

    def annotate(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        color: tuple[int, int, int] = (74, 144, 217),  # #4A90D9 in BGR
        thickness: int = 2,
        show_numbers: bool = True,
    ) -> np.ndarray:
        """Draw detected bounding boxes on a copy of the image.

        Args:
            image: Original BGR image.
            boxes: List of (x, y, w, h) bounding boxes.
            color: BGR color for the rectangles.
            thickness: Line thickness.
            show_numbers: If True, draw line numbers.

        Returns:
            Annotated copy of the image.
        """
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            if show_numbers:
                # Draw line number label
                label = str(i + 1)
                font_scale = 0.5
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )[0]
                # Background rectangle for readability
                cv2.rectangle(
                    annotated,
                    (x, y - label_size[1] - 6),
                    (x + label_size[0] + 6, y),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated, label,
                    (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                )

        return annotated

    def annotate_with_confidence(
        self,
        image: np.ndarray,
        results: list[dict],
        thickness: int = 2,
        show_numbers: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes with colour coded by confidence.

        Green: > 80%, Orange: 50–80%, Red: < 50%.
        """
        # BGR colours
        GREEN = (0, 200, 0)    # > 80%
        ORANGE = (0, 165, 255)  # 50–80%
        RED = (0, 0, 255)      # < 50%

        boxes_with_color = []
        for r in results:
            bbox = r.get("bbox")
            if bbox is None:
                continue
            conf = r.get("confidence", 0.0)
            if conf > 0.8:
                color = GREEN
            elif conf >= 0.5:
                color = ORANGE
            else:
                color = RED
            boxes_with_color.append((bbox, color))

        if not boxes_with_color:
            return image.copy()

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for i, ((x, y, w, h), color) in enumerate(boxes_with_color):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            if show_numbers:
                label = str(i + 1)
                font_scale = 0.5
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )[0]
                cv2.rectangle(
                    annotated,
                    (x, y - label_size[1] - 6),
                    (x + label_size[0] + 6, y),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated, label,
                    (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                )

        return annotated

    def has_multiple_lines(self, image: np.ndarray) -> bool:
        """Quick check whether the image likely contains more than one text line."""
        boxes = self.detect(image)
        return len(boxes) > 1


if __name__ == "__main__":
    # Quick test: create a multi-line dummy image
    img = np.ones((400, 600), dtype=np.uint8) * 255
    cv2.putText(img, "First line of text", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)
    cv2.putText(img, "Second line here", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)
    cv2.putText(img, "Third line too", (30, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)

    detector = LineDetector()
    boxes = detector.detect(img)
    print(f"Detected {len(boxes)} lines:")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"  Line {i+1}: x={x}, y={y}, w={w}, h={h}")

    crops = detector.detect_and_crop(img)
    for i, (crop, bbox) in enumerate(crops):
        print(f"  Crop {i+1}: shape={crop.shape}, bbox={bbox}")
