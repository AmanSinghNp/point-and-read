"""
predictor.py -- TrOCR inference wrapper for handwriting recognition.

Exposes:
    predict(image_input) -> str
    predict_with_confidence(image_input) -> tuple[str, float]
    predict_page(image_input) -> list[dict]      # multi-line detection + OCR
    set_model(size_or_path) -> None
    get_available_models() -> list[str]

Accepts a file path (str), a PIL Image, or a numpy array (e.g. from OpenCV).
Model loads lazily on the first call and is cached for all subsequent calls.
Uses the HuggingFace cache so the model is downloaded once and runs fully
offline after that.
"""

from __future__ import annotations

import os
import threading
from typing import Union

import numpy as np
import torch
from PIL import Image

from trocr.config import (
    MODELS,
    DEFAULT_MODEL,
    DEFAULT_NUM_BEAMS,
    DEFAULT_MAX_LENGTH,
    CHECKPOINT_DIR,
)
from preprocessing.clean import (
    preprocess_for_trocr,
    apply_exif_orientation,
    align_text_axis_minarearect,
)
from nlp.spell_check import OCRCorrector
from models.ocr_result import LineResult, PageResult

# ---------------------------------------------------------------------------
# Module-level cache (populated once on first predict() call)
# ---------------------------------------------------------------------------
_processor = None
_model = None
_device: torch.device | None = None
_current_model_id: str | None = None
_current_runtime: str | None = None
_model_lock = threading.RLock()

# Preprocessing cache: (image_id, preprocessed) to avoid re-running on same image
_preprocess_cache: tuple[int, np.ndarray] | None = None
_spell_checker: OCRCorrector | None = None
_spell_checker_init_failed = False


def _get_spell_checker() -> OCRCorrector | None:
    """Lazily initialize SymSpell post-processor."""
    global _spell_checker, _spell_checker_init_failed
    if _spell_checker is not None:
        return _spell_checker
    if _spell_checker_init_failed:
        return None
    try:
        checker = OCRCorrector()
        if checker.is_ready():
            _spell_checker = checker
            return _spell_checker
        _spell_checker_init_failed = True
        return None
    except Exception:
        _spell_checker_init_failed = True
        return None


def _compute_num_beams(line_image_width: int) -> int:
    """Choose beam count based on line crop width.

    Short lines have fewer tokens, so a large beam width wastes compute.
    """
    if line_image_width < 150:
        return 1   # Single word / very short
    elif line_image_width < 300:
        return 2
    return 4       # Full sentence lines

def _resolve_device() -> torch.device:
    """Detect the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_inference_backend() -> str:
    """Resolve model runtime backend from env var.

    Supported values:
        auto (default), pytorch, onnx
    """
    value = os.environ.get("POINTREAD_INFERENCE_BACKEND", "auto").strip().lower()
    if value in {"auto", "pytorch", "onnx", "int8"}:
        return value
    return "auto"


def _resolve_onnx_model_dir(model_id: str) -> str | None:
    """Resolve local ONNX export directory, if present."""
    env_path = os.environ.get("POINTREAD_ONNX_MODEL_DIR", "").strip()
    candidates: list[str] = []
    if env_path:
        candidates.append(env_path)
    # Built-in export script location
    candidates.append(os.path.join("weights", "trocr-onnx"))
    # Default export folder from docs / scripts.
    candidates.append("onnx_trocr_model")
    # Optional per-model export folder convention.
    safe_name = model_id.replace("/", "-").replace("\\", "-")
    candidates.append(os.path.join("weights", "onnx", safe_name))
    if os.path.isdir(model_id):
        candidates.append(os.path.join(model_id, "onnx"))

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _resolve_int8_model_dir(model_id: str) -> str | None:
    """Resolve local INT8 quantized export directory, if present."""
    env_path = os.environ.get("POINTREAD_INT8_MODEL_DIR", "").strip()
    candidates: list[str] = []
    if env_path:
        candidates.append(env_path)
    # Built-in export script location
    candidates.append(os.path.join("weights", "trocr-int8"))
    
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _resolve_model_id(size_or_path: str | None = None) -> str:
    """Resolve a model size name or path to a HuggingFace model ID or local path.

    Priority:
        1. If `size_or_path` is a local directory (fine-tuned checkpoint), use it.
        2. If `size_or_path` matches a key in MODELS ("small", "base", "large"), use the HF ID.
        3. Otherwise treat it as a HuggingFace model ID directly.
        4. If None, check for a local fine-tuned checkpoint first, then fall back to DEFAULT_MODEL.
    """
    if size_or_path is None:
        # Check for local fine-tuned checkpoint
        local_ckpt = os.path.join(CHECKPOINT_DIR, "model")
        if os.path.isdir(local_ckpt):
            return local_ckpt
        size_or_path = DEFAULT_MODEL

    # Named model size
    if size_or_path.lower() in MODELS:
        return MODELS[size_or_path.lower()]

    # Local directory path
    if os.path.isdir(size_or_path):
        return size_or_path

    # Direct HuggingFace model ID
    return size_or_path


def _load_model(model_id: str | None = None):
    """Load the TrOCR processor and model.

    Everything is cached at module level so loading happens only once per
    application session.  The HuggingFace hub cache (`~/.cache/huggingface/`)
    means the model is downloaded once and runs fully offline after that.

    Returns:
        A tuple of (processor, model, device).

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _processor, _model, _device, _current_model_id, _current_runtime

    resolved_id = _resolve_model_id(model_id)

    with _model_lock:
        # Already loaded this exact model
        if _processor is not None and _model is not None and _current_model_id == resolved_id:
            return _processor, _model, _device

        backend = _resolve_inference_backend()
        onnx_exc: Exception | None = None
        int8_exc: Exception | None = None
        processor = None
        model = None
        device = None
        runtime = None

        if backend in {"auto", "onnx"}:
            onnx_dir = _resolve_onnx_model_dir(resolved_id)
            if onnx_dir is not None:
                try:
                    from transformers import TrOCRProcessor
                    from optimum.onnxruntime import ORTModelForVision2Seq

                    try:
                        processor = TrOCRProcessor.from_pretrained(onnx_dir, use_fast=True)
                    except Exception:
                        processor = TrOCRProcessor.from_pretrained(resolved_id, use_fast=True)

                    model = ORTModelForVision2Seq.from_pretrained(onnx_dir)
                    device = torch.device("cpu")
                    runtime = "onnx"
                except Exception as exc:
                    onnx_exc = exc
                    if backend == "onnx":
                        raise RuntimeError(
                            f"Failed to load ONNX model from '{onnx_dir}'. "
                            "Install optimum[onnxruntime] and export the model first.\n"
                            f"Details: {exc}"
                        ) from exc

        if backend in {"auto", "int8"} and model is None:
            int8_dir = _resolve_int8_model_dir(resolved_id)
            if int8_dir is not None:
                try:
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                    
                    try:
                        processor = TrOCRProcessor.from_pretrained(int8_dir, use_fast=True)
                    except Exception:
                        processor = TrOCRProcessor.from_pretrained(resolved_id, use_fast=True)
                    
                    model = VisionEncoderDecoderModel.from_pretrained(int8_dir)
                    device = torch.device("cpu") # INT8 quantization is explicitly for CPU
                    model.to(device)
                    model.eval()
                    runtime = "int8"
                except Exception as exc:
                    int8_exc = exc
                    if backend == "int8":
                        raise RuntimeError(
                            f"Failed to load INT8 model from '{int8_dir}'. "
                            f"Details: {exc}"
                        ) from exc

        if model is None:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                device = _resolve_device()
                processor = TrOCRProcessor.from_pretrained(resolved_id, use_fast=True)
                model = VisionEncoderDecoderModel.from_pretrained(resolved_id)
                model.to(device)
                model.eval()
                runtime = "pytorch"
            except Exception as exc:
                details = f"Details: {exc}"
                if int8_exc is not None:
                    details += f"\nINT8 attempt failed first: {int8_exc}"
                if onnx_exc is not None:
                    details += f"\nONNX attempt failed first: {onnx_exc}"
                raise RuntimeError(
                    f"Failed to load model '{resolved_id}'. "
                    f"Check your internet connection (first run only).\n{details}"
                ) from exc

        _processor = processor
        _model = model
        _device = device
        _current_model_id = resolved_id
        _current_runtime = runtime
        return _processor, _model, _device


def _normalise_image(image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Convert any supported input type to an RGB PIL Image.

    Args:
        image_input: A file path, PIL Image, or numpy array (BGR or grayscale).

    Returns:
        An RGB PIL Image ready for the processor.

    Raises:
        FileNotFoundError: If a string path does not point to an existing file.
        ValueError: If the image cannot be opened or the type is unsupported.
    """
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        try:
            return Image.open(image_input).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Could not open image at '{image_input}': {exc}"
            ) from exc

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Received an empty numpy array.")
        # Grayscale (H, W) -> RGB
        if image_input.ndim == 2:
            rgb = np.stack([image_input] * 3, axis=-1)
            return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        # Colour (H, W, C)
        if image_input.ndim == 3:
            channels = image_input.shape[2]
            if channels == 3:
                # OpenCV default is BGR; convert to RGB
                rgb = image_input[:, :, ::-1]
                return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            if channels == 4:
                # BGRA -> RGB
                rgb = image_input[:, :, 2::-1]
                return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            raise ValueError(
                f"Unsupported channel count: {channels}. Expected 3 (BGR) or 4 (BGRA)."
            )
        raise ValueError(
            f"Unsupported array shape: {image_input.shape}. "
            "Expected (H, W) or (H, W, C)."
        )

    raise TypeError(
        f"Unsupported input type: {type(image_input).__name__}. "
        "Expected a file path (str), PIL Image, or numpy array."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    image_input: Union[str, Image.Image, np.ndarray],
    *,
    auto_detect_lines: bool = False,
    num_beams: int = DEFAULT_NUM_BEAMS,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> str:
    """Run TrOCR inference on a single image.

    On the very first call the model is loaded (downloaded if needed).
    Subsequent calls reuse the cached model.

    Args:
        image_input: One of:
            - A file path (str) to a .png/.jpg/.bmp/.tif image.
            - A PIL Image object.
            - A numpy array in BGR or grayscale format (e.g. from cv2.imread).
        auto_detect_lines: If True, detect text lines and recognize each; return
            joined multi-line text. If False, treat the image as a single line.
        num_beams: Number of beams for beam search (1 = greedy, 4 = default).
        max_length: Maximum number of tokens to generate.

    Returns:
        The recognised text as a plain string.

    Raises:
        FileNotFoundError: If a file path is given but does not exist.
        ValueError: If the image data is invalid or in an unsupported format.
        TypeError: If image_input is not a str, PIL Image, or numpy array.
        RuntimeError: If model loading or inference fails.
    """
    if auto_detect_lines:
        results = predict_page(
            image_input, num_beams=num_beams, max_length=max_length
        )
        return "\n".join(r.text for r in results)

    text, _ = predict_with_confidence(
        image_input, num_beams=num_beams, max_length=max_length
    )
    return text


def predict_with_confidence(
    image_input: Union[str, Image.Image, np.ndarray],
    *,
    num_beams: int = DEFAULT_NUM_BEAMS,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[str, float]:
    """Run TrOCR inference and return both text and a confidence score.

    The confidence score is derived from the sequence score (mean log-probability
    across generated tokens). Higher is better; typical range is 0.0 – 1.0.

    Args:
        image_input: File path, PIL Image, or numpy array.
        num_beams: Number of beams for beam search.
        max_length: Maximum number of tokens to generate.

    Returns:
        A tuple of (recognised_text, confidence_score).
    """
    processor, model, device = _load_model()
    image = _normalise_image(image_input)

    try:
        pixel_values = processor(
            images=image, return_tensors="pt",
        ).pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                num_beams=num_beams,
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Compute confidence from sequence scores
        # model.generate with return_dict returns sequences_scores when num_beams > 1
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            # sequences_scores is mean log-prob; convert to probability
            log_prob = outputs.sequences_scores[0].item()
            confidence = min(1.0, max(0.0, torch.exp(torch.tensor(log_prob)).item()))
        else:
            # Greedy decode: compute from per-token scores
            if outputs.scores:
                probs = torch.stack(outputs.scores, dim=0).softmax(dim=-1)
                token_ids = generated_ids[0, 1:]  # skip decoder_start_token
                token_confidences = []
                for t, tid in enumerate(token_ids):
                    if t < len(probs):
                        token_confidences.append(probs[t, 0, tid].item())
                if token_confidences:
                    confidence = sum(token_confidences) / len(token_confidences)
                else:
                    confidence = 0.0
            else:
                confidence = 0.0

    except Exception as exc:
        raise RuntimeError(
            f"Inference failed: {exc}"
        ) from exc

    return text, confidence


def batch_predict_with_confidence(
    images: list[Union[str, Image.Image, np.ndarray]],
    *,
    num_beams: int = 2,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[tuple[str, float]]:
    """Run batched TrOCR inference on a list of images.

    Args:
        images: List of file paths, PIL Images, or numpy arrays.
        num_beams: Number of beams for beam search.
        max_length: Maximum number of tokens to generate.

    Returns:
        A list of (recognised_text, confidence_score) tuples matching the input order.
    """
    if not images:
        return []

    processor, model, device = _load_model()
    norm_images = [_normalise_image(img) for img in images]

    try:
        pixel_values = processor(
            images=norm_images, return_tensors="pt",
        ).pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                num_beams=num_beams,
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        for i, text in enumerate(texts):
            # Compute confidence
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                log_prob = outputs.sequences_scores[i].item()
                confidence = min(1.0, max(0.0, torch.exp(torch.tensor(log_prob)).item()))
            else:
                if outputs.scores:
                    # scores holds one tensor (batch_size, vocab) per step
                    probs = torch.stack(outputs.scores, dim=0).softmax(dim=-1)
                    token_ids = generated_ids[i, 1:]  # skip decoder_start_token
                    token_confidences = []
                    for t, tid in enumerate(token_ids):
                        if t < len(probs) and tid != processor.tokenizer.pad_token_id:
                            token_confidences.append(probs[t, i, tid].item())
                    if token_confidences:
                        confidence = sum(token_confidences) / len(token_confidences)
                    else:
                        confidence = 0.0
                else:
                    confidence = 0.0
            results.append((text, confidence))

        return results

    except Exception as exc:
        raise RuntimeError(f"Batch inference failed: {exc}") from exc


def predict_page(
    image_input: Union[str, Image.Image, np.ndarray],
    *,
    num_beams: int = DEFAULT_NUM_BEAMS,
    max_length: int = DEFAULT_MAX_LENGTH,
    detection_mode: str = "line",
    use_webcam_config: bool = False,
    use_yolo_detector: bool = True,
    detector_backend: str | None = None,
    yolo_model_path: str | None = None,
    yolo_conf: float = 0.25,
    apply_spell_check: bool = True,
    include_crops: bool = False,
    auto_orient: bool = True,
    return_preprocessed: bool = False,
    return_dto: bool = False,
) -> list[dict] | tuple[list[dict], np.ndarray, dict] | PageResult:
    """Detect text lines in an image and recognize each one.

    Preprocesses (deskew, denoise, Otsu) then uses OpenCV morphological line
    detection to find text regions, then runs TrOCR on each cropped line.
    Works with full-page images, multi-line handwriting, and webcam captures.

    If only a single line (or no lines) is detected, falls back to
    running TrOCR on the entire preprocessed image.

    Args:
        image_input: File path, PIL Image, or numpy array.
        num_beams: Number of beams for beam search.
        max_length: Maximum tokens per line.
        detection_mode: "line" or "word". Word mode uses OpenCV detector.
        use_yolo_detector: If True, prefer YOLO line detection when available.
            Falls back to OpenCV detector if weights/dependency are unavailable.
        detector_backend: Explicit detector selection:
            - "auto": YOLO first, then OpenCV fallback.
            - "yolo": YOLO-only line detection (no OpenCV fallback).
            - "opencv": OpenCV-only detection.
            If None, inferred from use_yolo_detector for backward compatibility.
        yolo_model_path: Optional path to trained YOLO weights (best.pt).
        yolo_conf: YOLO confidence threshold for line detections.
        apply_spell_check: If True, run SymSpell correction on OCR output text.
        include_crops: If True, attach "crop" (np.ndarray) to each result item.
        auto_orient: If True, auto-correct orientation in two phases:
            (1) 90-degree axis alignment from minAreaRect on text blob, and
            (2) 0 vs 180 degree probe using TrOCR confidence on one line crop.

    Returns:
        List of dicts, each with keys:
            - "text" (str): Recognised text for the line.
            - "confidence" (float): Confidence score (0.0–1.0).
            - "bbox" (tuple[int,int,int,int] | None): (x, y, w, h) or None
              if the whole image was used.
        If return_preprocessed is True, returns (results, preprocessed, metadata) instead,
        where preprocessed is the image bboxes are defined in (for annotation)
        and metadata has "fallback_used" (bool) and "num_regions" (int) for feedback.
    """
    import cv2
    from detection.line_detector import (
        LineDetector,
        YOLOLineDetector,
        DetectionConfig,
        DetectionMode,
        crop_to_square,
        invert_if_needed,
        resize_for_trocr,
        tight_crop_with_padding,
    )

    # Ensure we have a numpy array for the detector
    if isinstance(image_input, str):
        # Use EXIF-aware loading so phone photos are rotated upright
        image_np = apply_exif_orientation(image_input)
        if image_np is None or image_np.size == 0:
            raise FileNotFoundError(f"Image file not found: {image_input}")
    elif isinstance(image_input, Image.Image):
        image_np = np.array(image_input.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    elif isinstance(image_input, np.ndarray):
        image_np = image_input
    else:
        raise TypeError(
            f"Unsupported input type: {type(image_input).__name__}"
        )

    if detector_backend is None:
        requested_detector_backend = "auto" if use_yolo_detector else "opencv"
    else:
        requested_detector_backend = str(detector_backend).strip().lower()
    if requested_detector_backend not in {"auto", "yolo", "opencv"}:
        raise ValueError(
            "detector_backend must be one of: 'auto', 'yolo', 'opencv'"
        )

    enable_yolo_detector = requested_detector_backend in {"auto", "yolo"}
    yolo_detector = (
        YOLOLineDetector(model_path=yolo_model_path, conf=yolo_conf)
        if enable_yolo_detector
        else None
    )
    yolo_model_found = bool(yolo_detector and yolo_detector.model_path)
    spell_checker = _get_spell_checker() if apply_spell_check else None
    spell_check_available = spell_checker is not None

    def _maybe_return(
        results: list[LineResult], preprocessed_img: np.ndarray, metadata: dict
    ) -> list[dict] | tuple | PageResult:
        if return_dto:
            for i, r in enumerate(results):
                r.line_index = i
            return PageResult(lines=results)
        result_dicts = [r.to_dict() for r in results]
        if return_preprocessed:
            return (result_dicts, preprocessed_img, metadata)
        return result_dicts

    def _detect_main_crops(
        img: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, tuple[int, int, int, int]]], str]:
        """Primary detection path with YOLO->OpenCV fallback."""
        mode = DetectionMode.WORD if detection_mode == "word" else DetectionMode.LINE
        config = DetectionConfig.webcam() if use_webcam_config else DetectionConfig()
        config.mode = DetectionMode.WORD if use_webcam_config else mode

        # YOLO model is trained for line detection, not word mode.
        if config.mode == DetectionMode.LINE and yolo_detector is not None:
            yolo_crops = yolo_detector.detect_and_crop(img)
            if yolo_crops:
                return yolo_crops, "yolo"
            if requested_detector_backend == "yolo":
                return [], "yolo"

        opencv_crops = LineDetector(config).detect_and_crop(img)
        return opencv_crops, "opencv"

    def _detect_probe_crops(
        img: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, tuple[int, int, int, int]]], str]:
        """Probe detection path (line mode) for orientation confidence test."""
        probe_cfg = DetectionConfig.webcam() if use_webcam_config else DetectionConfig()
        probe_cfg.mode = DetectionMode.LINE

        if yolo_detector is not None:
            yolo_crops = yolo_detector.detect_and_crop(img)
            if yolo_crops:
                return yolo_crops, "yolo"
            if requested_detector_backend == "yolo":
                return [], "yolo"

        opencv_crops = LineDetector(probe_cfg).detect_and_crop(img)
        return opencv_crops, "opencv"

    def _extract_crop_for_bbox(img: np.ndarray, bbox: tuple) -> np.ndarray:
        """Extract a padded square crop for a detected region."""
        return tight_crop_with_padding(img, bbox, margin_px=8)

    def _prepare_crop_image_for_trocr(square: np.ndarray) -> np.ndarray:
        """Prepare an already-cropped region for TrOCR inference."""
        square = invert_if_needed(square)
        return resize_for_trocr(square, size=384)

    def _prepare_crop_for_trocr(img: np.ndarray, bbox: tuple) -> np.ndarray:
        """Crop, invert if needed, resize to 384x384."""
        square = _extract_crop_for_bbox(img, bbox)
        return _prepare_crop_image_for_trocr(square)

    def _post_process_text(raw_text: str) -> tuple[str, bool]:
        """Apply language correction to OCR text when available."""
        if not apply_spell_check or spell_checker is None:
            return raw_text, False
        corrected = spell_checker.correct_text(raw_text)
        return corrected, corrected != raw_text

    def _build_result(
        *,
        text: str,
        raw_text: str,
        confidence: float,
        bbox: tuple[int, int, int, int] | None,
        spell_corrected: bool,
        crop: np.ndarray | None = None,
    ) -> LineResult:
        return LineResult(
            text=text,
            raw_text=raw_text,
            confidence=confidence,
            bbox=bbox,
            spell_corrected=spell_corrected,
            crop=crop if include_crops else None,
        )

    def _prepare_whole_image_for_trocr(img: np.ndarray) -> np.ndarray:
        """Prepare whole image as single crop for fallback/probing."""
        square = crop_to_square(img)
        return _prepare_crop_image_for_trocr(square)

    def _projection_line_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Fallback line chunking via horizontal projection profile.

        This is used when contour-based detection misses lines in low-light
        or shadow-heavy captures. It extracts text bands row-wise so TrOCR
        still receives line-sized chunks instead of a whole paragraph.
        """
        if img is None or img.size == 0:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            return []

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        # Wide-thin dilation to merge words within a line.
        kx = max(20, min(120, int(w * 0.08)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 2))
        merged = cv2.dilate(binary, kernel, iterations=1)

        row_density = (merged > 0).mean(axis=1).astype(np.float32)
        if row_density.size == 0:
            return []

        # Smooth density to avoid tiny fragmented bands.
        smooth_win = max(5, (h // 80) * 2 + 1)
        padded = np.pad(row_density, (smooth_win // 2,), mode="edge")
        smooth = np.convolve(padded, np.ones(smooth_win, dtype=np.float32) / smooth_win, mode="valid")

        thresh = max(0.01, float(np.percentile(smooth, 70)) * 0.45)
        active = smooth > thresh

        min_band_h = max(10, int(h * 0.03))
        boxes: list[tuple[int, int, int, int]] = []
        y = 0
        while y < h:
            if not active[y]:
                y += 1
                continue
            y1 = y
            while y < h and active[y]:
                y += 1
            y2 = y
            if y2 - y1 < min_band_h:
                continue

            band = merged[y1:y2, :]
            cols = np.where((band > 0).any(axis=0))[0]
            if cols.size == 0:
                continue
            x1 = max(0, int(cols[0]) - 8)
            x2 = min(w, int(cols[-1]) + 9)
            bw = x2 - x1
            bh = y2 - y1
            if bw < int(w * 0.12):
                continue

            # Ignore thin bands hugging page boundaries (typical edge/shadow artifacts).
            touches_edge = y1 <= 2 or y2 >= h - 2
            if touches_edge and bh < int(h * 0.08):
                continue
            boxes.append((x1, y1, bw, bh))

        boxes.sort(key=lambda b: b[1])
        return boxes

    def _apply_orientation_decision(img: np.ndarray, decision: dict) -> np.ndarray:
        """Apply cached orientation decision to another preprocessed image."""
        out = img
        if int(decision.get("axis_rotation_degrees", 0)) == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        if bool(decision.get("probe_flip_180", False)):
            out = cv2.rotate(out, cv2.ROTATE_180)
        return out

    def _auto_orient_preprocessed(
        preprocessed_img: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Orientation-agnostic correction: axis align + 0/180 confidence probe."""
        oriented, axis_meta = align_text_axis_minarearect(preprocessed_img)
        decision = {
            "auto_orient_enabled": True,
            "axis_rotation_degrees": int(axis_meta.get("axis_rotation_degrees", 0)),
            "axis_rotated_90": bool(axis_meta.get("axis_rotated_90", False)),
            "probe_flip_180": False,
            "probe_detector_backend": "opencv",
            "probe_conf_0": None,
            "probe_conf_180": None,
        }

        # Probe on a line crop (preferred) or whole image fallback.
        probe_crops, probe_backend = _detect_probe_crops(oriented)
        decision["probe_detector_backend"] = probe_backend

        if probe_crops:
            probe_bbox = probe_crops[0][1]
            probe_input = _prepare_crop_for_trocr(oriented, probe_bbox)
        else:
            probe_input = _prepare_whole_image_for_trocr(oriented)

        # Keep probe lightweight: tiny generation budget is enough for orientation.
        probe_num_beams = max(1, min(num_beams, 2))
        probe_max_length = max(8, min(max_length, 24))

        try:
            _, conf0 = predict_with_confidence(
                probe_input,
                num_beams=probe_num_beams,
                max_length=probe_max_length,
            )
            probe_180 = cv2.rotate(probe_input, cv2.ROTATE_180)
            _, conf180 = predict_with_confidence(
                probe_180,
                num_beams=probe_num_beams,
                max_length=probe_max_length,
            )

            decision["probe_conf_0"] = float(conf0)
            decision["probe_conf_180"] = float(conf180)

            # Add a small margin to avoid flip oscillation on near-tie confidence.
            if conf180 > conf0 + 0.02:
                oriented = cv2.rotate(oriented, cv2.ROTATE_180)
                decision["probe_flip_180"] = True
        except Exception:
            # Never fail OCR because orientation probing failed.
            pass

        return oriented, decision

    def _run_with_preprocessed(
        preprocessed_img: np.ndarray,
    ) -> tuple[list, np.ndarray, dict]:
        """Run detection + inference. Returns (results, preprocessed, metadata)."""
        line_crops, detector_backend = _detect_main_crops(preprocessed_img)
        num_regions = len(line_crops)

        # Webcam word mode: join words into single result
        if use_webcam_config and line_crops:
            raw_words = []
            confidences = []
            for _crop, bbox in line_crops:
                square = _extract_crop_for_bbox(preprocessed_img, bbox)
                prepared = _prepare_crop_image_for_trocr(square)
                raw_text, conf = predict_with_confidence(
                    prepared, num_beams=num_beams, max_length=16
                )
                raw_words.append(raw_text.strip())
                confidences.append(conf)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            raw_joined = " ".join(raw_words)
            corrected_joined, corrected_flag = _post_process_text(raw_joined)
            meta = {
                "fallback_used": False,
                "num_regions": num_regions,
                "projection_chunking_used": False,
                "detector_backend": detector_backend,
                "spell_corrected": corrected_flag,
            }
            joined_crop = crop_to_square(preprocessed_img)
            return (
                [
                    _build_result(
                        text=corrected_joined,
                        raw_text=raw_joined,
                        confidence=avg_conf,
                        bbox=None,
                        spell_corrected=corrected_flag,
                        crop=joined_crop,
                    )
                ],
                preprocessed_img,
                meta,
            )

        # Fallback path: if 0 or 1 regions detected, try projection chunking first.
        if len(line_crops) <= 1:
            projection_boxes = _projection_line_boxes(preprocessed_img)
            if len(projection_boxes) >= 2:
                crops_prepared = []
                squares = []
                for bbox in projection_boxes:
                    square = _extract_crop_for_bbox(preprocessed_img, bbox)
                    prepared = _prepare_crop_image_for_trocr(square)
                    squares.append(square)
                    crops_prepared.append(prepared)

                adaptive_beams = min(num_beams, max(_compute_num_beams(bbox[2]) for bbox in projection_boxes))
                batch_outputs = batch_predict_with_confidence(
                    crops_prepared, num_beams=adaptive_beams, max_length=max_length
                )

                chunk_results = []
                for i, bbox in enumerate(projection_boxes):
                    raw_text, confidence = batch_outputs[i]
                    corrected_text, corrected_flag = _post_process_text(raw_text)
                    chunk_results.append(
                        _build_result(
                            text=corrected_text,
                            raw_text=raw_text,
                            confidence=confidence,
                            bbox=bbox,
                            spell_corrected=corrected_flag,
                            crop=squares[i],
                        )
                    )
                meta = {
                    "fallback_used": False,
                    "num_regions": len(projection_boxes),
                    "projection_chunking_used": True,
                    "detector_backend": "projection",
                    "spell_corrected": any(r.spell_corrected for r in chunk_results),
                }
                return (chunk_results, preprocessed_img, meta)

            fallback_crop = crop_to_square(preprocessed_img)
            fallback_prepared = _prepare_crop_image_for_trocr(fallback_crop)
            raw_text, confidence = predict_with_confidence(
                fallback_prepared, num_beams=num_beams, max_length=max_length
            )
            corrected_text, corrected_flag = _post_process_text(raw_text)
            bbox = line_crops[0][1] if line_crops else None
            meta = {
                "fallback_used": True,
                "num_regions": num_regions,
                "projection_chunking_used": False,
                "detector_backend": detector_backend,
                "spell_corrected": corrected_flag,
            }
            return (
                [
                    _build_result(
                        text=corrected_text,
                        raw_text=raw_text,
                        confidence=confidence,
                        bbox=bbox,
                        spell_corrected=corrected_flag,
                        crop=fallback_crop,
                    )
                ],
                preprocessed_img,
                meta,
            )

        # Multi-line: recognise crops in batch
        crops_prepared = []
        squares = []
        for _crop, bbox in line_crops:
            square = _extract_crop_for_bbox(preprocessed_img, bbox)
            prepared = _prepare_crop_image_for_trocr(square)
            squares.append(square)
            crops_prepared.append(prepared)

        adaptive_beams = min(num_beams, max(_compute_num_beams(bbox[2]) for _crop, bbox in line_crops)) if line_crops else num_beams
        batch_outputs = batch_predict_with_confidence(
            crops_prepared, num_beams=adaptive_beams, max_length=max_length
        )

        results = []
        for i, (_crop, bbox) in enumerate(line_crops):
            raw_text, confidence = batch_outputs[i]
            corrected_text, corrected_flag = _post_process_text(raw_text)
            results.append(
                _build_result(
                    text=corrected_text,
                    raw_text=raw_text,
                    confidence=confidence,
                    bbox=bbox,
                    spell_corrected=corrected_flag,
                    crop=squares[i],
                )
            )
        meta = {
            "fallback_used": False,
            "num_regions": num_regions,
            "projection_chunking_used": False,
            "detector_backend": detector_backend,
            "spell_corrected": any(r.spell_corrected for r in results),
        }
        return (results, preprocessed_img, meta)

    # Confidence-based retry: Otsu -> adaptive -> raw
    global _preprocess_cache
    img_id = id(image_np)
    RETRY_MODES = ["adaptive", "otsu", "raw"] if use_webcam_config else ["otsu", "adaptive", "raw"]
    best_results: list | None = None
    best_preprocessed: np.ndarray | None = None
    best_metadata: dict = {
        "fallback_used": False,
        "num_regions": 0,
        "projection_chunking_used": False,
        "detector_backend": "opencv",
        "detector_requested": requested_detector_backend,
        "use_yolo_detector": enable_yolo_detector,
        "yolo_model_found": yolo_model_found,
        "apply_spell_check": apply_spell_check,
        "spell_check_available": spell_check_available,
        "spell_corrected": False,
        "auto_orient_enabled": auto_orient,
        "inference_runtime": _current_runtime,
    }
    best_avg_conf = -1.0
    orientation_decision: dict | None = None

    for binarization_mode in RETRY_MODES:
        if binarization_mode == "otsu" and _preprocess_cache is not None and _preprocess_cache[0] == img_id:
            preprocessed = _preprocess_cache[1]
        else:
            preprocessed = preprocess_for_trocr(image_np, binarization_mode=binarization_mode)
            if binarization_mode == "otsu":
                _preprocess_cache = (img_id, preprocessed)

        if auto_orient:
            if orientation_decision is None:
                oriented_preprocessed, orientation_decision = _auto_orient_preprocessed(preprocessed)
            else:
                oriented_preprocessed = _apply_orientation_decision(preprocessed, orientation_decision)
        else:
            oriented_preprocessed = preprocessed

        run_results, run_preprocessed, run_metadata = _run_with_preprocessed(oriented_preprocessed)
        run_metadata = {
            **run_metadata,
            "detector_requested": requested_detector_backend,
            "use_yolo_detector": enable_yolo_detector,
            "yolo_model_found": yolo_model_found,
            "apply_spell_check": apply_spell_check,
            "spell_check_available": spell_check_available,
            "inference_runtime": _current_runtime,
        }
        if orientation_decision is not None:
            run_metadata = {**run_metadata, **orientation_decision}

        avg_conf = sum(r.confidence for r in run_results) / len(run_results) if run_results else 0.0

        if avg_conf > best_avg_conf:
            best_avg_conf = avg_conf
            best_results = run_results
            best_preprocessed = run_preprocessed
            best_metadata = run_metadata

        if avg_conf >= 0.4:
            break

    final_preprocessed = best_preprocessed if best_preprocessed is not None else preprocessed
    return _maybe_return(best_results or [], final_preprocessed, best_metadata)


def set_model(size_or_path: str) -> None:
    """Switch to a different TrOCR model at runtime.

    Unloads the current model and loads the new one. This is called by
    the GUI when the user selects a different model from the dropdown.

    Args:
        size_or_path: One of "small", "base", "large", or a local directory path.
    """
    global _processor, _model, _device, _current_model_id, _current_runtime

    resolved_id = _resolve_model_id(size_or_path)

    with _model_lock:
        # No-op when the requested model is already active.
        if _processor is not None and _model is not None and _current_model_id == resolved_id:
            return

        # Free old model memory before switching.
        _processor = None
        _model = None
        _device = None
        _current_model_id = None
        _current_runtime = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load the new model under the same lock to avoid race conditions.
        _load_model(resolved_id)


def get_available_models() -> list[str]:
    """Return a list of available model identifiers.

    Includes the standard sizes ("small", "base", "large") and any
    detected fine-tuned checkpoint in CHECKPOINT_DIR.
    """
    available = list(MODELS.keys())

    local_ckpt = os.path.join(CHECKPOINT_DIR, "model")
    if os.path.isdir(local_ckpt):
        available.append("fine-tuned (local)")

    return available


def get_current_model() -> str | None:
    """Return the currently loaded model identifier, or None if no model is loaded."""
    return _current_model_id


def get_current_runtime() -> str | None:
    """Return loaded runtime backend ('pytorch' or 'onnx'), or None."""
    return _current_runtime
