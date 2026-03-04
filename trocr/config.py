"""
config.py — TrOCR model registry, defaults, and paths.

All model configuration lives here so swapping models or
adjusting generation parameters requires changing one file.
"""

# ---------------------------------------------------------------------------
# Pre-trained model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "small": "microsoft/trocr-small-handwritten",
    "base":  "microsoft/trocr-base-handwritten",
    "large": "microsoft/trocr-large-handwritten",
}

DEFAULT_MODEL: str = "base"
DEFAULT_MODEL_WEBCAM: str = "small"

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
DEFAULT_NUM_BEAMS: int = 4
DEFAULT_MAX_LENGTH: int = 64

# ---------------------------------------------------------------------------
# Local checkpoint directory (for future fine-tuned models)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR: str = "weights/trocr"
