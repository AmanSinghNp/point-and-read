import cv2
import numpy as np
import torch
import torchvision.transforms as T
import math
from PIL import Image as PILImage, ImageOps


def apply_exif_orientation(image_input) -> np.ndarray:
    """Apply EXIF orientation correction to an image.

    Phone cameras store rotation in EXIF metadata, but cv2.imread ignores it.
    This reads the EXIF tag and physically rotates the pixel data so the image
    is upright before any further processing.

    Args:
        image_input: A file path (str) or a BGR numpy array.
            If a file path, EXIF data is read from disk.
            If a numpy array, no EXIF data is available; returned unchanged.

    Returns:
        BGR numpy array with correct orientation applied.
    """
    if isinstance(image_input, str):
        try:
            pil_img = PILImage.open(image_input)
            # exif_transpose reads the EXIF Orientation tag and rotates/mirrors
            # the image to match, then strips the tag so it isn't applied twice.
            pil_img = ImageOps.exif_transpose(pil_img)
            # Convert to BGR numpy array for OpenCV pipeline
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            # If PIL can't read EXIF, fall back to cv2.imread (no correction)
            img = cv2.imread(image_input)
            return img if img is not None else np.array([])
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        return image_input


def remove_shadows(
    image: np.ndarray,
    *,
    dilate_kernel_size: int = 15,
    blur_kernel_size: int = 21,
) -> np.ndarray:
    """Flatten uneven lighting and heavy shadows.

    Uses background estimation:
      1. Heavily dilate to suppress thin text strokes.
      2. Median blur to obtain a smooth illumination map.
      3. Invert abs-difference and normalize contrast.

    Args:
        image: Grayscale or BGR uint8 image.
        dilate_kernel_size: Square dilation kernel size.
        blur_kernel_size: Median blur kernel size (must be odd).

    Returns:
        Shadow-corrected grayscale uint8 image.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    dks = max(3, int(dilate_kernel_size))
    bks = max(3, int(blur_kernel_size))
    if bks % 2 == 0:
        bks += 1

    dilated_bg = cv2.dilate(
        gray,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dks, dks)),
    )
    bg_map = cv2.medianBlur(dilated_bg, bks)
    diff = 255 - cv2.absdiff(gray, bg_map)
    normalized = cv2.normalize(
        diff,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )
    return normalized


def align_text_axis_minarearect(
    image: np.ndarray,
    *,
    kernel_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Rotate image by 90 degrees when the main text block is vertical.

    Workflow:
      1. Otsu-binarize (text as foreground).
      2. Apply heavy dilation so nearby words merge into one paragraph-like blob.
      3. Find largest contour and estimate dominant axis from minAreaRect.
      4. If dominant axis is vertical, rotate image 90 degrees clockwise.

    Args:
        image: Grayscale or BGR uint8 image.
        kernel_size: Optional fixed dilation kernel (w, h). If None, a
            dynamic large kernel is derived from image dimensions.

    Returns:
        (aligned_image, metadata)
        metadata keys:
            axis_rotation_degrees: 0 or 90
            axis_rotated_90: bool
            axis_rect_w: float
            axis_rect_h: float
            axis_rect_angle: float
            axis_detected: bool
    """
    meta = {
        "axis_rotation_degrees": 0,
        "axis_rotated_90": False,
        "axis_rect_w": 0.0,
        "axis_rect_h": 0.0,
        "axis_rect_angle": 0.0,
        "axis_long_edge_angle": 0.0,
        "axis_detected": False,
    }

    if image is None or image.size == 0:
        return image, meta

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = gray.shape
    if kernel_size is None:
        # Big kernel to merge words into one paragraph blob.
        kx = max(25, w // 12)
        ky = max(25, h // 12)
    else:
        kx = max(3, int(kernel_size[0]))
        ky = max(3, int(kernel_size[1]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    merged = cv2.dilate(binary_inv, kernel, iterations=1)

    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image.copy(), meta

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return image.copy(), meta

    rect = cv2.minAreaRect(largest)
    (_, _), (rw, rh), angle = rect
    meta["axis_rect_w"] = float(rw)
    meta["axis_rect_h"] = float(rh)
    meta["axis_rect_angle"] = float(angle)
    meta["axis_detected"] = True

    # Determine dominant axis from the longest edge direction of the box.
    box = cv2.boxPoints(rect)
    longest_len = -1.0
    longest_angle = 0.0
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        v = p2 - p1
        length = float(np.linalg.norm(v))
        if length > longest_len:
            longest_len = length
            longest_angle = float(np.degrees(np.arctan2(v[1], v[0])))

    # Normalize angle into [0, 180) so horizontal ~0/180 and vertical ~90.
    dominant_angle = (longest_angle + 180.0) % 180.0
    meta["axis_long_edge_angle"] = dominant_angle
    is_vertical = 45.0 <= dominant_angle <= 135.0

    if is_vertical:
        meta["axis_rotation_degrees"] = 90
        meta["axis_rotated_90"] = True
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), meta

    return image.copy(), meta


class Preprocessor:
    def __init__(self, target_height=64, is_training=False, apply_deskew=True):
        """
        Initializes the preprocessing pipeline.

        Args:
            target_height (int): The fixed height to resize all images to.
            is_training (bool): Whether to apply data augmentation.
            apply_deskew (bool): If True, run deskew (good for real-world photos).
                Set False for IAM and other already-cropped line images to avoid over-rotation.
        """
        self.target_height = target_height
        self.is_training = is_training
        self.apply_deskew = apply_deskew
        
        # ImageNet stats for standard normalization (though we pass 1 channel, it helps)
        self.mean = 0.485  # Using the first channel of ImageNet mean
        self.std = 0.229   # Using the first channel of ImageNet std
        
    def deskew(self, image):
        """
        Computes the dominant text angle and rotates to correct it.
        """
        # Compute projection profile based angle (simple deskew via moments or cv2.minAreaRect)
        # For handwriting, detecting the angle can be noisy. A robust way is finding contours.
        # Alternatively, we can use the hough transform on edges.
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        angle = 0.0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                # Calculate angle in degrees
                a = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
                # Filter out extreme vertical lines (likely borders, not text lines)
                if -45 < a < 45:
                    angles.append(a)
            
            if len(angles) > 0:
                angle = np.median(angles)
        
        if abs(angle) > 0.5: # Only rotate if the angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Use white background for rotation padding (since Otsu makes text white on black later,
            # we want the background to be distinct, but standard is white for paper).
            # We assume grayscale image (0-255). We will pad with 255 (white).
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255))
            
        return image

    def augment(self, image):
        """
        Applies random training augmentations.
        """
        h, w = image.shape
        
        # 1. Random rotation: +/- 3 degrees
        angle = np.random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        
        # 2. Random horizontal stretch (scaling width)
        stretch_factor = np.random.uniform(0.9, 1.2)
        new_w = int(w * stretch_factor)
        image = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_CUBIC)
        
        # 3. Brightness/contrast jitter (simulated via gamma/linear transforms since we're in grayscale)
        alpha = np.random.uniform(0.8, 1.2) # Contrast
        beta = np.random.randint(-20, 20)   # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 4. Gaussian noise
        if np.random.rand() > 0.5:
            # Generate noise with same shape
            noise = np.random.normal(0, 10, image.shape).astype(np.float32)
            noisy_image = cv2.add(image.astype(np.float32), noise)
            image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
        return image

    def process_array(self, image):
        """
        Preprocessing pipeline for a numpy array (grayscale uint8).

        Args:
            image: numpy array, grayscale (H, W) or BGR (H, W, 3).

        Returns:
            Preprocessed tensor of shape (1, target_height, W').
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image array")

        # Convert BGR to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Deskew (optional)
        if self.apply_deskew:
            image = self.deskew(image)

        # Apply augmentations if training
        if self.is_training:
            image = self.augment(image)

        # 3. Denoise
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # 4. Binarize (Otsu)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 5. Normalize height to 64px, keeping aspect ratio
        h, w = image.shape
        aspect_ratio = w / h
        new_w = int(self.target_height * aspect_ratio)
        new_w = max(1, new_w)

        image = cv2.resize(image, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        # 6. Normalize pixel values & Convert to tensor
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        # 7. Apply mean/std normalization
        normalize = T.Normalize([self.mean], [self.std])
        tensor = normalize(tensor)

        return tensor

    def process(self, image_path):
        """
        Full preprocessing pipeline from file path.
        """
        # 1. Load in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        return self.process_array(image)


def strip_dark_borders(image: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Remove black/near-black borders (e.g. from DroidCam) before detection.

    Dark borders corrupt adaptive threshold and cause wrong region detection.
    Call this first in the preprocessing pipeline.

    Args:
        image: BGR or grayscale numpy array.
        threshold: Pixels below this are considered black. Default 30.

    Returns:
        Cropped image with dark borders removed, or original if no content found.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    return image[y1:y2, x1:x2]


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Flatten perspective for photos taken at an angle using document contour detection.

    Finds the largest quad contour and warps to a rectangle. Returns image unchanged
    if no valid document outline is found.

    Args:
        image: BGR or grayscale numpy array.

    Returns:
        Perspective-corrected image (same type as input).
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area descending, find largest quad
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h_img, w_img = image.shape[:2]
    min_area = 0.1 * h_img * w_img  # Reject very small contours

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            src_pts = np.float32(approx.reshape(4, 2))
            # Order corners: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype=np.float32)
            s = src_pts.sum(axis=1)
            rect[0] = src_pts[np.argmin(s)]
            rect[2] = src_pts[np.argmax(s)]
            diff = np.diff(src_pts, axis=1)
            rect[1] = src_pts[np.argmin(diff)]
            rect[3] = src_pts[np.argmax(diff)]
            w = max(
                np.linalg.norm(rect[1] - rect[0]),
                np.linalg.norm(rect[2] - rect[3]),
            )
            h = max(
                np.linalg.norm(rect[2] - rect[1]),
                np.linalg.norm(rect[3] - rect[0]),
            )
            dst_pts = np.float32([
                [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
            ])
            M = cv2.getPerspectiveTransform(rect, dst_pts)
            if len(image.shape) == 3:
                return cv2.warpPerspective(image, M, (int(w), int(h)))
            return cv2.warpPerspective(image, M, (int(w), int(h)))

    return image


def preprocess_for_trocr(
    image: np.ndarray,
    *,
    apply_perspective: bool = True,
    apply_shadow_removal: bool = True,
    apply_deskew: bool = True,
    apply_clahe: bool = True,
    binarization_mode: str = "otsu",
) -> np.ndarray:
    """
    Preprocess image for TrOCR: perspective correction, deskew, denoise, binarize.

    Input: BGR (H,W,3) or grayscale (H,W). Output: grayscale uint8 (H,W)
    suitable for predictor._normalise_image (which stacks to RGB).

    Args:
        image: BGR or grayscale numpy array.
        apply_perspective: If True, try to flatten document perspective first.
        apply_shadow_removal: If True, normalize uneven illumination/shadows.
        apply_deskew: If True, run Hough-line deskew (critical for webcam shots).
        apply_clahe: If True, apply CLAHE for contrast normalization before Otsu.
        binarization_mode: "otsu" (default), "adaptive" (blockSize=15, C=5),
            or "raw" (no binarization, pass CLAHE-enhanced grayscale).

    Returns:
        Preprocessed grayscale uint8 array (H, W).
    """
    if image is None or image.size == 0:
        raise ValueError("Empty or invalid image array")

    # Strip black borders first (e.g. DroidCam) — before any other processing
    image = strip_dark_borders(image)

    # Perspective correction (then deskew)
    if apply_perspective:
        image = correct_perspective(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Shadow/illumination correction before thresholding.
    if apply_shadow_removal:
        gray = remove_shadows(gray)

    # Optional contrast enhancement before binarization
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Deskew (critical for angled paper)
    if apply_deskew:
        prep = Preprocessor(apply_deskew=True, is_training=False)
        gray = prep.deskew(gray)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarization
    if binarization_mode == "otsu":
        _, output = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif binarization_mode == "adaptive":
        output = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=5,
        )
    elif binarization_mode == "raw":
        output = gray
    else:
        raise ValueError(
            f"binarization_mode must be 'otsu', 'adaptive', or 'raw', got {binarization_mode!r}"
        )

    return output


if __name__ == "__main__":
    # Test the preprocessor
    import os
    
    # Create a dummy image
    dummy_img = np.ones((200, 800), dtype=np.uint8) * 255
    cv2.putText(dummy_img, "Test OpenCV Preprocessing", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Needs a real path to test cv2.imread
    os.makedirs("tests", exist_ok=True)
    test_path = "tests/test_preproc.jpg"
    cv2.imwrite(test_path, dummy_img)
    
    preprocessor = Preprocessor(target_height=64, is_training=True)
    try:
        tensor = preprocessor.process(test_path)
        print(f"Preprocessed tensor shape: {tensor.shape}")
        print(f"Tensor min: {tensor.min():.2f}, max: {tensor.max():.2f}")
    except Exception as e:
        print(f"Error: {e}")
