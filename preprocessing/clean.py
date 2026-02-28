import cv2
import numpy as np
import torch
import torchvision.transforms as T
import math

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

    def process(self, image_path):
        """
        Full preprocessing pipeline.
        """
        # 1. Load in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # 2. Deskew (optional; disable for IAM to avoid over-rotation)
        if self.apply_deskew:
            image = self.deskew(image)

        # Apply augmentations if training (before blurring/binarization to simulate real capture noise)
        if self.is_training:
            image = self.augment(image)
        
        # 3. Denoise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # 4. Binarize (Otsu) -> Text becomes white on black background
        # By default cv2.THRESH_BINARY makes dark text black.
        # THRESH_BINARY_INV makes dark text white, which CRNN prefers (dark background).
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 5. Normalize height to 64px, keeping aspect ratio
        h, w = image.shape
        aspect_ratio = w / h
        new_w = int(self.target_height * aspect_ratio)
        # Ensure minimum width of 1 and avoid width collapsing to 0
        new_w = max(1, new_w)
        
        image = cv2.resize(image, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
        
        # 6. Normalize pixel values & Convert to tensor
        # Scale 0-255 to 0.0-1.0
        image = image.astype(np.float32) / 255.0
        
        # Create tensor (Channels, Height, Width)
        tensor = torch.from_numpy(image).unsqueeze(0) # (1, H, W)
        
        # 7. Apply mean/std normalization
        normalize = T.Normalize([self.mean], [self.std])
        tensor = normalize(tensor)
        
        return tensor

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
