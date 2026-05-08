import cv2
import numpy as np
from PIL import Image, ImageOps


class ImageUtils:
    @staticmethod
    def pil_to_cv(image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image)
        rgb = image.convert("RGB")
        array = np.array(rgb)

        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv_to_pil(image: np.ndarray) -> Image.Image:
        if len(image.shape) == 2:
            return Image.fromarray(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(rgb)

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = Image.open(path)

        return ImageUtils.pil_to_cv(image)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def normalize_original(image: np.ndarray, max_side: int = 3200) -> np.ndarray:
        return ImageUtils.resize_if_too_large(image, max_side=max_side)

    @staticmethod
    def resize_if_too_large(
        image: np.ndarray,
        max_side: int = 3200,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)

        if longest <= max_side:
            return image

        scale = max_side / longest
        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=8,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=6,
            hColor=6,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)

        clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8),
        )

        return clahe.apply(gray)

    @staticmethod
    def soft_enhance_grayscale(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)

        clahe = cv2.createCLAHE(
            clipLimit=1.4,
            tileGridSize=(8, 8),
        )

        return clahe.apply(gray)

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            13,
        )

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresholded = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return thresholded

    @staticmethod
    def remove_small_noise(binary_image: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)

        return cv2.morphologyEx(
            binary_image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

    @staticmethod
    def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
        gray = ImageUtils.to_grayscale(image)

        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 100:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.25 or abs(angle) > 10:
            return image, 0.0

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated, float(angle)

    @staticmethod
    def encode_png(image: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".png", image)

        if not success:
            raise RuntimeError("Could not encode image as PNG")

        return buffer.tobytes()